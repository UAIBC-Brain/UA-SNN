import argparse
import datetime
import os
import sys
import time

import torch
import torch.nn.functional as F
from losses import Lossess
from monai.data import DataLoader
from spikingjelly.clock_driven import functional
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from utils.new_jsaon_data_utils import AgeData
from utils.runtime import build_model, evaluate, get_evidence


def parse_args():
    parser = argparse.ArgumentParser(description="UA-SNN Training")
    parser.add_argument("--split-json", default="./jsons/old/data_example.json")
    parser.add_argument("--root-dir", default="./dataset/data_round")
    parser.add_argument("--fold-index", default=1, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--device", "-device", default="cuda:0")
    parser.add_argument("--out-dir", "-out-dir", default="./logs")
    parser.add_argument("--batch-size", "--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", "--weight_decay", default=0.0, type=float)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--model", default="spike_basd_transformer")
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--num-heads", default=8, type=int)
    parser.add_argument("--in-channels", "--in_channels", default=1, type=int)
    parser.add_argument("--T", default=4, type=int)
    parser.add_argument("--drop", default=0.1, type=float)
    parser.add_argument("--opt", default="sgd", choices=("sgd", "adam"))
    parser.add_argument("--clf-type", "--clf_type", default="softplus", choices=("softplus", "exp"))
    parser.add_argument("--lambda-1", "--fisher_c", dest="lambda_1", default=0.01, type=float)
    parser.add_argument("--lambda-2", "--kl_c", dest="lambda_2", default=1.0, type=float)
    parser.add_argument("--u-base", default=0.95, type=float)
    parser.add_argument("--eta", default=0.2, type=float)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.lambda_1 < 0 or args.lambda_2 < 0:
        parser.error("lambda_1 and lambda_2 must be non-negative")
    return args


def main():
    args = parse_args()
    print(args)

    train_dataset = AgeData(args.split_json, args.root_dir, split="training")
    validation_dataset = AgeData(args.split_json, args.root_dir, split="validation")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=1)

    model = build_model(args).to(args.device)
    if args.opt == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    amp_enabled = args.amp and args.device.startswith("cuda")
    scaler = amp.GradScaler(enabled=amp_enabled)

    output_dir = os.path.join(
        args.out_dir,
        f"fold{args.fold_index}_lambda1_{args.lambda_1}_lambda2_{args.lambda_2}",
    )
    os.makedirs(output_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(output_dir, "checkpoint_best_validation.pth")
    latest_checkpoint_path = os.path.join(output_dir, "checkpoint_latest.pth")
    writer = SummaryWriter(output_dir)
    start_epoch = 0
    best_validation_acc = -1.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation_acc = checkpoint.get("best_validation_acc", -1.0)

    criterion = Lossess()
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_data in train_loader:
            image = batch_data["image"].unsqueeze(1).float().to(args.device)
            label = batch_data["label"].view(-1).long().to(args.device)
            label_onehot = F.one_hot(label, args.num_classes).float()
            optimizer.zero_grad()

            with amp.autocast(enabled=amp_enabled):
                output_sum = None
                for _ in range(args.T):
                    current_output = model(image)
                    output_sum = current_output if output_sum is None else output_sum + current_output
                output = output_sum / args.T
                evidence = get_evidence(output, args.clf_type)
                loss = criterion(
                    evidence,
                    output,
                    label,
                    args.lambda_1,
                    args.lambda_2,
                    label_onehot,
                    return_output="loss",
                    compute_loss=True,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = label.numel()
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_correct += (output.argmax(dim=1) == label).sum().item()
            functional.reset_net(model)

        scheduler.step()
        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        validation_metrics = evaluate(model, validation_loader, args)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("validation/acc", validation_metrics["acc"], epoch)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_validation_acc": max(best_validation_acc, validation_metrics["acc"]),
        }
        torch.save(checkpoint, latest_checkpoint_path)
        if validation_metrics["acc"] >= best_validation_acc:
            best_validation_acc = validation_metrics["acc"]
            checkpoint["best_validation_acc"] = best_validation_acc
            torch.save(checkpoint, best_checkpoint_path)

        elapsed = time.time() - start_time
        finish_time = datetime.datetime.now() + datetime.timedelta(
            seconds=elapsed * (args.epochs - epoch - 1)
        )
        print(
            f"epoch={epoch}, train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"validation_acc={validation_metrics['acc']:.4f}, finish={finish_time:%Y-%m-%d %H:%M:%S}"
        )

    with open(os.path.join(output_dir, "args.txt"), "w", encoding="utf-8") as file:
        file.write(" ".join(sys.argv))
    writer.close()


if __name__ == "__main__":
    main()
