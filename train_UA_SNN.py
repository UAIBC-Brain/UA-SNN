import os
import sys
import time
import torch
import argparse
import datetime
import model_test
import torch.nn as nn
from torch.cuda import amp
from losses import Lossess
import torch.nn.functional as F
from monai.data import  DataLoader
from timm.models import create_model
# from utils.new_jsaon_data_utils import AgeData
from utils.new_jsaon_data_utils import AgeData
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.clock_driven import neuron, functional, surrogate, layer

def main():

    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--epochs", default=400, type=int, help="number of training epochs")
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument('-amp', action='store_true', default=True, help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', default=True, help='use cupy backend')
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--batch_size", default=8, type=int, help="number of batch size")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--decay", default=0.1, type=float, help="decay rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--clf_type", default='softplus', type=str, help="the clf_type contains softplus and exp")
    parser.add_argument("--kl_c", default=-1, type=int, help="kl's weight")
    parser.add_argument("--fisher_c", default=0.01, type=int, help="fisher_c's weight")
    parser.add_argument("--weight_decay", type=float, default=0, metavar="W",help="weight decay (default: 0.1)")
    parser.add_argument('-save-es', default=None,help='dir for saving a batch spikes encoded by the first {Conv2d-BatchNorm2d-IFNode}')
    parser.add_argument( "--model",default="spike_basd_transformer",type=str,metavar="MODEL",help='Name of model to train (default: "spikformer")')
    parser.add_argument("--pooling-stat",default="1111",type=str,help="pooling layers in SPS moduls")
    parser.add_argument("--spike-mode",default="lif",type=str,help="")
    parser.add_argument("--layer",default=4,type=int,help="")
    parser.add_argument("--num-classes",type=int,default=2,metavar="N",help="number of label classes (Model default if None)")
    parser.add_argument("--T",type=int,default=4,metavar="N",help="")
    parser.add_argument( "--num-heads",type=int,default=8,metavar="N",help="")
    parser.add_argument("--opt",default="sgd",type=str,metavar="OPTIMIZER",help='Optimizer (default: "sgd")')
    parser.add_argument("--drop", type=float, default=0.1, metavar="PCT", help="Dropout rate (default: 0.)")
    parser.add_argument("--drop-path",type=float,default=0.2,metavar="PCT",help="Drop path rate (default: None)")
    parser.add_argument("--drop-block",type=float,default=None,metavar="PCT",help="Drop block rate (default: None)")
    args = parser.parse_args()
    print(args)

    model = create_model(
        args.model,
        img_size_d=128,  # 新增深度维度的尺寸
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=1,
        num_classes=2,
        embed_dims=64,
        num_heads=8,
        mlp_ratios=4,
        T=args.T,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        depths=8,
        sr_ratios=1,
        pretrained=False,
        pretrained_cfg=None,
    )
    print(model)
    model.to(args.device)

    #dataset
    json_file = "./jsons/old/dataset_mat - spilt3.json"
    root_dir = "./dataset/data_round"
    train_dataset = AgeData(json_file, root_dir, split='training', transform=None)
    train_loader = DataLoader(train_dataset, batch_size=15, shuffle=True, num_workers=1)

    test_dataset = AgeData(json_file, root_dir, split='validation', transform=None)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=1)


    scaler = None
    if args.amp:
        scaler = amp.GradScaler()
    start_epoch = 0
    max_test_acc = -1
    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    #pretrain
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    # logs
    out_dir = os.path.join(args.out_dir,
                           f'./pjs_new/T{args.T}_{args.batch_size}_{args.opt}_lr{args.lr}_c{args.in_channels}_drop{args.drop}_json3_{args.fisher_c}')

    if args.amp:
        out_dir += '_amp'
    if args.cupy:
        out_dir += '_cupy'

    # save result
    output_file_path = os.path.join(out_dir, 'resultss.txt')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')
    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    # pretrained_path = r'./logs/jsb/checkpoint_latest.pth'
    # # pretrained_path = r'./logs/T4_1_sgd_lr0.01_c1_drop0.1_fishe and data_split5_test11111110.01_amp_cupy/checkpoint_max.pth'
    # if os.path.exists(pretrained_path):
    #     checkpoint = torch.load(pretrained_path, map_location=args.device)
    #     if 'model' in checkpoint:
    #         model.load_state_dict(checkpoint['model'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #         epoch = checkpoint['epoch']
    #         max_test_acc = checkpoint['max_test_acc']
    #         print(f"成功加载预训练模型及相关信息: {pretrained_path}")
    #     else:
    #         model.load_state_dict(checkpoint)
    #         print(f"加载旧格式的预训练模型: {pretrained_path}")
    # else:
    #     raise FileNotFoundError(f"预训练模型未找到: {pretrained_path}")

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        model.train()
        criterion = Lossess()
        train_total_loss = 0
        train_acc = 0
        train_samples = 0
        mse_loss = 0
        var_loss = 0
        kl_loss = 0
        fisher_loss = 0
        softplus_ = nn.Softplus()
        clf_type = args.clf_type
        epochss = args.epochs
        fisher_c = args.fisher_c
        kl_c = args.kl_c

        for i_batch, batch_data in enumerate(train_loader):
            print(i_batch)
            image = batch_data['image']
            image = image.unsqueeze(1)
            image = image.to(torch.half)
            label = batch_data['label']
            label = torch.squeeze(label)
            label = label.long()
            image, label = image.cuda(args.rank), label.cuda(args.rank)
            optimizer.zero_grad()
            image = image.to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 2).float()

            if scaler is not None:
                with amp.autocast():
                   out_fr = 0
                   image = (image.unsqueeze(0)).repeat(args.T, 1, 1, 1, 1, 1)
                   for t in range(args.T):
                     image_x = image[t]
                     out_fr += model(image_x)
                     out_fr = out_fr.float()
                   out_fr = out_fr / args.T
                   if clf_type == "exp":
                        evi_alp_ = torch.exp(out_fr) + 1.0
                   elif clf_type == "softplus":
                        evi_alp_ = softplus_(out_fr) + 1.0
                   else:
                       raise NotImplementedError
                   criterion(evi_alp_, out_fr, label, fisher_c, kl_c, label_onehot, epochss, compute_loss=True)
                   grad_loss = criterion.grad_loss
                scaler.scale(grad_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = model(image)
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            # print('训练总样本数',train_samples)
            train_total_loss += grad_loss.item() * label.numel()
            mse_loss += criterion.loss_mse_.item() * label.numel()
            var_loss += criterion.loss_var_.item() * label.numel()
            kl_loss += criterion.loss_kl_.item() * label.numel()
            fisher_loss += criterion.loss_fisher_.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()
            # print('训练好的样本数', train_acc)

            functional.reset_net(model)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_total_loss /= train_samples
        mse_loss /= train_samples
        var_loss /= train_samples
        kl_loss /= train_samples
        fisher_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_total_loss', train_total_loss, epoch)
        writer.add_scalar('mse_loss', mse_loss, epoch)
        writer.add_scalar('var_loss', var_loss, epoch)
        writer.add_scalar('kl_loss', kl_loss, epoch)
        writer.add_scalar('fisher_loss', fisher_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        model.eval()
        test_acc = 0
        test_samples = 0
        noise_epsilon = 0.0
        test_total_loss = 0
        mse_loss1 = 0
        right_lable0 = 0
        right_lable1 = 0
        total_label0 = 0
        total_label1 = 0
        var_loss1 = 0
        kl_loss1 = 0
        fisher_loss1 = 0
        softplus_ = nn.Softplus()
        clf_type = args.clf_type
        epochss = args.epochs

        torch.set_printoptions(threshold=float('inf'))  # 设置打印所有元素

        with torch.no_grad():
            for i_batch, batch_data in enumerate(test_loader):
                image = batch_data['image']
                image = image.unsqueeze(1)
                label = batch_data['label']
                label = torch.squeeze(label)
                label = label.long()
                # 将输入张量转换为 FloatTensor
                image = image.float()
                image = (image + noise_epsilon * torch.randn_like(image)).to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 2).float()
                out_fr = 0
                image = (image.unsqueeze(0)).repeat(args.T, 1, 1, 1, 1, 1)
                for t in range(args.T):
                    image_x = image[t]
                    out_fr += model(image_x)
                    out_fr = out_fr.float()
                out_fr = out_fr / args.T
                if clf_type == "exp":
                    evi_alp_ = torch.exp(out_fr) + 1.0
                elif clf_type == "softplus":
                    evi_alp_ = softplus_(out_fr) + 1.0
                else:
                    raise NotImplementedError
                # print('测试输出', out_fr)
                # print('测试标签', label_onehot)
                criterion(evi_alp_, out_fr, label, fisher_c, kl_c, label_onehot, epochss, compute_loss=False)
                grad_loss = criterion.grad_loss
                test_samples += label.numel()
                # print('训练总样本数',train_samples)
                test_total_loss += grad_loss.item() * label.numel()
                mse_loss1 += criterion.loss_mse_.item() * label.numel()
                var_loss1 += criterion.loss_var_.item() * label.numel()
                kl_loss1 += criterion.loss_kl_.item() * label.numel()
                fisher_loss1 += criterion.loss_fisher_.item() * label.numel()
                test_acc += (evi_alp_.argmax(1) == label).float().sum().item()
                predicted_label = evi_alp_.argmax(1)
                right_lable0 += ((label == 0) & (predicted_label == label)).sum().item()
                right_lable1 += ((label == 1) & (predicted_label == label)).sum().item()
                total_label0 += (label == 0).sum().item()
                total_label1 += (label == 1).sum().item()

                functional.reset_net(model)


        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_total_loss /= test_samples
        mse_loss1 /= test_samples
        var_loss1 /= test_samples
        kl_loss1 /= test_samples
        fisher_loss1 /= test_samples
        test_acc /= test_samples
        # 计算 SEN（灵敏度）和 SPE（特异性）
        SEN = right_lable1 / total_label1 if total_label1 > 0 else 0  # 灵敏度 = TP / (TP + FN)
        SPE = right_lable0 / total_label0 if total_label0 > 0 else 0  # 特异性 = TN / (TN + FP)

        # 计算 F1 分数
        precision = right_lable1 / (right_lable1 + (total_label0 - right_lable0)) if (right_lable1 + (
                total_label0 - right_lable0)) > 0 else 0
        recall = SEN
        F1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


        writer.add_scalar('test_total_loss', test_total_loss, epoch)
        writer.add_scalar('mse_loss1', mse_loss1, epoch)
        writer.add_scalar('var_loss1', var_loss1, epoch)
        writer.add_scalar('kl_loss1', kl_loss1, epoch)
        writer.add_scalar('fisher_loss1', fisher_loss1, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)
        writer.add_scalar('test_sensitivity', SEN, epoch)
        writer.add_scalar('test_specificity', SPE, epoch)
        writer.add_scalar('test_f1_score', F1_score, epoch)

        save_max = False
        if test_acc >= max_test_acc:
            max_test_acc = test_acc
            save_max = True

        if test_acc >= 0.6:
            with open(output_file_path, 'a') as f:  # 打开文件，追加模式
                f.write(
                    f'(note_epoch:{epoch},note_acc:{test_acc:.2f},SEN:{SEN:.2f},SPE:{SPE:.2f},F1:{F1_score:.2f}\n')
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        print(args)
        print(out_dir)
        print(f'epoch = {epoch}, train_total_loss ={train_total_loss: .4f},train_acc ={train_acc: .4f}, test_total_loss ={test_total_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')
        print("right_label1:", right_lable1)
        print("right_label0:", right_lable0)
        print(f"SEN (Sensitivity): {SEN:.4f}")
        print(f"SPE (Specificity): {SPE:.4f}")
        print(f"F1 Score: {F1_score:.4f}")


if __name__ == '__main__':
    main()