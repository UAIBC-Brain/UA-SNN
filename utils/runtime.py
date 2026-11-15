import model_test
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional
from timm.models import create_model


def build_model(args):
    return create_model(
        args.model,
        img_size_d=128,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        embed_dims=64,
        num_heads=args.num_heads,
        mlp_ratios=4,
        T=args.T,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=args.drop,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        depths=8,
        sr_ratios=1,
        pretrained=False,
        pretrained_cfg=None,
    )


def get_evidence(logits, clf_type):
    if clf_type == "exp":
        return torch.exp(logits) + 1.0
    return F.softplus(logits) + 1.0


def evaluate(model, data_loader, args):
    model.eval()
    total = 0
    correct = 0
    true_positive = 0
    true_negative = 0
    positive = 0
    negative = 0
    total_time_steps = 0

    with torch.no_grad():
        for batch_data in data_loader:
            image = batch_data["image"].unsqueeze(1).float().to(args.device)
            label = batch_data["label"].view(-1).long().to(args.device)
            output_sum = None
            output = None
            best_time_step = args.T

            for step in range(1, args.T + 1):
                current_output = model(image)
                output_sum = current_output if output_sum is None else output_sum + current_output
                output = output_sum / step
                confidence = F.softmax(output, dim=1).amax(dim=1)
                threshold = args.u_base - args.eta * step / args.T
                if bool(torch.all(confidence > threshold)):
                    best_time_step = step
                    break

            prediction = get_evidence(output, args.clf_type).argmax(dim=1)
            batch_size = label.numel()
            total += batch_size
            correct += (prediction == label).sum().item()
            true_positive += ((label == 1) & (prediction == 1)).sum().item()
            true_negative += ((label == 0) & (prediction == 0)).sum().item()
            positive += (label == 1).sum().item()
            negative += (label == 0).sum().item()
            total_time_steps += best_time_step * batch_size
            functional.reset_net(model)

    false_positive = negative - true_negative
    sensitivity = true_positive / positive if positive else 0.0
    specificity = true_negative / negative if negative else 0.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0

    return {
        "acc": correct / total,
        "sen": sensitivity,
        "spe": specificity,
        "f1": f1,
        "avg_t": total_time_steps / total,
    }
