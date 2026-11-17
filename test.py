import argparse
import json
import os

import torch
from monai.data import DataLoader
from utils.new_jsaon_data_utils import AgeData
from utils.runtime import build_model, evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="UA-SNN Evaluation")
    parser.add_argument("--split-json", default="./jsons/old/data_example.json")
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="Partition in the split JSON to evaluate",
    )
    parser.add_argument("--root-dir", default="./dataset/data_round")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", "-device", default="cuda:0")
    parser.add_argument("--model", default="spike_basd_transformer")
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--num-heads", default=8, type=int)
    parser.add_argument("--in-channels", "--in_channels", default=1, type=int)
    parser.add_argument("--T", default=4, type=int)
    parser.add_argument("--drop", default=0.1, type=float)
    parser.add_argument("--clf-type", "--clf_type", default="softplus", choices=("softplus", "exp"))
    parser.add_argument("--u-base", default=0.95, type=float)
    parser.add_argument("--eta", default=0.2, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = AgeData(args.split_json, args.root_dir, split=args.split)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    model = build_model(args).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    metrics = evaluate(model, data_loader, args)
    print(
        f"ACC={metrics['acc']:.4f}, SEN={metrics['sen']:.4f}, "
        f"SPE={metrics['spe']:.4f}, F1={metrics['f1']:.4f}, Avg.T={metrics['avg_t']:.4f}"
    )
    output_path = args.output or os.path.join(
        os.path.dirname(args.checkpoint), f"{args.split}_results.json"
    )
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


if __name__ == "__main__":
    main()
