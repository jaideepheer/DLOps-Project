import sys, os
import importlib
from types import SimpleNamespace
import argparse, torch


parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config", default="socofing_optimized", help="config filename")
parser.add_argument("--model", default=None, type=str, help="args.model_name")
parser.add_argument("--wandb", default=True, type=bool, help="args.wandb")
parser.add_argument("--rootdir", default="/workspace", type=str, help="args.rootdir")
parser.add_argument(
    "--mode",
    default="train",
    type=str,
    help="Choose any of these {train, convert, pytorch, onnx, tensorrt}",
)
parser.add_argument("--distributed", help="Running DDP?", action="store_true")

parser_args, _ = parser.parse_known_args(sys.argv)
# parser_args = parser.parse_args()

print("Using config file", parser_args.config)

args = importlib.import_module(f"src.config.{parser_args.config}").build_args(
    model_name=parser_args.model,
    wandb=parser_args.wandb,
    mode=parser_args.mode,
    root_dir=parser_args.rootdir,
    distributed=parser_args.distributed,
)

if args["device"] == "cpu":
    args["device"] = torch.device("cpu")
else:
    args["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Args:-")
print(args)

args = SimpleNamespace(**args)
os.makedirs(args.out_weight_dir, exist_ok=True)
