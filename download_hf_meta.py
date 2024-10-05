import subprocess
import os
import argparse
from pathlib import Path

meta_file_names = [
    ".gitattributes",
    "LICENSE","README.md","USE_POLICY.md","generation_config.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "config.json",
    ]

parser = argparse.ArgumentParser()
parser.add_argument("--identifier", "-i", type=str)
parser.add_argument("--all", "-a", action="store_true")
args=parser.parse_args()

save_path = Path("/mnt/bs_fs/models/") / args.identifier.split('/')[-1]

if not args.all:
    for fn in meta_file_names:
        os.system(f"huggingface-cli download --local-dir {str(save_path)} {args.identifier} {fn}")
else:
    os.system(f"huggingface-cli download --local-dir {str(save_path)} {args.identifier}")
