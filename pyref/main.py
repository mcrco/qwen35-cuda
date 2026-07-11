import argparse
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from qwen35 import Qwen35Model


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models/Qwen3.5-4B"


def resolve_model_dir(model_dir_arg: str | None) -> Path:
    model_dir = (
        model_dir_arg
        or os.getenv("TRANSFORMER_MODEL_DIR")
        or os.getenv("QWEN35_MODEL_DIR")
    )
    if model_dir is not None:
        return Path(model_dir).expanduser()
    return DEFAULT_MODEL_DIR


def parse_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype {dtype_name!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", help="Local Qwen3.5 checkpoint directory.")
    parser.add_argument(
        "--tokenizer",
        default=os.getenv("QWEN35_TOKENIZER"),
        help="Tokenizer path or Hugging Face model id.",
    )
    parser.add_argument("--prompt", default="a")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default=os.getenv("QWEN35_DTYPE") or "bfloat16",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = resolve_model_dir(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"model directory does not exist: {model_dir}. "
            "Set TRANSFORMER_MODEL_DIR or pass --model-dir."
        )

    dtype = parse_dtype(args.dtype)
    model = Qwen35Model(model_dir, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or model_dir)

    input_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not input_ids:
        raise ValueError("prompt must encode to at least one token")

    max_seq_len = len(input_ids) + args.max_new_tokens
    cache = model.allocate_cache(max_seq_len, dtype=dtype)

    for token_id in input_ids[:-1]:
        model.forward(cache, token_id, 0.0)

    print(args.prompt, end="", flush=True)
    latest_token = input_ids[-1]
    for _ in range(args.max_new_tokens):
        new_token = model.forward(cache, latest_token, args.temperature)
        print(tokenizer.decode([new_token]), end="", flush=True)
        latest_token = new_token
    print()


if __name__ == "__main__":
    main()
