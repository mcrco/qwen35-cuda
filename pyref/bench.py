#!/usr/bin/env python3
"""Benchmark the PyTorch reference with the custom CUDA benchmark protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer

from main import parse_dtype, resolve_model_dir
from qwen35 import Qwen35Model


DEFAULT_PROMPT = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "Explain CUDA kernels briefly.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir")
    parser.add_argument("--tokenizer")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--measure-tokens", type=int, default=128)
    parser.add_argument("--input-token", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefill", action="store_true")
    parser.add_argument("--prompt", default="")
    parser.add_argument(
        "--dtype",
        choices=("fp32", "float32", "float16", "bfloat16"),
        default="fp32",
    )
    parser.add_argument("--json", default="bench-results")
    parser.add_argument("--label", default="")
    parser.add_argument("--git-commit", default="")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_seq_len <= 0 or args.measure_tokens <= 0:
        raise ValueError("--max-seq-len and --measure-tokens must be positive")
    if args.warmup_tokens < 0 or args.input_token < 0 or args.temperature < 0 or args.seed < 0:
        raise ValueError("warmup, input token, temperature, and seed must be non-negative")


def git_commit(value: str) -> str:
    if value:
        return value
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def output_path(json_arg: str, commit: str) -> Path | None:
    if not json_arg:
        return None
    path = Path(json_arg)
    directory = path.parent if path.suffix else path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return directory / commit[:8] / "forward" / "python_reference" / f"{timestamp}.json"


def main() -> int:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)

    model_dir = resolve_model_dir(args.model_dir)
    dtype_name = "float32" if args.dtype == "fp32" else args.dtype
    dtype = parse_dtype(dtype_name)
    model = Qwen35Model(model_dir, dtype=dtype)
    latest_token = args.input_token
    prefill_tokens = 0

    prompt_tokens: list[int] = []
    if args.prefill:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or model_dir)
        prompt_tokens = tokenizer.encode(args.prompt or DEFAULT_PROMPT, add_special_tokens=False)
        prefill_tokens = len(prompt_tokens)

    required_tokens = prefill_tokens + args.warmup_tokens + args.measure_tokens
    if required_tokens > args.max_seq_len:
        raise ValueError("prefill tokens + warmup tokens + measure tokens exceed --max-seq-len")
    cache = model.allocate_cache(args.max_seq_len, dtype=dtype)

    with torch.inference_mode():
        for token in prompt_tokens:
            latest_token = model.forward(cache, token, args.temperature)
        for _ in range(args.warmup_tokens):
            latest_token = model.forward(cache, latest_token, args.temperature)

        start_seq_len = cache.seq_len
        start = time.perf_counter()
        for _ in range(args.measure_tokens):
            latest_token = model.forward(cache, latest_token, args.temperature)
        elapsed_us = (time.perf_counter() - start) * 1_000_000

    commit = git_commit(args.git_commit)
    per_token_us = elapsed_us / args.measure_tokens
    output = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "label": args.label,
        "benchmark": {
            "name": "qwen35_forward_bench",
            "module": "python_reference",
            "implementation": "python_reference",
        },
        "git": {"commit": commit},
        "model": {
            "name": "qwen35",
            "size": model_dir.name.removeprefix("Qwen3.5-"),
            "dtype": {"weight": dtype_name, "hidden": dtype_name, "compute": dtype_name},
        },
        "config": {
            "max_seq_len": args.max_seq_len,
            "prefill_enabled": args.prefill,
            "prefill_tokens": prefill_tokens,
            "warmup_tokens": args.warmup_tokens,
            "measure_tokens": args.measure_tokens,
            "input_token": args.input_token,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        "result": {
            "start_seq_len": start_seq_len,
            "end_seq_len": cache.seq_len,
            "wall_total_us": elapsed_us,
            "wall_us_per_token": per_token_us,
            "tokens_per_sec": 1_000_000 / per_token_us,
        },
    }

    path = output_path(args.json, commit)
    if path is None:
        print(json.dumps(output, indent=2))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2) + "\n")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
