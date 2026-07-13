#!/usr/bin/env python3
"""Benchmark the fastest supported Hugging Face generation path on this GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    CompileConfig,
    Qwen3_5ForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

from main import parse_dtype, resolve_model_dir


DEFAULT_PROMPT = "Explain CUDA kernels briefly."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir")
    parser.add_argument("--tokenizer")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prefill", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--input-token", type=int, default=64, help=argparse.SUPPRESS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--measure-tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=("fp32", "float32", "float16", "bfloat16"), default="fp32")
    parser.add_argument("--json", default="bench-results")
    parser.add_argument("--label", default="")
    parser.add_argument("--git-commit", default="")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument(
        "--execution",
        choices=("auto", "eager", "compiled"),
        default="auto",
        help="auto uses compiled Inductor on compute capability >= 7.0, otherwise eager CUDA",
    )
    return parser.parse_args()


def commit_id(value: str) -> str:
    if value:
        return value
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def json_path(value: str, commit: str, execution: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    directory = path.parent if path.suffix else path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return directory / commit[:8] / "forward" / f"huggingface_{execution}" / f"{timestamp}.json"


class TokenEvents(StoppingCriteria):
    """Record a stream event after generate produces each token."""

    def __init__(self) -> None:
        self.events: list[torch.cuda.Event] = []

    def __call__(self, input_ids, scores, **kwargs) -> torch.BoolTensor:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.events.append(event)
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the Hugging Face baseline requires a CUDA device")
    if args.warmup_tokens < 0 or args.measure_tokens <= 0:
        raise ValueError("--warmup-tokens must be non-negative and --measure-tokens must be positive")
    if args.temperature != 0:
        raise ValueError("Hugging Face comparison currently requires --temperature 0")
    torch.manual_seed(args.seed)

    capability = torch.cuda.get_device_capability()
    execution = args.execution
    if execution == "auto":
        execution = "compiled" if capability >= (7, 0) else "eager"
    if execution == "compiled" and capability < (7, 0):
        raise RuntimeError(
            f"Torch Inductor/Triton requires compute capability >= 7.0; found {capability[0]}.{capability[1]}. "
            "Use --execution eager (or the default --execution auto)."
        )

    dtype_name = "float32" if args.dtype == "fp32" else args.dtype
    dtype = parse_dtype(dtype_name)
    model_dir = resolve_model_dir(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or model_dir)
    full_config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_dir,
        config=full_config.text_config,
        dtype=dtype,
        local_files_only=True,
        key_mapping={r"^model\.language_model\.": "model."},
    ).eval().to("cuda")
    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    prompt_tokens = int(inputs.input_ids.shape[1])
    generated_tokens = 1 + args.warmup_tokens + args.measure_tokens
    if prompt_tokens + args.warmup_tokens + args.measure_tokens > args.max_seq_len:
        raise ValueError("prompt tokens + warmup tokens + measure tokens exceed --max-seq-len")
    generation_args = {
        **inputs,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if execution == "compiled":
        generation_args.update(
            cache_implementation="static",
            compile_config=CompileConfig(
                mode=args.compile_mode, fullgraph=False, dynamic=False
            ),
        )

    # Exclude lazy initialization and, when enabled, compilation/autotuning.
    with torch.inference_mode():
        model.generate(
            **generation_args,
            min_new_tokens=generated_tokens,
            max_new_tokens=generated_tokens,
        )
    torch.cuda.synchronize()

    token_events = TokenEvents()
    start_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    wall_start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **generation_args,
            min_new_tokens=generated_tokens,
            max_new_tokens=generated_tokens,
            stopping_criteria=StoppingCriteriaList([token_events]),
        )
    torch.cuda.synchronize()
    wall_us = (time.perf_counter() - wall_start) * 1_000_000

    if len(token_events.events) != generated_tokens:
        raise RuntimeError(
            f"expected {generated_tokens} generated-token events, observed {len(token_events.events)}"
        )
    prefill_us = start_event.elapsed_time(token_events.events[0]) * 1000
    measured_events = token_events.events[args.warmup_tokens :]
    decode_us = sum(
        start.elapsed_time(end)
        for start, end in zip(measured_events, measured_events[1:])
    ) * 1000
    decode_tokens = args.measure_tokens
    commit = commit_id(args.git_commit)
    result = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "label": args.label,
        "benchmark": {
            "name": "qwen35_forward_bench",
            "module": f"huggingface_{execution}",
            "implementation": f"huggingface_{execution}",
        },
        "git": {"commit": commit},
        "model": {
            "name": "qwen35",
            "size": model_dir.name.removeprefix("Qwen3.5-"),
            "dtype": {"weight": dtype_name, "hidden": dtype_name, "compute": dtype_name},
        },
        "config": {
            "text_only": True,
            "prompt_tokens": prompt_tokens,
            "warmup_tokens": args.warmup_tokens,
            "measure_tokens": args.measure_tokens,
            "max_seq_len": args.max_seq_len,
            "temperature": args.temperature,
            "seed": args.seed,
            "execution": execution,
            "cuda_compute_capability": f"{capability[0]}.{capability[1]}",
            "cache_implementation": "static" if execution == "compiled" else "default_dynamic",
            "compile_backend": "inductor" if execution == "compiled" else None,
            "compile_mode": args.compile_mode if execution == "compiled" else None,
            "greedy": True,
        },
        "result": {
            "prefill_total_us": prefill_us,
            "prefill_us_per_token": prefill_us / prompt_tokens,
            "prefill_tokens_per_sec": prompt_tokens * 1_000_000 / prefill_us,
            "decode_total_us": decode_us,
            "decode_us_per_token": decode_us / decode_tokens if decode_tokens else None,
            "decode_tokens_per_sec": decode_tokens * 1_000_000 / decode_us if decode_us else None,
            "time_to_first_token_us": prefill_us,
            "wall_total_us": wall_us,
            "wall_tokens_per_sec": args.measure_tokens * 1_000_000 / wall_us,
            "generated_token_ids": output[0, prompt_tokens:].tolist(),
        },
    }

    path = json_path(args.json, commit, execution)
    if path is None:
        print(json.dumps(result, indent=2))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
