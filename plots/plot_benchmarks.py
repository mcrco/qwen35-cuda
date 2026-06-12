#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_RESULTS_DIR = Path("bench-results")
DEFAULT_OUT_DIR = Path("bench-plots")


def flatten_result(data: dict[str, Any], path: Path) -> dict[str, Any]:
    benchmark = data.get("benchmark", {})
    model = data.get("model", {})
    config = data.get("config", {})
    result = data.get("result", {})
    shapes = config.get("shapes", {})
    git = data.get("git", {})

    name = benchmark.get("name") or "qwen35_forward_bench"
    module = benchmark.get("module") or "forward"
    preset = benchmark.get("preset") or model.get("size", "")
    dtype = benchmark.get("dtype") or ".".join(str(v) for v in model.get("dtype", {}).values())

    return {
        "file": str(path),
        "timestamp": data.get("timestamp", ""),
        "label": data.get("label", ""),
        "git_commit": git.get("commit", ""),
        "git_short": str(git.get("commit", ""))[:8],
        "benchmark": name,
        "module": module,
        "preset": preset,
        "dtype": dtype,
        "gpu_us": result.get("gpu_us_per_iter", result.get("gpu_us_per_token")),
        "cpu_us": result.get("cpu_us_per_iter"),
        "speedup": result.get("speedup_cpu_over_gpu"),
        "tokens_per_sec": result.get("tokens_per_sec"),
        "seq_len": shapes.get("seq_len", config.get("max_seq_len")),
        "m": shapes.get("m"),
        "k": shapes.get("k"),
        "n": shapes.get("n"),
        "rows": shapes.get("rows"),
        "cols": shapes.get("cols"),
    }


def make_display_label(row: pd.Series) -> str:
    pieces = [str(row.get("module", "")), str(row.get("preset", ""))]
    if pd.notna(row.get("seq_len")):
        pieces.append(f"seq={int(row['seq_len'])}")
    if row.get("label"):
        pieces.append(str(row["label"]))
    if row.get("git_short"):
        pieces.append(str(row["git_short"]))
    return " ".join(piece for piece in pieces if piece and piece != "nan")


def load_results(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        with path.open() as f:
            data = json.load(f)
        items = data if isinstance(data, list) else [data]
        for item in items:
            rows.append(flatten_result(item, path))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    numeric_cols = [
        "gpu_us",
        "cpu_us",
        "speedup",
        "tokens_per_sec",
        "seq_len",
        "m",
        "k",
        "n",
        "rows",
        "cols",
    ]
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["display_label"] = df.apply(make_display_label, axis=1)
    return df


def write_summary(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"benchmark_summary_{artifact_identity(df)}.csv"
    df.to_csv(summary_path, index=False)
    return summary_path


def artifact_identity(df: pd.DataFrame) -> str:
    commits = sorted(str(commit) for commit in df["git_commit"].dropna().unique() if str(commit))
    if len(commits) == 1:
        return commits[0][:8]
    if len(commits) > 1:
        return "mixed"
    return "unknown"


def artifact_out_dir(df: pd.DataFrame, out_dir: Path) -> Path:
    return out_dir / artifact_identity(df)


def save_barplot(
    df: pd.DataFrame,
    *,
    y: str,
    title: str,
    filename: str,
    out_dir: Path,
    log_scale: bool = False,
) -> Path | None:
    plot_df = df.dropna(subset=[y]).copy()
    if plot_df.empty:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    height = max(4.0, 0.45 * len(plot_df))
    fig, ax = plt.subplots(figsize=(12, height))
    sns.barplot(data=plot_df, x=y, y="display_label", hue="module", dodge=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(y)
    ax.set_ylabel("")
    if log_scale:
        ax.set_xscale("log")
    ax.legend(loc="best", title="module")
    fig.tight_layout()

    out_path = out_dir / filename_for(filename, df)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def filename_for(filename: str, df: pd.DataFrame) -> Path:
    path = Path(filename)
    return path.with_name(f"{path.stem}_{artifact_identity(df)}{path.suffix}")


def write_plot_metadata(df: pd.DataFrame, out_dir: Path, paths: list[Path]) -> Path:
    identity = artifact_identity(df)
    metadata_path = out_dir / f"plot_metadata_{identity}.json"
    commits = sorted(str(commit) for commit in df["git_commit"].dropna().unique() if str(commit))
    input_files = sorted(str(path) for path in df["file"].dropna().unique())
    metadata = {
        "identity": identity,
        "commits": commits,
        "input_files": input_files,
        "artifacts": [str(path) for path in paths],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata_path


def generate_outputs(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir = artifact_out_dir(df, out_dir)
    paths: list[Path] = [write_summary(df, out_dir)]
    for maybe_path in [
        save_barplot(df, y="gpu_us", title="GPU time per iter/token", filename="gpu_time.png", out_dir=out_dir, log_scale=True),
        save_barplot(df, y="speedup", title="CPU/GPU speedup", filename="speedup.png", out_dir=out_dir, log_scale=True),
        save_barplot(df, y="tokens_per_sec", title="Forward tokens/sec", filename="tokens_per_sec.png", out_dir=out_dir),
    ]:
        if maybe_path is not None:
            paths.append(maybe_path)
    paths.append(write_plot_metadata(df, out_dir, paths))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", context="notebook")

    df = load_results(args.results_dir)
    if df.empty:
        print(f"No benchmark JSON files found in {args.results_dir}")
        return 1

    paths = generate_outputs(df, args.out_dir)
    print(df.to_string(index=False))
    print()
    for path in paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
