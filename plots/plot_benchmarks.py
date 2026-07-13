#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_RESULTS_DIR = Path("bench-results")
DEFAULT_OUT_DIR = Path("bench-plots")


def normalized_path(path: Path) -> str:
    return str(path.expanduser().resolve(strict=False))


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
        "implementation": benchmark.get("implementation") or module,
        "preset": preset,
        "dtype": dtype,
        "gpu_us": result.get("gpu_us_per_iter", result.get("gpu_us_per_token")),
        "wall_us": result.get("decode_us_per_token", result.get("wall_us_per_token")),
        "cpu_us": result.get("cpu_us_per_iter"),
        "speedup": result.get("speedup_cpu_over_gpu"),
        "tokens_per_sec": result.get("tokens_per_sec"),
        "prefill_tokens_per_sec": result.get("prefill_tokens_per_sec"),
        "decode_tokens_per_sec": result.get("decode_tokens_per_sec", result.get("tokens_per_sec")),
        "seq_len": shapes.get("seq_len", config.get("max_seq_len")),
        "m": shapes.get("m"),
        "k": shapes.get("k"),
        "n": shapes.get("n"),
        "rows": shapes.get("rows"),
        "cols": shapes.get("cols"),
    }


def make_display_label(row: pd.Series) -> str:
    pieces = [str(row.get("module", ""))]
    if pd.notna(row.get("seq_len")):
        pieces.append(f"seq={int(row['seq_len'])}")
    if row.get("label"):
        pieces.append(str(row["label"]))
    return " ".join(piece for piece in pieces if piece and piece != "nan")


def load_processed_input_files(out_dir: Path) -> set[str]:
    processed: set[str] = set()
    for path in sorted(out_dir.glob("**/plot_metadata_*.json")):
        with path.open() as f:
            metadata = json.load(f)
        if metadata.get("identity") == "mixed":
            continue
        input_files = metadata.get("input_files", [])
        if not isinstance(input_files, list):
            continue
        for input_file in input_files:
            if isinstance(input_file, str) and input_file:
                processed.add(normalized_path(Path(input_file)))
    return processed


def load_results(results_dir: Path, processed_input_files: set[str] | None = None) -> pd.DataFrame:
    processed_input_files = processed_input_files or set()
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        if normalized_path(path) in processed_input_files:
            continue
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
        "wall_us",
        "cpu_us",
        "speedup",
        "tokens_per_sec",
        "prefill_tokens_per_sec",
        "decode_tokens_per_sec",
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


def artifact_identity(df: pd.DataFrame) -> str:
    commits = sorted(str(commit) for commit in df["git_commit"].dropna().unique() if str(commit))
    if len(commits) == 1:
        return commits[0][:8]
    if len(commits) > 1:
        return "mixed"
    return "unknown"


def artifact_out_dir(df: pd.DataFrame, out_dir: Path) -> Path:
    return out_dir / artifact_identity(df)


def title_for(title: str, df: pd.DataFrame) -> str:
    identity = artifact_identity(df)
    title_pieces: list[str] = []
    presets = sorted(str(preset) for preset in df["preset"].dropna().unique() if str(preset))
    if len(presets) == 1:
        title_pieces.append(presets[0])
    if identity not in {"unknown", "mixed"}:
        title_pieces.append(identity)
    if not title_pieces:
        return title
    return f"{title} ({', '.join(title_pieces)})"


def log_axis_limit(df: pd.DataFrame, column: str) -> tuple[float, float] | None:
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    values = values[values > 0]
    if values.empty:
        return None

    lower = 10 ** math.floor(math.log10(values.min()))
    upper = 10 ** math.ceil(math.log10(values.max()))
    if values.max() / upper > 0.75:
        upper *= 10
    return min(lower, 1.0), max(upper, 1.0)


def log_axis_limits(df: pd.DataFrame, columns: list[str]) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for column in columns:
        limit = log_axis_limit(df, column)
        if limit is not None:
            limits[column] = limit
    return limits


def configure_log_x_axis(ax: plt.Axes, xlim: tuple[float, float] | None) -> None:
    ax.set_xscale("log")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=20))
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(2, 3, 4, 5, 6, 7, 8, 9), numticks=100))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, axis="x", which="major", linewidth=0.8)
    ax.grid(True, axis="x", which="minor", linewidth=0.4, alpha=0.35)


def add_bar_value_labels(ax: plt.Axes, values: pd.Series, *, suffix: str = "") -> None:
    labels = [f"{value:.2f}{suffix}" if pd.notna(value) else "" for value in values]
    for patch, label in zip(ax.patches, labels):
        if not label:
            continue
        width = patch.get_width()
        if not math.isfinite(width) or width <= 0:
            continue
        ax.annotate(
            label,
            xy=(width, patch.get_y() + patch.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
        )


def save_barplot(
    df: pd.DataFrame,
    *,
    y: str,
    title: str,
    filename: str,
    out_dir: Path,
    log_scale: bool = False,
    xlim: tuple[float, float] | None = None,
    show_value_labels: bool = False,
    value_suffix: str = "",
) -> Path | None:
    plot_df = df.dropna(subset=[y]).copy()
    if plot_df.empty:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    height = max(4.0, 0.45 * len(plot_df))
    fig, ax = plt.subplots(figsize=(12, height))
    sns.barplot(data=plot_df, x=y, y="display_label", hue="module", dodge=False, ax=ax)
    ax.set_title(title_for(title, df))
    ax.set_xlabel(y)
    ax.set_ylabel("")
    if log_scale:
        configure_log_x_axis(ax, xlim)
    if show_value_labels:
        add_bar_value_labels(ax, plot_df[y], suffix=value_suffix)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0, title="module")
    fig.tight_layout()

    out_path = out_dir / filename_for(filename, df)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
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


def generate_outputs(df: pd.DataFrame, out_dir: Path, axis_limits: dict[str, tuple[float, float]]) -> list[Path]:
    out_dir = artifact_out_dir(df, out_dir)
    paths: list[Path] = []
    for maybe_path in [
        save_barplot(
            df,
            y="gpu_us",
            title="GPU time per iter/token",
            filename="gpu_time.png",
            out_dir=out_dir,
            log_scale=True,
            xlim=axis_limits.get("gpu_us"),
        ),
        save_barplot(
            df,
            y="speedup",
            title="CPU/GPU speedup",
            filename="speedup.png",
            out_dir=out_dir,
            log_scale=True,
            xlim=axis_limits.get("speedup"),
            show_value_labels=True,
            value_suffix="x",
        ),
        save_barplot(df, y="tokens_per_sec", title="Forward tokens/sec", filename="tokens_per_sec.png", out_dir=out_dir),
        save_barplot(
            df[
                (df["module"] == "custom_cuda")
                | df["module"].str.startswith("huggingface_", na=False)
            ],
            y="wall_us",
            title="Hugging Face vs custom CUDA decode latency",
            filename="huggingface_vs_cuda.png",
            out_dir=out_dir,
            log_scale=True,
            xlim=axis_limits.get("wall_us"),
            show_value_labels=True,
            value_suffix=" us/token",
        ),
        save_barplot(
            df,
            y="prefill_tokens_per_sec",
            title="Prefill tokens/sec",
            filename="prefill_tokens_per_sec.png",
            out_dir=out_dir,
            show_value_labels=True,
        ),
        save_barplot(
            df,
            y="decode_tokens_per_sec",
            title="Cached decode tokens/sec",
            filename="decode_tokens_per_sec.png",
            out_dir=out_dir,
            show_value_labels=True,
        ),
    ]:
        if maybe_path is not None:
            paths.append(maybe_path)
    paths.append(write_plot_metadata(df, out_dir, paths))
    return paths


def generate_outputs_by_commit(df: pd.DataFrame, out_dir: Path, axis_limits: dict[str, tuple[float, float]]) -> list[Path]:
    paths: list[Path] = []
    commit_keys = df["git_commit"].fillna("")
    for _, group_df in df.groupby(commit_keys, sort=True):
        paths.extend(generate_outputs(group_df.copy(), out_dir, axis_limits))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--replot-all",
        action="store_true",
        help="Ignore existing plot metadata and regenerate plots for every benchmark JSON file.",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", context="notebook")

    all_df = load_results(args.results_dir)
    if all_df.empty:
        print(f"No benchmark JSON files found in {args.results_dir}")
        return 1

    processed_input_files = set() if args.replot_all else load_processed_input_files(args.out_dir)
    df = all_df if args.replot_all else load_results(args.results_dir, processed_input_files)
    if df.empty:
        if processed_input_files:
            print(f"No new benchmark JSON files found in {args.results_dir}")
            return 0
        print(f"No benchmark JSON files found in {args.results_dir}")
        return 1

    axis_limits = log_axis_limits(all_df, ["gpu_us", "speedup", "wall_us"])
    paths = generate_outputs_by_commit(df, args.out_dir, axis_limits)
    print(df.to_string(index=False))
    print()
    for path in paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
