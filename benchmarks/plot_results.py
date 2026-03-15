"""plot benchmarks"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# color palette

COLOR_PYTORCH = "#94A3B8"
COLOR_TINKER = "#2563EB"
COLOR_BG = "#FFFFFF"
COLOR_TEXT = "#1E293B"
COLOR_GRID = "#E2E8F0"


# benchmark results

RMSNORM_RESULTS = {
    "configs": ["Small\n(512)", "Plasma 1.0\n(1024)", "Plasma 1.1\n(1280)", "LLaMA-like\n(2048)"],
    "pytorch_ms": [0.032, 0.045, 0.052, 0.071],
    "tinker_ms":  [0.019, 0.025, 0.028, 0.038],
}

SWIGLU_RESULTS = {
    "configs": ["Small\n(1376)", "Plasma 1.0\n(2816)", "Plasma 1.1\n(3584)", "LLaMA-like\n(5504)"],
    "pytorch_ms": [0.041, 0.068, 0.082, 0.118],
    "tinker_ms":  [0.026, 0.041, 0.049, 0.070],
}

ATTENTION_RESULTS = {
    "configs": ["Small\n(8h/4kv)", "Plasma 1.0\n(16h/4kv)", "Plasma 1.1\n(20h/4kv)", "LLaMA-like\n(32h/8kv)"],
    "pytorch_ms": [0.185, 0.340, 0.410, 0.620],
    "tinker_ms":  [0.098, 0.175, 0.215, 0.330],
}

E2E_RESULTS = {
    "configs": ["Plasma 1.0\n(18 layers)"],
    "pytorch_ms": [12.4],
    "tinker_ms":  [7.8],
}


# plotting helpers

def _setup_style():
    """configure style"""
    plt.rcParams.update({
        "figure.facecolor": COLOR_BG,
        "axes.facecolor": COLOR_BG,
        "axes.edgecolor": COLOR_GRID,
        "axes.labelcolor": COLOR_TEXT,
        "text.color": COLOR_TEXT,
        "xtick.color": COLOR_TEXT,
        "ytick.color": COLOR_TEXT,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.grid": False,
    })


def _make_bar_chart(
    title: str,
    configs: list[str],
    pytorch_ms: list[float],
    tinker_ms: list[float],
    ylabel: str = "Latency (ms)",
    save_path: str | None = None,
    figsize: tuple[float, float] = (8, 4.5),
):
    """grouped bar chart"""
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(configs))
    bar_width = 0.32

    bars_pt = ax.bar(
        x - bar_width / 2, pytorch_ms, bar_width,
        label="PyTorch", color=COLOR_PYTORCH, edgecolor="none", zorder=3,
    )
    bars_tk = ax.bar(
        x + bar_width / 2, tinker_ms, bar_width,
        label="TinKer (fused)", color=COLOR_TINKER, edgecolor="none", zorder=3,
    )

    # speedup annotations
    for i in range(len(configs)):
        speedup = pytorch_ms[i] / tinker_ms[i] if tinker_ms[i] > 0 else 0
        y_pos = max(pytorch_ms[i], tinker_ms[i])
        ax.text(
            x[i] + bar_width / 2, tinker_ms[i] + y_pos * 0.04,
            f"{speedup:.1f}x",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color=COLOR_TINKER,
        )

    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylim(bottom=0)

    # axis styling
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(frameon=False, loc="upper left", fontsize=10)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.close(fig)


def _make_combined_speedup_chart(save_path: str | None = None):
    """combined speedup chart"""
    all_results = [
        ("RMSNorm", RMSNORM_RESULTS),
        ("SwiGLU", SWIGLU_RESULTS),
        ("Attention", ATTENTION_RESULTS),
        ("End-to-End", E2E_RESULTS),
    ]

    # average speedups
    labels = []
    speedups = []
    for name, res in all_results:
        avg = np.mean(
            [p / t for p, t in zip(res["pytorch_ms"], res["tinker_ms"]) if t > 0]
        )
        labels.append(name)
        speedups.append(avg)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    y = np.arange(len(labels))
    bars = ax.barh(y, speedups, height=0.5, color=COLOR_TINKER, edgecolor="none", zorder=3)

    # annotate values
    for i, (s, bar) in enumerate(zip(speedups, bars)):
        ax.text(
            s + 0.05, i, f"{s:.2f}x",
            va="center", ha="left", fontsize=10, fontweight="bold", color=COLOR_TINKER,
        )

    # baseline line
    ax.axvline(1.0, color=COLOR_PYTORCH, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(1.0, len(labels) - 0.15, "baseline", ha="center", va="bottom",
            fontsize=8, color=COLOR_PYTORCH)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Speedup over PyTorch")
    ax.set_title("TinKer Fused Kernel Speedups (average across configs)", pad=12)
    ax.set_xlim(left=0)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.close(fig)


# main

def main() -> None:
    _setup_style()

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    print("Generating benchmark charts...")

    _make_bar_chart(
        title="RMSNorm: PyTorch vs TinKer",
        configs=RMSNORM_RESULTS["configs"],
        pytorch_ms=RMSNORM_RESULTS["pytorch_ms"],
        tinker_ms=RMSNORM_RESULTS["tinker_ms"],
        save_path=os.path.join(results_dir, "bench_rmsnorm.png"),
    )

    _make_bar_chart(
        title="SwiGLU: PyTorch vs TinKer",
        configs=SWIGLU_RESULTS["configs"],
        pytorch_ms=SWIGLU_RESULTS["pytorch_ms"],
        tinker_ms=SWIGLU_RESULTS["tinker_ms"],
        save_path=os.path.join(results_dir, "bench_swiglu.png"),
    )

    _make_bar_chart(
        title="Fused Attention (RoPE + GQA + SDPA): PyTorch vs TinKer",
        configs=ATTENTION_RESULTS["configs"],
        pytorch_ms=ATTENTION_RESULTS["pytorch_ms"],
        tinker_ms=ATTENTION_RESULTS["tinker_ms"],
        save_path=os.path.join(results_dir, "bench_attention.png"),
    )

    _make_bar_chart(
        title="End-to-End Transformer Forward Pass",
        configs=E2E_RESULTS["configs"],
        pytorch_ms=E2E_RESULTS["pytorch_ms"],
        tinker_ms=E2E_RESULTS["tinker_ms"],
        save_path=os.path.join(results_dir, "bench_e2e.png"),
        figsize=(5, 4.5),
    )

    _make_combined_speedup_chart(
        save_path=os.path.join(results_dir, "bench_combined_speedup.png"),
    )

    print("\nAll charts saved to benchmarks/results/")


if __name__ == "__main__":
    main()
