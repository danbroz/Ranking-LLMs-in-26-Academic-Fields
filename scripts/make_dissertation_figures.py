"""Generate diagrams and figures for the dissertation.

Creates publication-quality PNGs in ../figures/ for inclusion in
Dissertation_Draft.lax (LaTeX). Uses results-apis.csv for empirical data.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
DATA = ROOT / "results-apis.csv"

DPI = 200
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def load_results():
    with DATA.open() as f:
        rows = list(csv.DictReader(f))
    fields = [k for k in rows[0].keys() if k not in {"model", "overall", "timestamp"}]

    def short(model_name: str) -> str:
        mapping = {
            "openai:gpt-5.1": "OpenAI GPT-5.1",
            "gemini:gemini-3-pro-preview": "Gemini 3 Pro",
            "together:meta-llama/Llama-4-Maverick-17B-128E-Instruct": "Llama-4 Maverick",
            "claude:claude-opus-4-1": "Claude Opus 4.1",
            "grok:grok-4-1-fast-non-reasoning": "Grok-4.1 Fast",
        }
        return mapping.get(model_name, model_name)

    data = []
    for r in rows:
        data.append(
            {
                "model": short(r["model"]),
                "overall": float(r["overall"]),
                "scores": {f: float(r[f]) for f in fields},
            }
        )
    data.sort(key=lambda d: d["overall"], reverse=True)
    return data, fields


# ----------------------------- CHAPTER 3 -----------------------------


def fig_methodology_flowchart():
    """Figure 3.1: End-to-end evaluation pipeline."""
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=DPI)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    steps = [
        ("Curate\n26-Field\nQuestion Bank", "#cfe8ff"),
        ("Standardize\nMCQ Format\n(1 correct + 3 distractors)", "#cfe8ff"),
        ("Configure\nDeterministic\nEvaluation (T=0)", "#fde2b3"),
        ("Administer\n1,000 Items\nper Field per Model", "#fde2b3"),
        ("Calibrated Scoring\n+0.1 / 0 / \u22120.1", "#d6f0d6"),
        ("Statistical Analysis\nANOVA, Tukey HSD,\nBootstrap", "#d6f0d6"),
        ("Field-Level\nLeaderboard\n& Reporting", "#f7c6c7"),
    ]

    n = len(steps)
    box_w, box_h = 12.0, 12.0
    y = 25
    xs = np.linspace(4, 100 - 4 - box_w, n)
    for (label, color), x in zip(steps, xs):
        box = FancyBboxPatch(
            (x, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            linewidth=1.0,
            edgecolor="#333333",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(
            x + box_w / 2,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            wrap=True,
        )
    for i in range(n - 1):
        x0 = xs[i] + box_w
        x1 = xs[i + 1]
        arrow = FancyArrowPatch(
            (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=14, color="#333333", linewidth=1.2
        )
        ax.add_patch(arrow)

    ax.text(
        50,
        46,
        "Figure 3.1. Quasi-Experimental Evaluation Pipeline for Ranking LLMs Across 26 Academic Fields",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.text(
        50,
        7,
        "Inputs (blue) \u2192 Standardized administration (orange) \u2192 Scoring & inference (green) \u2192 Reporting (red).",
        ha="center",
        va="center",
        fontsize=9,
        color="#444444",
    )

    out = FIG_DIR / "fig3_1_methodology_flowchart.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_variable_framework():
    """Figure 3.2: Independent / Dependent / Control variables diagram."""
    fig, ax = plt.subplots(figsize=(9, 5.0), dpi=DPI)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    iv = FancyBboxPatch(
        (4, 22),
        24,
        16,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        edgecolor="#1f4e79",
        facecolor="#cfe2f3",
        linewidth=1.2,
    )
    dv = FancyBboxPatch(
        (72, 22),
        24,
        16,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        edgecolor="#7a3e00",
        facecolor="#ffe599",
        linewidth=1.2,
    )
    ctrl = FancyBboxPatch(
        (35, 4),
        30,
        12,
        boxstyle="round,pad=0.4,rounding_size=1.0",
        edgecolor="#4d4d4d",
        facecolor="#efefef",
        linewidth=1.0,
    )
    ax.add_patch(iv)
    ax.add_patch(dv)
    ax.add_patch(ctrl)

    ax.text(
        16,
        33,
        "Independent Variable",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#1f4e79",
    )
    ax.text(
        16,
        28,
        "LLM evaluated\n(GPT-5.1, Gemini 3 Pro,\nLlama-4 Maverick,\nClaude Opus 4.1, Grok-4.1)",
        ha="center",
        va="center",
        fontsize=8.5,
    )

    ax.text(
        84,
        33,
        "Dependent Variable",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#7a3e00",
    )
    ax.text(
        84,
        28,
        "Mean accuracy score\nper academic field\n(percentage of\ncorrect responses)",
        ha="center",
        va="center",
        fontsize=8.5,
    )

    ax.text(50, 13, "Controls", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(
        50,
        8.5,
        "Same 1,000 items per field \u2022 Identical prompt template \u2022 Temperature = 0 \u2022 26 Scopus fields",
        ha="center",
        va="center",
        fontsize=8.2,
    )

    arr = FancyArrowPatch(
        (28, 30), (72, 30), arrowstyle="-|>", mutation_scale=18, color="#333333", linewidth=1.4
    )
    ax.add_patch(arr)
    ax.text(50, 33, "predicts", ha="center", va="center", fontsize=9, style="italic")

    arr2 = FancyArrowPatch(
        (50, 16), (50, 22), arrowstyle="-|>", mutation_scale=14, color="#666666", linewidth=1.0
    )
    ax.add_patch(arr2)

    ax.text(
        50,
        55,
        "Figure 3.2. Variable Framework: Independent, Dependent, and Control Variables",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    out = FIG_DIR / "fig3_2_variable_framework.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_scoring_schema():
    """Figure 3.3: Calibrated scoring schema visualization."""
    fig, ax = plt.subplots(figsize=(8, 4.0), dpi=DPI)
    categories = ["Correct", "Abstention", "Incorrect"]
    values = [0.1, 0.0, -0.1]
    colors = ["#4daf4a", "#999999", "#e41a1c"]
    bars = ax.bar(categories, values, color=colors, edgecolor="#333", linewidth=0.8, width=0.55)
    ax.axhline(0, color="#333", linewidth=0.7)
    ax.set_ylim(-0.18, 0.18)
    ax.set_ylabel("Score Awarded per Item")
    ax.set_title("Figure 3.3. Calibrated Scoring Schema for Multiple-Choice Items")
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + (0.012 if v >= 0 else -0.018),
            f"{v:+.1f}" if v != 0 else "0",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=11,
            fontweight="bold",
        )
    ax.text(
        0.5,
        -0.28,
        "Calibration discourages random guessing while permitting models to abstain when uncertain.",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9,
        color="#444",
    )
    out = FIG_DIR / "fig3_3_scoring_schema.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ----------------------------- CHAPTER 4 -----------------------------


def fig_overall_ranking(data):
    """Figure 4.1: Overall calibrated score by model."""
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=DPI)
    names = [d["model"] for d in data]
    overalls = [d["overall"] for d in data]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]
    bars = ax.barh(names[::-1], overalls[::-1], color=colors[: len(names)][::-1], edgecolor="#333")
    ax.set_xlabel("Overall Calibrated Score (0\u2013100)")
    ax.set_xlim(0, 100)
    ax.set_title("Figure 4.1. Overall Performance of Frontier LLMs Across 26 Academic Fields")
    for bar, v in zip(bars, overalls[::-1]):
        ax.text(v + 1.0, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    out = FIG_DIR / "fig4_1_overall_ranking.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_heatmap(data, fields):
    """Figure 4.2: Heatmap of model x field accuracy."""
    matrix = np.array([[d["scores"][f] for f in fields] for d in data])
    fig_w = max(12, 0.42 * len(fields) + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 4.6), dpi=DPI)
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=20, vmax=80)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([d["model"] for d in data])
    ax.set_xticks(range(len(fields)))
    pretty_fields = [f.replace("-", " ").title() for f in fields]
    ax.set_xticklabels(pretty_fields, rotation=55, ha="right", fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if 35 < matrix[i, j] < 70 else "white",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Calibrated Accuracy")
    ax.set_title(
        "Figure 4.2. Field-Level Calibrated Accuracy Heatmap for Five Frontier LLMs Across 26 Academic Fields"
    )
    out = FIG_DIR / "fig4_2_heatmap.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_grouped_bars(data, fields):
    """Figure 4.3: Grouped bar chart for a representative subset of fields."""
    selected = [
        "computer-science",
        "business-management-and-accounting",
        "mathematics",
        "medicine",
        "economics-econometrics-and-finance",
        "physics-and-astronomy",
        "psychology",
        "chemical-engineering",
    ]
    selected = [s for s in selected if s in fields]
    fig, ax = plt.subplots(figsize=(11, 5.0), dpi=DPI)
    x = np.arange(len(selected))
    width = 0.16
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]
    for i, d in enumerate(data):
        vals = [d["scores"][s] for s in selected]
        ax.bar(x + (i - 2) * width, vals, width, label=d["model"], color=palette[i % 5], edgecolor="#333", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("-", " ").title() for s in selected], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Calibrated Accuracy")
    ax.set_ylim(0, 90)
    ax.set_title(
        "Figure 4.3. Between-Model Performance Differences Across Eight Representative Academic Fields"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=5, frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    out = FIG_DIR / "fig4_3_grouped_bars.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_variability_box(data, fields):
    """Figure 4.4: Box plot of field-level variability per model (RQ1)."""
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=DPI)
    series = [[d["scores"][f] for f in fields] for d in data]
    labels = [d["model"] for d in data]
    bp = ax.boxplot(
        series,
        labels=labels,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color="black", linewidth=1.5),
    )
    colors = ["#cfe2f3", "#d9ead3", "#fde2b3", "#e6d5f0", "#f4cccc"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor("#333")
    for i, vals in enumerate(series, start=1):
        ax.scatter(
            np.full(len(vals), i) + np.random.uniform(-0.08, 0.08, len(vals)),
            vals,
            s=12,
            alpha=0.55,
            color="#333",
        )
    ax.set_ylabel("Field-Level Calibrated Accuracy")
    ax.set_ylim(20, 85)
    ax.set_title(
        "Figure 4.4. Distribution of Field-Level Accuracy per Model (Evidence for RQ1: Non-Uniform Performance)"
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right", fontsize=9)
    out = FIG_DIR / "fig4_4_variability_box.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_strength_weakness(data, fields):
    """Figure 4.5: Strongest vs weakest field per model."""
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=DPI)
    names = [d["model"] for d in data]
    bests = []
    worsts = []
    spreads = []
    best_labels = []
    worst_labels = []
    for d in data:
        items = list(d["scores"].items())
        items.sort(key=lambda kv: kv[1], reverse=True)
        bests.append(items[0][1])
        worsts.append(items[-1][1])
        spreads.append(items[0][1] - items[-1][1])
        best_labels.append(items[0][0].replace("-", " "))
        worst_labels.append(items[-1][0].replace("-", " "))

    y = np.arange(len(names))
    ax.hlines(y, worsts, bests, color="#888", linewidth=2.0)
    ax.scatter(bests, y, color="#2ca02c", s=70, zorder=3, label="Strongest field")
    ax.scatter(worsts, y, color="#d62728", s=70, zorder=3, label="Weakest field")
    for i, (b, w) in enumerate(zip(bests, worsts)):
        ax.text(b + 1.0, i, f"{best_labels[i]} ({b:.1f})", va="center", fontsize=8, color="#1b5e20")
        ax.text(w - 1.0, i, f"{worst_labels[i]} ({w:.1f})", va="center", ha="right", fontsize=8, color="#7f0000")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Calibrated Accuracy")
    ax.set_title(
        "Figure 4.5. Strongest and Weakest Academic Field per Model (Range of Domain-Level Performance)"
    )
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    out = FIG_DIR / "fig4_5_strength_weakness.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    data, fields = load_results()
    outputs = [
        fig_methodology_flowchart(),
        fig_variable_framework(),
        fig_scoring_schema(),
        fig_overall_ranking(data),
        fig_heatmap(data, fields),
        fig_grouped_bars(data, fields),
        fig_variability_box(data, fields),
        fig_strength_weakness(data, fields),
    ]
    for o in outputs:
        size_kb = os.path.getsize(o) / 1024
        print(f"{o.relative_to(ROOT)}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
