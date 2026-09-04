#!/usr/bin/env python3
"""Build exact blog figures from the project's raw JSONL records.

Run with a Python environment containing numpy and matplotlib. The figures are
derived from per-episode/per-cell records; no values are copied from Markdown.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "media" / "figures"
EPISODES = (
    ROOT
    / "artifacts"
    / "pi05_instruction_repair_2026-08-31"
    / "state_confirm"
    / "episodes.jsonl"
)
MEDIATION = ROOT / "artifacts" / "pi05_prefill_mediation_2026-08-31" / "rows.jsonl"

INK = "#172033"
MUTED = "#64748B"
GRID = "#D9E0EA"
RED = "#D95D5D"
GREEN = "#2B8A6E"
BLUE = "#3274C1"
ORANGE = "#D88932"
LIGHT_BLUE = "#7DB2E8"


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - radius, center + radius


def style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_closed_loop_figure(episodes: list[dict]) -> dict:
    order = ["conflict", "correct", "state_live", "state_early"]
    labels = ["Conflicting\nprompt", "Correct\nprompt", "Live-band\nrepair", "Early-band\ncontrol"]
    colors = [RED, GREEN, BLUE, ORANGE]

    grouped = {condition: [row for row in episodes if row["condition"] == condition] for condition in order}
    for condition, rows in grouped.items():
        if len(rows) != 20:
            raise AssertionError(f"expected 20 confirmation episodes for {condition}, found {len(rows)}")

    successes = np.array([sum(bool(row["success"]) for row in grouped[condition]) for condition in order])
    totals = np.array([len(grouped[condition]) for condition in order])
    rates = successes / totals
    intervals = [wilson(int(k), int(n)) for k, n in zip(successes, totals)]
    yerr = np.array([[rate - lo for rate, (lo, _) in zip(rates, intervals)],
                     [hi - rate for rate, (_, hi) in zip(rates, intervals)]])

    edit_magnitudes: dict[str, list[float]] = {}
    for condition in ("state_live", "state_early"):
        values = []
        for row in grouped[condition]:
            replan_values = [float(replan["direction_to_host_norm_ratio_median"]) for replan in row["replans"]]
            values.append(float(np.median(replan_values)))
        edit_magnitudes[condition] = values

    fig = plt.figure(figsize=(10.8, 4.8), constrained_layout=False)
    grid = fig.add_gridspec(
        1, 2, width_ratios=[1.35, 1.0], left=0.07, right=0.985, bottom=0.20, top=0.76, wspace=0.22
    )
    ax = fig.add_subplot(grid[0, 0])
    ax_mag = fig.add_subplot(grid[0, 1])

    x = np.arange(len(order))
    bars = ax.bar(x, rates, width=0.68, color=colors, edgecolor="white", linewidth=1.0)
    ax.errorbar(x, rates, yerr=yerr, fmt="none", ecolor=INK, elinewidth=1.2, capsize=4, zorder=5)
    ax.set_ylim(0.0, 1.15)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{int(value * 100)}%" for value in np.linspace(0, 1, 6)])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Task success", color=INK, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    style_axes(ax)
    ax.text(0.0, 1.11, "A  Closed-loop behavioral result", transform=ax.transAxes,
            color=INK, fontsize=12, fontweight="bold")
    ax.text(0.0, 1.045, "Held-out confirmation: 10 initial states on each of two tasks",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    for bar, k, n in zip(bars, successes, totals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.035, f"{k}/{n}",
                ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")

    rng = np.random.default_rng(20260902)
    mag_x = np.array([0.0, 1.0])
    mag_labels = ["Live band\nlayers 12-17", "Early band\nlayers 0-5"]
    mag_colors = [BLUE, ORANGE]
    for index, condition in enumerate(("state_live", "state_early")):
        values = np.asarray(edit_magnitudes[condition])
        jitter = rng.uniform(-0.10, 0.10, len(values))
        ax_mag.scatter(np.full(len(values), mag_x[index]) + jitter, values, s=27,
                       color=mag_colors[index], alpha=0.75, edgecolor="white", linewidth=0.45, zorder=3)
        median = float(np.median(values))
        ax_mag.plot([mag_x[index] - 0.18, mag_x[index] + 0.18], [median, median],
                    color=INK, linewidth=2.2, zorder=4)
        ax_mag.text(mag_x[index], median + 0.025, f"median {median:.3f}", ha="center", va="bottom",
                    color=INK, fontsize=9)
    ax_mag.set_xlim(-0.45, 1.45)
    ax_mag.set_ylim(0.0, 0.60)
    ax_mag.set_xticks(mag_x, mag_labels)
    ax_mag.set_ylabel("Edit norm / host-state norm", color=INK, fontsize=10)
    ax_mag.grid(axis="y", color=GRID, linewidth=0.8)
    style_axes(ax_mag)
    ax_mag.text(0.0, 1.11, "B  Intervention magnitude", transform=ax_mag.transAxes,
                color=INK, fontsize=12, fontweight="bold")
    ax_mag.text(0.0, 1.045, "Each point is one confirmation episode",
                transform=ax_mag.transAxes, color=MUTED, fontsize=9)
    ax_mag.text(0.5, -0.27, "Important caveat: the early control is not magnitude-matched.",
                transform=ax_mag.transAxes, ha="center", color=RED, fontsize=9, fontweight="bold")

    fig.suptitle("State-conditioned image-cache replacement repairs π0.5 behavior",
                 x=0.015, y=0.965, ha="left", color=INK, fontsize=16, fontweight="bold")
    fig.text(0.015, 0.035,
             "Bars and points are recomputed from state_confirm/episodes.jsonl. Error bars: 95% Wilson intervals.",
             color=MUTED, fontsize=8)
    save(fig, "closed-loop-repair")

    return {
        "conditions": {
            condition: {
                "n": int(totals[index]),
                "successes": int(successes[index]),
                "wilson_95": [float(intervals[index][0]), float(intervals[index][1])],
            }
            for index, condition in enumerate(order)
        },
        "episode_median_edit_norm_ratio": edit_magnitudes,
    }


def make_mediation_figure(rows: list[dict]) -> dict:
    conditions = [
        ("prefill_instr_src", "Swap instruction\nbefore prefill"),
        ("prefill_restore_instr_dst", "Then restore original\ntext-position K/V"),
        ("prefill_restore_noninstr_dst", "Restore original\nnon-text K/V"),
        ("prefill_restore_img_dst", "Restore original\nimage-position K/V"),
        ("prefill_restore_randimg_dst", "Restore matched small\nrandom image subset"),
        ("postcache_instr_src", "Insert donor text K/V\nafter prefill"),
    ]
    by_condition = {condition: [] for condition, _ in conditions}
    for row in rows:
        condition = row["condition"]
        if condition in by_condition:
            by_condition[condition].append(float(row["D_dst_10"]))
    for condition, values in by_condition.items():
        if len(values) != 150:
            raise AssertionError(f"expected 150 mediation cells for {condition}, found {len(values)}")

    fig, ax = plt.subplots(figsize=(11.5, 5.6), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.25, top=0.80)
    x = np.arange(len(conditions))
    rng = np.random.default_rng(20260902)
    colors = [GREEN, LIGHT_BLUE, RED, BLUE, ORANGE, MUTED]
    for index, ((condition, _), color) in enumerate(zip(conditions, colors)):
        values = np.asarray(by_condition[condition])
        jitter = rng.uniform(-0.18, 0.18, len(values))
        ax.scatter(np.full(len(values), index) + jitter, values, s=12, color=color,
                   alpha=0.28, edgecolors="none", rasterized=True)
        median = float(np.median(values))
        q1, q3 = np.quantile(values, [0.25, 0.75])
        ax.plot([index - 0.25, index + 0.25], [median, median], color=INK, linewidth=2.2, zorder=4)
        ax.plot([index, index], [q1, q3], color=INK, linewidth=5.0, solid_capstyle="round", zorder=4)
        ax.text(index, min(1.075, median + 0.055), f"{median:.3f}", ha="center", va="bottom",
                fontsize=9, color=INK, fontweight="bold")

    ax.axhline(0.0, color=RED, linewidth=1.1, linestyle="--")
    ax.axhline(1.0, color=GREEN, linewidth=1.1, linestyle="--")
    ax.text(-0.48, 0.025, "receiver-like", color=RED, fontsize=9, va="bottom")
    ax.text(-0.48, 0.975, "donor-like", color=GREEN, fontsize=9, va="top")
    ax.set_xlim(-0.55, len(conditions) - 0.45)
    ax.set_ylim(-0.08, 1.10)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("Normalized action distance from receiver", color=INK, fontsize=10)
    ax.set_xticks(x, [label for _, label in conditions], fontsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    style_axes(ax)
    fig.text(0.075, 0.955,
             "The instruction's causal effect moves off its original text positions during prefill",
             ha="left", color=INK, fontsize=15, fontweight="bold")
    fig.text(0.075, 0.90,
             "150 scene x direction cells per intervention. Dots are raw cells; thick marks show median and interquartile range.",
             ha="left", color=MUTED, fontsize=9)
    fig.text(0.53, 0.055,
             "Restoring image/non-text state reverses the prompt swap; restoring text state or a small random image subset does not.",
             ha="center", color=INK, fontsize=9)
    save(fig, "prefill-mediation")

    return {
        condition: {
            "n": len(values),
            "median_D_dst_10": float(np.median(values)),
            "q1_D_dst_10": float(np.quantile(values, 0.25)),
            "q3_D_dst_10": float(np.quantile(values, 0.75)),
        }
        for condition, values in by_condition.items()
    }


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    episodes = load_jsonl(EPISODES)
    mediation = load_jsonl(MEDIATION)
    summary = {
        "sources": {
            "closed_loop": str(EPISODES.relative_to(ROOT)),
            "mediation": str(MEDIATION.relative_to(ROOT)),
        },
        "closed_loop": make_closed_loop_figure(episodes),
        "mediation": make_mediation_figure(mediation),
    }
    (OUT / "data-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
