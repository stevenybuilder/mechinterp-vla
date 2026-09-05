#!/usr/bin/env python3
"""Build claim-first figures from preserved experiment records.

Each plot exposes the unit of evidence and writes its plotted values to CSV.
Unlike experiment families are kept on separate axes.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
AUDIT = ROOT / "artifacts" / "numbers-audit-derived.json"

INK = "#172033"
BLUE = "#1768AC"
SKY = "#74B3CE"
ORANGE = "#E07A3F"
PURPLE = "#775DA6"
TEAL = "#278C82"
GRAY = "#687386"
MID = "#A8B0BE"
LIGHT = "#E8ECF2"
PALE_BLUE = "#EAF3FA"
PALE_ORANGE = "#FCEFE7"
WHITE = "#FFFFFF"


def load_jsonl(relative: str) -> list[dict]:
    with (ROOT / relative).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.labelcolor": INK,
            "axes.edgecolor": MID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight", facecolor=WHITE)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.08, label, transform=ax.transAxes, fontsize=13, weight="bold", va="top")


def claim_title(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.055, y=0.99, ha="left", fontsize=19, weight="bold", color=INK)
    fig.text(0.055, 0.945, subtitle, ha="left", va="top", fontsize=11.5, color=GRAY)


def jitter(n: int, width: float = 0.15, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-width, width, n)


def grouped_medians(rows: list[dict], group_keys: tuple[str, ...], value: str) -> list[dict]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in group_keys)].append(float(row[value]))
    result = []
    for key, values in grouped.items():
        item = {k: v for k, v in zip(group_keys, key)}
        item[value] = float(np.median(values))
        result.append(item)
    return result


def figure_0_position_dose_hero() -> None:
    """Hero figure for the strongest compression failure."""
    dose = load_jsonl("artifacts/pi05_mediation_2026-08-31/dose1/rows.jsonl")
    counts = [4, 8, 16, 32, 64, 128, 256, 512]
    cell_rows = grouped_medians(dose, ("cell", "arm", "n_pos"), "R")
    cell_lookup = {(r["cell"], r["arm"], int(r["n_pos"])): r["R"] for r in cell_rows}
    pooled = {
        (arm, n): float(np.median([float(r["R"]) for r in dose if r["arm"] == arm and int(r["n_pos"]) == n]))
        for arm in ("LOC", "RAND")
        for n in counts
    }

    fig = plt.figure(figsize=(15.5, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.25, 1.05], wspace=0.14)
    ax = fig.add_subplot(gs[0, 0])
    ax_key = fig.add_subplot(gs[0, 1])
    claim_title(
        fig,
        "The target-relevant action shift stays small until all 512 image positions are replaced",
        "π0.5 · image-position K/V at layers 12–17 · 8 directed prompt-pair cells × 3 initial states; same pixels, state, and action noise within each comparison.",
    )

    x = np.arange(len(counts), dtype=float)
    ax.axhspan(-0.04, 0.20, color="#F4F5F7", zorder=-4)
    ax.axhspan(0.72, 1.08, color=PALE_BLUE, zorder=-4)
    ax.axvspan(6.68, 7.32, color="#E7EEF6", alpha=0.82, zorder=-3)
    ax.text(0.02, 0.08, "little target-axis shift", transform=ax.transAxes, color=GRAY, fontsize=10.2, weight="bold")
    ax.text(0.02, 0.91, "same target-axis projection as instruction B", transform=ax.transAxes, color=BLUE, fontsize=10.2, weight="bold")

    series = [
        ("LOC", "object-centered positions", TEAL, "o", -0.075),
        ("RAND", "count-matched random positions", ORANGE, "s", 0.075),
    ]
    for series_index, (arm, label, color, marker, offset) in enumerate(series):
        medians = np.array([pooled[(arm, n)] for n in counts])
        for i, n in enumerate(counts[:-1]):
            values = [r["R"] for r in cell_rows if r["arm"] == arm and int(r["n_pos"]) == n]
            ax.scatter(
                np.full(len(values), x[i] + offset) + jitter(len(values), 0.025, 910 + series_index * 20 + i),
                values,
                s=27,
                facecolors=WHITE,
                edgecolors=color,
                linewidth=1.05,
                alpha=0.70,
                zorder=3,
            )
        ax.plot(x, medians, color=color, marker=marker, lw=3.0, ms=7.3, label=label, zorder=5)

    full_values = [r["R"] for r in cell_rows if r["arm"] == "LOC" and int(r["n_pos"]) == 512]
    ax.scatter(
        np.full(len(full_values), 7.0) + jitter(len(full_values), 0.045, 997),
        full_values,
        s=34,
        facecolors=WHITE,
        edgecolors=BLUE,
        linewidth=1.2,
        alpha=0.85,
        zorder=6,
    )
    ax.scatter(7.0, pooled[("LOC", 512)], marker="D", s=105, color=BLUE, edgecolor=WHITE, linewidth=1.1, zorder=7)

    label_box = dict(boxstyle="round,pad=0.22", fc=WHITE, ec="none", alpha=0.92)
    ax.text(5.00, pooled[("LOC", 128)] + 0.055, "0.043", color=TEAL, ha="center", weight="bold", bbox=label_box)
    ax.text(5.83, pooled[("LOC", 256)] + 0.030, "0.082", color=TEAL, ha="right", weight="bold", bbox=label_box)
    ax.text(6.03, pooled[("RAND", 256)] + 0.070, "0.180", color=ORANGE, ha="center", weight="bold", bbox=label_box)
    ax.text(6.90, pooled[("LOC", 512)] + 0.105, "0.773", color=BLUE, ha="right", fontsize=13, weight="bold", bbox=label_box)
    ax.annotate(
        "the discontinuity is the full field",
        xy=(7.0, pooled[("LOC", 512)]),
        xytext=(5.05, 0.58),
        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6, connectionstyle="arc3,rad=-0.12"),
        color=BLUE,
        fontsize=11.3,
        weight="bold",
    )

    ax.axhline(0, color=MID, lw=1.0)
    ax.axhline(1, color=BLUE, lw=1.0, ls="--", alpha=0.65)
    ax.set_xlim(-0.42, 7.35)
    ax.set_ylim(-0.055, 1.10)
    ax.set_xticks(x, [str(n) for n in counts])
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xlabel("Image-token positions replaced  (of 512)", labelpad=10)
    ax.set_ylabel("Normalized target-axis shift  R\n0 = clean A projection · 1 = clean B projection", labelpad=10)
    ax.grid(axis="y", color=LIGHT, lw=0.9)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.84), fontsize=10.2, handlelength=2.7)
    panel_label(ax, "a")

    ax_key.set_xlim(0, 1)
    ax_key.set_ylim(0, 1)
    ax_key.axis("off")
    panel_label(ax_key, "b")
    ax_key.text(0.02, 0.985, "What each dose changed", va="top", fontsize=12.5, weight="bold")
    ax_key.text(0.02, 0.92, "K and V at every one of six layers", va="top", fontsize=10.0, color=GRAY)
    for j, layer in enumerate(range(12, 18)):
        y = 0.855 - j * 0.032
        ax_key.add_patch(FancyBboxPatch((0.02, y), 0.75, 0.020, boxstyle="round,pad=0.003", fc=PALE_BLUE, ec=LIGHT, lw=0.7))
        ax_key.text(0.81, y + 0.010, f"L{layer}", va="center", fontsize=8.5, color=GRAY)

    dose_specs = [
        ("object-centered", 128, pooled[("LOC", 128)], TEAL),
        ("object-centered", 256, pooled[("LOC", 256)], TEAL),
        ("random", 256, pooled[("RAND", 256)], ORANGE),
        ("all positions", 512, pooled[("LOC", 512)], BLUE),
    ]
    for i, (kind, n, value, color) in enumerate(dose_specs):
        y = 0.57 - i * 0.135
        ax_key.text(0.02, y + 0.068, f"{n}/512  {kind}", fontsize=10.2, weight="bold", color=INK)
        ax_key.add_patch(FancyBboxPatch((0.02, y), 0.76, 0.042, boxstyle="round,pad=0.003", fc="#F1F3F6", ec="none"))
        ax_key.add_patch(FancyBboxPatch((0.02, y), 0.76 * n / 512, 0.042, boxstyle="round,pad=0.003", fc=color, ec="none"))
        ax_key.text(0.82, y + 0.021, f"R={value:.3f}", va="center", fontsize=10.2, color=color, weight="bold")
    ax_key.text(
        0.02,
        0.055,
        "At 512 positions, object-centered\nand random are the same intervention.",
        fontsize=9.4,
        color=GRAY,
        linespacing=1.35,
    )

    exported = []
    for row in dose:
        arm = row["arm"]
        n = int(row["n_pos"])
        exported.append(
            {
                "cell": row["cell"],
                "init": row["init"],
                "arm": arm,
                "n_pos": n,
                "R": row["R"],
                "cell_median_R": cell_lookup[(row["cell"], arm, n)],
                "pooled_descriptive_median_R": pooled[(arm, n)],
            }
        )
    write_csv("00_position_dose_hero.csv", exported)
    fig.text(
        0.055,
        0.018,
        "Bold curves: descriptive median over 24 scene-direction rows. Hollow points: eight directed prompt-pair medians (three initial states each). "
        "R is target-axis progress, not full-vector similarity; the intervention edits positions, not individual neurons.",
        fontsize=9.0,
        color=GRAY,
    )
    fig.subplots_adjust(top=0.82, left=0.075, right=0.98, bottom=0.15)
    save(fig, "00_position_dose_hero")


def figure_1_stimulus_intervention_behavior() -> None:
    prefill = load_jsonl("artifacts/pi05_prefill_mediation_2026-08-31/rows.jsonl")
    episodes = load_jsonl("artifacts/pi05_instruction_repair_2026-08-31/state_confirm/episodes.jsonl")
    prompts = load_json("artifacts/vla_stage1/twins/t0_alphabet_soup_1_vs_cream_cheese_1/prompts.json")

    fig = plt.figure(figsize=(15.6, 8.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.03, 1], width_ratios=[0.92, 1.25, 1.25], hspace=0.44, wspace=0.42)
    ax_scene = fig.add_subplot(gs[0, 0])
    ax_prefill = fig.add_subplot(gs[0, 1:])
    ax_flow = fig.add_subplot(gs[1, 0])
    ax_behavior = fig.add_subplot(gs[1, 1:])
    claim_title(
        fig,
        "The readable instruction is not the state that later controls the robot",
        "Same pixels and robot state; only the instruction or a targeted internal state changes.",
    )

    image = mpimg.imread(ROOT / "artifacts/vla_stage1/twins/t0_alphabet_soup_1_vs_cream_cheese_1/ambiguous_init0.png")
    ax_scene.imshow(image)
    ax_scene.axis("off")
    panel_label(ax_scene, "a")
    ax_scene.set_title("One scene, two valid commands", loc="left", pad=9)
    ax_scene.text(
        0.0,
        -0.08,
        f"A  {prompts['A']}\nB  {prompts['B']}",
        transform=ax_scene.transAxes,
        va="top",
        fontsize=9.7,
        linespacing=1.45,
        wrap=True,
    )

    conditions = [
        ("Run B\nfrom input", "prefill_instr_src", MID),
        ("Restore A\ntext K/V", "prefill_restore_instr_dst", ORANGE),
        ("Restore A\nnon-text K/V", "prefill_restore_noninstr_dst", BLUE),
        ("Restore all A\nimage K/V", "prefill_restore_img_dst", BLUE),
        ("Matched small\nrandom subset", "prefill_restore_randimg_dst", MID),
    ]
    plotted = []
    for i, (label, condition, color) in enumerate(conditions):
        values = np.array([float(r["D_dst_10"]) for r in prefill if r["condition"] == condition])
        ax_prefill.scatter(i + jitter(len(values), seed=11 + i), values, s=13, color=color, alpha=0.22, edgecolors="none")
        median = float(np.median(values))
        q1, q3 = np.quantile(values, [0.25, 0.75])
        ax_prefill.vlines(i, q1, q3, color=color, lw=5, zorder=3)
        ax_prefill.scatter(i, median, s=72, color=color, edgecolor=WHITE, linewidth=1.2, zorder=4)
        ax_prefill.text(i, min(1.075, median + 0.07), f"{median:.3f}", ha="center", fontsize=9, weight="bold")
        plotted.extend({"condition": condition, "unit": j, "D_to_A": v} for j, v in enumerate(values))
    ax_prefill.axhspan(-0.04, 0.12, color=PALE_BLUE, zorder=-2)
    ax_prefill.axhspan(0.88, 1.07, color="#F2F3F5", zorder=-2)
    ax_prefill.text(4.47, 0.035, "A-like", ha="right", va="center", color=BLUE, weight="bold")
    ax_prefill.text(4.47, 0.965, "B-like", ha="right", va="center", color=GRAY, weight="bold")
    ax_prefill.set_ylim(-0.04, 1.08)
    ax_prefill.set_xlim(-0.5, 4.5)
    ax_prefill.set_xticks(range(5), [x[0] for x in conditions])
    ax_prefill.set_ylabel("Normalized distance from clean A action")
    ax_prefill.set_title("After B-prefill, restoring A's text state does almost nothing", loc="left")
    ax_prefill.grid(axis="y", color=LIGHT, lw=0.8)
    panel_label(ax_prefill, "b")
    write_csv("01b_prefill_units.csv", plotted)

    ax_flow.axis("off")
    panel_label(ax_flow, "c")
    ax_flow.set_title("The intervention", loc="left", pad=9)
    boxes = [
        (0.05, 0.77, "instruction", PALE_ORANGE, ORANGE),
        (0.05, 0.46, "prefill mixing", "#F3F0FA", PURPLE),
        (0.05, 0.15, "broad image-\nassociated K/V", PALE_BLUE, BLUE),
    ]
    for x, y, text, face, edge in boxes:
        patch = FancyBboxPatch((x, y), 0.78, 0.16, boxstyle="round,pad=0.025", facecolor=face, edgecolor=edge, lw=1.5)
        ax_flow.add_patch(patch)
        ax_flow.text(x + 0.39, y + 0.08, text, ha="center", va="center", fontsize=10.5, weight="bold")
    for y1, y2 in [(0.77, 0.62), (0.46, 0.31)]:
        ax_flow.add_patch(FancyArrowPatch((0.44, y1), (0.44, y2), arrowstyle="-|>", mutation_scale=14, color=GRAY, lw=1.4))
    ax_flow.text(0.44, 0.04, "replace K/V at every replan", ha="center", color=GRAY, fontsize=9.5)

    condition_order = ["correct", "conflict", "state_early", "state_live"]
    labels = ["Correct\nprompt", "Conflicting\nprompt", "Repair\nL0–5", "Repair\nL12–17"]
    metrics = [
        ("correct_target_first_touched", "Correct first touch", BLUE),
        ("success", "Task success", TEAL),
    ]
    records = []
    x = np.arange(4)
    width = 0.34
    for offset, (key, label, color) in zip([-width / 2, width / 2], metrics):
        rates = []
        local = []
        for condition in condition_order:
            values = [int(bool(r[key])) for r in episodes if r["condition"] == condition]
            rate = float(np.mean(values))
            rates.append(rate)
            record = {"condition": condition, "metric": key, "successes": sum(values), "episodes": len(values), "rate": rate}
            records.append(record)
            local.append(record)
        bars = ax_behavior.bar(x + offset, rates, width, color=color, label=label, zorder=2)
        for bar, record in zip(bars, local):
            ax_behavior.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.035, f"{record['successes']}/{record['episodes']}", ha="center", fontsize=9, weight="bold")
    ax_behavior.set_ylim(0, 1.12)
    ax_behavior.set_xticks(x, labels)
    ax_behavior.set_ylabel("Fraction of rollouts")
    ax_behavior.set_title("The broad late-state edit changes the robot's full behavior", loc="left")
    ax_behavior.legend(frameon=False, ncol=2, loc="upper center")
    ax_behavior.grid(axis="y", color=LIGHT, lw=0.8, zorder=0)
    ax_behavior.annotate("late repair ≈ clean behavior", xy=(3, 0.9), xytext=(2.18, 0.56), arrowprops=dict(arrowstyle="->", color=GRAY), color=GRAY)
    panel_label(ax_behavior, "d")
    write_csv("01d_closed_loop.csv", records)

    fig.subplots_adjust(top=0.88, left=0.06, right=0.98, bottom=0.08)
    save(fig, "01_stimulus_intervention_behavior")


def figure_2_layers_attention_causality() -> None:
    results = load_json("artifacts/vla_stage2/20260830-094027/libero_goal_confirm/results.json")
    audit = load_json("artifacts/numbers-audit-derived.json")
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.8), gridspec_kw={"width_ratios": [1.05, 1.12, 0.9]})
    claim_title(
        fig,
        "Readability and attention stay on text after causal control has moved elsewhere",
        "Metric: 10-way logistic-probe accuracy (5-fold by initial state), attention mass, and normalized causal repair R.",
    )

    probes = results["probes"]["resid"]
    layers = np.arange(18)
    probe_rows = []
    for segment, label, color, marker in [("INSTR", "instruction positions", ORANGE, "o"), ("IMG", "image positions", BLUE, "s")]:
        acc = np.array([probes[f"L{i}_{segment}"]["acc"] for i in layers])
        null = np.array([probes[f"L{i}_{segment}"]["null95"] for i in layers])
        axes[0].plot(layers, acc, color=color, marker=marker, ms=4.5, lw=2.4, label=label)
        if segment == "IMG":
            axes[0].plot(layers, null, color=MID, ls="--", lw=1.2, label="shuffled-label 95th pct")
        probe_rows.extend({"layer": int(i), "segment": segment, "accuracy": float(acc[i]), "shuffled_null_95": float(null[i])} for i in layers)
    axes[0].set(xlim=(-0.3, 17.3), ylim=(0, 1.04), xlabel="π0.5 prefix layer", ylabel="Held-out task-identity accuracy")
    axes[0].set_xticks(range(0, 18, 2))
    axes[0].set_title("Instruction identity appears at image positions by L1", loc="left")
    axes[0].legend(frameon=False, fontsize=8.8, loc="lower right")
    axes[0].grid(color=LIGHT, lw=0.8)
    axes[0].annotate("0.113 → 0.993", xy=(1, probe_rows[19]["accuracy"]), xytext=(2.2, 0.43), arrowprops=dict(arrowstyle="->", color=BLUE), color=BLUE, weight="bold")
    panel_label(axes[0], "a")
    write_csv("02a_layer_probe_accuracy.csv", probe_rows)

    segments = results["attention_mass_by_layer"]["segments"]
    attention = np.array(results["attention_mass_by_layer"]["clean_A"])
    image = axes[1].imshow(attention, aspect="auto", cmap="magma", origin="lower", vmin=0, vmax=np.quantile(attention, 0.97))
    axes[1].set_xticks(range(len(segments)), ["image", "format", "instruction", "object", "state"], rotation=32, ha="right")
    axes[1].set_yticks(range(18))
    axes[1].set_ylabel("Action-expert layer")
    axes[1].set_title("Where action queries attend in the prefix", loc="left")
    cbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.03)
    cbar.set_label("Mean attention mass")
    axes[1].text(0.02, -0.29, "Mean over 300 units, action steps and heads.\nGroups overlap: object tokens are within instruction.", transform=axes[1].transAxes, fontsize=8.7, color=GRAY)
    panel_label(axes[1], "b")
    attention_rows = [{"layer": layer, **{segments[j]: attention[layer, j] for j in range(len(segments))}} for layer in range(18)]
    write_csv("02b_attention_mass_by_layer.csv", attention_rows)

    stage2 = audit["stage2"]
    specs = [
        ("Text K/V\nall layers", "KV[INSTR]@all", ORANGE),
        ("Text key-output\nknockout", "KO[INSTR]", MID),
        ("Image K/V\nL12–17", "KV[IMG]@12-17", BLUE),
        ("Image K/V\nall layers", "KV[IMG]@all", BLUE),
    ]
    causal_rows = []
    for i, (label, key, color) in enumerate(specs):
        values = np.array(stage2["cell_R_values"][key])
        axes[2].scatter(i + jitter(len(values), 0.11, 50 + i), values, s=38, color=color, alpha=0.65, edgecolors=WHITE, linewidth=0.5)
        axes[2].scatter(i, np.median(values), marker="D", s=82, color=color, edgecolor=WHITE, linewidth=0.8, zorder=4)
        causal_rows.extend({"condition": key, "directed_prompt_pair": j, "repair_R": value} for j, value in enumerate(values))
    axes[2].axhspan(0.8, 1.08, color=PALE_BLUE, zorder=-2)
    axes[2].axhline(0, color=MID, lw=1)
    axes[2].set_ylim(-0.08, 1.08)
    axes[2].set_xticks(range(4), [x[0] for x in specs], rotation=24, ha="right")
    axes[2].set_ylabel("Normalized causal repair R")
    axes[2].set_title("But only image-state swaps move action", loc="left")
    axes[2].grid(axis="y", color=LIGHT, lw=0.8)
    axes[2].text(0.05, 0.93, "text ≈ 0", transform=axes[2].transAxes, color=ORANGE, weight="bold")
    axes[2].text(0.97, 0.93, "image ≈ 1", transform=axes[2].transAxes, color=BLUE, weight="bold", ha="right")
    panel_label(axes[2], "c")
    write_csv("02c_causal_repair_cells.csv", causal_rows)

    fig.subplots_adjust(top=0.78, left=0.06, right=0.98, bottom=0.22, wspace=0.34)
    save(fig, "02_layers_attention_causality")


def figure_3_distributed_field() -> None:
    dose = load_jsonl("artifacts/pi05_mediation_2026-08-31/dose1/rows.jsonl")
    mediation = load_jsonl("artifacts/pi05_mediation_2026-08-31/run2/rows.jsonl")
    fig, (ax_dose, ax_controls) = plt.subplots(1, 2, figsize=(15.2, 6.2), gridspec_kw={"width_ratios": [1.15, 1]})
    claim_title(
        fig,
        "The causal carrier behaves like a distributed field, not an object-local feature",
        "Selected image positions do not outperform count-matched random positions; the whole field is the discontinuity.",
    )

    denominator = defaultdict(float)
    for row in dose:
        key = (row["cell"], row["init"])
        denominator[key] = max(denominator[key], float(row["l2_to_B"]))
    normalized = []
    for row in dose:
        item = dict(row)
        item["D_B"] = float(row["l2_to_B"]) / denominator[(row["cell"], row["init"])]
        normalized.append(item)
    cell_dose = grouped_medians(normalized, ("cell", "arm", "n_pos"), "D_B")
    counts = [4, 8, 16, 32, 64, 128, 256, 512]
    for arm, label, color, marker in [("LOC", "object-selected", BLUE, "o"), ("RAND", "count-matched random", ORANGE, "s")]:
        for cell in sorted({r["cell"] for r in cell_dose}):
            values = [next(r["D_B"] for r in cell_dose if r["cell"] == cell and r["arm"] == arm and r["n_pos"] == n) for n in counts]
            ax_dose.plot(counts, 1 - np.array(values), color=color, alpha=0.12, lw=1)
        medians = np.array([np.median([r["D_B"] for r in cell_dose if r["arm"] == arm and r["n_pos"] == n]) for n in counts])
        ax_dose.plot(counts, 1 - medians, color=color, marker=marker, lw=2.7, ms=6, label=label, zorder=4)
    ax_dose.axvline(512, color=GRAY, lw=1, ls="--")
    ax_dose.set_xscale("log", base=2)
    ax_dose.set_xticks(counts, [str(n) for n in counts])
    ax_dose.set_ylim(-0.08, 0.86)
    ax_dose.set_xlabel("Image positions replaced (of 512)")
    ax_dose.set_ylabel("Donor recovery  (1 − distance to B)")
    ax_dose.set_title("Nothing smaller than the full 512-position field works", loc="left")
    ax_dose.legend(frameon=False, loc="upper left")
    ax_dose.grid(color=LIGHT, lw=0.8)
    ax_dose.annotate("selected ≈ random", xy=(128, 0.05), xytext=(22, 0.30), arrowprops=dict(arrowstyle="->", color=GRAY), color=GRAY)
    ax_dose.annotate("full-field jump", xy=(512, 0.744), xytext=(155, 0.66), arrowprops=dict(arrowstyle="->", color=INK), weight="bold")
    panel_label(ax_dose, "a")
    write_csv("03a_position_dose_cells.csv", cell_dose)

    baselines = {(r["cell"], r["init"]): float(r["l2_to_A"]) for r in mediation if r["condition"] == "C1_clean_B"}
    mediation_norm = []
    for row in mediation:
        item = dict(row)
        item["D_B"] = float(row["l2_to_B"]) / baselines[(row["cell"], row["init"])]
        mediation_norm.append(item)
    cell_medians = grouped_medians(mediation_norm, ("cell", "condition"), "D_B")
    specs = [
        ("whole image\ncarrier", "C3_carrier", BLUE),
        ("object-local\ncarrier", "C3L_carrier_local", MID),
        ("reset proposed\nmediator", "C4_reset", ORANGE),
        ("dead site", "C5_deadsite", MID),
        ("random carrier", "C6_rand_carrier", MID),
        ("unrelated\ndonor", "C7_unrel_carrier", MID),
    ]
    exported = []
    for i, (label, condition, color) in enumerate(specs):
        values = [r["D_B"] for r in cell_medians if r["condition"] == condition]
        ax_controls.scatter(i + jitter(len(values), 0.13, 31 + i), values, s=28, color=color, alpha=0.60, edgecolors=WHITE, linewidth=0.4)
        ax_controls.scatter(i, np.median(values), marker="D", s=78, color=color, edgecolor=INK if condition == "C3_carrier" else WHITE, linewidth=0.8, zorder=4)
        exported.extend({"condition": condition, "cell_index": j, "cell_median_D_B": value} for j, value in enumerate(values))
    ax_controls.axhspan(-0.08, 0.35, color=PALE_BLUE, zorder=-2)
    ax_controls.axhline(0.35, color=BLUE, lw=1, ls="--")
    ax_controls.text(5.48, 0.12, "B-like endpoint", ha="right", color=BLUE, fontsize=9.5, weight="bold")
    ax_controls.set_xlim(-0.5, 5.5)
    ax_controls.set_ylim(-0.08, 1.45)
    ax_controls.set_xticks(range(6), [x[0] for x in specs], rotation=20, ha="right")
    ax_controls.set_ylabel("Distance to donor B  (lower is better)")
    ax_controls.set_title("The broad effect survives matched causal controls", loc="left")
    ax_controls.grid(axis="y", color=LIGHT, lw=0.8)
    panel_label(ax_controls, "b")
    write_csv("03b_mediation_controls_cells.csv", exported)

    fig.subplots_adjust(top=0.82, left=0.07, right=0.98, bottom=0.17, wspace=0.30)
    save(fig, "03_distributed_image_field")


def figure_4_geometry_without_control() -> None:
    selection = load_jsonl("artifacts/pi05_sonar_lite_source_mediator_v1/selection_rows.jsonl")
    screen = load_jsonl("artifacts/pi05_sonar_lite_source_mediator_v1/screen_rows.jsonl")
    curvature = load_jsonl("artifacts/pi05_lean_midpoint_curvature_2026-09-04_v1/rows.jsonl")
    action = load_jsonl("artifacts/pi05_curvature_action_2026-09-04_v2/rows.jsonl")
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.8))
    claim_title(
        fig,
        "We repeatedly found real geometry that was not the causal mechanism",
        "Association beat matched controls, but it did not reach the intervention endpoint or reliably govern action.",
    )

    selected = [r for r in selection if int(r["layer"]) == 13 and int(r["rank"]) == 16]
    geometry_rows = []
    for cell in sorted({r["cell"] for r in selected}):
        rows = [r for r in selected if r["cell"] == cell]
        fit = float(np.median([r["cos_fit"] for r in rows]))
        wrong = float(np.median([r["cos_wrong"] for r in rows]))
        axes[0].scatter(wrong, fit, s=58, color=PURPLE, alpha=0.82, edgecolor=WHITE, linewidth=0.7)
        geometry_rows.append({"cell": cell, "median_cos_wrong": wrong, "median_cos_fit": fit})
    lo, hi = 0.12, 0.72
    axes[0].plot([lo, hi], [lo, hi], color=MID, ls="--", lw=1.2)
    axes[0].fill_between([lo, hi], [lo, hi], [hi, hi], color="#F3F0FA", alpha=0.7)
    axes[0].set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Wrong-prompt cosine", ylabel="Fitted-component cosine")
    axes[0].set_title("The rank-16 component is specific", loc="left")
    axes[0].text(0.17, 0.64, "fit > wrong\nin 12/12 cells", color=PURPLE, weight="bold")
    axes[0].grid(color=LIGHT, lw=0.8)
    panel_label(axes[0], "a")
    write_csv("04a_sonar_geometry_cells.csv", geometry_rows)

    screen_cells = grouped_medians(screen, ("cell", "condition"), "nL2_to_B")
    specs = [("clean B", "clean_B", BLUE), ("ceiling\ntransplant", "insert_ceiling", SKY), ("rank-16\ninsert", "insert_fit", PURPLE), ("wrong\nprompt", "insert_wrong", MID)]
    causal_rows = []
    for i, (label, condition, color) in enumerate(specs):
        values = [r["nL2_to_B"] for r in screen_cells if r["condition"] == condition]
        axes[1].scatter(i + jitter(len(values), 0.13, 71 + i), values, s=29, color=color, alpha=0.6, edgecolors=WHITE, linewidth=0.4)
        axes[1].scatter(i, np.median(values), marker="D", s=75, color=color, edgecolor=WHITE, linewidth=0.7, zorder=4)
        causal_rows.extend({"condition": condition, "cell_index": j, "cell_median_D_B": value} for j, value in enumerate(values))
    axes[1].axhspan(-0.05, 0.35, color=PALE_BLUE, zorder=-2)
    axes[1].axhline(0.35, color=BLUE, ls="--", lw=1)
    axes[1].set_ylim(-0.05, 1.28)
    axes[1].set_xticks(range(4), [x[0] for x in specs])
    axes[1].set_ylabel("Distance to donor B")
    axes[1].set_title("But inserting it lands at B in 0/12", loc="left")
    axes[1].grid(axis="y", color=LIGHT, lw=0.8)
    panel_label(axes[1], "b")
    write_csv("04b_sonar_causal_cells.csv", causal_rows)

    curvature_cells = grouped_medians(curvature, ("cell",), "curvature_to_chord_ratio")
    action_cells = grouped_medians(action, ("cell",), "action_change_over_endpoint_distance")
    curvature_map = {r["cell"]: r["curvature_to_chord_ratio"] for r in curvature_cells}
    action_map = {r["cell"]: r["action_change_over_endpoint_distance"] for r in action_cells}
    scatter_rows = []
    for cell in sorted(set(curvature_map) & set(action_map)):
        x, y = curvature_map[cell], action_map[cell]
        color = ORANGE if y >= 0.1 else MID
        axes[2].scatter(x, y, s=58, color=color, edgecolor=WHITE, linewidth=0.7)
        scatter_rows.append({"cell": cell, "curvature_ratio": x, "action_sensitivity": y, "passes_action_threshold": y >= 0.1})
    axes[2].axvline(0.1, color=GRAY, ls="--", lw=1)
    axes[2].axhline(0.1, color=GRAY, ls="--", lw=1)
    axes[2].set_xlim(0.08, 0.29)
    axes[2].set_ylim(-0.01, 0.56)
    axes[2].set_xlabel("Representation curvature / chord")
    axes[2].set_ylabel("Action change / A–B action distance")
    axes[2].set_title("All paths bend; only 5/12 affect action ≥ 0.1", loc="left")
    axes[2].grid(color=LIGHT, lw=0.8)
    panel_label(axes[2], "c")
    write_csv("04c_curvature_action_cells.csv", scatter_rows)

    fig.subplots_adjust(top=0.79, left=0.06, right=0.985, bottom=0.14, wspace=0.34)
    save(fig, "04_geometry_without_control")


def figure_5_route_and_reader() -> None:
    writer = load_jsonl("artifacts/pi05_attention_pathway_2026-09-04_v1/writer_screen_rows.jsonl")
    reader = load_jsonl("artifacts/pi05_attention_resolution_2026-09-04_v1/reader_development_rows.jsonl")
    fig, (ax_writer, ax_reader) = plt.subplots(1, 2, figsize=(15.3, 6.4), gridspec_kw={"width_ratios": [1.12, 1]})
    claim_title(
        fig,
        "The direct route is necessary; the writer is unresolved and the reader is distributed",
        "Blocking instruction→image communication removes the decision, but a compact seed does not restore it.",
    )

    writer_cells = grouped_medians(writer, ("cell", "condition"), "preference_A")
    specs = [
        ("Clean B", "clean_B", MID),
        ("Block instruction → other", "writer_block_other", MID),
        ("Block instruction → image", "writer_block_image", BLUE),
        ("Block all instruction → non-text", "writer_block_all_noninstruction", BLUE),
        ("Restore selected block-0 image heads", "writer_rescue_l0_image_selected", ORANGE),
    ]
    writer_export = []
    for i, (label, condition, color) in enumerate(specs):
        values = [r["preference_A"] for r in writer_cells if r["condition"] == condition]
        y = len(specs) - 1 - i
        ax_writer.scatter(values, y + jitter(len(values), 0.10, 110 + i), s=30, color=color, alpha=0.60, edgecolors=WHITE, linewidth=0.4)
        ax_writer.scatter(np.median(values), y, marker="D", s=82, color=color, edgecolor=INK if condition == "writer_block_image" else WHITE, linewidth=0.8, zorder=4)
        writer_export.extend({"condition": condition, "cell_index": j, "cell_median_preference_A": value} for j, value in enumerate(values))
    ax_writer.axvspan(-1.05, 0, color="#F2F3F5", zorder=-2)
    ax_writer.axvspan(0, 1.05, color=PALE_BLUE, zorder=-2)
    ax_writer.axvline(0, color=GRAY, lw=1)
    ax_writer.set_xlim(-1.05, 1.05)
    ax_writer.set_yticks(range(5), [x[0] for x in reversed(specs)])
    ax_writer.set_xlabel("Action preference   B-like  ←   0   →  A-like")
    ax_writer.set_title("Only blocking the image route flips B toward A", loc="left")
    ax_writer.grid(axis="x", color=LIGHT, lw=0.8)
    ax_writer.annotate("block-0 rescue still A-like", xy=(0.92, 0), xytext=(0.18, 0.78), arrowprops=dict(arrowstyle="->", color=ORANGE), color=ORANGE, weight="bold")
    panel_label(ax_writer, "a")
    write_csv("05a_writer_cells.csv", writer_export)

    ks = [1, 2, 4, 8, 13]
    reader_export = []
    for kind, endpoint, label, color, linestyle, marker in [
        ("top", "remove", "selected: removal reaches A", BLUE, "-", "o"),
        ("random", "remove", "random: removal reaches A", MID, "--", "o"),
        ("top", "rescue", "selected: rescue reaches B", PURPLE, "-", "s"),
        ("random", "rescue", "random: rescue reaches B", MID, ":", "s"),
    ]:
        fractions = []
        for k in ks:
            condition = f"reader_{endpoint}_{kind}_{k}"
            rows = grouped_medians([r for r in reader if r["condition"] == condition], ("cell",), "preference_A")
            if endpoint == "remove":
                fraction = float(np.mean([r["preference_A"] > 0 for r in rows]))
            else:
                fraction = float(np.mean([r["preference_A"] < 0 for r in rows]))
            fractions.append(fraction)
            reader_export.append({"edge_type": kind, "intervention": endpoint, "k_edges": k, "endpoint_fraction": fraction})
        ax_reader.plot(ks, fractions, color=color, ls=linestyle, marker=marker, lw=2.5 if kind == "top" else 1.7, ms=6, label=label)
    ax_reader.axhline(0.75, color=GRAY, lw=1, ls="--")
    ax_reader.axvline(8, color=ORANGE, lw=1, ls="--")
    ax_reader.set_xticks(ks)
    ax_reader.set_ylim(-0.03, 1.06)
    ax_reader.set_xlabel("Number of image→action layer/head edges")
    ax_reader.set_ylabel("Fraction of directed cells at intended side")
    ax_reader.set_title("A reader appears only after several edges", loc="left")
    ax_reader.legend(frameon=False, fontsize=9, loc="lower right")
    ax_reader.grid(color=LIGHT, lw=0.8)
    ax_reader.annotate("k=8 passes development gate\nnot prompt-pair confirmation", xy=(8, 0.83), xytext=(4.4, 0.51), arrowprops=dict(arrowstyle="->", color=ORANGE), color=ORANGE, weight="bold", fontsize=9.5)
    panel_label(ax_reader, "b")
    write_csv("05b_reader_topk.csv", reader_export)

    fig.subplots_adjust(top=0.80, left=0.13, right=0.98, bottom=0.13, wspace=0.32)
    save(fig, "05_route_and_reader")


def figure_6_architecture_boundary() -> None:
    audit = load_json("artifacts/numbers-audit-derived.json")
    oft = load_jsonl("artifacts/oft_downstream_kv/v2_20260831/rows.jsonl")
    fig, ax = plt.subplots(figsize=(10.8, 6.5))
    claim_title(
        fig,
        "The dominant route reverses across two VLA checkpoints",
        "The π0.5 image-carrier result is not a universal property of vision-language-action models.",
    )

    pi = audit["stage2"]["cell_R_values"]
    pi_data = {
        "Text positions": np.array(pi["KV[INSTR]@all"]),
        "Image positions": np.array(pi["KV[IMG]@all"]),
    }
    condition_map = {"Text positions": "kv_instr_8_31", "Image positions": "kv_img_8_31"}
    equal = [r for r in oft if r["panel"] == "equal"]
    oft_data = {}
    for label, condition in condition_map.items():
        task_medians = []
        for task in sorted({r["task"] for r in equal if r["condition"] == condition}):
            values = [1 - float(r["D_src"]) for r in equal if r["condition"] == condition and r["task"] == task]
            task_medians.append(float(np.median(values)))
        oft_data[label] = np.array(task_medians)

    x = np.array([0, 1])
    exported = []
    for offset, (model, data, color) in zip([-0.18, 0.18], [("π0.5", pi_data, BLUE), ("OpenVLA-OFT", oft_data, ORANGE)]):
        medians = []
        for i, label in enumerate(["Text positions", "Image positions"]):
            values = data[label]
            medians.append(float(np.median(values)))
            ax.scatter(np.full(len(values), x[i] + offset) + jitter(len(values), 0.055, 180 + i + (0 if model == "π0.5" else 10)), values, s=38, color=color, alpha=0.64, edgecolors=WHITE, linewidth=0.5)
            ax.scatter(x[i] + offset, medians[-1], marker="D", s=90, color=color, edgecolor=WHITE, linewidth=0.8, zorder=4)
            exported.extend({"model": model, "patched_positions": label, "unit_index": j, "normalized_donor_recovery": value} for j, value in enumerate(values))
        ax.plot(x + offset, medians, color=color, lw=2.2, alpha=0.85, label=model)
    ax.axhspan(0.8, 1.08, color=PALE_BLUE, zorder=-2)
    ax.axhline(0, color=MID, lw=1)
    ax.set_xlim(-0.52, 1.52)
    ax.set_ylim(-0.12, 1.08)
    ax.set_xticks(x, ["Patch text-position K/V", "Patch image-position K/V"])
    ax.set_ylabel("Normalized recovery toward donor action")
    ax.set_title("Same intervention label, opposite causal route", loc="left")
    ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.5, 0.52), ncol=2)
    ax.grid(axis="y", color=LIGHT, lw=0.8)
    ax.text(-0.40, 0.95, "OpenVLA: text route", color=ORANGE, weight="bold")
    ax.text(1.40, 0.95, "π0.5: image-associated route", color=BLUE, weight="bold", ha="right")
    write_csv("06_architecture_boundary.csv", exported)

    fig.subplots_adjust(top=0.78, left=0.11, right=0.97, bottom=0.14)
    save(fig, "06_architecture_boundary")


def figure_7_matched_band_transform() -> None:
    rows = load_jsonl("artifacts/pi05_matched_band_transform_2026-09-04_v1/rows.jsonl")
    summary = load_json("artifacts/pi05_matched_band_transform_2026-09-04_v1/summary.json")
    cells = sorted({row["cell"] for row in rows})

    def cell_values(condition: str) -> dict[str, float]:
        return {
            cell: float(np.median([
                row["progress_to_B"] for row in rows
                if row["cell"] == cell and row["condition"] == condition
            ]))
            for cell in cells
        }

    conditions = {
        "random_norm_matched": cell_values("random_norm_matched"),
        "mismatched_full_norm_matched": cell_values("mismatched_full_norm_matched"),
        "matched_full": cell_values("matched_full"),
        "matched_attention_only": cell_values("matched_attention_only"),
    }
    fig = plt.figure(figsize=(15.8, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.78, 1.18, 1.0], wspace=0.38)
    ax_flow = fig.add_subplot(gs[0, 0])
    ax_transfer = fig.add_subplot(gs[0, 1])
    ax_mlp = fig.add_subplot(gs[0, 2])
    claim_title(
        fig,
        "A broad layer-6→8 instruction update transfers across scenes—and the MLP matters",
        "Same instruction pair, new initial scene. The intervention is a 512-position message, not a compact feature.",
    )

    ax_flow.axis("off")
    panel_label(ax_flow, "a")
    ax_flow.set_title("The fixed test", loc="left", pad=8)
    boxes = [
        (0.06, 0.74, 0.84, 0.14, "hold entering image state fixed\nat L5", PALE_ORANGE, ORANGE),
        (0.06, 0.48, 0.84, 0.14, "run layers 6–8\nwith instruction A or B", "#F3F0FA", PURPLE),
        (0.06, 0.22, 0.84, 0.14, "add B−A image update\nto a clean-A host at L8", PALE_BLUE, BLUE),
    ]
    for x, y, w, h, text, face, edge in boxes:
        ax_flow.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025", facecolor=face, edgecolor=edge, lw=1.5))
        ax_flow.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.4, weight="bold", linespacing=1.2)
    for y1, y2 in [(0.74, 0.62), (0.48, 0.36)]:
        ax_flow.add_patch(FancyArrowPatch((0.48, y1), (0.48, y2), arrowstyle="-|>", mutation_scale=14, color=GRAY, lw=1.4))
    ax_flow.text(0.48, 0.10, "compare matching scene,\nnew scene, attention-only, and random", ha="center", va="center", color=GRAY, fontsize=9.2)

    specs = [
        ("Random", "random_norm_matched", MID),
        ("Other scene\nfull", "mismatched_full_norm_matched", TEAL),
        ("Matching scene\nfull", "matched_full", BLUE),
    ]
    x = np.arange(len(specs))
    export_transfer = []
    for cell in cells:
        values = [conditions[key][cell] for _, key, _ in specs]
        ax_transfer.plot(x, values, color=MID, alpha=0.32, lw=1.1, zorder=1)
        export_transfer.append({"cell": cell, **{key: conditions[key][cell] for _, key, _ in specs}})
    for i, (label, key, color) in enumerate(specs):
        values = np.asarray(list(conditions[key].values()))
        ax_transfer.scatter(i + jitter(len(values), 0.07, 240 + i), values, s=28, color=color, alpha=0.55, edgecolors=WHITE, linewidth=0.4, zorder=2)
        med = summary["conditions"][key]["median_cell_progress_to_B"]
        lo, hi = summary["conditions"][key]["bootstrap_95pct_interval"]
        ax_transfer.errorbar(i, med, yerr=[[med - lo], [hi - med]], color=color, lw=2.2, capsize=5, zorder=4)
        ax_transfer.scatter(i, med, marker="D", s=90, color=color, edgecolor=WHITE, linewidth=0.8, zorder=5)
        ax_transfer.text(i, hi + 0.055, f"{med:.3f}", ha="center", weight="bold", color=color)
    ax_transfer.axhline(0, color=GRAY, lw=1)
    ax_transfer.axhline(1, color=GRAY, lw=1, ls="--")
    ax_transfer.text(2.34, 1.0, "clean B", va="center", color=GRAY, fontsize=9)
    ax_transfer.set_xlim(-0.35, 2.35)
    ax_transfer.set_ylim(-0.12, 1.08)
    ax_transfer.set_xticks(x, [item[0] for item in specs])
    ax_transfer.set_ylabel("Progress along clean A→B action axis")
    ax_transfer.set_title("A message from another scene still moves action", loc="left")
    ax_transfer.grid(axis="y", color=LIGHT, lw=0.8)
    ax_transfer.text(0.02, 0.93, "other-scene > random in 12/12 cells\nexact sign p = 0.00049", transform=ax_transfer.transAxes, va="top", color=TEAL, weight="bold", fontsize=9.2)
    panel_label(ax_transfer, "b")
    write_csv("07b_matched_transform_cells.csv", export_transfer)

    attention = conditions["matched_attention_only"]
    full = conditions["matched_full"]
    export_mlp = []
    lo = min(min(attention.values()), min(full.values())) - 0.03
    hi = max(max(attention.values()), max(full.values())) + 0.04
    ax_mlp.plot([lo, hi], [lo, hi], color=MID, ls="--", lw=1.2)
    for cell in cells:
        ax_mlp.scatter(attention[cell], full[cell], s=62, color=PURPLE, alpha=0.82, edgecolor=WHITE, linewidth=0.7)
        export_mlp.append({"cell": cell, "attention_only_progress": attention[cell], "full_attention_plus_mlp_progress": full[cell]})
    ax_mlp.set_xlim(lo, hi)
    ax_mlp.set_ylim(lo, hi)
    ax_mlp.set_xlabel("Matching attention-only progress")
    ax_mlp.set_ylabel("Matching full attention+MLP progress")
    ax_mlp.set_title("Full blocks beat attention-only in 12/12", loc="left")
    ax_mlp.grid(color=LIGHT, lw=0.8)
    ax_mlp.text(0.04, 0.92, "median gain 0.224\nexact sign p = 0.00049", transform=ax_mlp.transAxes, va="top", color=PURPLE, weight="bold")
    panel_label(ax_mlp, "c")
    write_csv("07c_mlp_ablation_cells.csv", export_mlp)

    fig.subplots_adjust(top=0.80, left=0.06, right=0.98, bottom=0.14)
    save(fig, "07_matched_band_transform")


def figure_8_action_to_behavior() -> None:
    action_rows = load_jsonl("artifacts/pi05_matched_band_transform_2026-09-04_v1/rows.jsonl")
    rollout_rows = load_jsonl("artifacts/pi05_matched_band_rollout_screen_2026-09-04_v1/episodes.jsonl")
    cells = sorted({row["cell"] for row in rollout_rows})
    action_progress = {
        cell: float(np.median([
            row["progress_to_B"] for row in action_rows
            if row["cell"] == cell and row["condition"] == "matched_full"
        ]))
        for cell in cells
    }
    rollout = {(row["cell"], row["condition"]): row for row in rollout_rows}

    fig, (ax_cells, ax_behavior) = plt.subplots(
        1, 2, figsize=(14.8, 6.5), gridspec_kw={"width_ratios": [1.25, 0.9]}
    )
    claim_title(
        fig,
        "The edit sometimes changes the first target—but never completes the new task",
        "Exploratory breadth screen: one held-back initial state for each of 12 directed Goal prompt pairs.",
    )

    ordered = sorted(cells, key=lambda cell: action_progress[cell])
    export_cells = []
    for y, cell in enumerate(ordered):
        b_first = bool(rollout[(cell, "matched_full")]["B_target_first_touched"])
        color = TEAL if b_first else MID
        marker = "D" if b_first else "o"
        ax_cells.scatter(action_progress[cell], y, s=74 if b_first else 48, color=color, marker=marker,
                         edgecolor=WHITE, linewidth=0.7, zorder=3)
        export_cells.append({
            "cell": cell,
            "offline_matched_action_progress_to_B": action_progress[cell],
            "matched_rollout_B_target_first_touched": int(b_first),
            "matched_rollout_success_B": int(bool(rollout[(cell, "matched_full")]["success_B"])),
        })
    ax_cells.axvline(0, color=GRAY, lw=1)
    ax_cells.set_yticks(range(len(ordered)), [cell.replace("libero_goal_", "") for cell in ordered])
    ax_cells.set_xlabel("Offline progress along clean A→B action axis")
    ax_cells.set_ylabel("Directed prompt pair")
    ax_cells.set_title("Three initial trajectories touch B first", loc="left")
    ax_cells.grid(axis="x", color=LIGHT, lw=0.8)
    ax_cells.text(
        0.98, 0.03, "◆ B target first\n● not B target first",
        transform=ax_cells.transAxes, ha="right", va="bottom", color=GRAY, fontsize=9.3,
    )
    panel_label(ax_cells, "a")
    write_csv("08a_action_behavior_cells.csv", export_cells)

    condition_order = ["clean_A", "matched_full", "clean_B"]
    labels = ["Clean A\n(conflict)", "Layer-6→8\nedit", "Clean B\n(ceiling)"]
    metrics = [
        ("B_target_first_touched", "B target first", BLUE),
        ("success_B", "Task-B success", TEAL),
    ]
    x = np.arange(len(condition_order))
    width = 0.34
    export_behavior = []
    for offset, (metric, label, color) in zip([-width / 2, width / 2], metrics):
        counts = []
        for condition in condition_order:
            values = [int(bool(row[metric])) for row in rollout_rows if row["condition"] == condition]
            counts.append(sum(values))
            export_behavior.append({
                "condition": condition,
                "metric": metric,
                "count": sum(values),
                "episodes": len(values),
                "rate": float(np.mean(values)),
            })
        bars = ax_behavior.bar(x + offset, np.asarray(counts) / 12.0, width, color=color, label=label, zorder=2)
        for bar, count in zip(bars, counts):
            ax_behavior.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.035,
                f"{count}/12",
                ha="center",
                fontsize=9.5,
                weight="bold",
            )
    ax_behavior.set_ylim(0, 1.12)
    ax_behavior.set_xticks(x, labels)
    ax_behavior.set_ylabel("Fraction of prompt-pair rollouts")
    ax_behavior.set_title("Initial redirection does not become task execution", loc="left")
    ax_behavior.legend(frameon=False, loc="upper left")
    ax_behavior.grid(axis="y", color=LIGHT, lw=0.8, zorder=0)
    ax_behavior.annotate(
        "partial causal effect\nnot policy transfer",
        xy=(1 - width / 2, 0.25),
        xytext=(0.25, 0.58),
        arrowprops=dict(arrowstyle="->", color=ORANGE),
        color=ORANGE,
        weight="bold",
    )
    panel_label(ax_behavior, "b")
    write_csv("08b_rollout_outcomes.csv", export_behavior)

    fig.text(
        0.99, 0.02,
        "Screening result, not confirmation: n=1 initial state per pair; B-first exact sign p=0.25.",
        ha="right", fontsize=8.7, color=GRAY,
    )
    fig.subplots_adjust(top=0.79, left=0.14, right=0.98, bottom=0.14, wspace=0.34)
    save(fig, "08_action_to_behavior")


def figure_9_prompt_pairs_behavior_heatmap() -> None:
    """Show the controlled prompt manipulation and its task-level behavioral effect."""
    stage0 = load_json("artifacts/vla_stage0/20260830-094027/results.json")
    manifest = load_json("artifacts/vla_arbitration/20260830-192300/arbitration_manifest.json")
    raw_rows = load_jsonl("artifacts/vla_stage0/20260830-094027/per_episode.jsonl")

    fig = plt.figure(figsize=(18.0, 10.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.42, 1.25, 1.25], hspace=0.46, wspace=0.58)
    left = gs[:, 0].subgridspec(2, 1, height_ratios=[0.43, 0.57], hspace=0.08)
    ax_scene = fig.add_subplot(left[0])
    ax_prompts = fig.add_subplot(left[1])
    ax_object = fig.add_subplot(gs[0, 1:])
    ax_goal = fig.add_subplot(gs[1, 1:])
    claim_title(
        fig,
        "Changing only the instruction changed what the robot did",
        "π0.5 on LIBERO Object and Goal: 1,200 closed-loop rollouts; each heat-map cell is 20 official initial states.",
    )

    poster = mpimg.imread(ROOT / "media/videos/state-confirmation/poster.png")
    # Exact scene crop from the preserved task-1 poster; no synthetic illustration.
    scene = poster[156:457, 22:323]
    ax_scene.imshow(scene)
    ax_scene.axis("off")
    ax_scene.set_title("Controlled stimulus", loc="left", pad=8)
    panel_label(ax_scene, "a")
    ax_scene.text(
        0.0,
        -0.055,
        "Same camera image, robot state, and action noise;\nonly the instruction changes.",
        transform=ax_scene.transAxes,
        va="top",
        fontsize=9.5,
        color=GRAY,
        style="italic",
        linespacing=1.35,
    )

    ax_prompts.axis("off")
    prompt_blocks = [
        (
            0.93,
            "PAIR 1 — SAME OBJECT SCENE",
            "\u201cpick up the cream cheese and\nplace it in the basket\u201d\n"
            "versus\n"
            "\u201cpick up the tomato sauce and\nplace it in the basket\u201d",
            "Correct: 20/20 success\nConflict: 0/20 success; tomato sauce first 20/20",
        ),
        (
            0.48,
            "PAIR 2 — SAME OBJECT SCENE",
            "\u201cpick up the salad dressing and\nplace it in the basket\u201d\n"
            "versus\n"
            "\u201cpick up the ketchup and\nplace it in the basket\u201d",
            "Correct: 20/20 success\nConflict: 0/20 success; ketchup first 20/20",
        ),
    ]
    for y, heading, prompt_text, outcome in prompt_blocks:
        ax_prompts.text(0.0, y, heading, fontsize=9.0, weight="bold", color=BLUE, va="top")
        ax_prompts.text(0.0, y - 0.075, prompt_text, fontsize=9.2, va="top", linespacing=1.24)
        ax_prompts.text(0.0, y - 0.315, outcome, fontsize=8.7, va="top", color=GRAY, linespacing=1.22)
    ax_prompts.text(
        0.0,
        0.005,
        "\u201cA conflicting instruction can redirect first contact\neven when the physical scene is unchanged.\u201d",
        fontsize=10.2,
        color=ORANGE,
        weight="bold",
        va="bottom",
        linespacing=1.35,
    )

    condition_order = ["correct", "null", "wrong_object"]
    condition_labels = ["Correct\ninstruction", "Empty\ninstruction", "Conflicting\ninstruction"]
    export_rows = []
    images = []
    for ax, suite, title in [
        (ax_object, "libero_object", "LIBERO Object: some tasks persist without the right instruction"),
        (ax_goal, "libero_goal", "LIBERO Goal: behavior depends strongly on the instruction"),
    ]:
        per_task = stage0["suites"][suite]["per_task"]
        task_ids = list(range(10))
        values = np.array(
            [[per_task[str(task_id)][condition]["success"] / per_task[str(task_id)][condition]["n"]
              for condition in condition_order]
             for task_id in task_ids]
        )
        image = ax.imshow(values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        images.append(image)
        ax.set_xticks(range(3), condition_labels)
        labels = []
        for task_id in task_ids:
            prompt = manifest[suite]["tasks"][str(task_id)]["prompt"]
            if suite == "libero_object":
                prompt = prompt.removeprefix("pick up the ").removesuffix(" and place it in the basket")
            else:
                goal_labels = [
                    "open middle drawer",
                    "bowl → stove",
                    "wine bottle → cabinet",
                    "bowl → top drawer",
                    "bowl → cabinet",
                    "push plate forward",
                    "cream cheese → bowl",
                    "turn on stove",
                    "bowl → plate",
                    "wine bottle → rack",
                ]
                prompt = goal_labels[task_id]
            labels.append(f"t{task_id}  {prompt}")
        ax.set_yticks(task_ids, labels, fontsize=8.7)
        ax.set_title(title, loc="left", pad=9)
        ax.set_ylabel("Official LIBERO task")
        ax.tick_params(length=0)
        for task_id in task_ids:
            for j, condition in enumerate(condition_order):
                rec = per_task[str(task_id)][condition]
                rate = rec["success"] / rec["n"]
                ax.text(
                    j,
                    task_id,
                    f"{rec['success']}/{rec['n']}",
                    ha="center",
                    va="center",
                    fontsize=9.0,
                    weight="bold",
                    color=WHITE if rate < 0.25 or rate > 0.78 else INK,
                )
                matching = [
                    row for row in raw_rows
                    if row["suite"] == suite and int(row["task_id"]) == task_id and row["condition"] == condition
                ]
                export_rows.append(
                    {
                        "suite": suite,
                        "task_id": task_id,
                        "task_prompt": manifest[suite]["tasks"][str(task_id)]["prompt"],
                        "condition": condition,
                        "condition_prompt": matching[0]["prompt"] if matching else "",
                        "successes": rec["success"],
                        "episodes": rec["n"],
                        "success_rate": rate,
                    }
                )
        totals = [sum(per_task[str(i)][condition]["success"] for i in task_ids) for condition in condition_order]
        ax.text(
            1.0,
            -0.18,
            "Aggregate: " + "   |   ".join(f"{label.splitlines()[0]} {count}/200" for label, count in zip(condition_labels, totals)),
            transform=ax.transAxes,
            ha="right",
            fontsize=9.0,
            color=GRAY,
        )
        if suite == "libero_object":
            # These are the two concrete pairs printed at left.
            for task_id in [1, 2]:
                ax.add_patch(plt.Rectangle((-0.49, task_id - 0.48), 2.98, 0.96, fill=False, edgecolor=BLUE, lw=1.7))
        panel_label(ax, "b" if suite == "libero_object" else "c")

    cax = fig.add_axes([0.944, 0.235, 0.012, 0.52])
    cbar = fig.colorbar(images[-1], cax=cax)
    cbar.set_label("Simulator task-success rate")
    fig.text(
        0.44,
        0.022,
        "Under conflicting prompts, the robot touched the named alternative first in 55/200 Object rollouts and 185/200 Goal rollouts. "
        "The heat map shows success on the scene's original task, not generic motion.",
        ha="left",
        fontsize=9.0,
        color=GRAY,
    )
    write_csv("09_prompt_pairs_behavior_heatmap.csv", export_rows)
    fig.subplots_adjust(top=0.875, left=0.045, right=0.925, bottom=0.105)
    save(fig, "09_prompt_pairs_behavior_heatmap")


def figure_10_closed_loop_visual_comparison() -> None:
    """Turn the verified four-way rollout capture into a paper-ready example figure."""
    meta = load_json("media/videos/state-confirmation/meta.json")
    poster = mpimg.imread(ROOT / "media/videos/state-confirmation/poster.png")
    panel_by_condition = {panel["condition"]: panel for panel in meta["panels"]}
    specs = [
        (
            "conflict",
            (22, 323),
            "Conflicting prompt",
            "\u201cpick up the tomato sauce\u2026\u201d",
            "tomato sauce first  •  task failed",
            "#B33A3A",
        ),
        (
            "correct",
            (334, 635),
            "Correct prompt",
            "\u201cpick up the cream cheese\u2026\u201d",
            "cream cheese first  •  task complete",
            TEAL,
        ),
        (
            "state_live",
            (646, 947),
            "Late-state repair",
            "external prompt still names tomato sauce",
            "cream cheese first  •  task complete",
            BLUE,
        ),
        (
            "state_early",
            (958, 1259),
            "Early-layer control",
            "same transplant at layers 0–5",
            "tomato sauce first  •  task failed",
            ORANGE,
        ),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 5.5))
    claim_title(
        fig,
        "The same scene: prompt conflict fails, late-state repair succeeds",
        "π0.5, LIBERO Object task 1, initial state 20; independent closed-loop runs share the registered scene and noise-seed rule.",
    )
    exported = []
    for i, (condition, (x0, x1), title, context, outcome, color) in enumerate(specs):
        ax = axes[i]
        ax.imshow(poster[156:457, x0:x1])
        ax.axis("off")
        ax.set_title(title, loc="left", pad=8, color=color, fontsize=12.3, weight="bold")
        ax.text(0, -0.06, context, transform=ax.transAxes, va="top", fontsize=9.2, color=GRAY)
        ax.text(0, -0.13, outcome, transform=ax.transAxes, va="top", fontsize=9.3, color=color, weight="bold")
        panel_label(ax, "abcd"[i])
        saved = panel_by_condition[condition]["saved_record"]
        exported.append(
            {
                "panel": "abcd"[i],
                "condition": condition,
                "prompt": saved["prompt"],
                "donor_layers": panel_by_condition[condition]["donor_layers"],
                "success": int(bool(saved["success"])),
                "first_touch_set": saved["first_touch_set"],
                "correct_target_first_touched": int(bool(saved["correct_target_first_touched"])),
                "conflict_target_first_touched": int(bool(saved["conflict_target_first_touched"])),
                "seed": saved["seed"],
                "reproduction_check": panel_by_condition[condition]["reproduction_check"],
            }
        )
    fig.text(
        0.5,
        0.028,
        "At every replan, the repair copies only all 512 image-position K/V entries from the same-observation correct-prompt donor. "
        "No donor action or stored trajectory is copied. All four recaptures match the archived outcomes.",
        ha="center",
        fontsize=9.0,
        color=GRAY,
    )
    write_csv("10_closed_loop_visual_conditions.csv", exported)
    fig.subplots_adjust(top=0.80, left=0.045, right=0.98, bottom=0.17, wspace=0.075)
    save(fig, "10_closed_loop_visual_comparison")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    style()
    figure_0_position_dose_hero()
    figure_1_stimulus_intervention_behavior()
    figure_2_layers_attention_causality()
    figure_3_distributed_field()
    figure_4_geometry_without_control()
    figure_5_route_and_reader()
    figure_6_architecture_boundary()
    figure_7_matched_band_transform()
    figure_8_action_to_behavior()
    figure_9_prompt_pairs_behavior_heatmap()
    figure_10_closed_loop_visual_comparison()
    print("Wrote eleven claim-first figures and source CSVs to", OUT)


if __name__ == "__main__":
    main()
