#!/usr/bin/env python3
"""Compose captured state-confirmation frames into blog-ready video assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


BG = (255, 255, 255)
INK = (23, 32, 51)
MUTED = (95, 107, 128)
RULE = (219, 225, 234)
RED = (190, 67, 67)
GREEN = (43, 138, 110)
BLUE = (50, 116, 193)
ORANGE = (205, 125, 42)

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def wrap(draw: ImageDraw.ImageDraw, text: str, selected_font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or draw.textlength(candidate, font=selected_font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_lines(draw, x, y, text, selected_font, color, width, line_height):
    for line in wrap(draw, text, selected_font, width):
        draw.text((x, y), line, font=selected_font, fill=color)
        y += line_height
    return y


def panel_spec(record: dict) -> dict:
    saved = record["saved_record"]
    condition = record["condition"]
    first = (
        ", ".join(name.removesuffix("_1").replace("_", " ") for name in saved["first_touch_set"])
        if saved["first_touch_set"]
        else "none"
    )
    observed = f"Observed: first touch {first}; {'task complete' if saved['success'] else 'task failed'}"
    specs = {
        "conflict": {
            "title": "Conflicting prompt",
            "subtitle": "Prompt names tomato sauce; no edit",
            "color": RED,
        },
        "correct": {
            "title": "Correct prompt",
            "subtitle": "Prompt names cream cheese; no edit",
            "color": GREEN,
        },
        "state_live": {
            "title": "Live-band repair",
            "subtitle": "Wrong prompt; donor image K/V at 12-17",
            "color": BLUE,
        },
        "state_early": {
            "title": "Early-band control",
            "subtitle": "Wrong prompt; donor image K/V at 0-5",
            "color": ORANGE,
        },
    }
    return {**specs[condition], "observed": observed, "record": record}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.raw / "render_meta.json").read_text())
    records = {row["condition"]: row for row in metadata["panels"]}
    order = ["conflict", "correct", "state_live", "state_early"]
    if set(records) != set(order):
        raise AssertionError(f"missing expected panels: have {sorted(records)}")

    arrays = [np.load(args.raw / f"{condition}.npy")[:, ::-1].copy() for condition in order]
    panel_specs = [panel_spec(records[condition]) for condition in order]
    timeline = list(range(0, max(len(frames) for frames in arrays), args.stride))

    panel_size = 300
    margin = 22
    gap = 12
    title_font = font(FONT_BOLD, 22)
    subtitle_font = font(FONT_REGULAR, 16)
    panel_title_font = font(FONT_BOLD, 17)
    panel_subtitle_font = font(FONT_REGULAR, 14)
    outcome_font = font(FONT_REGULAR, 12)
    footer_font = font(FONT_REGULAR, 12)

    content_width = len(order) * panel_size + (len(order) - 1) * gap
    total_width = content_width + 2 * margin
    total_height = 22 + 30 + 23 + 18 + 24 + 48 + panel_size + 86 + margin
    total_width += total_width % 2
    total_height += total_height % 2

    base = Image.new("RGB", (total_width, total_height), BG)
    draw = ImageDraw.Draw(base)
    y = margin
    draw.text((margin, y), "Same registered scene, four inference conditions", font=title_font, fill=INK)
    y += 32
    task_id = metadata["task_id"]
    init_id = metadata["init_id"]
    y = draw_lines(
        draw,
        margin,
        y,
        f"pi0.5 on LIBERO-object task {task_id}, init state {init_id}. Independent closed-loop reruns share the same starting state and noise-seed rule.",
        subtitle_font,
        MUTED,
        content_width,
        21,
    )
    draw.line([(margin, y + 5), (margin + content_width, y + 5)], fill=RULE, width=1)
    caption_y = y + 16
    panel_y = caption_y + 66

    x_positions = []
    for index, spec in enumerate(panel_specs):
        x = margin + index * (panel_size + gap)
        x_positions.append(x)
        draw.text((x, caption_y), spec["title"], font=panel_title_font, fill=spec["color"])
        draw.text((x, caption_y + 24), spec["subtitle"], font=panel_subtitle_font, fill=MUTED)
        draw.rectangle([x - 1, panel_y - 1, x + panel_size, panel_y + panel_size], outline=RULE, width=1)
        draw_lines(
            draw,
            x,
            panel_y + panel_size + 10,
            spec["observed"],
            outcome_font,
            INK,
            panel_size,
            17,
        )

    footer_y = total_height - margin - 16
    draw.text(
        (margin, footer_y),
        "Captured by wrapping observation formatting only. Every outcome matches the pre-existing raw confirmation record.",
        font=footer_font,
        fill=MUTED,
    )

    mp4 = args.out / "combined.mp4"
    writer = imageio.get_writer(
        mp4,
        fps=args.fps,
        codec="libx264",
        macro_block_size=1,
        ffmpeg_params=["-crf", "20", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    for output_index, source_index in enumerate(timeline):
        canvas = base.copy()
        for array, x in zip(arrays, x_positions):
            frame = array[min(source_index, len(array) - 1)]
            tile = Image.fromarray(frame).resize((panel_size, panel_size), Image.Resampling.LANCZOS)
            canvas.paste(tile, (x, panel_y))
        if output_index == 0:
            canvas.save(args.out / "poster.png")
        writer.append_data(np.asarray(canvas))
    writer.close()

    webm = args.out / "combined.webm"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-c:v", "libvpx-vp9", "-crf", "32",
         "-b:v", "0", "-row-mt", "1", "-pix_fmt", "yuv420p", str(webm)],
        check=True,
    )
    gif = args.out / "combined.gif"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
         "fps=10,scale=900:-1:flags=lanczos", "-loop", "0", str(gif)],
        check=True,
    )

    composition = {
        **metadata,
        "composition": {
            "n_frames": len(timeline),
            "fps": args.fps,
            "duration_s": len(timeline) / args.fps,
            "stride": args.stride,
            "width": total_width,
            "height": total_height,
            "files": ["combined.mp4", "combined.webm", "combined.gif", "poster.png"],
        },
    }
    (args.out / "meta.json").write_text(json.dumps(composition, indent=2) + "\n")
    print(json.dumps(composition["composition"], indent=2))


if __name__ == "__main__":
    main()
