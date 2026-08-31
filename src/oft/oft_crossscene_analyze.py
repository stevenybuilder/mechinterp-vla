#!/usr/bin/env python
"""Cell-level read of the OFT cross-scene positive control, plus the A/B contrast it settles.

Unit is the CELL (task x init-pair x timepoint x direction) -- the cell-unit rule (docs/protocol.md). Endpoint is L2 in
action space -- the L2-endpoint rule (docs/protocol.md). The no-op anchor (self-patch) is printed on every line.
"""
import json, collections
import numpy as np

XS = "artifacts/vla_oft_stage2/xscene"
AB = "artifacts/vla_oft_stage2/imgsite"

rows = [json.loads(l) for l in open(f"{XS}/patching.jsonl")]
cells = set((r["task"], tuple(r["pair"]), r["t"], r["dir"]) for r in rows)
print("CROSS-SCENE POSITIVE CONTROL  (same prompt, different observation)")
print("rows=%d  cells=%d  tasks=%d" % (len(rows), len(cells), len(set(r["task"] for r in rows))))
print("same_layout True in %d/%d rows  (A/B run: False in 36000/36000 -- this run has no RoPE confound)"
      % (sum(1 for r in rows if r["same_layout"]), len(rows)))
sd = [r["l2_src_dst"] for r in rows if r["variant"] == "patch" and r["site"] == "IMG" and r["layer"] == 8]
print("scene-to-scene clean action distance |src-dst|: median %.3f  (min %.3f max %.3f)"
      % (np.median(sd), np.min(sd), np.max(sd)))


def cellstat(site, layer, variant, key):
    by = collections.defaultdict(list)
    for r in rows:
        if r["site"] == site and r["layer"] == layer and r["variant"] == variant and r[key] is not None:
            by[(r["task"], tuple(r["pair"]), r["t"], r["dir"])].append(r[key])
    if not by:
        return None
    per = np.array([np.median(v) for v in by.values()])
    return len(per), float(np.median(per)), float(np.percentile(per, 25)), float(np.percentile(per, 75))


print("\n%-8s %-3s %-7s %6s | %-22s | %-22s" % ("site", "L", "variant", "cells", "D_dst (patch efficacy)", "D_src (transfer)"))
print("-" * 88)
for site in ("IMG", "INSTR", "PROPRIO", "ACT"):
    for L in (8, 16, 24):
        for v in ("patch", "rand0", "self"):
            a, b = cellstat(site, L, v, "D_dst"), cellstat(site, L, v, "D_src")
            if a:
                print("%-8s %-3d %-7s %6d | %6.3f  [%.3f-%.3f]     | %6.3f  [%.3f-%.3f]"
                      % (site, L, v, a[0], a[1], a[2], a[3], b[1], b[2], b[3]))
    print()

# content vs norm: per cell, does the real patch beat an equal-norm random direction?
print("content vs equal-norm random direction, per cell (D_dst patch > D_dst rand0):")
for site in ("IMG", "INSTR", "PROPRIO", "ACT"):
    line = []
    for L in (8, 16, 24):
        by = collections.defaultdict(dict)
        for r in rows:
            if r["site"] == site and r["layer"] == L and r["variant"] in ("patch", "rand0"):
                by[(r["task"], tuple(r["pair"]), r["t"], r["dir"])][r["variant"]] = r["D_dst"]
        w = [1 for d in by.values() if "patch" in d and "rand0" in d and d["patch"] > d["rand0"]]
        line.append("L%-2d %2d/%-2d" % (L, len(w), len(by)))
    print("  %-8s %s" % (site, "   ".join(line)))

# the A/B run, re-derived at the cell level, for the contrast this run exists to settle
print("\n--- the A/B run (same observation, different prompt), cell level, for contrast ---")
ab = [json.loads(l) for l in open(f"{AB}/patching.jsonl")]


def abstat(site, layer, variant, key):
    by = collections.defaultdict(list)
    for r in ab:
        if r["site"] == site and r["layer"] == layer and r["variant"] == variant:
            den = r["l2_src_dst"] or 1e-9
            by[(r["task"], r["dir"], r["t"])].append(r[key] / den)
    if not by:
        return None
    per = np.array([np.median(v) for v in by.values()])
    return len(per), float(np.median(per))


print("%-8s %-3s %6s %10s %10s" % ("site", "L", "cells", "D_dst", "D_src"))
for site in ("IMG", "INSTR", "PROPRIO"):
    for L in (8, 16, 24):
        d, s = abstat(site, L, "patch", "l2_to_dst"), abstat(site, L, "patch", "l2_to_src")
        if d:
            print("%-8s %-3d %6d %10.3f %10.3f" % (site, L, d[0], d[1], s[1]))

print("\nREADING: in the A/B run the IMG patch moved the output by D_dst~0.015 -- a near-no-op -- so")
print("'the image positions are inert' and 'the patch was undone downstream' were not separable.")
print("This run replaces the SAME 512 positions with activations from a genuinely different scene.")
print("If IMG D_dst here is large, the site has teeth and the A/B near-zero is a property of the")
print("CONTRAST (both prompts saw the same image), not of the method.")
