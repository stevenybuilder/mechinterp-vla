"""Re-derive the OFT numbers at the prespecified unit (the cell), with the L2 endpoint as primary.
The unit is the cell, never the episode; aggregating over episodes inflates n and can flip a
preregistered pass/fail. See docs/protocol.md."""
import json, collections
import numpy as np
D = "artifacts/vla_oft_stage2/imgsite"
rows = [json.loads(l) for l in open(f"{D}/patching.jsonl")]
print("rows=%d  tasks=%d" % (len(rows), len(set(r["task"] for r in rows))))
print("same_layout False in %d/%d rows" % (sum(1 for r in rows if not r["same_layout"]), len(rows)))

# cell = task x direction x timepoint ; median within cell, then median across cells
def cellstat(site, layer, variant):
    by = collections.defaultdict(list)
    for r in rows:
        if r["site"] == site and r["layer"] == layer and r["variant"] == variant:
            den = r["l2_src_dst"] or 1e-9
            by[(r["task"], r["dir"], r["t"])].append(
                (r["R_axis"] if r["R_axis"] is not None else np.nan,
                 r["l2_to_src"] / den, r["l2_to_dst"] / den))
    if not by: return None
    per = [np.nanmedian(np.array(v), axis=0) for v in by.values()]
    a = np.array(per)
    return len(per), np.nanmedian(a[:, 0]), np.nanmedian(a[:, 1]), np.nanmedian(a[:, 2])

print("\n%-8s %-4s %-10s %5s %8s %10s %10s" % ("site", "L", "variant", "cells", "R", "D_src", "D_dst"))
print("-" * 62)
for site in ("IMG", "INSTR", "ACT", "PROPRIO"):
    for L in (8, 16, 24):
        for v in ("patch", "rand0", "resample"):
            s = cellstat(site, L, v)
            if s: print("%-8s %-4d %-10s %5d %8.3f %10.3f %10.3f" % (site, L, v, *s))
    print()
print("D_src = |patched-src|/|src-dst|  (0 = perfect transfer, 1 = no transfer)")
print("D_dst = |patched-dst|/|src-dst|  (0 = no-op, large = the patch had teeth)")
print("\n--- PREREG P5 re-derived at the CELL level, full 10-task run ---")
for L in (8, 16, 24):
    s = cellstat("INSTR", L, "rand0")
    if s: print("  INSTR L%-3d rand0: R=%.3f  (prereg bar 0.20)  -> %s" % (L, s[1], "PASS" if abs(s[1]) < 0.20 else "FAIL"))
