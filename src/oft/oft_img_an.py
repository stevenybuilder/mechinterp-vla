"""Partial read of the OFT IMG replication against docs/protocol.md
(sha 0d21e476...). Predictions were fixed before any of these numbers were seen."""
import json, collections
import numpy as np
rows = [json.loads(l) for l in open("artifacts/vla_oft_stage2/imgsite/patching.jsonl")]
print("rows=%d  tasks so far=%s" % (len(rows), sorted(set(r["task"] for r in rows))))
P = collections.defaultdict(lambda: collections.defaultdict(list))
R = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    (P if r["variant"] == "patch" else R)[r["layer"]][r["site"]].append(float(r["R_axis"]))
print("\n%6s %10s %10s %10s %10s" % ("layer", "R_IMG", "R_INSTR", "R_ACT", "R_PROPRIO"))
print("-" * 52)
for L in sorted(P):
    v = lambda s: np.median(P[L][s]) if P[L].get(s) else float("nan")
    print("%6d %10.3f %10.3f %10.3f %10.3f" % (L, v("IMG"), v("INSTR"), v("ACT"), v("PROPRIO")))
Ls = sorted(P)
deep = Ls[-1]
ri, rn = np.median(P[deep]["IMG"]), np.median(P[deep]["INSTR"])
print("\n--- preregistered checks (PARTIAL DATA, %d of ~250 inits) ---" % len(set((r["task"], r["init"]) for r in rows)))
print("P3  R_IMG(L%d)=%.3f > R_INSTR(L%d)=%.3f ?  %s" % (deep, ri, deep, rn, "YES" if ri > rn else "NO"))
print("    R_IMG(L%d) >= 0.30 ?                  %s" % (deep, "YES" if ri >= 0.30 else "NO"))
shallow = Ls[0]
print("P4  R_IMG rises with depth: L%d=%.3f -> L%d=%.3f  %s"
      % (shallow, np.median(P[shallow]["IMG"]), deep, ri,
         "YES" if ri > np.median(P[shallow]["IMG"]) else "NO"))
print("P5  random controls < 0.20 at every layer:")
ok = True
for L in sorted(R):
    for s in ("IMG", "INSTR"):
        if R[L].get(s):
            m = np.median(R[L][s]); ok &= abs(m) < 0.20
            print("      L%-3d %-6s rand=%+.3f  (patch=%+.3f)" % (L, s, m, np.median(P[L][s])))
print("    -> %s" % ("PASS" if ok else "FAIL"))
