"""Side-by-side of two Stage 0 results.json files (analyze_stage0.py output), e.g. v044 vs pi05_libero_base."""
import argparse, json

def rate(s, cond, m):
    r = s["rates"][cond][m]
    return f"{r['rate']*100:.1f} [{r['ci'][0]*100:.1f},{r['ci'][1]*100:.1f}] ({r['k']}/{r['n']})"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--a_name", default="A")
    ap.add_argument("--b", required=True); ap.add_argument("--b_name", default="B")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    A, B = json.load(open(args.a)), json.load(open(args.b))
    L = ["# Checkpoint diffing: Stage 0 side-by-side", "",
         f"- {args.a_name}: policy={A['policy']} rev={A['revision']} n_episodes={A['n_episodes']}",
         f"- {args.b_name}: policy={B['policy']} rev={B['revision']} n_episodes={B['n_episodes']}", ""]
    for suite in A["suites"]:
        if suite not in B["suites"]: continue
        sa, sb = A["suites"][suite], B["suites"][suite]
        L += [f"## {suite}  (units: {args.a_name} n={sa['n_units']}, {args.b_name} n={sb['n_units']})", ""]
        for m in ["success", "touched_target_first"]:
            L += [f"### {m} (% [Wilson 95% CI])", "", f"| condition | {args.a_name} | {args.b_name} |", "|---|---|---|"]
            for c in ["correct", "null", "wrong_object"]:
                L.append(f"| {c} | {rate(sa,c,m)} | {rate(sb,c,m)} |")
            L.append("")
        L += ["### paired gaps (McNemar exact)", "", f"| contrast | {args.a_name} | {args.b_name} |", "|---|---|---|"]
        for k in sa["paired"]:
            pa, pb = sa["paired"][k], sb["paired"].get(k)
            fa = f"{pa['gap_pp']:+.1f} pp (p={pa['p']:.1e}, {pa['a_only']}/{pa['b_only']}, n={pa['n_pairs']})"
            fb = f"{pb['gap_pp']:+.1f} pp (p={pb['p']:.1e}, {pb['a_only']}/{pb['b_only']}, n={pb['n_pairs']})" if pb else "-"
            L.append(f"| {k} | {fa} | {fb} |")
        L += ["", f"swap effect (correct − wrong_object, success): {args.a_name} {sa['swap_effect_success_pp']:+.1f} pp vs "
              f"{args.b_name} {sb['swap_effect_success_pp']:+.1f} pp; null gap: {sa['null_gap_success_pp']:+.1f} vs {sb['null_gap_success_pp']:+.1f} pp", ""]
    open(args.out, "w").write("\n".join(L))
    print("\n".join(L))

if __name__ == "__main__":
    main()
