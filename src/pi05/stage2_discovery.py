#!/usr/bin/env python
"""Stage 2 discovery: prefix-segment KV decomposition of pi0.5's prompt channel (protocol §4.2/§4.3/§4.4).

For each pair (A prompt = task's instruction, B prompt = an equal-token-count alternative naming a different object
present in the scene), init state and timepoint along the clean-A closed-loop trajectory, all with A's images and a
fixed noise:
  clean_A, clean_B (= full KV swap)               both directions (A<-B "noising", B<-A "denoising")
  KV[seg] for seg in INSTR, OBJ, FMT, IMG, INSTR+FMT, IMG+FMT  x layer bands {all, 0-5, 6-11, 12-17}
  RD[seg] (equal-norm random direction, all layers), RS[seg] (K/V from a length-matched unrelated prompt)
  KO[seg] attention knockout (action tokens cannot attend to seg) + KOrand[seg] matched-count random keys
  attention mass per expert layer/step/head/segment on clean_A; K=5 noise seeds on clean_A/clean_B for SD_noise
  VLM residual + K/V segment-mean features for all suite prompts (readout probes, analysed separately)
Metric: m = sum_{t<k} a_t[:3].u, u = unit(pB - pA), k = 10 (primary) and 50; L2 to clean chunks; gripper separately.
Rows -> <out>/rows.jsonl ; features -> <out>/features_<pair>_<init>_<tp>.npz ; manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch

from hooks import LAYER_BANDS, Pi05Harness, axis_metric, cache_kv_lists, clone_cache
from lerobot.envs.libero import LiberoEnv, _get_suite

OBJ_NAMES = {  # object suite: task -> object name in prompt; goal suite handled by GOAL_OBJ
    "alphabet_soup_1": "alphabet soup", "cream_cheese_1": "cream cheese", "salad_dressing_1": "salad dressing",
    "bbq_sauce_1": "bbq sauce", "ketchup_1": "ketchup", "tomato_sauce_1": "tomato sauce", "butter_1": "butter",
    "milk_1": "milk", "chocolate_pudding_1": "chocolate pudding", "orange_juice_1": "orange juice",
}
GOAL_OBJ = {0: ("wooden_cabinet_1", "cabinet"), 1: ("akita_black_bowl_1", "bowl"), 2: ("wine_bottle_1", "wine bottle"),
            3: ("wooden_cabinet_1", "drawer"), 4: ("akita_black_bowl_1", "bowl"), 5: ("plate_1", "plate"),
            6: ("cream_cheese_1", "cream cheese"), 7: ("flat_stove_1", "stove"), 8: ("akita_black_bowl_1", "bowl"),
            9: ("wine_bottle_1", "wine bottle")}
SEGS = ["INSTR", "OBJ", "FMT", "IMG", "INSTR+FMT", "IMG+FMT"]


def seg_positions(seg, segd):
    return sorted(set(sum([segd[s] for s in seg.split("+")], [])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--pairs", required=True, help="JSON list of [taskA, taskB, taskRS] (task ids in the suite)")
    ap.add_argument("--init_ids", default="0-24")
    ap.add_argument("--tps", default="0,10")
    ap.add_argument("--n_noise_sd", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="lerobot/pi05_libero_finetuned_v044")
    ap.add_argument("--revision", default="8e174154ef5f6c60a8da12ae99c303d8963138c1")
    ap.add_argument("--processor_path", default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--skip_bands", action="store_true")
    ap.add_argument("--features", action="store_true", help="capture probe features for all suite prompts")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_f = (out / "rows.jsonl").open("a")
    done = set()
    if (out / "rows.jsonl").exists():
        for l in open(out / "rows.jsonl"):
            try:
                r = json.loads(l)
                done.add((r["pair"], r["init"], r["tp"]))
            except Exception:
                pass
    a, b = args.init_ids.split("-")
    init_ids = list(range(int(a), int(b) + 1))
    tps = [int(x) for x in args.tps.split(",")]
    pairs = json.loads(args.pairs)
    H = Pi05Harness(args.policy, revision=args.revision, dtype=args.dtype, processor_path=args.processor_path)
    suite = _get_suite(args.suite)
    prompts = [suite.tasks[i].language for i in range(len(suite.tasks))]
    manifest = {"args": vars(args), "module_paths": H.module_paths, "pairs": [], "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}

    def obj_of(task_id, scene_objects):
        if args.suite == "libero_goal":
            return GOAL_OBJ[task_id]
        import re
        mm = re.match(r"pick_up_the_(.+?)_and_place_it_in_the_basket", suite.tasks[task_id].name)
        o = mm.group(1) + "_1"
        return o, OBJ_NAMES[o]

    bands = LAYER_BANDS if not args.skip_bands else {"all": LAYER_BANDS["all"]}
    for tA, tB, tRS in pairs:
        pair = f"{args.suite}_t{tA}_vs_t{tB}"
        env = LiberoEnv(task_suite=suite, task_id=tA, task_suite_name=args.suite, obs_type="pixels_agent_pos",
                        observation_height=360, observation_width=360, init_states=True, episode_index=0, n_envs=1)
        rs = env._env.env
        scene_objects = list(rs.obj_body_id)
        objA, nameA = obj_of(tA, scene_objects)
        objB, nameB = obj_of(tB, scene_objects)
        assert objA in scene_objects and objB in scene_objects, (objA, objB, scene_objects)
        pA_text, pB_text, pRS_text = prompts[tA], prompts[tB], prompts[tRS]
        manifest["pairs"].append({"pair": pair, "A": [tA, objA, pA_text], "B": [tB, objB, pB_text], "RS": [tRS, pRS_text]})
        print(f"[s2] {pair}: A='{pA_text}' ({objA}) B='{pB_text}' ({objB}) RS='{pRS_text}'", flush=True)
        for init in init_ids:
            if all((pair, init, tp) in done for tp in tps):
                continue
            # clean-A closed-loop trajectory to collect observations at the timepoints
            torch.manual_seed(1000 + tA * 1000 + init)
            env.init_state_id = init
            obs, _ = env.reset(seed=1000 + init)
            obs_at, pos_at = {}, {}
            step = 0
            for tp in sorted(tps):
                while step < tp:
                    bt = H.build_batch(obs, pA_text)
                    ch = H.chunk(bt, H.make_noise(seed=50_000 + step))[0].cpu().numpy()
                    for k in range(10):
                        raw, *_ = env._env.step(ch[k].astype(np.float32))
                        step += 1
                    obs = env._format_raw_obs(raw)
                obs_at[tp] = obs
                pos_at[tp] = (np.array(rs.sim.data.body_xpos[rs.obj_body_id[objA]]),
                              np.array(rs.sim.data.body_xpos[rs.obj_body_id[objB]]))
            for tp in tps:
                if (pair, init, tp) in done:
                    continue
                t0 = time.time()
                o = obs_at[tp]
                pA, pB = pos_at[tp]
                bA, bB, bRS = H.build_batch(o, pA_text), H.build_batch(o, pB_text), H.build_batch(o, pRS_text)
                prA, prB, prRS = H.prefix_forward(bA), H.prefix_forward(bB), H.prefix_forward(bRS)
                n_img = prA["n_img_slots"]
                segA, decA = H.segments(bA, object_name=nameA, n_img=n_img)
                segB, decB = H.segments(bB, object_name=nameB, n_img=n_img)
                segA["IMG"] = list(range(prA["n_img_valid"]))
                segB["IMG"] = segA["IMG"]
                equal_len = len(segA["TEXT_VALID"]) == len(segB["TEXT_VALID"]) == len(H.segments(bRS, n_img=n_img)[0]["TEXT_VALID"])
                noise = H.make_noise(seed=init * 100 + tp)
                base = {"pair": pair, "init": init, "tp": tp, "objA": objA, "objB": objB, "pA": pA.tolist(),
                        "pB": pB.tolist(), "equal_len": equal_len, "segA": {k: v for k, v in segA.items() if k not in ("IMG", "TEXT_VALID")},
                        "segB": {k: v for k, v in segB.items() if k not in ("IMG", "TEXT_VALID")}, "n_img_valid": prA["n_img_valid"]}
                chunks = {}

                def run(name, prefix, direction, seed_tag=0, **kw):
                    ch, cap = H.action_forward(prefix, noise, **kw)
                    env_ch = H.unnormalize(ch)[0].cpu().numpy()
                    chunks[(direction, name, seed_tag)] = env_ch
                    row = dict(base)
                    row.update({"direction": direction, "condition": name, "seed": seed_tag,
                                "m10": axis_metric(env_ch, pA, pB, 10), "m50": axis_metric(env_ch, pA, pB, 50),
                                "grip10": float(env_ch[:10, 6].mean()), "chunk10": env_ch[:10].round(5).tolist()})
                    rows_f.write(json.dumps(row) + "\n")
                    return env_ch, cap

                for direction, (src_pr, dst_pr, segS, segD) in {
                    "A<-B": (prB, prA, segB, segA), "B<-A": (prA, prB, segA, segB)}.items():
                    # segS: positions in the source prompt's own segmentation; segD: destination's. Equal length ->
                    # identical index sets; if not, patch by role using the destination's positions (exploratory).
                    run("clean", dst_pr, direction)
                    full = clone_cache(dst_pr["cache"])
                    H.kv_swap(full, src_pr["cache"], list(range(dst_pr["prefix_len"])), LAYER_BANDS["all"])
                    run("full_swap", {**dst_pr, "cache": full}, direction)
                    for seg in SEGS:
                        pos = seg_positions(seg, segD)
                        for bname, layers in bands.items():
                            c = clone_cache(dst_pr["cache"])
                            H.kv_swap(c, src_pr["cache"], pos, layers)
                            run(f"KV[{seg}]@{bname}", {**dst_pr, "cache": c}, direction)
                        c = clone_cache(dst_pr["cache"])
                        H.rd_patch(c, pos, LAYER_BANDS["all"], seed=init * 7 + tp)
                        run(f"RD[{seg}]@all", {**dst_pr, "cache": c}, direction)
                        c = clone_cache(dst_pr["cache"])
                        H.kv_swap(c, prRS["cache"], pos, LAYER_BANDS["all"])
                        run(f"RS[{seg}]@all", {**dst_pr, "cache": c}, direction)
                        # knockout + matched-count random keys (from valid prefix positions outside seg)
                        am = dst_pr["prefix_pad_masks"].clone()
                        am[:, pos] = False
                        run(f"KO[{seg}]", dst_pr, direction, attend_mask=am)
                        valid = torch.nonzero(dst_pr["prefix_pad_masks"][0]).flatten().tolist()
                        cand = [p for p in valid if p not in set(pos)]
                        g = np.random.RandomState(init * 13 + tp)
                        rnd = g.choice(cand, size=min(len(pos), len(cand)), replace=False).tolist()
                        am = dst_pr["prefix_pad_masks"].clone()
                        am[:, rnd] = False
                        run(f"KOrand[{seg}]", dst_pr, direction, attend_mask=am)
                    # noise-seed SD (clean + full swap)
                    for s in range(1, args.n_noise_sd + 1):
                        nz = H.make_noise(seed=900_000 + init * 100 + tp * 10 + s)
                        for nm, pr in (("clean", dst_pr), ("full_swap", {**dst_pr, "cache": full})):
                            ch, _ = H.action_forward(pr, nz)
                            env_ch = H.unnormalize(ch)[0].cpu().numpy()
                            row = dict(base)
                            row.update({"direction": direction, "condition": nm, "seed": s,
                                        "m10": axis_metric(env_ch, pA, pB, 10), "m50": axis_metric(env_ch, pA, pB, 50),
                                        "grip10": float(env_ch[:10, 6].mean())})
                            rows_f.write(json.dumps(row) + "\n")
                # attention mass on clean A (and clean B) by segment
                attn_segs = {k: segA[k] for k in ("IMG", "FMT", "INSTR", "OBJ", "STATE")}
                _, capA = H.action_forward(prA, noise, attn_segments=attn_segs)
                attn_segsB = {k: segB[k] for k in ("IMG", "FMT", "INSTR", "OBJ", "STATE")}
                _, capB = H.action_forward(prB, noise, attn_segments=attn_segsB)
                feats = {"attn_A": capA["attn"].numpy().astype(np.float16), "attn_B": capB["attn"].numpy().astype(np.float16),
                         "attn_segments": np.array(capA["attn_segments"])}
                if args.features:
                    # readout features: all suite prompts on this observation; segment means of residual + K/V per layer
                    R, K, V, labels = [], [], [], []
                    for ti, ptxt in enumerate(prompts):
                        bt = H.build_batch(o, ptxt)
                        pr = H.prefix_forward(bt, capture=True)
                        oname = obj_of(ti, scene_objects)[1] if args.suite == "libero_goal" or True else None
                        try:
                            sg, _ = H.segments(bt, object_name=oname, n_img=n_img)
                        except AssertionError:
                            sg, _ = H.segments(bt, object_name=None, n_img=n_img)
                        sg["IMG"] = list(range(pr["n_img_valid"]))
                        ks, vs = cache_kv_lists(pr["cache"])
                        segs_use = {"INSTR": sg["INSTR"], "OBJ": sg["OBJ"] or sg["INSTR"][-1:], "FMT": sg["FMT"],
                                    "IMG": sg["IMG"], "LAST": [sg["TEXT_VALID"][-1]]}
                        r_l, k_l, v_l = [], [], []
                        for L in range(18):
                            h = pr["resid"][L][0]
                            r_l.append(np.stack([h[idx].mean(0).numpy() for idx in segs_use.values()]))
                            k_l.append(np.stack([ks[L][0, 0, idx].float().mean(0).cpu().numpy() for idx in segs_use.values()]))
                            v_l.append(np.stack([vs[L][0, 0, idx].float().mean(0).cpu().numpy() for idx in segs_use.values()]))
                        R.append(np.stack(r_l)); K.append(np.stack(k_l)); V.append(np.stack(v_l)); labels.append(ti)
                    feats.update({"resid": np.stack(R).astype(np.float16), "K": np.stack(K).astype(np.float16),
                                  "V": np.stack(V).astype(np.float16), "labels": np.array(labels),
                                  "feat_segments": np.array(list(segs_use))})
                np.savez_compressed(out / f"features_{pair}_i{init}_tp{tp}.npz", **feats)
                rows_f.flush()
                print(f"[s2] {pair} init {init} tp {tp} done in {time.time()-t0:.1f}s "
                      f"(mA={chunks[('A<-B','clean',0)][:10,:3].sum():.3f}) equal_len={equal_len}", flush=True)
        env.close()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[s2] DONE", flush=True)


if __name__ == "__main__":
    main()
