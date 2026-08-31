#!/usr/bin/env python
"""Build Stage 1 scene twins for libero_object by BDDL edit + init-state edit.

For each pair (A = task's target object, B = a co-present training-target distractor) built from task A's scene:
  ambiguous    : the shipped scene (A at target_object_region, B among distractors) + shipped init states 0..N-1
  swap         : same BDDL; init states with A and B free-joint qpos exchanged (layout now favours B)
  determinate  : BDDL with the 5 distractors replaced by never-target objects, positioned exactly where the removed
                 distractors were (A, basket, robot identical); init states derived from the shipped ones.
Outputs per twin: <out>/<pair>/<twin>.bddl (copy for ambiguous/swap), <twin>_init.pt (N,110), agentview PNGs for
init 0, and pixel-diff stats vs the ambiguous render (fraction of changed pixels inside the changed objects' masks,
from MuJoCo segmentation rendering). Writes <out>/manifest.json with prompts (incl. the negation set) and token counts.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch
from PIL import Image

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

NEVER_TARGETS = ["macaroni_and_cheese", "popcorn", "white_bowl", "yellow_book", "new_salad_dressing"]


def obj_joint(sim, obj_name):
    names = [n for n in sim.model.joint_names if n.startswith(obj_name + "_joint")]
    assert len(names) == 1, (obj_name, names)
    return names[0]


def qpos_slice(sim, obj_name):
    a = sim.model.get_joint_qpos_addr(obj_joint(sim, obj_name))
    return slice(1 + a[0], 1 + a[1])  # +1: flattened state = [time, qpos, qvel]


def qvel_slice(sim, obj_name):
    a = sim.model.get_joint_qvel_addr(obj_joint(sim, obj_name))
    return slice(1 + sim.model.nq + a[0], 1 + sim.model.nq + a[1])


def _patched_read_pixels(self, width, height, depth=False, segmentation=False):
    """robosuite 1.4 read_pixels with the numpy-2 uint8 overflow fixed (int cast before *256)."""
    import mujoco
    viewport = mujoco.MjrRect(0, 0, width, height)
    rgb_img = np.empty((height, width, 3), dtype=np.uint8)
    depth_img = np.empty((height, width), dtype=np.float32) if depth else None
    mujoco.mjr_readPixels(rgb=rgb_img, depth=depth_img, viewport=viewport, con=self.con)
    ret_img = rgb_img
    if segmentation:
        r = rgb_img.astype(np.int64)
        seg_img = r[:, :, 0] + r[:, :, 1] * (2**8) + r[:, :, 2] * (2**16)
        seg_img[seg_img >= (self.scn.ngeom + 1)] = 0
        seg_ids = np.full((self.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32)
        for i in range(self.scn.ngeom):
            geom = self.scn.geoms[i]
            if geom.segid != -1:
                seg_ids[geom.segid + 1, 0] = geom.objtype
                seg_ids[geom.segid + 1, 1] = geom.objid
        ret_img = seg_ids[seg_img]
    return (ret_img, depth_img) if depth else ret_img


def render(env, state, hw=360):
    from robosuite.utils import binding_utils
    binding_utils.MjRenderContext.read_pixels = _patched_read_pixels
    env.reset()  # full reset first: the OSC controller's internal goal is not part of the sim state
    env.set_init_state(state)
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    img = obs["agentview_image"]
    rgb = env.sim.render(width=hw, height=hw, camera_name="agentview")
    seg = env.sim.render(width=hw, height=hw, camera_name="agentview", segmentation=True)  # (H,W,2) objtype, objid
    best, bd = None, None
    for f in ("id", "ud", "lr", "udlr"):
        t = {"id": lambda x: x, "ud": lambda x: x[::-1], "lr": lambda x: x[:, ::-1], "udlr": lambda x: x[::-1, ::-1]}[f]
        d = np.abs(t(rgb).astype(int) - img.astype(int)).mean()
        if bd is None or d < bd:
            best, bd = t, d
    assert bd < 3.0, f"could not align sim.render with obs image (mean abs diff {bd})"
    return img, best(seg)


def seg_mask(env, seg, objects):
    import mujoco
    core = getattr(env, "env", env)
    m = core.sim.model
    roots = {core.obj_body_id[o] for o in objects}
    geom_ids = set()
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        while b > 0:
            if b in roots:
                geom_ids.add(g)
                break
            b = m.body_parentid[b]
    is_geom = seg[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM)
    return is_geom & np.isin(seg[..., 1], list(geom_ids))


def make_determinate_bddl(src_txt, keep, replacements):
    """Replace every object (except `keep`) in :objects/:init with never-target objects, same regions."""
    m = re.search(r"\(:objects(.*?)\)\s*\(:obj_of_interest", src_txt, re.S)
    lines = [l.strip() for l in m.group(1).strip().splitlines() if l.strip()]
    mapping = {}
    new_lines = []
    ri = 0
    for l in lines:
        inst, cls = [x.strip() for x in l.split("-")]
        if inst in keep:
            new_lines.append(f"    {inst} - {cls}")
        else:
            new_cls = replacements[ri]
            ri += 1
            new_inst = f"{new_cls}_1"
            mapping[inst] = new_inst
            new_lines.append(f"    {new_inst} - {new_cls}")
    txt = src_txt[: m.start(1)] + "\n" + "\n".join(new_lines) + "\n  " + src_txt[m.end(1):]
    for old, new in mapping.items():
        txt = re.sub(rf"\b{re.escape(old)}\b", new, txt)
    return txt, mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="0,6,7", help="libero_object task ids whose target is A")
    ap.add_argument("--n_init", type=int, default=20)
    ap.add_argument("--out", default="stage1")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    bddl_dir = Path(get_libero_path("bddl_files")) / "libero_object"
    init_dir = Path(get_libero_path("init_states")) / "libero_object"
    targets = {}
    for i in range(len(suite.tasks)):
        mm = re.match(r"pick_up_the_(.+?)_and_place_it_in_the_basket", suite.tasks[i].name)
        targets[i] = mm.group(1) + "_1"
    manifest = {"pairs": {}}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    for tid in [int(x) for x in args.tasks.split(",")]:
        task = suite.tasks[tid]
        src = (bddl_dir / task.bddl_file).read_text()
        A = targets[tid]
        objs = re.findall(r"^\s*(\w+_1) - \w+\s*$", re.search(r"\(:objects(.*?)\)\s*\(:obj_of_interest", src, re.S).group(1), re.M)
        # B = same rule as the runner: first cyclic other task whose target is present
        B = None
        for k in range(1, 10):
            j = (tid + k) % 10
            if targets[j] in objs:
                B, btid = targets[j], j
                break
        pair = f"t{tid}_{A}_vs_{B}"
        pdir = out / pair
        pdir.mkdir(exist_ok=True)
        states = torch.load(init_dir / task.init_states_file, weights_only=False)[: args.n_init]
        # ambiguous (shipped)
        amb_bddl = pdir / "ambiguous.bddl"
        shutil.copy(bddl_dir / task.bddl_file, amb_bddl)
        torch.save(states, pdir / "ambiguous_init.pt")
        env = OffScreenRenderEnv(bddl_file_name=str(amb_bddl), camera_heights=360, camera_widths=360)
        env.reset()
        sim = env.sim
        # swap: exchange A and B free-joint qpos/qvel
        sw = states.copy()
        sa, sb = qpos_slice(sim, A), qpos_slice(sim, B)
        va, vb = qvel_slice(sim, A), qvel_slice(sim, B)
        sw[:, sa], sw[:, sb] = states[:, sb], states[:, sa]
        sw[:, va], sw[:, vb] = states[:, vb], states[:, va]
        shutil.copy(amb_bddl, pdir / "swap.bddl")
        torch.save(sw, pdir / "swap_init.pt")
        img_amb, seg_amb = render(env, states[0])
        img_sw, seg_sw = render(env, sw[0])
        mask_sw = seg_mask(env, seg_amb, {A, B}) | seg_mask(env, seg_sw, {A, B})
        distractors = [o for o in objs if o not in (A, "basket_1")]
        mask_dist_amb = seg_mask(env, seg_amb, set(distractors))
        # determinate
        det_txt, mapping = make_determinate_bddl(src, keep={A, "basket_1"}, replacements=NEVER_TARGETS[: len(distractors)])
        det_bddl = pdir / "determinate.bddl"
        det_bddl.write_text(det_txt)
        sim = env.sim  # env.reset() inside render() rebuilds the MjSim; re-grab before reading joint addresses
        old_sl = {o: (qpos_slice(sim, o), qvel_slice(sim, o)) for o in list(mapping) + [A, "basket_1"]}
        old_nq = sim.model.nq
        env.close()
        del env, sim  # only one robosuite EGL context per process, else renders are corrupted
        denv = OffScreenRenderEnv(bddl_file_name=str(det_bddl), camera_heights=360, camera_widths=360)
        denv.reset()
        dsim = denv.sim
        assert dsim.model.nq == old_nq, (dsim.model.nq, old_nq)
        det = states.copy()
        for old, new in mapping.items():
            det[:, qpos_slice(dsim, new)] = states[:, old_sl[old][0]]
            det[:, qvel_slice(dsim, new)] = states[:, old_sl[old][1]]
        for o in (A, "basket_1"):
            det[:, qpos_slice(dsim, o)] = states[:, old_sl[o][0]]
            det[:, qvel_slice(dsim, o)] = states[:, old_sl[o][1]]
        torch.save(det, pdir / "determinate_init.pt")
        img_det, seg_det = render(denv, det[0])
        mask_det = mask_dist_amb | seg_mask(denv, seg_det, set(mapping.values()))
        denv.close()

        from scipy.ndimage import binary_dilation
        mask_sw = binary_dilation(mask_sw, iterations=3)
        mask_det = binary_dilation(mask_det, iterations=3)

        def diffstats(a, b, mask):
            d = np.abs(a.astype(int) - b.astype(int)).max(-1) > 8
            inside = float((d & mask).sum() / max(1, d.sum()))
            return {"changed_px": int(d.sum()), "mask_px": int(mask.sum()), "frac_changed_inside_mask": inside,
                    "changed_outside_mask_px": int((d & ~mask).sum())}

        for name, img in (("ambiguous", img_amb), ("swap", img_sw), ("determinate", img_det)):
            Image.fromarray(img[::-1, ::-1]).save(pdir / f"{name}_init0.png")
        Image.fromarray((mask_det * 255).astype(np.uint8)[::-1, ::-1]).save(pdir / "determinate_mask.png")
        stats = {"swap_vs_ambiguous": diffstats(img_amb, img_sw, mask_sw),
                 "determinate_vs_ambiguous": diffstats(img_amb, img_det, mask_det)}
        Aname, Bname = A[:-2].replace("_", " "), B[:-2].replace("_", " ")
        prompts = {
            "A": f"pick up the {Aname} and place it in the basket",
            "B": f"pick up the {Bname} and place it in the basket",
            "except_A": f"pick up anything except the {Aname} and place it in the basket",
            "notA_B": f"do not pick up the {Aname}; pick up the {Bname} and place it in the basket",
        }
        ntok = {k: len(tok(v)["input_ids"]) for k, v in prompts.items()}
        manifest["pairs"][pair] = {"task_id": tid, "A": A, "B": B, "B_task_id": btid, "distractors": distractors,
                                   "replacements": mapping, "prompts": prompts, "prompt_tokens": ntok,
                                   "pixel_diff": stats, "n_init": int(len(states))}
        print(pair, json.dumps(stats), ntok, flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("manifest ->", out / "manifest.json")


if __name__ == "__main__":
    main()
