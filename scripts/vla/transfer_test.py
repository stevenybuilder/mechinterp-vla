#!/usr/bin/env python
"""CROSS-SCENE DIRECTION TRANSFER TEST.

The claim under test: obedience has no scene-transferable representation.
Probe evidence is correlational (cross-scene cosine -0.03, LOSO below chance).
This makes it CAUSAL.

Four arms, same episodes, same seeds, same site (image positions, VLM layers 12-17):
  SELF   additive steer with the obey-vs-ignore direction fitted IN THIS SCENE   -> positive control
  CROSS  additive steer with the direction fitted in a DIFFERENT scene           -> the test
  RAND   additive steer with a random direction, norm-matched to SELF            -> negative control
  SWAP   the cross-prompt KV edit at lambda=1 (known to flip)                    -> upper anchor

Prediction: SWAP flips, SELF may flip, CROSS does not, RAND does not.
If CROSS fails while SWAP succeeds on the SAME episodes, the direction does not transfer causally.
"""
import os, sys, json, time, argparse, collections
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/root/vla/scripts/vla")
import numpy as np, torch
torch.set_num_threads(4)
from lerobot.envs.libero import _get_suite
from lerobot.envs.utils import preprocess_observation
from lerobot.utils.constants import ACTION
from run_libero_prompt_conditions import load_policy, make_env, batchify, ContactTracker

REV = "8e174154ef5f6c60a8da12ae99c303d8963138c1"
LAYERS = [12,13,14,15,16,17]          # residual ENTERING these == output of layer L-1
SUITE = "libero_object"


def unit(v):
    n = np.linalg.norm(v); return v/n if n>0 else v


class ResidSteer:
    """h <- h + lam * scale * d_hat at the valid image positions, on the output of layers 11..16."""
    def __init__(self, policy, n_img_valid=512):
        self.pwe = policy.model.paligemma_with_expert
        self.layers = list(self.pwe.paligemma.model.language_model.layers)
        self.pwe.paligemma.language_model.config._attn_implementation = "eager"
        self.dt = None           # {L: torch.Tensor(2048)} on device
        self.scale = 1.0         # set once per episode from the prefix residual norm
        self.lam = 0.0
        self.nimg = n_img_valid
        self.handles = []
        self.n_edits = 0
        for L in LAYERS:
            self.handles.append(self.layers[L-1].register_forward_hook(self._mk(L)))

    def set_direction(self, d, device, dtype):
        """Move the direction to GPU ONCE. Doing torch.as_tensor(numpy) inside the hook forces a
        host->device copy and a sync on every forward pass, which made episodes ~50x slower."""
        self.dt = {L: torch.as_tensor(d[L], dtype=dtype, device=device) for L in LAYERS} if d else None

    def _mk(self, L):
        def hook(mod, inp, out):
            if self.dt is None or self.lam == 0.0:
                return None
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] < self.nimg:          # not the prefix pass
                return None
            h[:, :self.nimg, :].add_(self.dt[L], alpha=self.lam * self.scale)
            self.n_edits += 1
            return None                          # edited in place; no need to rebuild the tuple
        return hook

    def close(self):
        for h in self.handles: h.remove()


def fit_directions(npz):
    """Per-scene obey-vs-ignore direction, per layer, from the cached activations."""
    z = np.load(npz, allow_pickle=True)
    X, y, sc = z["X_img"].astype(np.float64), z["y"], z["scene"]
    D = X.shape[1] // len(LAYERS)
    out = {}
    for s in np.unique(sc):
        m = sc == s
        ig, ob = m & (y == 0), m & (y == 1)
        if ig.sum() < 3 or ob.sum() < 3: continue
        diff = X[ig].mean(0) - X[ob].mean(0)
        out[int(s)] = {L: unit(diff[i*D:(i+1)*D]) for i, L in enumerate(LAYERS)}
    return out


def rollout(env, policy, pre, post, epre, epost, prompt, init, seed, max_steps, tracker, steer, lam, d):
    torch.manual_seed(seed); np.random.seed(seed); policy.reset()
    steer.set_direction(d, next(policy.parameters()).device, next(policy.parameters()).dtype)
    steer.scale = 30.0        # fixed scale: residual norms at image positions are ~O(30)
    steer.lam, steer.n_edits = lam, 0
    env.init_state_id = init
    obs, _ = env.reset(seed=seed)
    rs = env._env
    first = None
    for step in range(max_steps):
        o = preprocess_observation(batchify(obs)); o["task"] = [prompt]
        o = pre(epre(o))
        with torch.inference_mode():
            a = policy.select_action(o)
        a = epost({ACTION: post(a)})[ACTION].cpu().numpy()[0]
        raw, _, done, _ = rs.step(a); obs = env._format_raw_obs(raw)
        t, _g = tracker.contacts()
        for ob in t:
            if first is None: first = ob
        if bool(rs.check_success()) or done: break
    steer.lam = 0.0
    return {"first_touch": first, "success": bool(rs.check_success()), "n_steps": step+1,
            "n_edits": steer.n_edits}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/root/vla/artifacts/obedience_probe_feats.npz")
    ap.add_argument("--out", default="/root/vla/artifacts/transfer_test.jsonl")
    ap.add_argument("--inits", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--lambdas", default="0,1,2,4")
    ap.add_argument("--cells", default="3:4,2:4,1:5")   # task:wrong_task, from control_pilot CELLS
    a = ap.parse_args()
    inits = [int(x) for x in a.inits.split(",")]
    lams = [float(x) for x in a.lambdas.split(",")]

    D = fit_directions(a.npz)
    print(f"[xfer] directions fitted for scenes {sorted(D)}", flush=True)
    policy, pcfg, pre, post, epre, epost = load_policy("lerobot/pi05_libero_finetuned_v044", REV, 10, "cuda")
    suite = _get_suite(SUITE)
    steer = ResidSteer(policy)
    rng = np.random.default_rng(0)
    f = open(a.out, "a")

    for spec in a.cells.split(","):
        t, w = (int(x) for x in spec.split(":"))
        if t not in D:
            print(f"[xfer] scene {t} has no direction, skipping", flush=True); continue
        env = make_env(suite, SUITE, t); tracker = ContactTracker(env._env)
        promptA = suite.tasks[w].language           # the wrong instruction it normally follows/ignores
        others = [s for s in D if s != t]
        cross_from = others[0]
        d_self  = D[t]
        d_cross = D[cross_from]
        d_rand  = {L: unit(rng.standard_normal(len(d_self[L]))) for L in LAYERS}
        print(f"[xfer] scene t{t}  promptA={promptA!r}  cross-direction from scene {cross_from}", flush=True)
        for arm, d in (("SELF", d_self), ("CROSS", d_cross), ("RAND", d_rand)):
            for lam in lams:
                for i in inits:
                    r = rollout(env, policy, pre, post, epre, epost, promptA, i, 1000+i, 280, tracker, steer, lam, d)
                    rec = {"cell": f"t{t}_w{w}", "task_id": t, "wrong_task_id": w, "arm": arm,
                           "cross_from": cross_from if arm == "CROSS" else None, "lam": lam,
                           "init_id": i, **r}
                    f.write(json.dumps(rec)+"\n"); f.flush()
                    print(f"[ep] t{t} {arm:5s} lam={lam:<4} i={i} touch={r['first_touch']} "
                          f"succ={int(r['success'])} edits={r['n_edits']}", flush=True)
        env.close()
    f.close()
    print("[xfer] DONE", flush=True)


if __name__ == "__main__":
    main()
