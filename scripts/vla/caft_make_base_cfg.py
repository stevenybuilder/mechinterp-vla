"""Build a pi05_base checkpoint dir whose config can TRAIN on LIBERO.

pi05_base ships a generic config with no LIBERO input/output features and no normalisation mapping.
v044 (the released LIBERO fine-tune) ships exactly those. We copy pi05_base's WEIGHTS and take the
five config keys that describe the LIBERO interface from v044, then drop `pretrained_path` so the
trainer treats this as a starting point rather than a resume.

We deliberately do NOT copy v044's normalisation STATS: LeRobot computes MEAN_STD stats from the
LeRobotDataset at train time (feasibility doc SS1.1), so the stats in the checkpoint are irrelevant
for training and copying them would silently leak v044's data statistics into the 'base' arm.
"""
import os, json, shutil, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "0")
from huggingface_hub import snapshot_download

SRC_REV  = "b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba"
V044_REV = "8e174154ef5f6c60a8da12ae99c303d8963138c1"
TAKE = ["input_features", "output_features", "empty_cameras", "normalization_mapping", "dtype"]

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()

src  = snapshot_download("lerobot/pi05_base", revision=SRC_REV)
v044 = snapshot_download("lerobot/pi05_libero_finetuned_v044", revision=V044_REV)
print("pi05_base   :", src)
print("v044        :", v044)

if os.path.exists(a.out): shutil.rmtree(a.out)
shutil.copytree(src, a.out, symlinks=False)

cs = json.load(open(f"{a.out}/config.json"))
cv = json.load(open(f"{v044}/config.json"))
print("\nkey                     base -> v044")
for k in TAKE:
    if k not in cv:
        raise SystemExit(f"FATAL: v044 config has no key {k!r}")
    before = cs.get(k, "<absent>")
    cs[k] = cv[k]
    bs = str(before); vs = str(cv[k])
    print(f"  {k:22s} {bs[:38]:38s} -> {vs[:60]}")
cs.pop("pretrained_path", None)
json.dump(cs, open(f"{a.out}/config.json", "w"), indent=2)

# receipts: the weights must still be pi05_base's, not v044's
import hashlib
def h(p):
    m = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): m.update(b)
    return m.hexdigest()[:16]
for fn in sorted(os.listdir(a.out)):
    if fn.endswith(".safetensors"):
        same = os.path.exists(f"{v044}/{fn}") and h(f"{a.out}/{fn}") == h(f"{v044}/{fn}")
        print(f"  weights {fn}: sha16={h(f'{a.out}/{fn}')}  identical_to_v044={same}  (MUST be False)")
print("\nwrote", a.out)
