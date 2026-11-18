"""Cache all FLAIR slices/masks and write the patient-level deep-learning split.

Split (seed 42): the 12 development patients are the validation set (early stopping only);
the 98 held-out patients are divided into 70 training and 28 test patients.
All methods, classical and deep learning, are compared on the same 28 test patients.

Usage: python experiments/dl_prepare.py --data <path to kaggle_3m>
"""
import argparse
import json
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_patient  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default=os.path.join(ROOT, "data", "dl_cache.npz"))
args = ap.parse_args()

dev = [l.strip() for l in open(os.path.join(ROOT, "experiments", "dev_patients.txt")) if l.strip()]
patients = sorted(d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d)))
held = [p for p in patients if p not in dev]
rng = random.Random(42)
test = sorted(rng.sample(held, 28))
train = sorted(p for p in held if p not in test)
split = {"seed": 42, "train": train, "val": dev, "test": test}
with open(os.path.join(ROOT, "experiments", "dl_split.json"), "w") as f:
    json.dump(split, f, indent=1)
print(f"train {len(train)}  val {len(dev)}  test {len(test)}")

imgs, masks, pid, sno = [], [], [], []
for p in patients:
    for s in load_patient(os.path.join(args.data, p), "flair"):
        imgs.append(s["image"])
        masks.append(s["gt"])
        pid.append(p)
        sno.append(s["slice_number"])
os.makedirs(os.path.dirname(args.out), exist_ok=True)
np.savez_compressed(args.out, images=np.stack(imgs), masks=np.stack(masks), patient=np.array(pid),
                    slice_number=np.array(sno))
print("cached", len(imgs), "slices to", args.out)
