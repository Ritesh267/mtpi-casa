"""Prepare the supervision pool for the GPU LoRA-DPO study (supervisor comments 2, 3, 7).

Scores every source prompt with two detectors so that the GPU script never needs
scikit-learn or the 188 MB stacked detector:

  strong  the deployed stacked turn detector (results/stack.pkl), as in Section 6.6
  weak    the engineered 24-feature set (R4) with logistic regression, the weakest
          configuration in Table 6.1. It is a real detector that a deployer could
          plausibly ship, not a synthetic corruption of the strong one.

Partitioning is by source prompt, exactly as in data.split, so no test prompt or
fragment of one is seen during detector fitting or preference training.

Writes results2/dpo_pool.json, which holds the full text of every source prompt and
is therefore not shipped; run this script to rebuild it. The stored agreement of each
detector with the ground truth (agreement_train_pool: 94.4% strong, 63.1% weak) is
measured on the training pool, where the strong detector is in-sample. The agreement
on the held-out test prompts is not stored; recomputed from the p_strong and p_weak
scores of the test rows it is 83.5% and 62.5%.
"""
import sys, json, warnings; sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
import numpy as np, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import data as D, reps as R

convs, df = D.load(); df = D.split(df)
srcs = df.groupby("src").agg(y=("conv_label", "first"), part=("part", "first")).reset_index()
tr = srcs[srcs.part == "train"].reset_index(drop=True)
te = srcs[srcs.part == "test"].reset_index(drop=True)
va = srcs[srcs.part == "val"].reset_index(drop=True)
print(f"source prompts: train {len(tr)} (attack rate {tr.y.mean():.2f})  test {len(te)} (attack rate {te.y.mean():.2f})")

stack = joblib.load("results/stack.pkl")
p_strong_tr = stack.predict_proba(tr.src.tolist())[:, 1]
p_strong_te = stack.predict_proba(te.src.tolist())[:, 1]

# weak detector: engineered features, logistic regression, fitted on training TURNS
# (the same data the stacked detector was fitted on), then applied to source prompts
ttr = df[df.part == "train"]
eng = R.Engineered().fit(ttr.text.tolist())
lr = LogisticRegression(max_iter=3000, C=2.0, random_state=42).fit(eng.transform(ttr.text.tolist()), ttr.y.values)
p_weak_tr = lr.predict_proba(eng.transform(tr.src.tolist()))[:, 1]
p_weak_te = lr.predict_proba(eng.transform(te.src.tolist()))[:, 1]

out = {"train": [], "test": [], "val": [], "detectors": {}}
for name, ptr, pte in [("strong", p_strong_tr, p_strong_te), ("weak", p_weak_tr, p_weak_te)]:
    agree_tr = float(((ptr >= 0.5).astype(int) == tr.y.values).mean())
    out["detectors"][name] = {
        "auc_train_pool": float(roc_auc_score(tr.y.values, ptr)),
        "auc_test": float(roc_auc_score(te.y.values, pte)),
        "agreement_train_pool": agree_tr,
    }
    print(f"{name:>6}: AUC train-pool {out['detectors'][name]['auc_train_pool']:.4f}  "
          f"test {out['detectors'][name]['auc_test']:.4f}  agreement with ground truth {agree_tr*100:.1f}%")

for i, r in tr.iterrows():
    out["train"].append({"text": r.src, "y": int(r.y), "p_strong": float(p_strong_tr[i]), "p_weak": float(p_weak_tr[i])})
for i, r in te.iterrows():
    out["test"].append({"text": r.src, "y": int(r.y), "p_strong": float(p_strong_te[i]), "p_weak": float(p_weak_te[i])})

# validation prompts, label-free use only: calibrating one fixed refusal threshold on the base model
for i, r in va.iterrows():
    out["val"].append({"text": r.src, "y": int(r.y)})

import os; os.makedirs("results2", exist_ok=True)
json.dump(out, open("results2/dpo_pool.json", "w"))
print("wrote results2/dpo_pool.json")
