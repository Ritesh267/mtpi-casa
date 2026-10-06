"""Repeat the headline conversation-level comparison over further partitions (comment 7).

The report gives Table 6.2 for a single partition (seed 42). Here the whole clean
pipeline is re-run for further partition seeds: a fresh source-prompt split, a fresh
stacked turn detector, fresh accumulator tuning and a fresh linear meta-classifier.
Only unmodified benchmark conversations are used. For each seed the script records
the stateless baseline (B1), CASA with the linear meta-classifier (B7), the
fragmentation ceiling and the proportion of it recovered, all at a 5% false-positive
budget set on benign validation conversations exactly as in src/eval_casa.py.

The partitions are fresh random re-splits of the same corpus, not independent samples:
any two of the five test sets (seeds 42 and 1 to 4) share 79 to 102 of their 461
source prompts, so an interval across them is indicative only. The realised test
false-positive rates also differ from seed to seed.
"""
import sys, os, json, time, warnings, argparse
sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import data as D, casa as K
from ensemble import StackedTurnDetector

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True)
a = ap.parse_args(); SEED = a.seed; B = 0.05
os.makedirs("results2/multiseed_casa", exist_ok=True)
out_path = f"results2/multiseed_casa/seed{SEED}.json"
if os.path.exists(out_path): print("exists", out_path); sys.exit()

t0 = time.time()
convs, df = D.load(); df = D.split(df, seed=SEED); convs = D.conv_split(convs, df)
tr = df[df.part == "train"]
det = StackedTurnDetector(seed=SEED).fit(tr.text.tolist(), tr.y.values)
print(f"seed {SEED}: stack fitted ({time.time()-t0:.0f}s)", flush=True)
scorer = lambda t: det.predict_proba(list(t))[:, 1]

texts = [t["text"] for c in convs for t in c["turns"]]
P = scorer(texts); S = {}; i = 0
for c in convs: S[c["conv_id"]] = P[i:i + len(c["turns"])]; i += len(c["turns"])
WIN = {c["conv_id"]: K.window_scores([t["text"] for t in c["turns"]], [t["channel"] for t in c["turns"]], scorer)
       for c in convs}
print(f"seed {SEED}: turns and windows scored ({time.time()-t0:.0f}s)", flush=True)

C = {p: [c for c in convs if c["part"] == p] for p in ("train", "val", "test")}
Y = {p: np.array([c["label"] for c in C[p]]) for p in C}
te_turn = df[df.part == "test"]; turn_auc = float(roc_auc_score(te_turn.y.values, scorer(te_turn.text.tolist())))

cfg, _ = K.tune_sprt(C["val"], S, Y["val"])
X = {p: K.conv_matrix(C[p], S, cfg, win_cache=WIN) for p in C}
lin = LogisticRegression(max_iter=3000).fit(X["train"], Y["train"])

def thr(y, s): return float(np.quantile(np.asarray(s)[np.asarray(y) == 0], 1 - B))
def m(y, s, t):
    yh = s >= t
    return {"recall": float(yh[y == 1].mean()), "fpr": float(yh[y == 0].mean()), "auc": float(roc_auc_score(y, s))}

sl = {p: np.array([float(np.max(S[c["conv_id"]])) for c in C[p]]) for p in ("val", "test")}
ca = {p: lin.predict_proba(X[p])[:, 1] for p in ("val", "test")}
res = {"seed": SEED, "turn_auc": turn_auc,
       "sprt_cfg": {"leak": cfg.leak, "alarm": cfg.alarm, "prior_user": cfg.prior_user},
       "stateless": m(Y["test"], sl["test"], thr(Y["val"], sl["val"])),
       "casa_linear": m(Y["test"], ca["test"], thr(Y["val"], ca["val"])),
       "n_test": int(len(Y["test"])), "n_test_attack": int(Y["test"].sum())}
orc = float(roc_auc_score(Y["test"], scorer([c["source_prompt"] for c in C["test"]])))
res["oracle_auc"] = orc
res["recovered_pct"] = 100 * (res["casa_linear"]["auc"] - res["stateless"]["auc"]) / max(orc - res["stateless"]["auc"], 1e-9)
res["time_s"] = time.time() - t0
json.dump(res, open(out_path, "w"), indent=2)
print(json.dumps(res, indent=2))
