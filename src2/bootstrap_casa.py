"""Paired bootstrap confidence intervals for Table 6.2 (comment 7).

Uses the seed-42 artefacts written by src/eval_casa.py (cached turn scores, window
scores and the linear meta-classifier). The saved artefacts in results/ come from the
later re-run of the pipeline, so the point estimates equal the printed Table 6.2
except for the CASA AUC in the fourth decimal (0.9079 here, 0.9077 printed).
Test conversations are resampled with replacement 2,000 times; stateless
and CASA are evaluated on the same resample each time, so the interval on their
difference is paired. Thresholds stay fixed at their validation values, as deployed.
"""
import sys, json, warnings; sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
import numpy as np, joblib
from sklearn.metrics import roc_auc_score
import data as D, casa as K

convs, df = D.load(); df = D.split(df); convs = D.conv_split(convs, df)
S = {k: np.array(v) for k, v in json.load(open("results/turn_scores.json")).items()}
WIN = json.load(open("results/window_scores.json"))
M = joblib.load("results/casa_meta_linear.pkl")
main = json.load(open("results/casa_main.json"))
thr_sl = [r for r in main["results"] if r["config"].startswith("B1")][0]["thr"]
Cte = [c for c in convs if c["part"] == "test"]
y = np.array([c["label"] for c in Cte])
sl = np.array([float(np.max(S[c["conv_id"]])) for c in Cte])
ca = M["meta"].predict_proba(K.conv_matrix(Cte, S, M["cfg"], win_cache=WIN))[:, 1]

def stats(idx):
    yy, a, b = y[idx], sl[idx], ca[idx]
    ra = float((a[yy == 1] >= thr_sl).mean()); rb = float((b[yy == 1] >= M["thr"]).mean())
    fa = float((a[yy == 0] >= thr_sl).mean()); fb = float((b[yy == 0] >= M["thr"]).mean())
    aa = roc_auc_score(yy, a); ab = roc_auc_score(yy, b)
    return [ra, rb, rb - ra, fa, fb, aa, ab, ab - aa]

names = ["stateless_recall", "casa_recall", "recall_gain", "stateless_fpr", "casa_fpr",
         "stateless_auc", "casa_auc", "auc_gain"]
point = stats(np.arange(len(y)))
rs = np.random.RandomState(42); boots = []
for _ in range(2000):
    boots.append(stats(rs.randint(0, len(y), len(y))))
boots = np.array(boots)
out = {}
for j, n in enumerate(names):
    lo, hi = np.percentile(boots[:, j], [2.5, 97.5])
    out[n] = {"point": point[j], "ci95": [float(lo), float(hi)]}
    print(f"{n:<18} {point[j]:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
out["p_auc_gain_le_0"] = float((boots[:, 7] <= 0).mean())
out["p_recall_gain_le_0"] = float((boots[:, 2] <= 0).mean())
print(f"bootstrap P(AUC gain <= 0) = {out['p_auc_gain_le_0']:.4f}   P(recall gain <= 0) = {out['p_recall_gain_le_0']:.4f}")
json.dump(out, open("results2/bootstrap_casa.json", "w"), indent=2)
