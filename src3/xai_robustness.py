"""How much of the explainability audit (src2/xai_audit.py, report Section 6.11) is a property
of the detector and how much a property of one fitted model.

The audit attributes 52.9% of the deployed linear meta-classifier's output to the accumulator
features. Three checks on that reading:

  1. scale      the deployed model is an L2-regularised logistic regression fitted on unscaled
                features, so features with a large numeric range (the accumulator is a sum of
                log-odds) are cheap for it to use. Refit on standardised features and recompute
                the shares.
  2. removal    refit without each feature group and see what test AUC is lost. Attribution
                says where a model puts its weight; removal says whether it needed to.
  3. redundancy correlations among the accumulator features and with the pooled scores, and
                the AUC lost when a whole group is permuted jointly in the deployed model.

Writes results3/xai_robustness.json. Seconds on a CPU; uses the cached turn and window scores.

Settings that the output file does not record: the deployed model is scikit-learn's
LogisticRegression with its default L2 penalty and C = 1.0 (fitted in src/eval_casa.py);
the joint permutation here averages 50 shuffles, while the single-feature permutation
figures of src2/xai_audit.py average 20.
"""
import sys, json, warnings; sys.path.insert(0, "src"); sys.path.insert(0, "src2"); warnings.filterwarnings("ignore")
import numpy as np, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import data as D, casa as K

convs, df = D.load(); df = D.split(df); convs = D.conv_split(convs, df)
S = {k: np.array(v) for k, v in json.load(open("results/turn_scores.json")).items()}
WIN = json.load(open("results/window_scores.json"))
M = joblib.load("results/casa_meta_linear.pkl"); meta, cfg = M["meta"], M["cfg"]
part = lambda p: [c for c in convs if c["part"] == p]
Xtr, Xte = (K.conv_matrix(part(p), S, cfg, win_cache=WIN) for p in ("train", "test"))
ytr, yte = (np.array([c["label"] for c in part(p)]) for p in ("train", "test"))
names = list(K.FEATNAMES)
GROUP = {"log_turns": "structure", "p_mean": "pooled", "p_max": "pooled", "p_min": "pooled", "p_std": "pooled", "p_top2": "pooled",
         "p_median": "pooled", "sprt_final": "accumulator", "sprt_max": "accumulator", "sprt_mean": "accumulator",
         "sprt_slope": "accumulator", "alarm_pos_norm": "accumulator", "alarm_fired": "accumulator", "p_slope": "trajectory",
         "max_run_norm": "trajectory", "high_rate": "trajectory", "frac_high": "trajectory", "second_half_lift": "trajectory",
         "frac_retrieved": "provenance", "imp_retrieved": "provenance", "imp_user": "provenance", "imp_gap": "provenance",
         "imp_mean": "imperative", "imp_max": "imperative", "concat_all": "reassembly", "concat_user": "reassembly",
         "concat_retrieved": "provenance", "win2_max": "reassembly", "win2_mean": "reassembly", "win3_max": "reassembly", "win3_mean": "reassembly"}
groups = sorted(set(GROUP.values()))
idx = {g: [j for j, n in enumerate(names) if GROUP[n] == g] for g in groups}

def shares(coef, X):
    s = np.abs(coef) * np.mean(np.abs(X - X.mean(0)), axis=0)
    return {g: float(s[idx[g]].sum() / s.sum()) for g in groups}
auc = lambda m, X: float(roc_auc_score(yte, m.predict_proba(X)[:, 1]))

out = {"deployed": {"test_auc": auc(meta, Xte), "group_share": shares(meta.coef_[0], Xte)}}
# 1. scale
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
std = LogisticRegression(max_iter=3000).fit(Ztr, ytr)
out["standardised_refit"] = {"test_auc": auc(std, Zte), "group_share": shares(std.coef_[0], Zte)}
# 2. removal (same recipe as the deployed model: unscaled features)
full = LogisticRegression(max_iter=3000).fit(Xtr, ytr)
out["refit_full_auc"] = auc(full, Xte); out["removal"] = {}
for g in groups:
    keep = [j for j in range(len(names)) if j not in idx[g]]
    m = LogisticRegression(max_iter=3000).fit(Xtr[:, keep], ytr)
    out["removal"][g] = {"test_auc": auc(m, Xte[:, keep]), "delta": auc(m, Xte[:, keep]) - out["refit_full_auc"], "n_features": len(idx[g])}
# 3. redundancy
C = np.corrcoef(Xte.T); j = names.index
out["correlation"] = {"sprt_final~sprt_max": float(C[j("sprt_final"), j("sprt_max")]), "sprt_final~p_mean": float(C[j("sprt_final"), j("p_mean")]),
                      "sprt_final~concat_all": float(C[j("sprt_final"), j("concat_all")]), "sprt_max~p_max": float(C[j("sprt_max"), j("p_max")])}
rng = np.random.RandomState(42); out["joint_permutation"] = {}
for g in groups:
    drops = []
    for _ in range(50):
        Xp = Xte.copy(); perm = rng.permutation(len(Xte)); Xp[:, idx[g]] = Xte[perm][:, idx[g]]
        drops.append(out["deployed"]["test_auc"] - auc(meta, Xp))
    out["joint_permutation"][g] = float(np.mean(drops))
json.dump(out, open("results3/xai_robustness.json", "w"), indent=1)
d, s = out["deployed"], out["standardised_refit"]
print(f"deployed model   AUC {d['test_auc']:.4f}  accumulator share {d['group_share']['accumulator']*100:.1f}%")
print(f"standardised fit AUC {s['test_auc']:.4f}  accumulator share {s['group_share']['accumulator']*100:.1f}%  shares:", {g: round(v*100,1) for g, v in s['group_share'].items()})
print("removal (AUC change):", {g: round(v['delta'], 4) for g, v in out['removal'].items()})
print("joint permutation (AUC drop, deployed):", {g: round(v, 4) for g, v in out['joint_permutation'].items()})
print("correlations:", {k: round(v, 3) for k, v in out['correlation'].items()})
