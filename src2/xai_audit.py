"""Explainability audit of the conversation-level detector (supervisor comment 4).

The predecessor dissertation used SHAP attributions on its prompt-injection detector
and found that a single feature, prompt length, dominated the decision, which it
reported as a dual-use result: the attribution that lets a defender audit a block
also tells an attacker the cheapest feature to move. This script performs the
analogous audit for CASA, so the same reasoning is carried into the present work.

Three attributions of the deployed linear meta-classifier over the 31 conversation
features are computed on the held-out test conversations:

  1. standardised linear coefficients (exact for a linear model),
  2. permutation importance (drop in ROC-AUC when one feature is shuffled),
  3. mean absolute SHAP value via the exact linear/Shapley identity
     phi_j(x) = w_j * (x_j - E[x_j]), so |phi_j| averaged over x is |w_j| * E|x_j - mean|.

How to read the output. The attribution describes one fitted model, an L2-regularised
logistic regression on unscaled features. It shows where that model puts its weight,
not which component the detector needs. src3/xai_robustness.py checks the reading
against feature scale, removal and collinearity, and src3/pace_mechanism.py tests
directly whether the accumulator explains the pacing attack. An earlier version of
this script stored a note that linked the component with the largest share to the
adaptive strategy named after it; those checks do not support that conclusion and the
note has been removed (result files written before the change may still carry it).
No new attack is run here.
"""
import sys, json, warnings; sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
import numpy as np, joblib
from sklearn.metrics import roc_auc_score
import data as D, casa as K

convs, df = D.load(); df = D.split(df); convs = D.conv_split(convs, df)
S = {k: np.array(v) for k, v in json.load(open("results/turn_scores.json")).items()}
WIN = json.load(open("results/window_scores.json"))
M = joblib.load("results/casa_meta_linear.pkl"); meta, cfg = M["meta"], M["cfg"]
Cte = [c for c in convs if c["part"] == "test"]
y = np.array([c["label"] for c in Cte])
X = K.conv_matrix(Cte, S, cfg, win_cache=WIN)
names = list(K.FEATNAMES)

# standardise for comparable coefficients
mu, sd = X.mean(0), X.std(0) + 1e-9
w = meta.coef_[0] * sd                      # effect of a 1-sd change in each feature
base_auc = roc_auc_score(y, meta.predict_proba(X)[:, 1])

# permutation importance
rng = np.random.RandomState(42)
perm = np.zeros(len(names))
for j in range(len(names)):
    drops = []
    for _ in range(20):
        Xp = X.copy(); Xp[:, j] = X[rng.permutation(len(X)), j]
        drops.append(base_auc - roc_auc_score(y, meta.predict_proba(Xp)[:, 1]))
    perm[j] = np.mean(drops)

# exact mean|SHAP| for a linear model
shap = np.abs(meta.coef_[0]) * np.mean(np.abs(X - mu), axis=0)

# component grouping, aligned with the feature table of the report
GROUP = {"log_turns": "structure",
         "p_mean": "pooled", "p_max": "pooled", "p_min": "pooled", "p_std": "pooled",
         "p_top2": "pooled", "p_median": "pooled",
         "sprt_final": "accumulator", "sprt_max": "accumulator", "sprt_mean": "accumulator",
         "sprt_slope": "accumulator", "alarm_pos_norm": "accumulator", "alarm_fired": "accumulator",
         "p_slope": "trajectory", "max_run_norm": "trajectory", "high_rate": "trajectory",
         "frac_high": "trajectory", "second_half_lift": "trajectory",
         "frac_retrieved": "provenance", "imp_retrieved": "provenance", "imp_user": "provenance",
         "imp_gap": "provenance", "imp_mean": "imperative", "imp_max": "imperative",
         "concat_all": "reassembly", "concat_user": "reassembly", "concat_retrieved": "provenance",
         "win2_max": "reassembly", "win2_mean": "reassembly", "win3_max": "reassembly", "win3_mean": "reassembly"}

order = np.argsort(-shap)
rows = []
for j in order:
    rows.append({"feature": names[j], "group": GROUP.get(names[j], "other"),
                 "coef_std": float(w[j]), "perm_importance": float(perm[j]), "mean_abs_shap": float(shap[j])})

# aggregate by component
grp = {}
for r in rows:
    grp.setdefault(r["group"], 0.0)
    grp[r["group"]] += r["mean_abs_shap"]
tot = sum(grp.values())
grp_share = {g: v / tot for g, v in sorted(grp.items(), key=lambda kv: -kv[1])}

print("top 10 features by mean|SHAP|:")
for r in rows[:10]:
    print(f"  {r['feature']:<16} {r['group']:<12} coef(sd) {r['coef_std']:+.3f}  perm {r['perm_importance']:+.4f}  |shap| {r['mean_abs_shap']:.4f}")
print("\ncomponent share of total attribution:")
for g, s in grp_share.items():
    print(f"  {g:<12} {s*100:5.1f}%")

top_group = next(iter(grp_share))
out = {"base_auc": float(base_auc), "features": rows, "group_share": grp_share,
       "top_feature": rows[0]["feature"], "top_group": top_group,
       "note": ("Attribution of one fitted model (L2 logistic regression on unscaled features). "
                "See results3/xai_robustness.json for scale, removal and collinearity checks "
                "before reading it as component importance.")}
json.dump(out, open("results2/xai_audit.json", "w"), indent=2)
print(f"\nlargest share of mean |SHAP| in this fitted model: {top_group} ({grp_share[top_group]*100:.1f}%)")
print("wrote results2/xai_audit.json")
