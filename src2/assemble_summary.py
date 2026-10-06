"""Collect every result file into one summary.json that the deliverable documents read."""
import json, os, glob, numpy as np
from scipy.stats import t as _tdist
tq = lambda n: float(_tdist.ppf(0.975, n - 1))     # two-sided 95% t quantile, n-1 degrees of freedom

R = "results2"
def load(p, d=None):
    return json.load(open(p)) if os.path.exists(p) else d

S = {}

# --- comment 7: bootstrap CIs on Table 6.2 ---
S["bootstrap"] = load(f"{R}/bootstrap_casa.json")

# --- comment 7: multi-partition CASA ---
ms = [load(p) for p in sorted(glob.glob(f"{R}/multiseed_casa/seed*.json"))]
ms = [m for m in ms if m]
if ms:
    def ci(vals):
        vals = np.array(vals); m = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        h = tq(len(vals)) * se
        return {"mean": float(m), "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "lo": float(m - h), "hi": float(m + h), "n": len(vals)}
    S["multiseed_casa"] = {
        "seeds": [m["seed"] for m in ms],
        "stateless_auc": ci([m["stateless"]["auc"] for m in ms]),
        "casa_auc": ci([m["casa_linear"]["auc"] for m in ms]),
        "auc_gain": ci([m["casa_linear"]["auc"] - m["stateless"]["auc"] for m in ms]),
        "casa_recall": ci([m["casa_linear"]["recall"] for m in ms]),
        "recovered_pct": ci([m["recovered_pct"] for m in ms]),
        "per_seed": ms}

# --- comments 2,3,7: GPU LoRA-DPO grid ---
runs = [load(p) for p in sorted(glob.glob(f"{R}/dpo_gpu/*seed*.json"))]
runs = [r for r in runs if r and "eval" in r]
if runs:
    # Use only seeds for which every one of the nine trained configurations has finished, so
    # that every row of the table rests on the same seeds.
    def _tag(r): return r["arm"] if r["arm"] in ("A0", "A1") else f"{r['arm']}_{r.get('det')}_{r.get('sel')}"
    _by = {}
    for r in runs:
        if r["arm"] != "A0": _by.setdefault(_tag(r), set()).add(r["seed"])
    _complete = sorted(set.intersection(*_by.values())) if len(_by) == 9 else sorted(set.union(*_by.values()))
    runs = [r for r in runs if r["arm"] == "A0" or r["seed"] in _complete]
    def agg(sel):
        rs = sorted([r for r in runs if sel(r)], key=lambda r: r["seed"])
        if not rs: return None
        h = lambda k: np.array([r["eval"]["heldout"][k] for r in rs], dtype=float)
        v = h("separation") * 100
        n = len(v); t = tq(n)   # two-sided 95% t, n-1 degrees of freedom
        se = v.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        return {"sep_mean": float(v.mean()), "sep_lo": float(v.mean() - t * se),
                "sep_hi": float(v.mean() + t * se), "auc_mean": float(h("refusal_auc").mean()),
                "nll_mean": float(np.mean([r["eval"]["lm_nll_benign"] for r in rs])), "n": n,
                "per_seed": [float(x) for x in v], "seeds": [r["seed"] for r in rs],
                "refuse_attack": float(h("refuse_attack").mean()), "refuse_benign": float(h("refuse_benign").mean()),
                "raw_sep": float((h("raw_refuse_attack") - h("raw_refuse_benign")).mean() * 100),
                "label_agreement": rs[0].get("label_agreement_in_pairs")}
    A0 = next((r for r in runs if r["arm"] == "A0"), None)
    _tr = [r.get("trainable_params") or 0 for r in runs]
    _to = [r.get("total_params") or 0 for r in runs]
    grid = {"base": runs[0]["base"], "trainable": max(_tr) or None,
            "total": max(_to) or None,
            "seeds": _complete,
            "A0": ({"sep": A0["eval"]["heldout"]["separation"] * 100,
                    "auc": A0["eval"]["heldout"]["refusal_auc"],
                    "refuse_attack": A0["eval"]["heldout"]["refuse_attack"],
                    "refuse_benign": A0["eval"]["heldout"]["refuse_benign"],
                    "nll": A0["eval"]["lm_nll_benign"]} if A0 else None),
            "A1": agg(lambda r: r["arm"] == "A1")}
    for det in ("strong", "weak"):
        for sel in ("confident", "boundary"):
            for arm in ("A2", "A3"):
                grid[f"{arm}_{det}_{sel}"] = agg(lambda r, a=arm, d=det, s=sel:
                                                 r["arm"] == a and r.get("det") == d and r.get("sel") == s)
    # detector pool agreement
    pool = load(f"{R}/dpo_pool.json")
    if pool: grid["detector_agreement"] = {k: v["agreement_train_pool"] for k, v in pool["detectors"].items()}
    S["dpo_gpu"] = grid

# --- comments 2,7: CPU seeded DPO ---
cpu = [load(p) for p in sorted(glob.glob(f"{R}/dpo_cpu_seeds/seed*.json"))]
cpu = [c for c in cpu if c]
if cpu:
    def cagg(arm, det, sel):
        seps, aucs = [], []
        for c in cpu:
            for r in c["rows"]:
                if r["arm"] == arm and r.get("det") == det and r.get("sel") == sel:
                    seps.append(r["separation"] * 100); aucs.append(r["refusal_auc"])
        if not seps: return None
        v = np.array(seps); n = len(v); t = tq(n)   # two-sided 95% t, n-1 degrees of freedom
        se = v.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        return {"sep_mean": float(v.mean()), "sep_lo": float(v.mean() - t * se), "sep_hi": float(v.mean() + t * se),
                "auc_mean": float(np.mean(aucs)), "n": n}
    cd = {"seeds": [c["seed"] for c in cpu], "A0": cagg("A0", None, None), "A1": cagg("A1", None, None)}
    for det in ("strong", "weak"):
        for sel in ("confident", "boundary"):
            for arm in ("A2", "A3"):
                cd[f"{arm}_{det}_{sel}"] = cagg(arm, det, sel)
    S["dpo_cpu"] = cd

# --- comment 4: XAI audit ---
S["xai"] = load(f"{R}/xai_audit.json")

# --- comment 1: live ASR ---
asr = load(f"{R}/asr_live.json")
if asr and asr.get("models"):
    S["asr"] = asr

# --- comment 5: CASA hardening (if present) ---
S["casa_fix"] = load(f"{R}/casa_fix.json")
# --- comment 6: composed attacks (if present) ---
S["composed"] = load(f"{R}/composed_eval.json")

json.dump(S, open(f"{R}/summary.json", "w"), indent=2)
present = [k for k, v in S.items() if v]
print("summary.json written. sections present:", present)
