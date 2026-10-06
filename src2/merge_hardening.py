"""Merge per-seed CASA hardening results into results2/casa_fix.json.

casa_hardening.py can be run one seed at a time (--seeds N --out ...seedN.json) so that
seeds run in parallel. This combines every casa_fix_seed*.json present and recomputes
the per-configuration summary: mean with a 95% t-interval when two or more seeds are
available, and the plain mean (lo = hi = mean, n = 1) otherwise.
"""
import json, glob, numpy as np

files = sorted(glob.glob("results2/casa_fix_seed*.json"))
per_seed = []
for f in files:
    d = json.load(open(f))
    per_seed.extend(d["per_seed"])
# de-duplicate by seed, keep first occurrence
seen, uniq = set(), []
for ps in per_seed:
    if ps["seed"] not in seen:
        seen.add(ps["seed"]); uniq.append(ps)
per_seed = sorted(uniq, key=lambda ps: ps["seed"])

from scipy.stats import t as _tdist      # two-sided 95% t quantile, n-1 degrees of freedom
def ci(vals):
    v = np.array(vals, dtype=float); n = len(v); m = float(v.mean())
    if n < 2:
        return {"mean": m, "lo": m, "hi": m, "n": n}
    se = v.std(ddof=1) / np.sqrt(n); h = float(_tdist.ppf(0.975, n - 1)) * se
    # rates live in [0, 1]; the t-interval is clipped to that range rather than printed below zero
    return {"mean": m, "lo": float(max(0.0, m - h)), "hi": float(min(1.0, m + h)), "n": n}

CONFIGS = ["AT", "+P", "+C", "CASA-R"]
STRATS = ["none", "dilute", "combined", "split", "pace", "pace-wide", "launder"]
summary = {}
for c in CONFIGS:
    summary[c] = {s: ci([ps["configs"][c]["detection"][s] for ps in per_seed]) for s in STRATS}
    summary[c]["clean_fp"] = ci([ps["configs"][c]["fpr_control"]["none"] for ps in per_seed])
    summary[c]["combined_fp"] = ci([ps["configs"][c]["fpr_control"]["combined"] for ps in per_seed])

base = json.load(open(files[0]))
json.dump({"per_seed": per_seed, "summary": summary, "passage": base.get("passage", {}),
           "n_seeds": len(per_seed), "seeds": [ps["seed"] for ps in per_seed]},
          open("results2/casa_fix.json", "w"), indent=2)
r = summary["CASA-R"]
print(f"merged seeds {[ps['seed'] for ps in per_seed]} -> results2/casa_fix.json; "
      f"CASA-R dilute {r['dilute']['mean']*100:.1f} [{r['dilute']['lo']*100:.1f},{r['dilute']['hi']*100:.1f}] "
      f"combined {r['combined']['mean']*100:.1f} [{r['combined']['lo']*100:.1f},{r['combined']['hi']*100:.1f}]")
