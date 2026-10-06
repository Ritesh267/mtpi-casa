"""CASA-R: two additions to the conversation detector, measured (supervisor comment 5).

Section 6.5.2 left two conditions unrepaired and diagnosed each. This script adds the
two fixes Section 8.3 specifies, and measures detection recovery against the
benign-transformation control that Section 6.5.1 shows is essential.

Fix P (turn layer). Padding a short payload span inside a long benign turn lowers the
whole-turn score. Each turn is therefore additionally scored in overlapping word
windows (40 words, stride 20) and four features are added to the conversation vector:
the highest passage score, the mean of the two highest per-turn passage scores, and the
largest and the mean gain of a turn's best passage over its whole-turn score. A padded
turn is exactly the case where that gain is large.

Fix C (training distribution). Each augmented copy is a composition of one, two or
three of the single transformations at randomised strength (about half of the copies
receive one step), applied to benign traffic at the same rate. The full four-way
test-time combination is never generated for training here, so the combined column
measures generalisation to an unseen composition.

Configurations, all at a matched 5% false-positive budget on clean benign validation:
  AT       a baseline refitted inside this script (31 features): every copy is
           transformed by ONE of the five single strategies, drawn with replacement,
           with "combined" excluded. It is not the CASA-AT of src/adv_train.py, which
           draws its two copies without replacement from all six strategies.
  +P       AT plus the passage features (35 features)
  +C       AT plus compositional augmentation
  CASA-R   both

Attack variants come from src/adaptive.py unchanged. Repeated over training seeds: a
seed changes the draw of the training augmentation and the random state of the
meta-classifier only. The rewritten test attacks use one fixed seed (99) and are the
same for every seed and configuration. Mean and 95% t-interval are reported; the stored
results2/casa_fix.json is written by src2/merge_hardening.py, which recomputes the
intervals from the per-seed files. The benign-transformation false-positive control is
reported for every configuration.

Known limitations of the construction (see the README). After its first step compose()
sets the lead-in count to zero, so later steps treat the whole conversation as body:
lead-in turns and inserted filler are rewritten and, in attack conversations, flagged
as payload, and a retrieval conversation with a lead-in loses its retrieved channel.
"pace" and "pace-wide" are the same transformation there, because the gap is drawn at
random. The benign control creates its generator afresh for every conversation, so
every benign conversation receives the same filler sequence.
"""
import sys, os, json, time, argparse, warnings
sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
import numpy as np, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
import data as D, casa as K, adaptive as A

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--pw", type=int, default=40)     # passage window (words)
ap.add_argument("--ps", type=int, default=20)     # passage stride
ap.add_argument("--out", default="results2/casa_fix.json")
a = ap.parse_args()

FPR = 0.05
convs, df = D.load(); df = D.split(df); convs = D.conv_split(convs, df)
stack = joblib.load("results/stack.pkl")
cfg = joblib.load("results/casa_meta_linear.pkl")["cfg"]
scorer = lambda t: stack.predict_proba(list(t))[:, 1]
Ctr = [c for c in convs if c["part"] == "train"]
Cva = [c for c in convs if c["part"] == "val"]
Cte = [c for c in convs if c["part"] == "test"]
benign_pool = [c["source_prompt"] for c in convs if c["label"] == 0 and c["part"] == "train"]

STR = ["split", "pace", "pace-wide", "dilute", "launder", "combined"]
PAR = {"split": {"factor": 3}, "pace": {"gap": 2}, "pace-wide": {"gap": 4},
       "dilute": {"ratio": 1.5}, "launder": {}, "combined": {"factor": 3, "gap": 2, "ratio": 1.0}}

def variant(c, s, rng):
    p = PAR[s]
    return A.build_variant(c, benign_pool, rng, s.replace("-wide", ""),
                           {"factor": p.get("factor", 2), "gap": p.get("gap", 2), "ratio": p.get("ratio", 1.0)})

# ---- passage features -------------------------------------------------------
def passages(text):
    w = str(text).split()
    if len(w) <= a.pw:
        return [" ".join(w)] if w else [str(text)]
    out = [" ".join(w[i:i + a.pw]) for i in range(0, len(w) - a.pw + 1, a.ps)]
    if (len(w) - a.pw) % a.ps:
        out.append(" ".join(w[-a.pw:]))
    return out

PNAMES = ["pass_max", "pass_top2", "pass_gain_max", "pass_gain_mean"]

def passage_feats(turns, turn_scores):
    """Four scalars summarising the best sub-passage evidence per conversation."""
    gains = []; maxes = []
    all_pass = []
    for t, ts in zip(turns, turn_scores):
        ps = passages(t["text"])
        all_pass.extend(ps)
    if all_pass:
        sp = scorer(all_pass)
    idx = 0; pm = []
    for t, ts in zip(turns, turn_scores):
        ps = passages(t["text"]); pv = sp[idx:idx + len(ps)]; idx += len(ps)
        m = float(np.max(pv)) if len(pv) else float(ts)
        pm.append(m); maxes.append(m); gains.append(m - float(ts))
    pm = np.array(pm) if pm else np.array([0.0])
    return np.array([float(np.max(pm)), float(np.mean(np.sort(pm)[-2:])),
                     float(np.max(gains)) if gains else 0.0, float(np.mean(gains)) if gains else 0.0], dtype=np.float32)

def featurise(turnlists, with_pass):
    flat = [t["text"] for tl in turnlists for t in tl]
    P = scorer(flat); S = {}; i = 0
    for j, tl in enumerate(turnlists): S[j] = P[i:i + len(tl)]; i += len(tl)
    fake = [{"conv_id": j, "turns": tl} for j, tl in enumerate(turnlists)]
    WIN = {j: K.window_scores([t["text"] for t in tl], [t["channel"] for t in tl], scorer) for j, tl in enumerate(turnlists)}
    X = K.conv_matrix(fake, S, cfg, win_cache=WIN)
    if not with_pass:
        return X
    PX = np.vstack([passage_feats(tl, S[j]) for j, tl in enumerate(turnlists)])
    return np.hstack([X, PX])

def compose(c, rng):
    """One augmented copy: a single transformation, or a composition of two or three."""
    k = rng.choice([1, 1, 2, 3])
    picks = list(rng.choice([s for s in STR if s != "combined"], size=min(k, 5), replace=False))
    v = c["turns"]
    tmp = {"turns": v, "lead_turns": c.get("lead_turns"), "label": c["label"]}
    for s in picks:
        tmp = {"turns": A.build_variant(tmp, benign_pool, rng, s.replace("-wide", ""),
               {"factor": rng.choice([2, 3]), "gap": int(rng.choice([2, 3, 4])), "ratio": float(rng.choice([1.0, 1.5]))}),
               "lead_turns": 0, "label": c["label"]}
    return tmp["turns"]

def augment(cs, rng, compositional):
    tl = []; y = []
    for c in cs:
        tl.append(c["turns"]); y.append(c["label"])
        for _ in range(2):
            v = compose(c, rng) if compositional else variant(c, rng.choice([s for s in STR if s != "combined"]), rng)
            if c["label"] == 1:
                o = [t["text"] for t in c["turns"] if t.get("carries_payload", 0) == 1]
                nw = [t["text"] for t in v if t.get("carries_payload", 0) == 1]
                if not A.payload_preserved(o, nw): continue
            tl.append(v); y.append(c["label"])
    return tl, np.array(y)

def pick_thr(y, s): return float(np.quantile(np.asarray(s)[np.asarray(y) == 0], 1 - FPR))

CONFIGS = [("AT", False, False), ("+P", True, False), ("+C", False, True), ("CASA-R", True, True)]
atk = [c for c in Cte if c["label"] == 1]; ben = [c for c in Cte if c["label"] == 0]

per_seed = []
for seed in a.seeds:
    t0 = time.time(); rng = np.random.RandomState(seed)
    seedres = {"seed": seed, "configs": {}}
    for cname, wp, comp in CONFIGS:
        TL, Y = augment(Ctr, np.random.RandomState(seed), comp)
        Xtr = featurise(TL, wp)
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, random_state=seed).fit(Xtr, Y)
        Xva = featurise([c["turns"] for c in Cva if c["label"] == 0], wp)
        thr = float(np.quantile(m.predict_proba(Xva)[:, 1], 1 - FPR))
        row = {}
        for s in ["none", "dilute", "combined", "split", "pace", "pace-wide", "launder"]:
            rng2 = np.random.RandomState(99)
            tls = []; keep = []
            for c in atk:
                v = c["turns"] if s == "none" else variant(c, s, rng2)
                ok = True
                if s != "none":
                    o = [t["text"] for t in c["turns"] if t.get("carries_payload", 0) == 1]
                    nw = [t["text"] for t in v if t.get("carries_payload", 0) == 1]
                    ok = A.payload_preserved(o, nw)
                tls.append(v); keep.append(ok)
            keep = np.array(keep); X = featurise(tls, wp)
            row[s] = float((m.predict_proba(X)[:, 1][keep] >= thr).mean())
        # benign-transformation control (fp on benign transformed like the attacks).
        # Known limitation: RandomState(5) is created afresh for every conversation,
        # so every benign conversation receives the same filler sequence.
        fpr_ctrl = {}
        for s in ["dilute", "combined"]:
            TLb = [variant(c, s, np.random.RandomState(5)) for c in ben]
            fpr_ctrl[s] = float((m.predict_proba(featurise(TLb, wp))[:, 1] >= thr).mean())
        fpr_ctrl["none"] = float((m.predict_proba(featurise([c["turns"] for c in ben], wp))[:, 1] >= thr).mean())
        seedres["configs"][cname] = {"detection": row, "fpr_control": fpr_ctrl, "threshold": thr}
        print(f"seed {seed} {cname:<7} dilute {row['dilute']*100:5.1f}  combined {row['combined']*100:5.1f}  "
              f"clean-fp {fpr_ctrl['none']*100:4.1f}  ({time.time()-t0:.0f}s)", flush=True)
    per_seed.append(seedres)

# aggregate. Two-sided 95% t multipliers for n values (n - 1 degrees of freedom), the same
# values as src2/merge_hardening.py, which writes the stored results2/casa_fix.json.
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
def ci(vals):
    v = np.array(vals); n = len(v)
    if n in T95:
        t = T95[n]
    elif n > 5:
        from scipy.stats import t as _t
        t = float(_t.ppf(0.975, n - 1))
    else:
        t = 0.0                         # a single value has no interval
    se = v.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return {"mean": float(v.mean()), "lo": float(v.mean() - t * se), "hi": float(v.mean() + t * se), "n": n}
summary = {}
for cname, _, _ in CONFIGS:
    summary[cname] = {s: ci([ps["configs"][cname]["detection"][s] for ps in per_seed])
                      for s in ["none", "dilute", "combined", "split", "pace", "pace-wide", "launder"]}
    summary[cname]["clean_fp"] = ci([ps["configs"][cname]["fpr_control"]["none"] for ps in per_seed])
    summary[cname]["combined_fp"] = ci([ps["configs"][cname]["fpr_control"]["combined"] for ps in per_seed])
os.makedirs("results2", exist_ok=True)
json.dump({"per_seed": per_seed, "summary": summary, "passage": {"window": a.pw, "stride": a.ps}},
          open(a.out, "w"), indent=2)
print("wrote", a.out)
