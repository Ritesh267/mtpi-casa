"""Offline versus online decisions (report Section 6.14).

Sections 6.3 to 6.5 score each conversation once, after its last turn ("offline"). A filter
in front of a model cannot wait for the last turn: it decides after every turn, and a
conversation is blocked as soon as one decision fires ("online"). The two are different
tests. Online, a detector gets one chance per turn to fire, which raises both detection and
false alarms, and an attacker can no longer erase earlier evidence by appending text.

This script repeats the adaptive evaluation of recalib.py with both rules, for the stateless
filter, CASA and adversarially trained CASA:

  offline              the published rule and thresholds (reproduces results/adaptive_final.json)
  online               one decision per turn, thresholds re-calibrated so that 5% of clean
                       benign VALIDATION conversations are blocked at any turn
  online, old thr      one decision per turn with the offline thresholds left unchanged, which
                       is what Section 6.8 deployed

and for each attacker strategy also the false-positive rate on benign conversations
transformed the same way. It then measures one further manipulation the offline rule
permits: appending benign turns after the attack is complete.

Writes results3/online_eval.json. CPU only; about 40 minutes on one core.
"""
import hashlib, json, sys, time, warnings
sys.path.insert(0, "src"); sys.path.insert(0, "src3"); warnings.filterwarnings("ignore")
import numpy as np, joblib
import data as D, casa as K, adaptive as A
from rag_detect import Scorer

BUDGET = 0.05
t0 = time.time()
convs, df = D.load(); df = D.split(df); convs = D.conv_split(convs, df)
stack = joblib.load("results/stack.pkl")
base = joblib.load("results/casa_meta_linear.pkl"); cfg = base["cfg"]
AT = joblib.load("results/casa_at.pkl")["models"]
fin = json.load(open("results/adaptive_final.json"))
thr_off = {"stateless": json.load(open("results/casa_main.json"))["results"][0]["thr"], "CASA-base": base["thr"],
           **{n: fin["thresholds"][n] for n in AT}}
MODELS = {"CASA-base": base["meta"], **{n: d["m"] for n, d in AT.items()}}
DEFS = ["stateless"] + list(MODELS)
sc = Scorer(stack)

def prefix_scores(turnlists):
    """For every conversation: per-prefix score of every defence (list over prefixes)."""
    for tl in turnlists: sc.collect([t["text"] for t in tl])
    sc.flush()
    P = [sc([t["text"] for t in tl]) for tl in turnlists]
    def feats(tl, p, scorer):
        ch = [t["channel"] for t in tl]; tx = [t["text"] for t in tl]; rows = []
        for i in range(1, len(tl) + 1):
            rows.append(K.conv_features(p[:i], ch[:i], tx[:i], cfg, K.window_scores(tx[:i], ch[:i], scorer)))
        return np.nan_to_num(np.vstack(rows), nan=0.0, posinf=0.0, neginf=0.0)
    for tl, p in zip(turnlists, P): feats(tl, p, sc.collect)
    sc.flush()
    out = []
    for tl, p in zip(turnlists, P):
        X = feats(tl, p, sc)
        out.append({"stateless": np.maximum.accumulate(p), **{n: m.predict_proba(X)[:, 1] for n, m in MODELS.items()}})
    return out

def rates(S, thr_on):
    """Share of conversations flagged under each rule."""
    r = {}
    for d in DEFS:
        last = np.array([s[d][-1] for s in S]); mx = np.array([np.max(s[d]) for s in S])
        r[d] = {"offline": float((last >= thr_off[d]).mean()), "online": float((mx >= thr_on[d]).mean()),
                "online_old_thr": float((mx >= thr_off[d]).mean())}
    return r

Cva = [c for c in convs if c["part"] == "val"]; Cte = [c for c in convs if c["part"] == "test"]
atk = [c for c in Cte if c["label"] == 1]; ben = [c for c in Cte if c["label"] == 0]
Sv = prefix_scores([c["turns"] for c in Cva if c["label"] == 0])
thr_on = {d: float(np.quantile([np.max(s[d]) for s in Sv], 1 - BUDGET)) for d in DEFS}
print("online thresholds:", {k: round(v, 4) for k, v in thr_on.items()}, f"({time.time()-t0:.0f}s)", flush=True)

benign_pool = [c["source_prompt"] for c in convs if c["label"] == 0 and c["part"] == "train"]
STRATS = ["split", "pace", "pace-wide", "dilute", "launder", "combined"]
PARAMS = {"split": {"factor": 3}, "pace": {"gap": 2}, "pace-wide": {"gap": 4}, "dilute": {"ratio": 1.5}, "launder": {},
          "combined": {"factor": 3, "gap": 2, "ratio": 1.0}}
def variant(c, s, rng):
    p = PARAMS.get(s, {})
    return A.build_variant(c, benign_pool, rng, s.replace("-wide", ""),
                           {"factor": p.get("factor", 2), "gap": p.get("gap", 2), "ratio": p.get("ratio", 1.0)})

res = {"budget": BUDGET, "thresholds": {"offline": thr_off, "online": thr_on}, "attack": {}, "benign": {}, "append": {}}
rng = np.random.RandomState(99)       # the same stream as recalib.py, so the offline column reproduces Table 6.9
rngb = np.random.RandomState(5)       # one stream for the benign control (each conversation gets different filler)
for s in ["none"] + STRATS:
    tls, keep = [], []
    for c in atk:
        v = c["turns"] if s == "none" else variant(c, s, rng)
        ok = True
        if s != "none":
            ok = A.payload_preserved([t["text"] for t in c["turns"] if t["carries_payload"] == 1],
                                     [t["text"] for t in v if t["carries_payload"] == 1])
        tls.append(v); keep.append(ok)
    S = prefix_scores(tls); S = [x for x, k in zip(S, keep) if k]
    res["attack"][s] = {"kept": int(sum(keep)), "mean_turns": float(np.mean([len(t) for t in tls])), **rates(S, thr_on)}
    bl = [c["turns"] if s == "none" else variant(c, s, rngb) for c in ben]
    res["benign"][s] = {"n": len(bl), **rates(prefix_scores(bl), thr_on)}
    a, b = res["attack"][s], res["benign"][s]
    print(f"{s:<10} " + "  ".join(f"{d}: off {a[d]['offline']*100:5.1f} on {a[d]['online']*100:5.1f} [fp {b[d]['online']*100:4.1f}]" for d in DEFS)
          + f"  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open("results3/online_eval.json", "w"), indent=1)

# appending benign turns after a complete attack: an edit that costs the attacker nothing
rnga = np.random.RandomState(11)
for k in (1, 3):
    def pad(c):
        extra = [{"text": " ".join(str(benign_pool[int(rnga.randint(len(benign_pool)))]).split()[:45]), "carries_payload": 0, "channel": "user"}
                 for _ in range(k)]
        return c["turns"] + extra
    res["append"][f"{k}"] = {"attack": rates(prefix_scores([pad(c) for c in atk]), thr_on),
                             "benign": rates(prefix_scores([pad(c) for c in ben]), thr_on)}
    a = res["append"][f"{k}"]["attack"]
    print(f"append {k}: " + "  ".join(f"{d}: off {a[d]['offline']*100:5.1f} on {a[d]['online']*100:5.1f}" for d in DEFS), flush=True)

# sanity: the offline column must reproduce the published table
chk = {r["strategy"]: r for r in fin["rows"]}
dev = max(abs(res["attack"][s][d]["offline"] - chk[s][d]) for s in chk for d in DEFS)
res["max_abs_deviation_from_adaptive_final"] = float(dev)
print(f"largest difference between the offline column and results/adaptive_final.json: {dev:.4f}")
json.dump(res, open("results3/online_eval.json", "w"), indent=1)
print(f"wrote results3/online_eval.json ({time.time()-t0:.0f}s)")
