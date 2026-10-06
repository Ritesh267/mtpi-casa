"""Adaptive evaluation of CASA.

The attacker knows the architecture and its published parameters. Each strategy
targets a named component; the combined strategy uses all four. Rewritten
conversations are screened for payload preservation before they count, on the
same principle that a rewrite which no longer carries the instruction is not an
attack.

Status. This is the first adaptive evaluation. It rewrites the test attacks
with seed 42 and writes results/adaptive.json and results/adaptive.csv. It is
NOT the source of any printed table: Table 6.7 of the report is the stateless
and CASA-base columns of results/adaptive_final.json, which src/recalib.py
writes with seed-99 rewrites. The adaptive.csv files shipped in results/ and
results_v1_original/ come from an earlier run of this script against an
earlier CASA model, and adaptive.json is not shipped."""
import sys, json, time, warnings; sys.path.insert(0,"src"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score
import data as D, casa as K, adaptive as A

SEED=42
convs,df=D.load(); df=D.split(df); convs=D.conv_split(convs,df)
stack=joblib.load("results/stack.pkl")
import os
MP="results/casa_meta_linear.pkl" if os.path.exists("results/casa_meta_linear.pkl") else "results/casa_meta.pkl"
M=joblib.load(MP); meta,thr_casa,cfg=M["meta"],M["thr"],M["cfg"]
print("CASA meta-classifier:",MP)
base=json.load(open("results/casa_main.json"))
thr_stateless=[r for r in base["results"] if r["config"].startswith("B1")][0]["thr"]
M2=joblib.load("results/casa_meta_linear.pkl") if __import__("os").path.exists("results/casa_meta_linear.pkl") else None

Cte=[c for c in convs if c["part"]=="test"]
Yte=np.array([c["label"] for c in Cte])
benign_pool=[c["source_prompt"] for c in convs if c["label"]==0 and c["part"]=="train"]
print(f"test conversations {len(Cte)} ({int(Yte.sum())} attack)  benign pool {len(benign_pool)}")

def score_conv(turnlists):
    """Turn scores, window scores and the two decisions, for a list of turn lists."""
    flat=[t["text"] for tl in turnlists for t in tl]
    p=stack.predict_proba(flat)[:,1]
    S={}; i=0
    for j,tl in enumerate(turnlists):
        S[j]=p[i:i+len(tl)]; i+=len(tl)
    fake=[{"conv_id":j,"turns":tl} for j,tl in enumerate(turnlists)]
    scorer=lambda texts: stack.predict_proba(list(texts))[:,1]
    WIN={j:K.window_scores([t["text"] for t in tl],[t["channel"] for t in tl],scorer)
         for j,tl in enumerate(turnlists)}
    X=K.conv_matrix(fake,S,cfg,win_cache=WIN)
    casa=meta.predict_proba(X)[:,1]
    stateless=np.array([float(np.max(S[j])) for j in range(len(turnlists))])
    return stateless,casa,S

STRATS=[("none",{}),("split",{"factor":3}),("pace",{"gap":2}),("pace-wide",{"gap":4}),
        ("dilute",{"ratio":1.5}),("launder",{}),("combined",{"factor":3,"gap":2,"ratio":1.0})]
rng=np.random.RandomState(SEED)
atk=[c for c in Cte if c["label"]==1]
ben=[c for c in Cte if c["label"]==0]
rows=[]
print(f"\n{'strategy':<12} {'kept':>6} {'turns':>6} | {'stateless':>10} | {'CASA':>8} | {'CASA DSR':>9}")
for name,params in STRATS:
    strat = name.replace("-wide","")
    variants=[]; keep=[]
    for c in atk:
        tl = c["turns"] if name=="none" else A.build_variant(c,benign_pool,rng,strat,
                {"factor":params.get("factor",2),"gap":params.get("gap",2),"ratio":params.get("ratio",1.0)})
        orig=[t["text"] for t in c["turns"] if t["carries_payload"]==1]
        new=[t["text"] for t in tl if t["carries_payload"]==1]
        ok = True if name=="none" else A.payload_preserved(orig,new)
        variants.append(tl); keep.append(ok)
    keep=np.array(keep)
    sl,ca,S=score_conv(variants)
    # benign side is not rewritten: the false-positive rate is the deployed one
    slb,cab,_=score_conv([c["turns"] for c in ben])
    det_sl=(sl[keep]>=thr_stateless).mean()
    det_ca=(ca[keep]>=thr_casa).mean()
    # delivery success: did every payload turn reach the model?
    dsr=[]
    for j,(tl,k) in enumerate(zip(variants,keep)):
        if not k: continue
        fire=None
        # CASA runs online: the accumulator decision is taken turn by turn
        tr_,f=K.sprt_trace(S[j],[t["channel"] for t in tl],cfg)
        fire=f if ca[j]>=thr_casa else None
        if ca[j]>=thr_casa and fire is None: fire=len(tl)-1
        dsr.append(K.delivery_success({"turns":tl},fire))
    rows.append({"strategy":name,"kept":int(keep.sum()),"of":len(atk),
                 "mean_turns":float(np.mean([len(v) for v in variants])),
                 "stateless_detect":float(det_sl),"casa_detect":float(det_ca),
                 "casa_dsr":float(np.mean(dsr)) if dsr else 0.0,
                 "stateless_fpr":float((slb>=thr_stateless).mean()),
                 "casa_fpr":float((cab>=thr_casa).mean())})
    print(f"{name:<12} {int(keep.sum()):>6} {np.mean([len(v) for v in variants]):>6.1f} | "
          f"{det_sl:>10.3f} | {det_ca:>8.3f} | {np.mean(dsr) if dsr else 0:>9.3f}")

json.dump(rows,open("results/adaptive.json","w"),indent=2)
pd.DataFrame(rows).to_csv("results/adaptive.csv",index=False)
print(f"\nfalse-positive rate on unmodified benign traffic: "
      f"stateless {rows[0]['stateless_fpr']:.3f}  CASA {rows[0]['casa_fpr']:.3f}")
