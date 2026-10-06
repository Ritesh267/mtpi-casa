"""CASA-AT: adversarial training of the conversation layer.

The base meta-classifier is fitted only on conversations from the natural
distribution, and Table 6.7 of the report shows what an attacker who paces or
fragments the conversation does to it. The remedy is the standard one for a
learned detector facing a distribution the attacker controls: put the attacker's
distribution into the training set.

Two details decide whether this works.

  Benign traffic is augmented too. Augmenting only attacks teaches the model
  that long, fragmented conversations are attacks, which trades detection
  against a false-positive rate on long benign sessions. Every attacker
  transformation is therefore applied to benign conversations at the same rate.

  Features are scale-free. Counts that grow with conversation length were
  replaced by rates before any augmentation, so pacing cannot move the decision
  by length alone.

Notes on what the script does and stores.

  The two transformed copies of each training conversation are drawn without
  replacement from all six strategies, the combined strategy included.

  The thresholds saved in results/casa_at.pkl, and the detection rates in
  results/adv_train.json, are set on AUGMENTED validation data. The operating
  thresholds behind the printed tables are recalibrated on clean benign
  validation traffic by src/recalib.py and stored in
  results/adaptive_final.json under "thresholds". A user who loads casa_at.pkl
  and applies its stored threshold gets a higher false-positive rate.

  Known limitations (see the README). The benign-transformation control at the
  end of the script creates its random generator afresh for every conversation,
  so every benign conversation receives the same filler sequence. The filler
  the attacker inserts is taken from the benign training prompts, which are
  in-sample for the turn detector.
"""
import sys, json, time, os, warnings; sys.path.insert(0,"src"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import data as D, casa as K, adaptive as A

SEED=42; FPR_BUDGET=0.05
convs,df=D.load(); df=D.split(df); convs=D.conv_split(convs,df)
stack=joblib.load("results/stack.pkl")
cfgM=joblib.load("results/casa_meta_linear.pkl"); cfg=cfgM["cfg"]
scorer=lambda texts: stack.predict_proba(list(texts))[:,1]

Ctr=[c for c in convs if c["part"]=="train"]
Cva=[c for c in convs if c["part"]=="val"]
Cte=[c for c in convs if c["part"]=="test"]
benign_pool=[c["source_prompt"] for c in convs if c["label"]==0 and c["part"]=="train"]

STRATS=["split","pace","pace-wide","dilute","launder","combined"]
PARAMS={"split":{"factor":3},"pace":{"gap":2},"pace-wide":{"gap":4},
        "dilute":{"ratio":1.5},"launder":{},"combined":{"factor":3,"gap":2,"ratio":1.0}}

def variant(c, strat, rng):
    p=PARAMS[strat]
    return A.build_variant(c,benign_pool,rng,strat.replace("-wide",""),
        {"factor":p.get("factor",2),"gap":p.get("gap",2),"ratio":p.get("ratio",1.0)})

def featurise(turnlists):
    """Score turns and windows for a list of turn lists, then build features."""
    flat=[t["text"] for tl in turnlists for t in tl]
    P=scorer(flat); S={}; i=0
    for j,tl in enumerate(turnlists):
        S[j]=P[i:i+len(tl)]; i+=len(tl)
    fake=[{"conv_id":j,"turns":tl} for j,tl in enumerate(turnlists)]
    WIN={j:K.window_scores([t["text"] for t in tl],[t["channel"] for t in tl],scorer)
         for j,tl in enumerate(turnlists)}
    return K.conv_matrix(fake,S,cfg,win_cache=WIN), S

def augment(cs, rng, per_conv=2):
    """Original plus per_conv attacker-transformed copies, for BOTH classes."""
    tl=[]; y=[]
    for c in cs:
        tl.append(c["turns"]); y.append(c["label"])
        picks=rng.choice(STRATS,size=per_conv,replace=False)
        for s in picks:
            v=variant(c,s,rng)
            if c["label"]==1:
                orig=[t["text"] for t in c["turns"] if t["carries_payload"]==1]
                new=[t["text"] for t in v if t["carries_payload"]==1]
                if not A.payload_preserved(orig,new): continue
            tl.append(v); y.append(c["label"])
    return tl,np.array(y)

rng=np.random.RandomState(SEED)
t0=time.time()
print("building the adversarially augmented training set ...")
TLtr,Ytr_a=augment(Ctr,rng,per_conv=2)
print(f"  {len(Ctr)} -> {len(TLtr)} conversations  ({time.time()-t0:.0f}s)")
Xtr_a,_=featurise(TLtr); print(f"  featurised in {time.time()-t0:.0f}s")

TLva,Yva_a=augment(Cva,np.random.RandomState(SEED+1),per_conv=2)
Xva_a,_=featurise(TLva)
Xva0,_=featurise([c["turns"] for c in Cva]); Yva0=np.array([c["label"] for c in Cva])

def pick_thr(y,s,b=FPR_BUDGET):
    bs=np.asarray(s)[np.asarray(y)==0]
    return float(np.quantile(bs,1-b)) if len(bs) else float(np.median(s))

models={}
for name,mk in [("CASA-AT (linear)",lambda: LogisticRegression(max_iter=4000,C=1.0)),
                ("CASA-AT (gbm)",lambda: HistGradientBoostingClassifier(max_iter=400,
                                          learning_rate=0.06,random_state=SEED))]:
    m=mk().fit(Xtr_a,Ytr_a)
    thr=pick_thr(Yva_a,m.predict_proba(Xva_a)[:,1])
    models[name]=(m,thr)
    print(f"  {name}: threshold {thr:.3f}  (val AUC clean {roc_auc_score(Yva0,m.predict_proba(Xva0)[:,1]):.4f}"
          f"  augmented {roc_auc_score(Yva_a,m.predict_proba(Xva_a)[:,1]):.4f})")

base=joblib.load("results/casa_meta_linear.pkl")
Yte=np.array([c["label"] for c in Cte])
atk=[c for c in Cte if c["label"]==1]; ben=[c for c in Cte if c["label"]==0]
Xben,_=featurise([c["turns"] for c in ben])

rows=[]
print(f"\n{'strategy':<12} {'kept':>5} | {'stateless':>9} | {'CASA-base':>10} | "
      f"{'CASA-AT lin':>12} | {'CASA-AT gbm':>12}")
st_base=json.load(open("results/casa_main.json"))
thr_sl=[r for r in st_base["results"] if r["config"].startswith("B1")][0]["thr"]
rng2=np.random.RandomState(99)
for strat in ["none"]+STRATS:
    tls=[]; keep=[]
    for c in atk:
        v=c["turns"] if strat=="none" else variant(c,strat,rng2)
        ok=True
        if strat!="none":
            orig=[t["text"] for t in c["turns"] if t["carries_payload"]==1]
            new=[t["text"] for t in v if t["carries_payload"]==1]
            ok=A.payload_preserved(orig,new)
        tls.append(v); keep.append(ok)
    keep=np.array(keep)
    X,S=featurise(tls)
    sl=np.array([float(np.max(S[j])) for j in range(len(tls))])
    row={"strategy":strat,"kept":int(keep.sum()),
         "stateless":float((sl[keep]>=thr_sl).mean()),
         "CASA-base":float((base["meta"].predict_proba(X)[:,1][keep]>=base["thr"]).mean())}
    for name,(m,thr) in models.items():
        row[name]=float((m.predict_proba(X)[:,1][keep]>=thr).mean())
    rows.append(row)
    print(f"{strat:<12} {int(keep.sum()):>5} | {row['stateless']:>9.3f} | {row['CASA-base']:>10.3f} | "
          f"{row['CASA-AT (linear)']:>12.3f} | {row['CASA-AT (gbm)']:>12.3f}")

fpr={"stateless":float((np.array([float(np.max(featurise([c['turns']])[1][0])) for c in ben])>=thr_sl).mean()),
     "CASA-base":float((base["meta"].predict_proba(Xben)[:,1]>=base["thr"]).mean())}
for name,(m,thr) in models.items():
    fpr[name]=float((m.predict_proba(Xben)[:,1]>=thr).mean())
print("\nfalse-positive rate on unmodified benign traffic:")
for k,v in fpr.items(): print(f"  {k:<18} {v:.3f}")

# and on benign traffic that has itself been transformed the way an attacker
# transforms an attack. This is the control that decides whether adversarial
# training bought robustness or merely taught the model that long conversations
# are attacks, so it is measured per strategy rather than in aggregate.
# Known limitation: RandomState(5) is created afresh for every conversation,
# so every benign conversation receives the same filler sequence.
fpr2={}
for strat in STRATS:
    TLb=[variant(c,strat,np.random.RandomState(5)) for c in ben]
    Xb2,_=featurise(TLb)
    row={"stateless":None,
         "CASA-base":float((base["meta"].predict_proba(Xb2)[:,1]>=base["thr"]).mean())}
    for name,(m,thr) in models.items():
        row[name]=float((m.predict_proba(Xb2)[:,1]>=thr).mean())
    fpr2[strat]=row
print("false-positive rate on benign sessions transformed the same way:")
hdr=["CASA-base"]+list(models)
print("  "+f"{'strategy':<12}"+" ".join(f"{h:>18}" for h in hdr))
for strat,row in fpr2.items():
    print("  "+f"{strat:<12}"+" ".join(f"{row[h]:>18.3f}" for h in hdr))

joblib.dump({"models":{k:{"m":v[0],"thr":v[1]} for k,v in models.items()},"cfg":cfg},
            "results/casa_at.pkl")
json.dump({"rows":rows,"fpr_clean":fpr,"fpr_fragmented_benign":fpr2,
           "n_train_aug":len(TLtr)},open("results/adv_train.json","w"),indent=2)
