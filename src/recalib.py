"""Recalibrate CASA-AT on clean benign validation traffic, so every
configuration in the adaptive table is compared at the same false-positive
budget rather than at whatever budget its own training distribution implied.

Writes results/adaptive_final.json: the rows behind Tables 6.7 and 6.9 of the
report (test attacks rewritten with seed 99), the false-positive rates on clean
benign test traffic (first row of Table 6.10), and the recalibrated thresholds
under "thresholds". The thresholds are NOT written back into
results/casa_at.pkl, which keeps the values chosen on augmented validation
data: take the operating thresholds from adaptive_final.json."""
import sys, json, time, warnings; sys.path.insert(0,"src"); warnings.filterwarnings("ignore")
import numpy as np, joblib
import data as D, casa as K, adaptive as A

FPR_BUDGET=0.05
convs,df=D.load(); df=D.split(df); convs=D.conv_split(convs,df)
stack=joblib.load("results/stack.pkl")
AT=joblib.load("results/casa_at.pkl"); cfg=AT["cfg"]
base=joblib.load("results/casa_meta_linear.pkl")
scorer=lambda t: stack.predict_proba(list(t))[:,1]

def featurise(tls):
    flat=[t["text"] for tl in tls for t in tl]
    P=scorer(flat); S={}; i=0
    for j,tl in enumerate(tls): S[j]=P[i:i+len(tl)]; i+=len(tl)
    fake=[{"conv_id":j,"turns":tl} for j,tl in enumerate(tls)]
    WIN={j:K.window_scores([t["text"] for t in tl],[t["channel"] for t in tl],scorer)
         for j,tl in enumerate(tls)}
    return K.conv_matrix(fake,S,cfg,win_cache=WIN),S

Cva=[c for c in convs if c["part"]=="val"]; Cte=[c for c in convs if c["part"]=="test"]
benv=[c for c in Cva if c["label"]==0]
Xbv,_=featurise([c["turns"] for c in benv])
thr={}
for name,d in AT["models"].items():
    p=d["m"].predict_proba(Xbv)[:,1]
    thr[name]=float(np.quantile(p,1-FPR_BUDGET))
    print(f"{name}: recalibrated threshold {thr[name]:.4f}")

benign_pool=[c["source_prompt"] for c in convs if c["label"]==0 and c["part"]=="train"]
STRATS=["split","pace","pace-wide","dilute","launder","combined"]
PARAMS={"split":{"factor":3},"pace":{"gap":2},"pace-wide":{"gap":4},
        "dilute":{"ratio":1.5},"launder":{},"combined":{"factor":3,"gap":2,"ratio":1.0}}
atk=[c for c in Cte if c["label"]==1]; ben=[c for c in Cte if c["label"]==0]
st=json.load(open("results/casa_main.json"))
thr_sl=[r for r in st["results"] if r["config"].startswith("B1")][0]["thr"]

Xben,Sben=featurise([c["turns"] for c in ben])
fpr={"stateless":float((np.array([float(np.max(Sben[j])) for j in range(len(ben))])>=thr_sl).mean()),
     "CASA-base":float((base["meta"].predict_proba(Xben)[:,1]>=base["thr"]).mean())}
for n,d in AT["models"].items(): fpr[n]=float((d["m"].predict_proba(Xben)[:,1]>=thr[n]).mean())

rng=np.random.RandomState(99); rows=[]
print(f"\n{'strategy':<12} {'kept':>5} | {'stateless':>9} | {'CASA-base':>10} | "+
      " | ".join(f"{n:>16}" for n in AT["models"]))
for s in ["none"]+STRATS:
    tls=[];keep=[]
    for c in atk:
        p=PARAMS.get(s,{})
        v=c["turns"] if s=="none" else A.build_variant(c,benign_pool,rng,s.replace("-wide",""),
            {"factor":p.get("factor",2),"gap":p.get("gap",2),"ratio":p.get("ratio",1.0)})
        ok=True
        if s!="none":
            o=[t["text"] for t in c["turns"] if t["carries_payload"]==1]
            nw=[t["text"] for t in v if t["carries_payload"]==1]
            ok=A.payload_preserved(o,nw)
        tls.append(v);keep.append(ok)
    keep=np.array(keep); X,S=featurise(tls)
    sl=np.array([float(np.max(S[j])) for j in range(len(tls))])
    r={"strategy":s,"kept":int(keep.sum()),"mean_turns":float(np.mean([len(t) for t in tls])),
       "stateless":float((sl[keep]>=thr_sl).mean()),
       "CASA-base":float((base["meta"].predict_proba(X)[:,1][keep]>=base["thr"]).mean())}
    for n,d in AT["models"].items():
        r[n]=float((d["m"].predict_proba(X)[:,1][keep]>=thr[n]).mean())
    rows.append(r)
    print(f"{s:<12} {int(keep.sum()):>5} | {r['stateless']:>9.3f} | {r['CASA-base']:>10.3f} | "+
          " | ".join(f"{r[n]:>16.3f}" for n in AT["models"]))
print("\nfalse-positive rate on unmodified benign traffic (all recalibrated to a 5% budget)")
for k,v in fpr.items(): print(f"  {k:<20} {v:.3f}")
json.dump({"rows":rows,"fpr":fpr,"thresholds":thr},open("results/adaptive_final.json","w"),indent=2)
