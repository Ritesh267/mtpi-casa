"""The control that justifies symmetric augmentation.

An earlier implementation of Section 4.7 transformed only attack conversations,
because the transformation was keyed on the payload flag and benign
conversations do not carry one. The resulting detector looked excellent under
adaptive attack. This script rebuilds that variant deliberately and measures
what the attack-only version does to the false-positive rate on benign traffic
that has itself been paced or fragmented, which is the measurement that reveals
it had learned conversation length rather than attack structure.
"""
import sys, json, time, warnings; sys.path.insert(0,"src"); warnings.filterwarnings("ignore")
import numpy as np, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
import data as D, casa as K, adaptive as A

SEED=42; FPR_BUDGET=0.05
convs,df=D.load(); df=D.split(df); convs=D.conv_split(convs,df)
stack=joblib.load("results/stack.pkl")
cfg=joblib.load("results/casa_meta_linear.pkl")["cfg"]
scorer=lambda t: stack.predict_proba(list(t))[:,1]
Ctr=[c for c in convs if c["part"]=="train"]
Cva=[c for c in convs if c["part"]=="val"]
Cte=[c for c in convs if c["part"]=="test"]
benign_pool=[c["source_prompt"] for c in convs if c["label"]==0 and c["part"]=="train"]
STRATS=["split","pace","pace-wide","dilute","launder","combined"]
P={"split":{"factor":3},"pace":{"gap":2},"pace-wide":{"gap":4},
   "dilute":{"ratio":1.5},"launder":{},"combined":{"factor":3,"gap":2,"ratio":1.0}}

def variant(c,s,rng):
    p=P[s]
    return A.build_variant(c,benign_pool,rng,s.replace("-wide",""),
        {"factor":p.get("factor",2),"gap":p.get("gap",2),"ratio":p.get("ratio",1.0)})

def featurise(tls):
    flat=[t["text"] for tl in tls for t in tl]
    S0=scorer(flat); S={}; i=0
    for j,tl in enumerate(tls): S[j]=S0[i:i+len(tl)]; i+=len(tl)
    fake=[{"conv_id":j,"turns":tl} for j,tl in enumerate(tls)]
    W={j:K.window_scores([t["text"] for t in tl],[t["channel"] for t in tl],scorer)
       for j,tl in enumerate(tls)}
    return K.conv_matrix(fake,S,cfg,win_cache=W)

def augment(cs,rng,attacks_only):
    tl=[];y=[]
    for c in cs:
        tl.append(c["turns"]); y.append(c["label"])
        if attacks_only and c["label"]==0:
            tl+= [c["turns"],c["turns"]]; y+=[0,0]      # benign triplicated, untransformed
            continue
        for s in rng.choice(STRATS,size=2,replace=False):
            v=variant(c,s,rng)
            if c["label"]==1:
                o=[t["text"] for t in c["turns"] if t["carries_payload"]==1]
                nw=[t["text"] for t in v if t["carries_payload"]==1]
                if not A.payload_preserved(o,nw): continue
            tl.append(v); y.append(c["label"])
    return tl,np.array(y)

t0=time.time()
print("building the attack-only augmented training set ...")
TL,Y=augment(Ctr,np.random.RandomState(SEED),attacks_only=True)
ta=np.mean([len(t) for t,l in zip(TL,Y) if l==1]); tb=np.mean([len(t) for t,l in zip(TL,Y) if l==0])
print(f"  {len(Ctr)} -> {len(TL)} conversations; mean turns attack {ta:.2f}  benign {tb:.2f}")
X=featurise(TL); print(f"  featurised in {time.time()-t0:.0f}s")
m=HistGradientBoostingClassifier(max_iter=400,learning_rate=0.06,random_state=SEED).fit(X,Y)

benv=[c for c in Cva if c["label"]==0]
thr=float(np.quantile(m.predict_proba(featurise([c["turns"] for c in benv]))[:,1],1-FPR_BUDGET))
print(f"  threshold recalibrated on clean benign validation: {thr:.4f}")

atk=[c for c in Cte if c["label"]==1]; ben=[c for c in Cte if c["label"]==0]
rng=np.random.RandomState(99)
det={};
for s in ["none"]+STRATS:
    tls=[c["turns"] if s=="none" else variant(c,s,rng) for c in atk]
    det[s]=float((m.predict_proba(featurise(tls))[:,1]>=thr).mean())
fpr={"none":float((m.predict_proba(featurise([c["turns"] for c in ben]))[:,1]>=thr).mean())}
# Known limitation: RandomState(5) is created afresh for every conversation,
# so every benign conversation receives the same filler sequence (see README).
for s in STRATS:
    tls=[variant(c,s,np.random.RandomState(5)) for c in ben]
    fpr[s]=float((m.predict_proba(featurise(tls))[:,1]>=thr).mean())

print(f"\n{'strategy':<12} {'detection':>10} {'FPR on benign transformed the same way':>42}")
for s in ["none"]+STRATS:
    print(f"{s:<12} {det[s]:>10.3f} {fpr[s]:>42.3f}")
json.dump({"detection":det,"fpr":fpr,"threshold":thr,
           "mean_turns_attack":float(ta),"mean_turns_benign":float(tb),"n_train":len(TL)},
          open("results/asym_control.json","w"),indent=2)
