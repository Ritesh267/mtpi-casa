"""Baseline grid: six representations x five classifiers, turn level.

This replaces the single Random Forest of the predecessor study. The purpose is
to establish what turn-level detection can achieve at all, so that the gain
attributable to conversation-level aggregation in the next chapter is measured
against the best available stateless detector rather than against a weak one.
"""
import sys, json, time, warnings; sys.path.insert(0,"src"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, joblib
torch.set_num_threads(2)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import reps as R, data as D, encoder as E

SEED=42
convs,df=D.load(); df=D.split(df)
tr=df[df.part=="train"]; va=df[df.part=="val"]; te=df[df.part=="test"]
print(f"turns  train {len(tr)}  val {len(va)}  test {len(te)}")

# ---- R5 encoder, trained once on the training turns only -------------------
enc_path="results/encoder.pt"
t0=time.time()
print("\ntraining the from-scratch transformer encoder (R5) ...")
m,vocab=E.train(tr.text.tolist(),tr.y.values,epochs=6,bs=32,maxlen=192,time_budget=1500)
torch.save({"model":m.state_dict(),"itos":vocab.itos},enc_path)
print(f"encoder done in {time.time()-t0:.0f}s")

class EncRep(R.Rep):
    def __init__(self): super().__init__("R5 transformer")
    def fit(self,texts,y=None): return self
    def transform(self,texts): return E.embed_all(m,vocab,list(texts))

REPS=[R.TfidfWord(), R.TfidfChar(), R.LSA(256), R.Engineered(), EncRep(), R.Hybrid(EncRep())]
DENSE_ONLY={"R3 lsa-256","R4 engineered","R5 transformer","R6 hybrid"}

def clfs(dense):
    out=[("LogisticRegression",LogisticRegression(max_iter=3000,C=2.0,random_state=SEED)),
         ("LinearSVM",CalibratedClassifierCV(LinearSVC(C=0.5,random_state=SEED),cv=3))]
    if dense:
        out+=[("RandomForest",RandomForestClassifier(n_estimators=300,min_samples_leaf=2,
                                                     random_state=SEED,n_jobs=-1)),
              ("GradientBoosting",HistGradientBoostingClassifier(max_iter=300,random_state=SEED)),
              ("MLP",MLPClassifier(hidden_layer_sizes=(256,64),max_iter=400,random_state=SEED,
                                   early_stopping=True))]
    return out

def metrics(y,p,thr=0.5):
    yh=(p>=thr).astype(int)
    tp=int(((yh==1)&(y==1)).sum()); fp=int(((yh==1)&(y==0)).sum())
    fn=int(((yh==0)&(y==1)).sum()); tn=int(((yh==0)&(y==0)).sum())
    return {"auc":float(roc_auc_score(y,p)),"ap":float(average_precision_score(y,p)),
            "acc":float((yh==y).mean()),"recall":float(tp/max(tp+fn,1)),
            "fpr":float(fp/max(fp+tn,1)),"precision":float(tp/max(tp+fp,1)),
            "f1":float(f1_score(y,yh))}

rows=[]; cache={}
for rep in REPS:
    t=time.time(); rep.fit(tr.text.tolist(),tr.y.values)
    Xtr=rep.transform(tr.text.tolist()); Xva=rep.transform(va.text.tolist()); Xte=rep.transform(te.text.tolist())
    cache[rep.name]=(Xtr,Xva,Xte)
    dim=Xtr.shape[1]
    print(f"\n{rep.name}  dim={dim}  ({time.time()-t:.0f}s)")
    for cn,clf in clfs(rep.name in DENSE_ONLY):
        t1=time.time()
        try:
            clf.fit(Xtr,tr.y.values)
            pv=clf.predict_proba(Xva)[:,1]; pt=clf.predict_proba(Xte)[:,1]
        except Exception as ex:
            print(f"   {cn:<20} FAILED {ex}"); continue
        # threshold chosen on validation at the lowest FPR reaching 0.90 recall
        best,thr=None,0.5
        for q in np.linspace(0.02,0.98,97):
            mm=metrics(va.y.values,pv,q)
            if mm["recall"]>=0.90 and (best is None or mm["fpr"]<best["fpr"]): best,thr=mm,q
        if best is None:
            thr=float(np.quantile(pv,1-va.y.mean()))
        mt=metrics(te.y.values,pt,thr)
        rows.append({"rep":rep.name,"dim":int(dim),"clf":cn,"thr":float(thr),
                     "fit_s":round(time.time()-t1,1),**mt})
        print(f"   {cn:<20} AUC {mt['auc']:.4f}  AP {mt['ap']:.4f}  "
              f"recall {mt['recall']:.3f}  FPR {mt['fpr']:.3f}  F1 {mt['f1']:.3f}  "
              f"({time.time()-t1:.0f}s)")
        joblib.dump({"clf":clf,"thr":thr},f"results/clf_{rep.name.split()[0]}_{cn}.pkl")

# the encoder's own classification head
pv=E.predict_proba(m,vocab,va.text.tolist()); pt=E.predict_proba(m,vocab,te.text.tolist())
best,thr=None,0.5
for q in np.linspace(0.02,0.98,97):
    mm=metrics(va.y.values,pv,q)
    if mm["recall"]>=0.90 and (best is None or mm["fpr"]<best["fpr"]): best,thr=mm,q
mt=metrics(te.y.values,pt,thr if best else 0.5)
rows.append({"rep":"R5 transformer","dim":192,"clf":"TransformerHead","thr":float(thr),"fit_s":0.0,**mt})
print(f"\n   {'TransformerHead':<20} AUC {mt['auc']:.4f}  recall {mt['recall']:.3f}  FPR {mt['fpr']:.3f}")

pd.DataFrame(rows).to_csv("results/grid.csv",index=False)
json.dump(rows,open("results/grid.json","w"),indent=2)
b=max(rows,key=lambda r:r["auc"])
print(f"\nbest turn-level detector: {b['rep']} + {b['clf']}  AUC {b['auc']:.4f}  "
      f"recall {b['recall']:.3f} at FPR {b['fpr']:.3f}")
