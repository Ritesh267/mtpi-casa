"""Conversation-level evaluation: stateless baselines against CASA.

Every configuration is given the same turn-level detector and the same
threshold-selection budget on validation, so the comparison isolates the
aggregation rule.

Caches. The script reuses results/stack.pkl, results/turn_scores.json and
results/window_scores.json when they exist. The two score caches are not tied
to the detector that produced them. To rebuild from nothing, delete all three
files together: deleting only stack.pkl refits the detector and then silently
reuses the scores of the old one.

Two things to keep in mind when reading the output (see the README, known
limitations). The meta-classifier is fitted on training conversations whose
turn scores are in-sample for the turn detector. The ablation is a single run
whose differences are within run-to-run variation, and its "no trajectory"
group also removes p_std, the standard deviation of the turn scores."""
import sys, json, time, warnings; sys.path.insert(0,"src"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, joblib
torch.set_num_threads(2)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import data as D, casa as K, reps as R, encoder as E

SEED=42
convs,df=D.load(); df=D.split(df); convs=D.conv_split(convs,df)

# ---- turn-level detector: the grid's winner, refitted on train ----
from ensemble import StackedTurnDetector
from sklearn.metrics import roc_auc_score as _auc
tr0=df[df.part=="train"]; te0=df[df.part=="test"]
t0=time.time()
import os
if os.path.exists("results/stack.pkl"):
    stack=joblib.load("results/stack.pkl"); print("loaded cached stacked detector")
else:
    stack=StackedTurnDetector().fit(tr0.text.tolist(),tr0.y.values)
p_te=stack.predict_proba(te0.text.tolist())[:,1]
print(f"stacked turn detector: test AUC {_auc(te0.y.values,p_te):.4f}  ({time.time()-t0:.0f}s)")
joblib.dump(stack,"results/stack.pkl")
clf=stack
class _B: pass
best={"rep":"stacked (R2+R1+R3)","clf":"LogReg meta","auc":float(_auc(te0.y.values,p_te))}

_SKIP_ENC=True   # the R5 encoder is not used by CASA; it is loaded only if present
if os.path.exists("results/encoder.pt"):
    ck=torch.load("results/encoder.pt",map_location="cpu",weights_only=False)
    class V:
        def __init__(self,itos): self.itos=itos; self.stoi={w:i for i,w in enumerate(itos)}
    vocab=V(ck["itos"]); enc=E.Encoder(len(vocab.itos),maxlen=192)
    enc.load_state_dict(ck["model"]); enc.eval()

def score_turns(convs_subset):
    texts=[t["text"] for c in convs_subset for t in c["turns"]]
    p=clf.predict_proba(texts)[:,1]
    out={}; i=0
    for c in convs_subset:
        n=len(c["turns"]); out[c["conv_id"]]=p[i:i+n]; i+=n
    return out

t0=time.time()
if os.path.exists("results/turn_scores.json") and os.path.exists("results/stack.pkl"):
    S={k:np.array(v) for k,v in json.load(open("results/turn_scores.json")).items()}
    print("loaded cached turn scores")
else:
    S=score_turns(convs)
print(f"scored {sum(len(v) for v in S.values())} turns in {time.time()-t0:.0f}s")
np.save("results/turn_scores.npy",np.array([1]))
json.dump({k:v.tolist() for k,v in S.items()},open("results/turn_scores.json","w"))

part={c["conv_id"]:c["part"] for c in convs}
Ytr=np.array([c["label"] for c in convs if c["part"]=="train"])
Yva=np.array([c["label"] for c in convs if c["part"]=="val"])
Yte=np.array([c["label"] for c in convs if c["part"]=="test"])
Ctr=[c for c in convs if c["part"]=="train"]
Cva=[c for c in convs if c["part"]=="val"]
Cte=[c for c in convs if c["part"]=="test"]
print(f"conversations  train {len(Ctr)}  val {len(Cva)}  test {len(Cte)}")

def conv_metrics(y,score,thr):
    yh=(score>=thr).astype(int)
    tp=int(((yh==1)&(y==1)).sum()); fp=int(((yh==1)&(y==0)).sum())
    fn=int(((yh==0)&(y==1)).sum()); tn=int(((yh==0)&(y==0)).sum())
    return {"recall":tp/max(tp+fn,1),"fpr":fp/max(fp+tn,1),
            "precision":tp/max(tp+fp,1),"acc":(tp+tn)/len(y),
            "auc":float(roc_auc_score(y,score)),"ap":float(average_precision_score(y,score))}

def pick_thr(y,score,target_fpr=0.05):
    """Operational rule: set the threshold at the (1 - budget) quantile of the
    scores of benign validation traffic. This is how a deployed filter is
    calibrated, and unlike a search over thresholds it does not overfit the few
    false positives a tight budget allows on a validation set of this size."""
    b=np.asarray(score)[np.asarray(y)==0]
    if len(b)==0: return float(np.median(score))
    return float(np.quantile(b,1.0-target_fpr))

# ------------------------------------------------------------ configurations
# window / reassembly scores, computed once per conversation
def scorer(texts): return clf.predict_proba(list(texts))[:,1]
t0=time.time()
if os.path.exists("results/window_scores.json"):
    WIN=json.load(open("results/window_scores.json")); print("loaded cached window scores")
else:
    WIN={}
    for c in convs:
        WIN[c["conv_id"]]=K.window_scores([t["text"] for t in c["turns"]],
                                          [t["channel"] for t in c["turns"]], scorer)
    print(f"reassembly and window scores computed in {time.time()-t0:.0f}s")
json.dump({k:v for k,v in WIN.items()},open("results/window_scores.json","w"))

cfg_default=K.CasaConfig()
cfg,sprt_auc=K.tune_sprt(Cva,S,Yva)
print(f"SPRT tuned on validation: leak {cfg.leak} alarm {cfg.alarm} "
      f"prior_user {cfg.prior_user} prior_retrieved {cfg.prior_retrieved:.2f}  (val AUC {sprt_auc:.4f})")
def agg_scores(cs, how):
    out=[]
    for c in cs:
        p=S[c["conv_id"]]
        if how=="any":  out.append(float(np.max(p)))
        elif how=="mean": out.append(float(np.mean(p)))
        elif how=="top2": out.append(float(np.mean(np.sort(p)[-2:])))
        elif how=="concat": out.append(float(WIN[c["conv_id"]]["concat_all"]))
        elif how=="sprt":
            tr_,_=K.sprt_trace(p,[t["channel"] for t in c["turns"]],cfg)
            out.append(float(np.max(tr_)))
    return np.array(out)

results=[]; FPR_BUDGET=0.05
for how,label in [("any","B1 stateless: any turn over threshold"),
                  ("mean","B2 mean pooling"),
                  ("top2","B3 top-2 pooling"),
                  ("concat","B4 reassembly only (concatenate and re-score)"),
                  ("sprt","B5 sequential accumulator only (tuned)")]:
    sv=agg_scores(Cva,how); st=agg_scores(Cte,how)
    thr=pick_thr(Yva,sv,FPR_BUDGET)
    m=conv_metrics(Yte,st,thr)
    results.append({"config":label,**m,"thr":float(thr)})
    print(f"{label:<44} recall {m['recall']:.3f}  FPR {m['fpr']:.3f}  AUC {m['auc']:.4f}")

# CASA-full: meta-classifier over the 31 conversation features (casa.FEATNAMES)
Xtr=K.conv_matrix(Ctr,S,cfg,win_cache=WIN); Xva=K.conv_matrix(Cva,S,cfg,win_cache=WIN); Xte=K.conv_matrix(Cte,S,cfg,win_cache=WIN)
meta=HistGradientBoostingClassifier(max_iter=400,learning_rate=0.06,random_state=SEED)
meta.fit(Xtr,Ytr)
sv=meta.predict_proba(Xva)[:,1]; st=meta.predict_proba(Xte)[:,1]
thr=pick_thr(Yva,sv,FPR_BUDGET); m=conv_metrics(Yte,st,thr)
results.append({"config":"B6 CASA-full",**m,"thr":float(thr)})
print(f"{'B6 CASA-full':<44} recall {m['recall']:.3f}  "
      f"FPR {m['fpr']:.3f}  AUC {m['auc']:.4f}")
joblib.dump({"meta":meta,"thr":thr,"cfg":cfg},"results/casa_meta.pkl")

# a linear meta-classifier, to show the gain is not from model capacity
lin=LogisticRegression(max_iter=3000).fit(Xtr,Ytr)
svl=lin.predict_proba(Xva)[:,1]; stl=lin.predict_proba(Xte)[:,1]
thl=pick_thr(Yva,svl,FPR_BUDGET); ml=conv_metrics(Yte,stl,thl)
results.append({"config":"B7 CASA-full, linear meta-classifier",**ml,"thr":float(thl)})
joblib.dump({"meta":lin,"thr":thl,"cfg":cfg},"results/casa_meta_linear.pkl")
print(f"{'B7 CASA-full, linear meta-classifier':<44} recall {ml['recall']:.3f}  FPR {ml['fpr']:.3f}  AUC {ml['auc']:.4f}")

# ---- ablations: remove one component group at a time ----
_WIN_NO_RET=[k for k in K.WIN_KEYS if k!="concat_retrieved"]
GROUPS={"no reassembly/windows":[K.FEATNAMES.index(k) for k in _WIN_NO_RET],
        "no sequential accumulator":[K.FEATNAMES.index(k) for k in
            ("sprt_final","sprt_max","sprt_mean","sprt_slope","alarm_pos_norm","alarm_fired")],
        "no trajectory":[K.FEATNAMES.index(k) for k in
            ("p_slope","max_run_norm","high_rate","frac_high","second_half_lift","p_std")],
        "no provenance":[K.FEATNAMES.index(k) for k in
            ("frac_retrieved","imp_retrieved","imp_user","imp_gap","concat_retrieved")],
        "pooled turn scores only":[i for i,n in enumerate(K.FEATNAMES)
            if n not in ("p_mean","p_max","p_min","p_std","p_top2","p_median","log_turns")]}
full_gbm=[r for r in results if r["config"].startswith("B6")][0]
print(f"\nablations (each row removes one component group; baseline is the "
      f"gradient-boosted full model, AUC {full_gbm['auc']:.4f})")
for name,drop in GROUPS.items():
    keep=[i for i in range(len(K.FEATNAMES)) if i not in set(drop)]
    mm=HistGradientBoostingClassifier(max_iter=400,learning_rate=0.06,random_state=SEED).fit(Xtr[:,keep],Ytr)
    sv2=mm.predict_proba(Xva[:,keep])[:,1]; st2=mm.predict_proba(Xte[:,keep])[:,1]
    th2=pick_thr(Yva,sv2,FPR_BUDGET); m2=conv_metrics(Yte,st2,th2)
    results.append({"config":f"ABL {name}",**m2,"thr":float(th2),
                    "delta_auc":float(m2["auc"]-full_gbm["auc"])})
    print(f"  {name:<32} recall {m2['recall']:.3f}  FPR {m2['fpr']:.3f}  "
          f"AUC {m2['auc']:.4f}  dAUC {m2['auc']-full_gbm['auc']:+.4f}")

# ---- operating curve: recall at several false-positive budgets ----
def curve(y_va,s_va,y_te,s_te):
    out=[]
    for b in (0.01,0.02,0.05,0.10,0.20):
        t=pick_thr(y_va,s_va,b); m=conv_metrics(y_te,s_te,t)
        out.append({"budget":b,"recall":m["recall"],"fpr":m["fpr"]})
    return out
sv_any=agg_scores(Cva,"any"); st_any=agg_scores(Cte,"any")
OP={"B1 stateless":curve(Yva,sv_any,Yte,st_any),
    "B6 CASA-full":curve(Yva,meta.predict_proba(Xva)[:,1],Yte,meta.predict_proba(Xte)[:,1]),
    "B7 CASA-full linear":curve(Yva,svl,Yte,stl)}
print("\noperating curve (conversation-level recall at a false-positive budget)")
print(f"{'budget':>7} " + " ".join(f"{k:>22}" for k in OP))
for i,b in enumerate((0.01,0.02,0.05,0.10,0.20)):
    print(f"{b:7.2f} " + " ".join(f"{OP[k][i]['recall']:>21.3f} " for k in OP))

# ---- breakdown by delivery channel and fragmentation depth ----
stB=stl; thB=thl   # the linear meta-classifier, matching the headline row B7
BRK=[]
for key_,sel in [("direct",lambda c:c["delivery"]=="direct"),("rag",lambda c:c["delivery"]=="rag"),
                 ("1-2 payload turns",lambda c:c["payload_turns"]<=2),
                 ("3-4 payload turns",lambda c:3<=c["payload_turns"]<=4),
                 ("6-8 payload turns",lambda c:c["payload_turns"]>=6)]:
    m=np.array([sel(c) for c in Cte])
    if m.sum()<20 or len(set(Yte[m]))<2: continue
    a=conv_metrics(Yte[m],st_any[m],pick_thr(Yva,sv_any,FPR_BUDGET))
    b_=conv_metrics(Yte[m],stB[m],thB)
    BRK.append({"slice":key_,"n":int(m.sum()),"stateless_recall":a["recall"],
                "casa_recall":b_["recall"],"stateless_auc":a["auc"],"casa_auc":b_["auc"]})
    print(f"  {key_:<20} n={int(m.sum()):3d}  stateless recall {a['recall']:.3f} AUC {a['auc']:.4f}"
          f"   CASA recall {b_['recall']:.3f} AUC {b_['auc']:.4f}")

# ---- the fragmentation ceiling ----
src=[c["source_prompt"] for c in Cte]
po=clf.predict_proba(src)[:,1]
from sklearn.metrics import roc_auc_score as _ra
orc=float(_ra(Yte,po))
sl=float(_ra(Yte,st_any)); cs=float(_ra(Yte,stl))
print(f"\nfragmentation ceiling: oracle(unfragmented) AUC {orc:.4f}   "
      f"stateless {sl:.4f}   CASA {cs:.4f}   recovered {(cs-sl)/max(orc-sl,1e-9)*100:.1f}%")

json.dump({"results":results,"operating_curve":OP,"breakdown":BRK,
           "ceiling":{"oracle":orc,"stateless":sl,"casa":cs,
                      "recovered_pct":(cs-sl)/max(orc-sl,1e-9)*100},"sprt_cfg":{"leak":cfg.leak,"alarm":cfg.alarm,
           "prior_user":cfg.prior_user,"prior_retrieved":cfg.prior_retrieved},
           "turn_detector":str(best["rep"])+" + "+str(best["clf"]),
           "turn_auc":float(best["auc"])},open("results/casa_main.json","w"),indent=2)
pd.DataFrame(results).to_csv("results/casa_main.csv",index=False)
