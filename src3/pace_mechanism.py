"""Read-only check: is pacing's success explained by the accumulator features?
Reproduces the pace and pace-wide rows of Table 6.7 (recalib.py, seed 99), then scores the same
paced conversations with (a) the deployed linear model, (b) a refit without the accumulator group,
(c) a refit on standardised features, and (d) the deployed model with feature groups swapped
between the clean and paced versions of each conversation. Run from the code folder; pass an output path as the first argument (default results3/pace_mechanism.json).
About 7 minutes on a busy CPU (it re-scores the paced conversations with results/stack.pkl)."""
import sys, json, time, warnings; sys.path.insert(0,"src"); warnings.filterwarnings("ignore")
import numpy as np, joblib
from sklearn.linear_model import LogisticRegression
import data as D, casa as K, adaptive as A
OUT=sys.argv[1] if len(sys.argv)>1 else "results3/pace_mechanism.json"   # run from the code folder
convs,df=D.load(); df=D.split(df); convs=D.conv_split(convs,df)
t0=time.time(); stack=joblib.load("results/stack.pkl"); print("stack loaded",round(time.time()-t0),"s",flush=True)
base=joblib.load("results/casa_meta_linear.pkl"); cfg=base["cfg"]; meta=base["meta"]
AT=joblib.load("results/casa_at.pkl"); print("cfg equal:", AT["cfg"]==cfg, flush=True)
scorer=lambda t: stack.predict_proba(list(t))[:,1]
def featurise(tls):
    flat=[t["text"] for tl in tls for t in tl]
    P=scorer(flat); S={}; i=0
    for j,tl in enumerate(tls): S[j]=P[i:i+len(tl)]; i+=len(tl)
    fake=[{"conv_id":j,"turns":tl} for j,tl in enumerate(tls)]
    WIN={j:K.window_scores([t["text"] for t in tl],[t["channel"] for t in tl],scorer) for j,tl in enumerate(tls)}
    return K.conv_matrix(fake,S,cfg,win_cache=WIN),S
names=list(K.FEATNAMES)
ACC=[names.index(n) for n in ("sprt_final","sprt_max","sprt_mean","sprt_slope","alarm_pos_norm","alarm_fired")]
keep=[i for i in range(len(names)) if i not in ACC]
S0={k:np.array(v) for k,v in json.load(open("results/turn_scores.json")).items()}
WIN0=json.load(open("results/window_scores.json"))
part=lambda p:[c for c in convs if c["part"]==p]
Xtr,Xva=(K.conv_matrix(part(p),S0,cfg,win_cache=WIN0) for p in ("train","val"))
ytr,yva=(np.array([c["label"] for c in part(p)]) for p in ("train","val"))
thr=lambda m,f: float(np.quantile(m.predict_proba(f(Xva[yva==0]))[:,1],0.95))
ident=lambda X:X; noacc_f=lambda X:X[:,keep]
mu,sd=Xtr.mean(0),Xtr.std(0)+1e-9; std_f=lambda X:(X-mu)/sd
m_noacc=LogisticRegression(max_iter=3000).fit(Xtr[:,keep],ytr)
m_std=LogisticRegression(max_iter=3000).fit(std_f(Xtr),ytr)
models={"deployed":(meta,ident,base["thr"]),"refit_no_accumulator":(m_noacc,noacc_f,thr(m_noacc,noacc_f)),
        "refit_standardised":(m_std,std_f,thr(m_std,std_f))}
Cte=part("test"); atk=[c for c in Cte if c["label"]==1]; ben=[c for c in Cte if c["label"]==0]
benign_pool=[c["source_prompt"] for c in convs if c["label"]==0 and c["part"]=="train"]
PARAMS={"split":{"factor":3},"pace":{"gap":2},"pace-wide":{"gap":4}}
rng=np.random.RandomState(99); out={"n_attacks":len(atk)}
variants={}
for s in ["split","pace","pace-wide"]:
    p=PARAMS[s]; tls=[]
    for c in atk:
        tls.append(A.build_variant(c,benign_pool,rng,s.replace("-wide",""),{"factor":p.get("factor",2),"gap":p.get("gap",2),"ratio":p.get("ratio",1.0)}))
    variants[s]=tls
t0=time.time(); Xclean,_=featurise([c["turns"] for c in atk]); print("clean featurised",round(time.time()-t0),"s",flush=True)
Xben,_=featurise([c["turns"] for c in ben])
det=lambda m,f,t,X: float((m.predict_proba(f(X))[:,1]>=t).mean())
out["clean"]={k:{"detection":det(m,f,t,Xclean),"fpr":det(m,f,t,Xben),"thr":t} for k,(m,f,t) in models.items()}
print("clean",out["clean"],flush=True)
G={"accumulator":ACC,"pooled":[names.index(n) for n in ("p_mean","p_max","p_min","p_std","p_top2","p_median")],
   "trajectory":[names.index(n) for n in ("p_slope","max_run_norm","high_rate","frac_high","second_half_lift")],
   "structure":[names.index("log_turns")],
   "provenance":[names.index(n) for n in ("frac_retrieved","imp_retrieved","imp_user","imp_gap","concat_retrieved")],
   "reassembly":[names.index(n) for n in ("concat_all","concat_user","win2_max","win2_mean","win3_max","win3_mean")]}
for s in ["pace","pace-wide"]:
    t0=time.time(); X,_=featurise(variants[s]); print(s,"featurised",round(time.time()-t0),"s",flush=True)
    r={"mean_turns":float(np.mean([len(t) for t in variants[s]]))}
    for k,(m,f,t) in models.items(): r[k]=det(m,f,t,X)
    # deployed model, paced conversation but one group restored to its clean value
    r["deployed_paced_with_clean_group"]={}; r["deployed_clean_with_paced_group"]={}
    for g,ix in G.items():
        Xa=X.copy(); Xa[:,ix]=Xclean[:,ix]; r["deployed_paced_with_clean_group"][g]=det(meta,ident,base["thr"],Xa)
        Xb=Xclean.copy(); Xb[:,ix]=X[:,ix]; r["deployed_clean_with_paced_group"][g]=det(meta,ident,base["thr"],Xb)
    # mean logit contribution change by group in the deployed model (paced minus clean)
    dz=(X-Xclean)*meta.coef_[0]
    r["deployed_mean_logit_shift_by_group"]={g:float(dz[:,ix].sum(1).mean()) for g,ix in G.items()}
    r["deployed_mean_logit_shift_total"]=float(dz.sum(1).mean())
    r["mean_feature_change"]={names[j]:float((X-Xclean)[:,j].mean()) for j in range(len(names))}
    out[s]=r; print(s,json.dumps({k:v for k,v in r.items() if k!="mean_feature_change"},indent=1),flush=True)
json.dump(out,open(OUT,"w"),indent=1); print("ALLDONE")
