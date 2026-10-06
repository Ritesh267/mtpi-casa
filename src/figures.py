"""Figures 3.1, 4.1 and 6.1 to 6.8 of the report. Run from the repository root:

    python src/figures.py

Reads data/mtpi_bench.jsonl and the result files in results/, writes figures/*.png.

Note. results/ holds the later re-run of the pipeline, not the original run
behind the printed tables (results_v1_original/), and the gradient-boosted
rows differ between the two. Running this script therefore redraws every
figure from the re-run. The ablation, operating-curve, adaptive and
augmentation-control figures change visibly and will no longer match the
images shipped in figures/, which are the ones used in the report (the shipped
ceiling image also predates the "(main partition)" wording of its annotation).
The recovered share printed on the ceiling
figure is for the main partition (seed 42), the highest of the five partitions
on which the comparison was run.
"""
import json, warnings, numpy as np, pandas as pd, matplotlib; warnings.filterwarnings("ignore")
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.grid":True,"grid.alpha":.25,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":200})
C={"bad":"#C0392B","ok":"#2E7D8F","warn":"#D98C24","n":"#6B7280","good":"#3E7A52","alt":"#6B4C9A"}
R=lambda f: json.load(open(f"results/{f}")); O="figures"
import os; os.makedirs(O,exist_ok=True)

# F1 benchmark length distributions
import data as D
convs=[json.loads(l) for l in open("data/mtpi_bench.jsonl")]
la=[len(c["source_prompt"].split()) for c in convs if c["label"]==1]
lb=[len(c["source_prompt"].split()) for c in convs if c["label"]==0]
f,ax=plt.subplots(1,2,figsize=(7.6,2.9))
ax[0].hist(np.log10(np.array(lb)+1),bins=30,alpha=.65,color=C["good"],label=f"benign (median {int(np.median(lb))})")
ax[0].hist(np.log10(np.array(la)+1),bins=30,alpha=.65,color=C["bad"],label=f"attack (median {int(np.median(la))})")
ax[0].set_xlabel("log10(source prompt length in tokens)"); ax[0].set_ylabel("prompts")
ax[0].legend(fontsize=7.5,frameon=False); ax[0].set_title("MTPI-Bench (length-stratified)",fontsize=9.5,fontweight="bold")
ax[1].bar([0,1],[np.mean(lb),np.mean(la)],color=[C["good"],C["bad"]],width=.55)
ax[1].bar([2.4,3.4],[223.2,344.3],color=[C["good"],C["bad"]],width=.55,alpha=.55)
ax[1].set_xticks([0,1,2.4,3.4]); ax[1].set_xticklabels(["benign","attack","benign","attack"],fontsize=8)
ax[1].set_ylabel("mean tokens")
ax[1].text(0.5,-0.22,"after stratification",ha="center",transform=ax[1].get_xaxis_transform(),fontsize=8)
ax[1].text(2.9,-0.22,"source pool",ha="center",transform=ax[1].get_xaxis_transform(),fontsize=8)
ax[1].set_title("Length signal removed by stratification",fontsize=9.5,fontweight="bold")
f.tight_layout(); f.savefig(f"{O}/fig_bench.png"); plt.close(f)

# F2 grid heatmap
g=pd.read_csv("results/grid.csv"); g=g[g.clf!="TransformerHead"]
piv=g.pivot_table(index="rep",columns="clf",values="auc")
order=["LogisticRegression","LinearSVM","RandomForest","GradientBoosting","MLP"]
piv=piv[[c for c in order if c in piv.columns]]
f,ax=plt.subplots(figsize=(6.2,3.0))
im=ax.imshow(piv.values,cmap="YlGnBu",vmin=0.65,vmax=0.90,aspect="auto")
ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns,rotation=22,ha="right",fontsize=8)
ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index,fontsize=8)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v=piv.values[i,j]
        if not np.isnan(v):
            ax.text(j,i,f"{v:.3f}",ha="center",va="center",fontsize=7.5,
                    color="white" if v>0.83 else "black")
ax.grid(False); f.colorbar(im,label="turn-level AUC",fraction=.035)
ax.set_title("Turn-level detection: representation x classifier",fontsize=9.5,fontweight="bold")
f.tight_layout(); f.savefig(f"{O}/fig_grid.png"); plt.close(f)

# F3 fragmentation ceiling
cm=R("casa_main.json"); ce=cm["ceiling"]
f,ax=plt.subplots(figsize=(4.6,3.0))
v=[ce["stateless"],ce["casa"],ce["oracle"]]
b=ax.bar([0,1,2],v,color=[C["n"],C["ok"],C["good"]],width=.6)
for r,x in zip(b,v): ax.text(r.get_x()+r.get_width()/2,x+0.002,f"{x:.4f}",ha="center",fontsize=8.5,fontweight="bold")
ax.set_xticks([0,1,2]); ax.set_xticklabels(["stateless\n(per turn)","CASA","oracle\n(unfragmented)"],fontsize=8)
ax.set_ylim(0.84,0.94); ax.set_ylabel("conversation-level AUC")
ax.annotate("",xy=(2,ce["oracle"]-0.001),xytext=(0,ce["stateless"]+0.001),
            arrowprops=dict(arrowstyle="<->",color=C["warn"],lw=1.2))
ax.text(1,0.928,f"CASA recovers {ce['recovered_pct']:.0f}% of the\nfragmentation gap (main partition)",
        ha="center",fontsize=8,color=C["warn"],fontweight="bold")
ax.set_title("What fragmentation costs, and what CASA recovers",fontsize=9.5,fontweight="bold")
f.tight_layout(); f.savefig(f"{O}/fig_ceiling.png"); plt.close(f)

# F4 operating curve
op=cm["operating_curve"]
f,ax=plt.subplots(figsize=(4.9,3.1))
sty={"B1 stateless":(C["n"],"o--"),"B6 CASA-full":(C["ok"],"s-"),"B7 CASA-full linear":(C["alt"],"^-")}
for k,rows in op.items():
    col,m=sty.get(k,(C["bad"],"d-"))
    ax.plot([r["budget"]*100 for r in rows],[r["recall"]*100 for r in rows],m,color=col,lw=1.8,ms=4.5,label=k)
ax.set_xscale("log"); ax.set_xticks([1,2,5,10,20]); ax.set_xticklabels(["1","2","5","10","20"])
ax.set_xlabel("false-positive budget (%)"); ax.set_ylabel("conversation-level recall (%)")
ax.legend(fontsize=7.5,frameon=False,loc="upper left")
ax.set_title("Operating curve",fontsize=9.5,fontweight="bold")
f.tight_layout(); f.savefig(f"{O}/fig_opcurve.png"); plt.close(f)

# F5 breakdown
brk=cm["breakdown"]
f,ax=plt.subplots(figsize=(6.0,3.0))
x=np.arange(len(brk)); w=.36
ax.bar(x-w/2,[b["stateless_recall"]*100 for b in brk],w,color=C["n"],label="stateless")
ax.bar(x+w/2,[b["casa_recall"]*100 for b in brk],w,color=C["ok"],label="CASA (linear meta)")
for i,b in enumerate(brk):
    d=(b["casa_recall"]-b["stateless_recall"])*100
    ax.text(i,max(b["casa_recall"],b["stateless_recall"])*100+1.5,f"{d:+.1f}",ha="center",
            fontsize=7.5,color=C["good"] if d>0 else C["bad"],fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([b["slice"].replace(" payload turns","\npayload turns") for b in brk],fontsize=7.5)
ax.set_ylabel("recall (%)"); ax.legend(fontsize=7.5,frameon=False)
ax.set_title("Where the conversation layer helps",fontsize=9.5,fontweight="bold")
f.tight_layout(); f.savefig(f"{O}/fig_breakdown.png"); plt.close(f)

# F6 ablations
res=cm["results"]; abl=[r for r in res if r["config"].startswith("ABL")]
full=[r for r in res if r["config"].startswith("B6")][0]
f,ax=plt.subplots(figsize=(5.4,2.9))
names=["full CASA (GBM)"]+[a["config"].replace("ABL ","") for a in abl]
vals=[full["auc"]]+[a["auc"] for a in abl]
cols=[C["ok"]]+[(C["good"] if a["auc"]>full["auc"] else C["warn"]) for a in abl]
b=ax.barh(range(len(vals)),vals,color=cols,height=.6)
for r,x in zip(b,vals): ax.text(x+0.0006,r.get_y()+r.get_height()/2,f"{x:.4f}",va="center",fontsize=7.5)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names,fontsize=8); ax.invert_yaxis()
ax.set_xlim(0.880,0.910); ax.set_xlabel("conversation-level AUC")
ax.set_title("Component ablation",fontsize=9.5,fontweight="bold")
f.tight_layout(); f.savefig(f"{O}/fig_ablation.png"); plt.close(f)

# F7 adaptive
ad=R("adaptive_final.json"); rows=ad["rows"]
f,ax=plt.subplots(figsize=(7.2,3.2))
x=np.arange(len(rows)); w=.27
ax.bar(x-w,[r["stateless"]*100 for r in rows],w,color=C["n"],label="stateless")
ax.bar(x,[r["CASA-base"]*100 for r in rows],w,color=C["warn"],label="CASA (natural training)")
ax.bar(x+w,[r["CASA-AT (gbm)"]*100 for r in rows],w,color=C["ok"],label="CASA-AT (adversarially trained)")
ax.set_xticks(x); ax.set_xticklabels([r["strategy"] for r in rows],fontsize=8)
ax.set_ylabel("detection rate (%)"); ax.set_ylim(0,100)
ax.legend(fontsize=7.5,frameon=False,ncol=3,loc="upper center")
ax.set_title("Detection under an attacker who adapts to the defence",fontsize=10,fontweight="bold")
f.tight_layout(); f.savefig(f"{O}/fig_adaptive.png"); plt.close(f)

# F8 DPO
try:
    dp=R("dpo.json"); rws=dp["rows"]
    f,ax=plt.subplots(1,2,figsize=(7.4,3.0))
    nm=[r["arm"].split(" ",1)[0] for r in rws]
    ax[0].bar(np.arange(len(rws))-0.2,[r["refuse_attack"] for r in rws],0.4,color=C["bad"],label="attacks")
    ax[0].bar(np.arange(len(rws))+0.2,[r["refuse_benign"] for r in rws],0.4,color=C["good"],label="benign")
    ax[0].set_xticks(range(len(rws))); ax[0].set_xticklabels(nm,fontsize=8)
    ax[0].set_ylabel("refusal rate (%)"); ax[0].legend(fontsize=7.5,frameon=False)
    ax[0].set_title("Refusal behaviour by supervision source",fontsize=9.5,fontweight="bold")
    g_=[r["gap"] for r in rws]
    b=ax[1].bar(range(len(rws)),g_,color=[C["n"]]+[C["ok"],C["warn"],C["alt"]][:len(rws)-1],width=.6)
    for r,x in zip(b,g_): ax[1].text(r.get_x()+r.get_width()/2,x+(0.6 if x>=0 else -1.8),f"{x:+.1f}",ha="center",fontsize=8,fontweight="bold")
    ax[1].set_xticks(range(len(rws))); ax[1].set_xticklabels(nm,fontsize=8)
    ax[1].set_ylabel("separation (pp)"); ax[1].axhline(0,color="k",lw=.8)
    ax[1].set_title("Safety separation (attack - benign refusal)",fontsize=9.5,fontweight="bold")
    f.tight_layout(); f.savefig(f"{O}/fig_dpo.png"); plt.close(f)
except Exception as e: print("dpo figure skipped:",e)

# F10 the augmentation control
try:
    az=R("asym_control.json"); sy=R("adv_train.json"); af=R("adaptive_final.json")
    ks=["none","split","pace","pace-wide","dilute","launder","combined"]
    f,ax=plt.subplots(1,2,figsize=(7.8,3.1),sharey=True)
    x=np.arange(len(ks)); w=.38
    symd={r["strategy"]:r["CASA-AT (gbm)"] for r in af["rows"]}
    symf=sy["fpr_fragmented_benign"]
    ax[0].bar(x-w/2,[az["detection"][k]*100 for k in ks],w,color=C["ok"],label="detection")
    ax[0].bar(x+w/2,[az["fpr"][k]*100 for k in ks],w,color=C["bad"],label="FPR on benign\ntransformed the same way")
    ax[0].set_title("Attack-only augmentation",fontsize=9.5,fontweight="bold")
    ax[1].bar(x-w/2,[symd[k]*100 for k in ks],w,color=C["ok"],label="detection")
    ax[1].bar(x+w/2,[(sy["fpr_clean"]["CASA-AT (gbm)"] if k=="none" else symf[k]["CASA-AT (gbm)"])*100
                     for k in ks],w,color=C["bad"])
    ax[1].set_title("Symmetric augmentation",fontsize=9.5,fontweight="bold")
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(ks,fontsize=7.5,rotation=18,ha="right"); a.set_ylim(0,100)
    ax[0].set_ylabel("%"); ax[0].legend(fontsize=7,frameon=False,loc="upper left")
    f.suptitle("Why the benign-transformation control decides the result",fontsize=10,fontweight="bold")
    f.tight_layout(); f.savefig(f"{O}/fig_augcontrol.png"); plt.close(f)
except Exception as e: print("aug control figure skipped:",e)

# F9 architecture
f,ax=plt.subplots(figsize=(7.4,3.6)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,6)
def box(x,y,w,h,t,c,fs=8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.06",lw=1.2,
                                edgecolor=c,facecolor=c+"22"))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,color="black")
def arr(x1,y1,x2,y2,c="#555"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=11,color=c,lw=1.1))
box(0.2,4.2,1.7,1.0,"user turn",C["n"]); box(0.2,2.6,1.7,1.0,"retrieved\ncontent",C["alt"])
box(2.4,3.4,1.9,1.8,"turn-level\ndetector\n(stacked)",C["ok"])
box(4.8,4.3,2.1,0.9,"sequential\naccumulator",C["warn"])
box(4.8,3.25,2.1,0.9,"trajectory\nfeatures",C["warn"])
box(4.8,2.2,2.1,0.9,"provenance\nseparation",C["warn"])
box(7.4,3.25,1.4,1.9,"CASA\ndecision",C["bad"])
box(4.8,0.7,4.0,0.9,"internal layer: LoRA-DPO adapter trained on detector-generated pairs",C["good"],7.5)
box(2.4,0.7,1.9,0.9,"local LLM",C["n"])
for y in (4.7,3.1): arr(1.9,y,2.4,4.3 if y>4 else 4.3)
arr(4.3,4.3,4.8,4.75); arr(4.3,4.3,4.8,3.7); arr(4.3,4.3,4.8,2.65)
for y in (4.75,3.7,2.65): arr(6.9,y,7.4,4.2)
arr(8.1,3.25,8.1,1.6,C["bad"]); arr(4.8,1.15,4.3,1.15)
ax.text(8.25,2.4,"verdict,\nmargin",fontsize=7,color=C["bad"])
ax.text(0.2,5.5,"Runtime path",fontsize=9,fontweight="bold")
ax.text(4.8,0.25,"offline loop",fontsize=7.5,color=C["good"])
f.tight_layout(); f.savefig(f"{O}/fig_arch.png"); plt.close(f)

print("figures:", sorted(os.listdir(O)))
