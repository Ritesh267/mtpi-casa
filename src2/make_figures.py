"""Figures for the revision (supervisor comments 1, 4, 5, 6, 7, 9).

Reads results2/summary.json and writes figures2/*.png. Each figure is skipped if its
result section is not present yet, so the script can be re-run as jobs complete.
Palette is colour-blind-safe and consistent across figures.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.edgecolor": "#33383F",
                     "axes.linewidth": 0.8, "axes.grid": True, "grid.color": "#E3E6EA",
                     "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.dpi": 150})
NAVY, TEAL, AMBER, RED, BLUE, GREY = "#1F2A44", "#2A9D8F", "#E9A23B", "#C1432E", "#3B6EA5", "#8A94A6"

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(f"{CODE}/results2/summary.json"))
OUT = f"{CODE}/figures2"; os.makedirs(OUT, exist_ok=True)
SHORT = {"Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B", "Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
         "Qwen2.5-3B-Instruct": "Qwen2.5-3B", "SmolLM2-1.7B-Instruct": "SmolLM2-1.7B",
         "Phi-3.5-mini-instruct": "Phi-3.5-mini"}
def short(n): return SHORT.get(n.split("/")[-1], n.split("/")[-1])
made = []


def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
    made.append(name)


# ---- S1: ASR by model and defence (comment 1) ----
asr = S.get("asr")
if asr and asr.get("models"):
    models = list(asr["models"]); labels = [short(m) for m in models]
    defs = [d for d in ["none", "stateless", "casa_base", "casa_at"] if d in next(iter(asr["models"].values()))["defences"]]
    dname = {"none": "Undefended", "stateless": "Stateless filter", "casa_base": "CASA", "casa_at": "CASA-AT"}
    dcol = {"none": RED, "stateless": AMBER, "casa_base": TEAL, "casa_at": NAVY}
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    x = np.arange(len(models)); w = 0.8 / len(defs)
    for i, d in enumerate(defs):
        vals = [asr["models"][m]["defences"][d]["asr"] * 100 for m in models]
        bbr = np.mean([asr["models"][m]["defences"][d]["benign_block_rate"] for m in models]) * 100
        lab = dname[d] if d == "none" else f"{dname[d]} ({bbr:.1f}% benign blocked)"
        bars = ax.bar(x + (i - (len(defs) - 1) / 2) * w, vals, w, label=lab, color=dcol[d], edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Attack success rate (%)"); ax.set_ylim(0, 105)
    ax.set_title("Attack success rate against served models, by defence", fontweight="bold", color=NAVY)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    save(fig, "figS1_asr_by_model")

    # S1b: DSR vs ASR to show delivery-block vs response effect for CASA
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    d = "casa_base" if "casa_base" in defs else defs[-1]
    dsr = [asr["models"][m]["defences"][d]["dsr"] * 100 for m in models]
    asrv = [asr["models"][m]["defences"][d]["asr"] * 100 for m in models]
    und = [asr["models"][m]["defences"]["none"]["asr"] * 100 for m in models]
    x = np.arange(len(models))
    ax.bar(x - 0.27, und, 0.27, label="Undefended ASR", color=RED, edgecolor="white")
    ax.bar(x, dsr, 0.27, label="CASA delivery rate", color=GREY, edgecolor="white")
    ax.bar(x + 0.27, asrv, 0.27, label="CASA ASR", color=TEAL, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("%"); ax.set_ylim(0, 105)
    ax.set_title("Delivery and attack success with CASA deciding after every turn", fontweight="bold", color=NAVY)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    save(fig, "figS1b_dsr_asr")


# ---- S2: LoRA-DPO separation at scale, every arm of Table 6.17 ----
gpu = S.get("dpo_gpu")
if gpu:
    cells = [("strong", "confident"), ("strong", "boundary"), ("weak", "confident"), ("weak", "boundary")]
    order = [("A0", "A0 unaligned base", GREY), ("A1", "A1 curated (random pairs)", BLUE)]
    for d_, s_ in cells:
        order.append((f"A2_{d_}_{s_}", f"A2 gated · {d_} · {s_}", TEAL))
        order.append((f"A3_{d_}_{s_}", f"A3 generated · {d_} · {s_}", NAVY))
    ys, labs, cols, los, his, nll = [], [], [], [], [], []
    for key, lab, col in order:
        node = gpu.get(key)
        if not node: continue
        if key == "A0":
            m = node["sep"]; lo = hi = m; nll.append(node.get("nll"))
        else:
            m = node["sep_mean"]; lo = node["sep_lo"]; hi = node["sep_hi"]; nll.append(node.get("nll_mean"))
        ys.append(m); labs.append(lab); cols.append(col); los.append(m - lo); his.append(hi - m)
    if ys:
        nseed = len(gpu.get("seeds", [])) or 2
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.6), gridspec_kw={"width_ratios": [2.2, 1]}, sharey=True)
        yy = np.arange(len(ys))[::-1]
        ax.barh(yy, ys, color=cols, edgecolor="white", height=0.66)
        ax.errorbar(ys[1:], yy[1:], xerr=[los[1:], his[1:]], fmt="none", ecolor="#333", elinewidth=1.1, capsize=3)
        ax.set_yticks(yy); ax.set_yticklabels(labs, fontsize=8.8)
        ax.set_xlabel("Safety separation at the base-median threshold\n(percentage points; 95% t-interval over seeds)")
        ax.axvline(0, color="#888", lw=0.8)
        lo_x = min(-5, min(y - l for y, l in zip(ys, los)) - 3); hi_x = max(y + h for y, h in zip(ys, his)) + 3
        ax.set_xlim(max(lo_x, -40), min(hi_x, 130))
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        if all(v is not None for v in nll):
            ax2.barh(yy, nll, color=cols, edgecolor="white", height=0.66)
            ax2.set_xlim(min(nll) - 0.15, max(nll) + 0.1)
            ax2.set_xlabel("Language-modelling loss\non benign prompts")
            for s in ("top", "right"): ax2.spines[s].set_visible(False)
        base = short(gpu.get("base", ""))
        fig.suptitle(f"Detector-supervised LoRA-DPO on {base} ({nseed} seeds per arm; A0 is one run)",
                     fontweight="bold", color=NAVY, fontsize=11, y=1.0)
        save(fig, "figS2_dpo_scale")


# ---- S3: detector feature attribution (comment 4) ----
xai = S.get("xai")
if xai:
    feats = xai["features"][:10][::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    names = [f["feature"] for f in feats]; vals = [f["mean_abs_shap"] for f in feats]
    gcol = {"accumulator": NAVY, "provenance": TEAL, "reassembly": BLUE, "trajectory": AMBER,
            "pooled": GREY, "structure": "#B0B7C3", "imperative": RED}
    cols = [gcol.get(f["group"], GREY) for f in feats]
    a1.barh(range(len(feats)), vals, color=cols, edgecolor="white")
    a1.set_yticks(range(len(feats))); a1.set_yticklabels(names, fontsize=8.5)
    a1.set_xlabel("mean |SHAP|"); a1.set_title("Largest mean |SHAP| (deployed model)", fontweight="bold", color=NAVY, fontsize=10.5)
    for s in ("top", "right"): a1.spines[s].set_visible(False)
    gs = xai["group_share"]
    gk = list(gs); gv = [gs[k] * 100 for k in gk]
    rob_p = f"{CODE}/results3/xai_robustness.json"
    rob = json.load(open(rob_p)) if os.path.exists(rob_p) else None
    refit = None
    if rob:
        for key in ("standardised_refit", "standardized_refit"):
            if key in rob and "group_share" in rob[key]: refit = rob[key]["group_share"]
    yy = np.arange(len(gk))[::-1]
    if refit:
        a2.barh(yy + 0.2, gv, 0.38, color=NAVY, edgecolor="white", label="deployed model")
        a2.barh(yy - 0.2, [refit.get(k, 0) * 100 for k in gk], 0.38, color=AMBER, edgecolor="white", label="refit on standardised features")
        a2.legend(frameon=False, fontsize=7.5, loc="lower right")
    else:
        a2.barh(yy, gv, color=[gcol.get(k, GREY) for k in gk], edgecolor="white")
    a2.set_yticks(yy); a2.set_yticklabels(gk, fontsize=8.5)
    a2.set_xlabel("share of mean |SHAP| (%)")
    a2.set_title("By component", fontweight="bold", color=NAVY, fontsize=10.5)
    for s in ("top", "right"): a2.spines[s].set_visible(False)
    fig.suptitle("Attribution in the deployed linear meta-classifier",
                 fontweight="bold", color=NAVY, fontsize=11.5, y=1.02)
    save(fig, "figS3_xai")


# ---- S4: CASA hardening recovery (comment 5) ----
cf = S.get("casa_fix")
if cf and cf.get("summary"):
    sm = cf["summary"]; conds = ["dilute", "combined"]; cfgs = ["AT", "+P", "+C", "CASA-R"]
    ccol = {"AT": GREY, "+P": BLUE, "+C": AMBER, "CASA-R": NAVY}
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(conds)); w = 0.2
    for i, cg in enumerate(cfgs):
        vals = [sm[cg][c]["mean"] * 100 for c in conds]
        err = [[(sm[cg][c]["mean"] - sm[cg][c]["lo"]) * 100 for c in conds],
               [(sm[cg][c]["hi"] - sm[cg][c]["mean"]) * 100 for c in conds]]
        nseed = sm[cg]["dilute"].get("n", 1)
        if nseed >= 2:
            err = [[(sm[cg][c]["mean"] - sm[cg][c]["lo"]) * 100 for c in conds],
                   [(sm[cg][c]["hi"] - sm[cg][c]["mean"]) * 100 for c in conds]]
            ax.bar(x + (i - 1.5) * w, vals, w, label=cg, color=ccol[cg], edgecolor="white", yerr=err, capsize=2, ecolor="#555")
        else:
            ax.bar(x + (i - 1.5) * w, vals, w, label=cg, color=ccol[cg], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(["Dilution", "Combined"])
    ax.set_ylabel("Detection rate (%)")
    _n = sm["CASA-R"]["dilute"].get("n", 1)
    ax.set_title("Repairing the two surviving attacks " + (f"(mean, 95% CI over {_n} seeds)" if _n >= 2 else "(single seed)"),
                 fontweight="bold", color=NAVY)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    save(fig, "figS4_casa_fix")


# ---- S5: composed vs fragmented vs original (comment 6) ----
comp = S.get("composed")
if comp:
    order = [k for k in ["original", "fragmented", "composed"] if k in comp]
    labs = {"original": "Original\n(single split)", "fragmented": "Deeply\nfragmented", "composed": "Composed\n(escalating)"}
    defs = [d for d in ["stateless", "casa", "casa_at"] if d in comp[order[0]]]
    dname = {"stateless": "Stateless", "casa": "CASA", "casa_at": "CASA-AT"}
    dcol = {"stateless": AMBER, "casa": TEAL, "casa_at": NAVY}
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    x = np.arange(len(order)); w = 0.8 / len(defs)
    for i, d in enumerate(defs):
        vals = [comp[k][d] * 100 for k in order]
        ax.bar(x + (i - (len(defs) - 1) / 2) * w, vals, w, label=dname[d], color=dcol[d], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([labs[k] for k in order])
    ax.set_ylabel("Detection rate (%)")
    ax.set_title("Detection on composed, escalating conversations", fontweight="bold", color=NAVY)
    ax.legend(frameon=False, ncol=len(defs), loc="upper center", bbox_to_anchor=(0.5, -0.14))
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    save(fig, "figS5_composed")


# ---- S6: bootstrap + multi-seed CIs on CASA (comment 7) ----
bs = S.get("bootstrap"); ms = S.get("multiseed_casa")
if bs:
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    metrics = [("stateless_auc", "Stateless AUC", GREY), ("casa_auc", "CASA AUC", TEAL),
               ("stateless_recall", "Stateless recall", "#C9CDD4"), ("casa_recall", "CASA recall", NAVY)]
    ys = np.arange(len(metrics))[::-1]
    for i, (k, lab, col) in enumerate(metrics):
        node = bs[k]; m = node["point"]; lo, hi = node["ci95"]
        ax.barh(ys[i], m, color=col, edgecolor="white", height=0.6)
        ax.errorbar(m, ys[i], xerr=[[m - lo], [hi - m]], fmt="none", ecolor="#333", capsize=3, elinewidth=1.1)
        ax.text(hi + 0.03, ys[i], f"{m:.3f}", va="center", fontsize=8.5, color="#333")
    ax.set_yticks(ys); ax.set_yticklabels([m[1] for m in metrics])
    ax.set_xlim(0, 1.0); ax.set_xlabel("value (95% bootstrap CI)")
    ax.set_title("CASA vs stateless with confidence intervals (seed-42 test set)", fontweight="bold", color=NAVY, fontsize=10.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    save(fig, "figS6_bootstrap")

print("figures written:", made)
