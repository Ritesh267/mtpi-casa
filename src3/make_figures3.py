"""Figures for the live-RAG study (report Section 6.13). Reads results3/*.json, writes figures3/*.png.
Same palette and style as src2/make_figures.py. Each figure is skipped if its results are absent."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.edgecolor": "#33383F",
                     "axes.linewidth": 0.8, "axes.grid": True, "grid.color": "#E3E6EA",
                     "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.dpi": 150})
NAVY, TEAL, AMBER, RED, BLUE, GREY = "#1F2A44", "#2A9D8F", "#E9A23B", "#C1432E", "#3B6EA5", "#8A94A6"
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{CODE}/figures3"; os.makedirs(OUT, exist_ok=True)
made = []
def load(n):
    p = f"{CODE}/results3/{n}"
    return json.load(open(p)) if os.path.exists(p) else None
def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", facecolor="white"); plt.close(fig); made.append(name)
MODE_NAME = {"targeted": "Query-targeted", "chunkaware": "Chunk-aware", "hijack": "Document hijack"}
MODE_COL = {"targeted": AMBER, "chunkaware": RED, "hijack": BLUE}

# ---- R1: how much of the payload reaches the context window, by placement, retriever and k
ret = load("rag_retrieval.json")
if ret and all(m in ret.get("planted", {}) for m in MODE_NAME):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    ks = [1, 3, 5, 10]
    for ax, r, title in zip(axes, ("bm25", "dense"), ("BM25 retriever", "Dense retriever (MiniLM)")):
        for m in MODE_NAME:
            y = [ret["planted"][m][r][f"k{k}"]["session_full_payload_delivered"] * 100 for k in ks]
            ax.plot(ks, y, marker="o", color=MODE_COL[m], label=MODE_NAME[m], linewidth=2)
        g = [ret["gold_recall"][r][f"recall@{k}"] * 100 for k in ks]
        ax.plot(ks, g, marker="s", color=GREY, linestyle="--", label="Gold paragraph retrieved (clean)", linewidth=1.5)
        ax.set_title(title, fontweight="bold", color=NAVY); ax.set_xlabel("Chunks retrieved per turn (k)"); ax.set_xticks(ks)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Sessions (%)"); axes[0].set_ylim(0, 102)
    axes[1].legend(frameon=False, fontsize=8.5, loc="center right")
    fig.suptitle("Share of attack sessions in which the complete payload reaches the model", fontweight="bold", color=NAVY, y=1.02)
    save(fig, "figR1_rag_delivery")

# ---- R2: online detection by defence and scoring unit (5% budget)
det = {r: load(f"rag_detection_{r}.json") for r in ("bm25", "dense")}
if all(det.values()):
    rows = [("stateless_prompt", "Stateless\nprompt-level", GREY), ("stateless_chunk", "Stateless\nchunk-level", AMBER),
            ("casa_prompt_zs", "CASA\nprompt-level", BLUE), ("casa_chunk_zs", "CASA\nchunk-level", TEAL),
            ("casa_chunk_rag", "CASA-RAG\nchunk-level", NAVY)]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=True)
    for ax, r, title in zip(axes, ("bm25", "dense"), ("BM25 retriever", "Dense retriever")):
        x = np.arange(3); w = 0.16
        for i, (key, lab, col) in enumerate(rows):
            vals = [det[r]["by_mode"][m][key]["recalibrated"]["detection"] * 100 for m in MODE_NAME]
            ax.bar(x + (i - 2) * w, vals, w, color=col, edgecolor="white", linewidth=0.5, label=lab.replace("\n", ", "))
        ax.set_xticks(x); ax.set_xticklabels([MODE_NAME[m] for m in MODE_NAME]); ax.set_title(title, fontweight="bold", color=NAVY)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Attack sessions flagged (%)")
    axes[1].legend(frameon=False, fontsize=8.5, ncol=1, loc="upper right")
    fig.suptitle("Online detection in live-RAG sessions at a 5% false-positive budget", fontweight="bold", color=NAVY, y=1.02)
    save(fig, "figR2_rag_detection")

# ---- R3: how often each served model obeys the planted instruction
asr = load("rag_asr.json")
if asr and asr.get("models"):
    order = ["Qwen2.5-0.5B-Instruct", "SmolLM2-1.7B-Instruct", "Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct", "Phi-3.5-mini-instruct"]
    ms = [m for m in order if m in asr["models"]] + [m for m in asr["models"] if m not in order]
    labs = [m.replace("-Instruct", "").replace("-instruct", "") for m in ms]
    ttl = {"chunkaware_override": "Benchmark payload with\noverride-style instruction", "directive_override": "Override-style instruction alone",
           "directive_plain": "Plainly worded instruction alone"}
    fams = [f for f in ttl if all(f in asr["models"][m]["attacks"] for m in ms)]
    fig, axes = plt.subplots(1, len(fams), figsize=(3.9 * len(fams), 4.1), sharey=True, squeeze=False)
    series = [("undefended", "No defence", RED), ("stateless_chunk", "Stateless chunk filter", AMBER),
              ("casa_chunk_rag", "CASA-RAG (chunk-level)", NAVY)]
    for ax, f in zip(axes[0], fams):
        x = np.arange(len(ms)); w = 0.26
        for i, (key, lab, col) in enumerate(series):
            vals = [(asr["models"][m]["attacks"][f]["undefended"] if key == "undefended" else asr["models"][m]["attacks"][f][key]["asr"]) * 100 for m in ms]
            ax.bar(x + (i - 1) * w, vals, w, color=col, edgecolor="white", linewidth=0.5, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(labs, rotation=20, ha="right", fontsize=8.5); ax.set_title(ttl[f], fontweight="bold", color=NAVY, fontsize=9.5)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[0][0].set_ylabel("Sessions in which the final reply\ncontains the planted code (%)")
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Served models in the live RAG pipeline (dense retriever, five chunks per turn)", fontweight="bold", color=NAVY, y=1.02)
    save(fig, "figR3_rag_asr")

print("figures written:", made)
