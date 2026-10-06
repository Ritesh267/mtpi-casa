"""Paired comparisons between defences on the live-RAG test sessions.

Detection rates of two defences on the same 237 attack sessions are not independent, so
their difference is tested on the paired outcomes: a bootstrap interval over sessions for
the difference in detection rate, and an exact McNemar test on the discordant pairs.

Writes results3/rag_paired.json. Reads results3/rag_session_stats_{bm25,dense}.json.
"""
import json, os
import numpy as np
from scipy.stats import binomtest

PAIRS = [("casa_chunk_rag", "stateless_chunk", "conversation layer (refitted) vs stateless, both chunk-level"),
         ("casa_chunk_zs", "stateless_chunk", "conversation layer (benchmark-trained) vs stateless, both chunk-level"),
         ("stateless_chunk", "stateless_prompt", "chunk-level vs prompt-level scoring, stateless"),
         ("casa_chunk_zs", "casa_prompt_zs", "chunk-level vs prompt-level scoring, CASA as trained"),
         ("casa_chunk_rag", "casa_prompt_rag", "chunk-level vs prompt-level scoring, CASA refitted")]
rng = np.random.RandomState(42)
out = {}
for retr in ("bm25", "dense"):
    p = f"results3/rag_session_stats_{retr}.json"
    if not os.path.exists(p): continue
    d = json.load(open(p)); thr = d["thresholds"]; out[retr] = {}
    for mode in ("targeted", "chunkaware", "hijack"):
        att = [v for v in d["sessions"][mode].values() if v["label"] == 1]
        deep = np.array([v["payload_turns"] >= 5 for v in att])
        rows = {}
        for a, b, what in PAIRS:
            fa = np.array([v["stat"][a] >= thr[a] for v in att]); fb = np.array([v["stat"][b] >= thr[b] for v in att])
            def summarise(mask):
                x, y = fa[mask], fb[mask]; n = len(x)
                diffs = []
                for _ in range(5000):
                    i = rng.randint(0, n, n); diffs.append(x[i].mean() - y[i].mean())
                only_a, only_b = int((x & ~y).sum()), int((~x & y).sum())
                pval = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
                return {"n": int(n), "a": float(x.mean()), "b": float(y.mean()), "diff": float(x.mean() - y.mean()),
                        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                        "only_a": only_a, "only_b": only_b, "mcnemar_p": float(pval)}
            rows[f"{a}__vs__{b}"] = {"what": what, "all": summarise(np.ones(len(att), bool)), "five_or_more_payload_turns": summarise(deep)}
        out[retr][mode] = rows
        r = rows["casa_chunk_rag__vs__stateless_chunk"]["all"]; q = rows["stateless_chunk__vs__stateless_prompt"]["all"]
        print(f"[{retr}] {mode:<10} CASA-RAG(chunk) - stateless(chunk): {r['diff']*100:+5.1f} pp  CI [{r['ci95'][0]*100:+.1f}, {r['ci95'][1]*100:+.1f}]  p={r['mcnemar_p']:.3f}"
              f"   | chunk - prompt (stateless): {q['diff']*100:+5.1f} pp CI [{q['ci95'][0]*100:+.1f}, {q['ci95'][1]*100:+.1f}] p={q['mcnemar_p']:.4f}")
json.dump(out, open("results3/rag_paired.json", "w"), indent=1)
print("wrote results3/rag_paired.json")
