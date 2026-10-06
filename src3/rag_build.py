"""Build the live-RAG sessions: plant documents, index, retrieve.

For every MTPI-Bench conversation (all three partitions) and every placement strategy,
the conversation is re-delivered as a multi-turn RAG session against a Wikipedia
knowledge base, and the top-10 chunks returned by each retriever are recorded for every
turn. Three kinds of session are written:

  planted   attack and benign conversations, both planted by the same procedure
  clean     the same questions against the unpoisoned knowledge base (nothing planted)
  canary    test attacks only, with a verification instruction in the final fragment,
            used by the served-model study (rag_asr.py)

It also writes results3/rag_retrieval.json: how often planted fragments are retrieved,
how much of each payload reaches the context window, and how well the pipeline retrieves
the paragraph that actually answers the question (so that "the RAG system works" is a
measured statement).

    python src3/rag_build.py                 # everything
    python src3/rag_build.py --splits test   # a subset
"""
import argparse, json, os, sys, time, warnings
sys.path.insert(0, "src"); sys.path.insert(0, "src3"); warnings.filterwarnings("ignore")
import numpy as np
import data as D, rag_lib as R

OUT = "results3/rag"

def build_wiki(paras):
    chunks = []
    for p in paras:
        for text, s, e in R.chunk_doc(p["text"]):
            chunks.append({"text": text, "pid": p["pid"]})
    return chunks

def interval_union(iv):
    iv = sorted(iv); tot = 0; cur = None
    for a, b in iv:
        if cur is None: cur = [a, b]
        elif a <= cur[1]: cur[1] = max(cur[1], b)
        else: tot += cur[1] - cur[0]; cur = [a, b]
    return tot + (cur[1] - cur[0] if cur else 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--modes", nargs="+", default=list(R.MODES))
    ap.add_argument("--served-only", action="store_true",
                    help="build only the session files used by the served-model study (test attacks), leaving other files untouched")
    a = ap.parse_args()
    if a.served_only: a.splits = ["test"]
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    paras, quests = R.load_squad()
    wiki = build_wiki(paras)
    wiki_text = [c["text"] for c in wiki]
    wiki_pid = np.array([c["pid"] for c in wiki])
    print(f"knowledge base: {len(paras)} paragraphs, {len(wiki)} chunks, {len(quests)} questions", flush=True)

    convs, df = D.load(); df = D.split(df); convs = D.conv_split(convs, df)
    by_split = {s: [c for c in convs if c["part"] == s] for s in ("train", "val", "test")}

    dense = R.Dense()
    cache = f"{OUT}/wiki_emb.npy"
    if os.path.exists(cache) and np.load(cache, mmap_mode="r").shape[0] == len(wiki):
        E_wiki = np.load(cache)
    else:
        E_wiki = dense.encode(wiki_text); np.save(cache, E_wiki)
    bm_wiki = R.BM25(wiki_text)
    print(f"indexes ready ({time.time()-t0:.0f}s)", flush=True)

    stats = {"kb": {"paragraphs": len(paras), "chunks": len(wiki), "questions": len(quests),
                    "chunk_words": R.CHUNK_WORDS, "chunk_overlap": R.CHUNK_OVERLAP, "dense_model": R.DENSE_MODEL},
             "gold_recall": {}, "planted": {}}

    for split in a.splits:
        cs = by_split[split]
        plan = R.assign_questions(cs, paras, quests, split)
        order = [c["conv_id"] for c in cs]
        queries, qmap = [], []                       # flat list of every turn's question
        for c in cs:
            for ti, t in enumerate(plan[c["conv_id"]]["turns"]):
                queries.append(t["question"]); qmap.append((c["conv_id"], ti))
        Q = dense.encode(queries)

        # ---- clean: unpoisoned knowledge base
        ret = {"bm25": bm_wiki.search(queries), "dense": dense.search(Q, E_wiki)}
        sessions = {}
        for c in cs:
            pl = plan[c["conv_id"]]
            sessions[c["conv_id"]] = {"label": 0, "orig_label": int(c["label"]), "title": pl["title"],
                "lead_turns": int(c["lead_turns"]), "payload_turns": len(pl["turns"]) - int(c["lead_turns"]),
                "turns": [{"question": t["question"], "gold_pid": t["gold_pid"], "answers": t["answers"], "ret": {}} for t in pl["turns"]]}
        for r in R.RETRIEVERS:
            for (cid, ti), ids in zip(qmap, ret[r]):
                sessions[cid]["turns"][ti]["ret"][r] = [["w", int(i)] for i in ids]
        if not a.served_only:
            json.dump({"split": split, "mode": "clean", "planted": [], "sessions": sessions}, open(f"{OUT}/sessions_{split}_clean.json", "w"))
        if split == "test" and not a.served_only:
            for r in R.RETRIEVERS:
                hit = {k: [] for k in (1, 3, 5, 10)}
                for (cid, ti), ids in zip(qmap, ret[r]):
                    g = sessions[cid]["turns"][ti]["gold_pid"]
                    for k in hit:
                        hit[k].append(float(any(wiki_pid[i] == g for i in ids[:k])))
                stats["gold_recall"][r] = {f"recall@{k}": float(np.mean(v)) for k, v in hit.items()}
                stats["gold_recall"][r]["n_questions"] = len(queries)
                print(f"  [{split}] gold-paragraph recall {r}: " + "  ".join(f"@{k} {np.mean(v)*100:.1f}%" for k, v in hit.items()), flush=True)

        # ---- planted, per placement strategy (and the canary variant for test attacks)
        # (placement, instruction style or None, file tag). Instruction-carrying variants exist only
        # for test attacks: "<mode>_canary" adds a plainly worded instruction to the benchmark payload
        # (used for the delivery statistics); the three files used by the served-model study follow.
        SERVED = [("chunkaware", "override", "chunkaware_override"), ("directive", "plain", "directive_plain"),
                  ("directive", "override", "directive_override")]
        variants = [(m, None, m) for m in a.modes]
        if split == "test": variants += [(m, "plain", f"{m}_canary") for m in a.modes] + SERVED
        if a.served_only: variants = SERVED
        for mode, canary, tag in variants:
            planted, replaced = [], set()
            use = [c for c in cs if (c["label"] == 1 or not canary)]
            for c in use:
                can = R.canary_for(c["conv_id"], canary) if canary else (None, None)
                docs, frags = R.plant_documents(c, plan[c["conv_id"]], mode, paras, canary=can[0])
                for d in docs:
                    if d["replaces"] is not None: replaced.add(d["replaces"])
                    for ch in R.chunk_planted(d):
                        ch.update({"conv_id": c["conv_id"], "label": int(c["label"])}); planted.append(ch)
            fw_of = {(p["conv_id"], p["turn"]): p["frag_words"] for p in planted}
            # One index holds the whole knowledge base plus every planted chunk. Each session is
            # then evaluated as an independent scenario: it can retrieve the documents planted for
            # it, never those planted for another session, and (hijack) it sees its own poisoned
            # paragraphs in place of the originals.
            own_replaced = {}
            for c in use:
                docs, _ = R.plant_documents(c, plan[c["conv_id"]], mode, paras)
                own_replaced[c["conv_id"]] = {d["replaces"] for d in docs if d["replaces"] is not None}
            kb_text = wiki_text + [p["text"] for p in planted]
            nW = len(wiki_text); DEEP = 80
            bm = R.BM25(kb_text)
            E = np.vstack([E_wiki, dense.encode([p["text"] for p in planted])])
            use_ids = {c["conv_id"] for c in use}
            sel = [i for i, (cid, _) in enumerate(qmap) if cid in use_ids]
            raw = {"bm25": bm.search([queries[i] for i in sel], k=DEEP), "dense": dense.search(Q[sel], E, k=DEEP)}
            def view(cid, ids):
                out = []
                for i in ids:
                    if i < nW:
                        if wiki_pid[i] in own_replaced[cid]: continue
                        out.append(["w", int(i)])
                    else:
                        if planted[i - nW]["conv_id"] != cid: continue
                        out.append(["p", int(i - nW)])
                    if len(out) == R.TOPK_MAX: break
                return out
            ret = {r: [view(qmap[i][0], ids) for i, ids in zip(sel, raw[r])] for r in R.RETRIEVERS}
            sessions = {}
            for c in use:
                pl = plan[c["conv_id"]]; lead = int(c["lead_turns"])
                sessions[c["conv_id"]] = {"label": int(c["label"]), "title": pl["title"], "lead_turns": lead,
                    "payload_turns": len(pl["turns"]) - lead, "delivery_orig": c["delivery"],
                    "canary": (R.canary_for(c["conv_id"], canary)[1] if canary else None),
                    "turns": [{"question": t["question"], "gold_pid": t["gold_pid"], "answers": t["answers"],
                               "is_payload_turn": (ti >= lead if mode != "directive" else ti == len(pl["turns"]) - 1),
                               "ret": {}} for ti, t in enumerate(pl["turns"])]}
            for r in R.RETRIEVERS:
                for i, ids in zip(sel, ret[r]):
                    cid, ti = qmap[i]
                    sessions[cid]["turns"][ti]["ret"][r] = ids
            json.dump({"split": split, "mode": tag, "planted": planted, "sessions": sessions},
                      open(f"{OUT}/sessions_{split}_{tag}.json", "w"))

            # ---- retrieval and delivery statistics (test attacks)
            if split == "test":
                st = {}
                for r in R.RETRIEVERS:
                    st[r] = {}
                    for k in (1, 3, 5, 10):
                        frag_hit, frag_cov, sess_all, sess_full, sess_cov, can_hit = [], [], [], [], [], []
                        for cid, s in sessions.items():
                            if s["label"] != 1: continue
                            covs = []
                            for ti, t in enumerate(s["turns"]):
                                if not t["is_payload_turn"]: continue
                                own = [planted[j] for kind, j in t["ret"][r][:k] if kind == "p"
                                       and planted[j]["conv_id"] == cid and planted[j]["turn"] == ti and planted[j]["fi"]]
                                fw = fw_of[(cid, ti)]
                                cov = interval_union([tuple(p["fi"]) for p in own]) / max(fw, 1)
                                frag_hit.append(float(len(own) > 0)); frag_cov.append(cov); covs.append(cov)
                                if canary and ti == len(s["turns"]) - 1:
                                    can_hit.append(float(any(p["has_canary"] for p in own)))
                            sess_all.append(float(all(c > 0 for c in covs))); sess_full.append(float(min(covs) >= 0.99))
                            sess_cov.append(float(np.mean(covs)))
                        st[r][f"k{k}"] = {"fragment_retrieved": float(np.mean(frag_hit)), "fragment_coverage": float(np.mean(frag_cov)),
                                          "session_all_fragments_retrieved": float(np.mean(sess_all)),
                                          "session_full_payload_delivered": float(np.mean(sess_full)),
                                          "session_mean_coverage": float(np.mean(sess_cov)), "n_sessions": len(sess_all),
                                          **({"canary_delivered": float(np.mean(can_hit))} if canary else {})}
                    v = st[r]["k5"]
                    print(f"  [{split}] {tag:<18} {r:<5} k=5: fragment retrieved {v['fragment_retrieved']*100:5.1f}%  "
                          f"coverage {v['fragment_coverage']*100:5.1f}%  full payload {v['session_full_payload_delivered']*100:5.1f}%"
                          + (f"  canary delivered {v['canary_delivered']*100:5.1f}%" if canary else ""), flush=True)
                stats["planted"][tag] = st
                stats["planted"][tag]["n_planted_chunks"] = len(planted)
        print(f"[{split}] done ({time.time()-t0:.0f}s)", flush=True)

    prev = json.load(open("results3/rag_retrieval.json")) if os.path.exists("results3/rag_retrieval.json") else {}
    for k, v in stats.items():
        if not v: continue
        if k == "planted": prev.setdefault("planted", {}).update(v)
        else: prev[k] = v
    json.dump(prev, open("results3/rag_retrieval.json", "w"), indent=1)
    print("wrote results3/rag_retrieval.json")

if __name__ == "__main__":
    main()
