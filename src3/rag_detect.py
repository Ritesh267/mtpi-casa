"""Detection on live-RAG sessions (report Section 6.13).

Every defence sees the session as it unfolds: one decision after each turn, and a session
counts as flagged if any of those decisions fires ("online"). Two ways of presenting a
RAG turn to the detector are compared:

  prompt   the turn detector scores the whole assembled prompt (question plus the top-5
           retrieved chunks), which is what a filter in front of the model sees
  chunk    the turn detector scores every retrieved chunk separately; the turn's score is
           the highest chunk score and the turn's text, for reassembly, is that chunk.
           This is provenance separation taken to the granularity a RAG pipeline actually
           has: each retrieved chunk is its own unit of untrusted content.

and three conversation layers:

  stateless   any turn score over the threshold (no memory)
  CASA        the conversation layer exactly as trained on MTPI-Bench (no adaptation)
  CASA-RAG    the same features and the same logistic meta-classifier, refitted on live-RAG
              sessions built from the TRAINING partition (the turn detector is unchanged)

Thresholds are reported two ways: carried over unchanged from MTPI-Bench, and
recalibrated to a 5% false-positive budget on benign live-RAG VALIDATION sessions.

    python src3/rag_detect.py --retriever bm25
    python src3/rag_detect.py --retriever dense
    python src3/rag_detect.py --merge
"""
import argparse, hashlib, json, os, sys, time, warnings
sys.path.insert(0, "src"); sys.path.insert(0, "src3"); warnings.filterwarnings("ignore")
import numpy as np, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import casa as K, rag_lib as R

OUT = "results3/rag"; KTOP = 5; BUDGET = 0.05
LIMIT = int(os.environ.get("RAG_LIMIT", "0"))      # smoke-test switch: sessions per file
MODES = list(R.MODES)

class Scorer:
    """Turn-detector scores with a text-hash cache. `collect` queues texts and returns
    zeros, so the feature code can be run once to discover what needs scoring; `flush`
    scores the queue in bulk; calling the object looks scores up."""
    def __init__(self, stack):
        self.stack, self.cache, self.pending = stack, {}, {}
    @staticmethod
    def key(t): return hashlib.md5(t.encode("utf-8", "ignore")).digest()
    def collect(self, texts):
        for t in texts:
            k = self.key(t)
            if k not in self.cache: self.pending[k] = t
        return np.zeros(len(texts))
    def flush(self, bs=512):
        keys = list(self.pending)
        for i in range(0, len(keys), bs):
            ks = keys[i:i + bs]
            p = self.stack.predict_proba([self.pending[k] for k in ks])[:, 1]
            self.cache.update(zip(ks, p.tolist()))
        n = len(keys); self.pending = {}; return n
    def __call__(self, texts):
        return np.array([self.cache[self.key(t)] for t in texts], dtype=float)


def turn_inputs(sess, planted, wiki_text, retr):
    """For each turn: the texts of the top-K retrieved chunks and the assembled prompt."""
    out = []
    for t in sess["turns"]:
        ch = [(wiki_text[j] if kind == "w" else planted[j]["text"]) for kind, j in t["ret"][retr][:KTOP]]
        out.append((ch, R.assemble_prompt(t["question"], ch)))
    return out

def representations(inputs, sc):
    """(p, texts) per representation. Channels are all 'retrieved'."""
    p_prompt = sc([a for _, a in inputs]); t_prompt = [a for _, a in inputs]
    p_chunk, t_chunk = [], []
    for ch, a in inputs:
        s = sc(ch) if ch else np.array([0.0])
        i = int(np.argmax(s)); p_chunk.append(float(s[i])); t_chunk.append(ch[i] if ch else "")
    return {"prompt": (np.asarray(p_prompt), t_prompt), "chunk": (np.asarray(p_chunk), t_chunk)}

def features(p, texts, cfg, scorer, online):
    """Conversation features for the full session (online=False) or for every prefix."""
    n = len(p); ch = ["retrieved"] * n; rows = []
    for i in (range(1, n + 1) if online else [n]):
        W = K.window_scores(texts[:i], ch[:i], scorer)
        rows.append(K.conv_features(p[:i], ch[:i], texts[:i], cfg, W))
    return np.nan_to_num(np.vstack(rows), nan=0.0, posinf=0.0, neginf=0.0)


def run(retr):
    t0 = time.time()
    stack = joblib.load("results/stack.pkl")
    M = joblib.load("results/casa_meta_linear.pkl"); cfg = M["cfg"]
    MA = joblib.load("results/casa_at.pkl")["models"]["CASA-AT (gbm)"]
    thr_at = json.load(open("results/adaptive_final.json"))["thresholds"]["CASA-AT (gbm)"]
    thr_sl = json.load(open("results/casa_main.json"))["results"][0]["thr"]
    paras, _ = R.load_squad()
    wiki_text = [t for p in paras for t, _, _ in R.chunk_doc(p["text"])]
    sc = Scorer(stack)
    cache_path = f"{OUT}/score_cache_{retr}.joblib"          # turn-detector scores, reused across runs
    if os.path.exists(cache_path) and not LIMIT: sc.cache = joblib.load(cache_path)

    def process(split, tag, online):
        """Returns {conv_id: {label, feats: {rep: X}, p: {rep: p}}} for one session file."""
        d = json.load(open(f"{OUT}/sessions_{split}_{tag}.json")); planted = d["planted"]
        if LIMIT: d["sessions"] = dict(list(d["sessions"].items())[:LIMIT])
        ins = {cid: turn_inputs(s, planted, wiki_text, retr) for cid, s in d["sessions"].items()}
        for v in ins.values():
            for ch, a in v:
                sc.collect(ch); sc.collect([a])
        sc.flush()
        reps = {cid: representations(v, sc) for cid, v in ins.items()}
        for cid, r in reps.items():                       # discover the window texts
            for rep, (p, tx) in r.items():
                features(p, tx, cfg, sc.collect, online)
        sc.flush()
        out = {}
        for cid, r in reps.items():
            cov = R.payload_coverage(d["sessions"][cid], cid, planted, retr, KTOP) if planted else []
            out[cid] = {"label": d["sessions"][cid]["label"], "n_turns": len(d["sessions"][cid]["turns"]),
                        "payload_turns": d["sessions"][cid]["payload_turns"],
                        "full": bool(cov and min(cov) >= 0.99), "cov": float(np.mean(cov)) if cov else 0.0,
                        "p": {rep: r[rep][0] for rep in r},
                        "X": {rep: features(r[rep][0], r[rep][1], cfg, sc, online) for rep in r}}
        print(f"  [{retr}] {split:<5} {tag:<18} {len(out)} sessions  ({time.time()-t0:.0f}s)", flush=True)
        return out

    # ---- training features (full sessions), pooled over placement strategies
    Xtr = {"prompt": [], "chunk": []}; Ytr = []
    for mode in MODES:
        tr = process("train", mode, online=False)
        for cid, v in tr.items():
            Ytr.append(v["label"])
            for rep in Xtr: Xtr[rep].append(v["X"][rep][0])
    Ytr = np.array(Ytr)
    refit = {rep: LogisticRegression(max_iter=3000).fit(np.vstack(Xtr[rep]), Ytr) for rep in Xtr}

    # ---- online scores on validation and test
    def score_file(split, tag):
        data = process(split, tag, online=True); res = {}
        for cid, v in data.items():
            s = {"stateless_prompt": v["p"]["prompt"].tolist(), "stateless_chunk": v["p"]["chunk"].tolist()}
            for rep in ("prompt", "chunk"):
                s[f"casa_{rep}_zs"] = M["meta"].predict_proba(v["X"][rep])[:, 1].tolist()
                s[f"casa_{rep}_rag"] = refit[rep].predict_proba(v["X"][rep])[:, 1].tolist()
            s["casa_at_chunk_zs"] = MA["m"].predict_proba(v["X"]["chunk"])[:, 1].tolist()
            res[cid] = {"label": v["label"], "payload_turns": v["payload_turns"], "full": v["full"], "cov": v["cov"],
                        "scores": s, "X": v["X"]}
        return res

    S = {}
    for split in ("val", "test"):
        for tag in MODES + ["clean"]:
            S[(split, tag)] = score_file(split, tag)
    # test attacks that carry a verifiable instruction (see rag_build.py): the three placements with
    # a plainly worded instruction, and the three session files used by the served-model study
    CANARY = [f"{m}_canary" for m in MODES] + ["chunkaware_override", "directive_plain", "directive_override"]
    for tag in CANARY:
        S[("test", tag)] = score_file("test", tag)
    if not LIMIT: joblib.dump(sc.cache, cache_path, compress=3)

    # ---- robustness check on the fitting recipe. The turn detector was fitted on the training
    # partition, so its scores on training fragments are in-sample. "ragcv" fits the same
    # conversation layer on VALIDATION sessions instead (out-of-sample turn scores), with
    # 5-fold cross-fitting grouped by conversation so that the validation scores used for the
    # threshold never come from a model that saw that conversation.
    from sklearn.model_selection import GroupKFold
    for rep in ("prompt", "chunk"):
        keys = [(mode, cid) for mode in MODES for cid in S[("val", mode)]]
        Xf = np.vstack([S[("val", m)][c]["X"][rep][-1] for m, c in keys])
        yf = np.array([S[("val", m)][c]["label"] for m, c in keys]); groups = np.array([c for _, c in keys])
        for tr_i, te_i in GroupKFold(5).split(Xf, yf, groups):
            mdl = LogisticRegression(max_iter=3000).fit(Xf[tr_i], yf[tr_i])
            for i in te_i:
                m, c = keys[i]
                S[("val", m)][c]["scores"][f"casa_{rep}_ragcv"] = mdl.predict_proba(S[("val", m)][c]["X"][rep])[:, 1].tolist()
            held = {groups[i] for i in te_i}
            for c, v in S[("val", "clean")].items():
                if c in held: v["scores"][f"casa_{rep}_ragcv"] = mdl.predict_proba(v["X"][rep])[:, 1].tolist()
        full = LogisticRegression(max_iter=3000).fit(Xf, yf)
        for key, data in S.items():
            if key[0] == "test":
                for v in data.values():
                    v["scores"][f"casa_{rep}_ragcv"] = full.predict_proba(v["X"][rep])[:, 1].tolist()
        refit[f"{rep}_cv"] = full

    DEF = ["stateless_prompt", "stateless_chunk", "casa_prompt_zs", "casa_chunk_zs", "casa_at_chunk_zs",
           "casa_prompt_rag", "casa_chunk_rag", "casa_prompt_ragcv", "casa_chunk_ragcv"]
    zs_thr = {"stateless_prompt": thr_sl, "stateless_chunk": thr_sl, "casa_prompt_zs": M["thr"], "casa_chunk_zs": M["thr"],
              "casa_at_chunk_zs": thr_at, "casa_prompt_rag": None, "casa_chunk_rag": None,
              "casa_prompt_ragcv": None, "casa_chunk_ragcv": None}
    stat = lambda v, d: float(np.max(v["scores"][d]))
    # recalibrated threshold: 95th percentile of the session statistic over benign validation
    # sessions planted by the same procedure as the attacks (pooled over placement strategies)
    recal = {}
    for d in DEF:
        b = [stat(v, d) for mode in MODES for v in S[("val", mode)].values() if v["label"] == 0]
        recal[d] = float(np.quantile(b, 1 - BUDGET))

    def fire_turn(v, d, thr):
        for i, x in enumerate(v["scores"][d]):
            if x >= thr: return i
        return None

    results = {"retriever": retr, "k": KTOP, "budget": BUDGET, "thresholds": {"benchmark": zs_thr, "recalibrated": recal},
               "n_train_sessions": int(len(Ytr)), "by_mode": {}, "clean": {}, "canary": {}}
    for mode in MODES:
        te = S[("test", mode)]; row = {}
        y = np.array([v["label"] for v in te.values()])
        for d in DEF:
            st = np.array([stat(v, d) for v in te.values()])
            e = {"auc": float(roc_auc_score(y, st)), "n_attack": int(y.sum()), "n_benign": int((1 - y).sum())}
            for name, thr in (("benchmark", zs_thr[d]), ("recalibrated", recal[d])):
                if thr is None: continue
                e[name] = {"detection": float((st[y == 1] >= thr).mean()), "fpr_planted_benign": float((st[y == 0] >= thr).mean())}
            # payload fully delivered and not flagged: the delivery success rate under this defence
            full = np.array([v["full"] for v in te.values()])
            e["full_delivery"] = float(full[y == 1].mean())
            e["dsr_under_defence"] = float((full & (st < recal[d]))[y == 1].mean())
            e["detection_when_fully_delivered"] = float((st[(y == 1) & full] >= recal[d]).mean()) if ((y == 1) & full).sum() else None
            # breakdown by number of payload turns, at the recalibrated threshold
            pt = np.array([v["payload_turns"] for v in te.values()])
            e["by_depth"] = {lab: float((st[(y == 1) & m] >= recal[d]).mean()) for lab, m in
                             (("1-2", pt <= 2), ("3-4", (pt >= 3) & (pt <= 4)), ("5-8", pt >= 5)) if ((y == 1) & m).sum() > 0}
            row[d] = e
        results["by_mode"][mode] = row
    for split in ("val", "test"):
        cl = S[(split, "clean")]
        results["clean"][split] = {d: {"n": len(cl),
            **({"benchmark": float(np.mean([stat(v, d) >= zs_thr[d] for v in cl.values()]))} if zs_thr[d] is not None else {}),
            "recalibrated": float(np.mean([stat(v, d) >= recal[d] for v in cl.values()]))} for d in DEF}
    # decisions for the served-model study: turn at which each defence fires
    dec = {"thresholds": recal, "attack": {}, "clean": {}}
    for tag in CANARY:
        ca = S[("test", tag)]
        dec["attack"][tag] = {cid: {d: fire_turn(v, d, recal[d]) for d in DEF} for cid, v in ca.items()}
        results["canary"][tag] = {d: float(np.mean([fire_turn(v, d, recal[d]) is not None for v in ca.values()])) for d in DEF}
    dec["clean"] = {cid: {d: fire_turn(v, d, recal[d]) for d in DEF} for cid, v in S[("test", "clean")].items()}
    sfx = "_smoke" if LIMIT else ""
    # per-session statistics on the test partition, for paired comparisons between defences
    per = {mode: {cid: {"label": v["label"], "payload_turns": v["payload_turns"], "full": v["full"],
                        "stat": {d: stat(v, d) for d in DEF}} for cid, v in S[("test", mode)].items()} for mode in MODES}
    per["clean"] = {cid: {"label": 0, "stat": {d: stat(v, d) for d in DEF}} for cid, v in S[("test", "clean")].items()}
    json.dump({"thresholds": recal, "sessions": per}, open(f"results3/rag_session_stats_{retr}{sfx}.json", "w"))
    json.dump(results, open(f"results3/rag_detection_{retr}{sfx}.json", "w"), indent=1)
    json.dump(dec, open(f"results3/rag_decisions_{retr}{sfx}.json", "w"))
    joblib.dump({"refit": refit, "recal": recal, "cfg": cfg}, f"results3/rag/casa_rag_{retr}{sfx}.pkl")
    print(f"\n[{retr}] online detection at the recalibrated 5% budget (planted-benign FPR in brackets), k={KTOP}")
    for d in DEF:
        print(f"  {d:<18} " + "  ".join(f"{m}: {results['by_mode'][m][d]['recalibrated']['detection']*100:5.1f}% "
              f"[{results['by_mode'][m][d]['recalibrated']['fpr_planted_benign']*100:4.1f}%]" for m in MODES)
              + f"   clean FPR {results['clean']['test'][d]['recalibrated']*100:4.1f}%")
    print(f"[{retr}] done in {time.time()-t0:.0f}s")


def merge():
    out = {r: json.load(open(f"results3/rag_detection_{r}.json")) for r in R.RETRIEVERS if os.path.exists(f"results3/rag_detection_{r}.json")}
    json.dump(out, open("results3/rag_detection.json", "w"), indent=1)
    print("wrote results3/rag_detection.json with", list(out))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", choices=R.RETRIEVERS)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge: merge()
    else: run(a.retriever)
