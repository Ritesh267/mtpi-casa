"""Precompute, once, the turn at which each defence blocks each conversation.

The defence gates the model's input, so its block decision depends only on the
conversation, not on which model is served. Computing it once here and caching the
result lets the served-model study (measure_asr_hf.py) reuse it across every model
instead of recomputing the detector on the CPU five times over.

For each defence and conversation the earliest alarm turn is recorded (or null if it
never fires). measure_asr_hf.py then replays the delivered prefix to each model.
"""
import sys, json, os, time, warnings; sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
import numpy as np, joblib
import data as D, casa as K

def alarm_turn(defence, stack, turns, meta, thr, cfg):
    """Earliest turn index at which the defence fires, or None."""
    texts = [t["text"] for t in turns]; chans = [t["channel"] for t in turns]
    if defence == "stateless":
        p = stack.predict_proba(texts)[:, 1]
        for i in range(len(turns)):
            if p[i] >= thr:
                return i
        return None
    # conversation-level defences run online, one decision per turn
    p = stack.predict_proba(texts)[:, 1]
    scorer = lambda t: stack.predict_proba(list(t))[:, 1]
    for i in range(len(turns)):
        W = K.window_scores(texts[:i + 1], chans[:i + 1], scorer)
        X = K.conv_matrix([{"conv_id": 0, "turns": turns[:i + 1]}], {0: p[:i + 1]}, cfg, win_cache={0: W})
        if meta.predict_proba(X)[0, 1] >= thr:
            return i
    return None


def main():
    out_path = "results2/defence_decisions.json"
    convs, df = D.load(); df = D.split(df); convs = D.conv_split(convs, df)
    atk = [c for c in convs if c["part"] == "test" and c["label"] == 1]
    ben = [c for c in convs if c["part"] == "test" and c["label"] == 0]

    stack = joblib.load("results/stack.pkl")
    defs = {}
    thr_sl = json.load(open("results/casa_main.json"))["results"][0]["thr"]
    defs["stateless"] = (None, thr_sl, None)
    M = joblib.load("results/casa_meta_linear.pkl"); defs["casa_base"] = (M["meta"], M["thr"], M["cfg"])
    if os.path.exists("results/casa_at.pkl"):
        MA = joblib.load("results/casa_at.pkl")
        thr = json.load(open("results/adaptive_final.json"))["thresholds"]["CASA-AT (gbm)"]
        defs["casa_at"] = (MA["models"]["CASA-AT (gbm)"]["m"], thr, MA["cfg"])

    cache = json.load(open(out_path)) if os.path.exists(out_path) else {"attack": {}, "benign": {}}
    t0 = time.time()
    for grp, cs in [("attack", atk), ("benign", ben)]:
        for dfn, (meta, thr, cfg) in defs.items():
            key = dfn
            done = cache[grp].get(key)
            if done is not None and len(done) == len(cs):
                continue
            fires = []
            for c in cs:
                fires.append(alarm_turn(dfn, stack, c["turns"], meta, thr, cfg))
            cache[grp][key] = fires
            n_block = sum(1 for f in fires if f is not None)
            print(f"  {grp:<7} {dfn:<10} fired on {n_block}/{len(cs)}  ({time.time()-t0:.0f}s)", flush=True)
            json.dump(cache, open(out_path, "w"))
    cache["ids"] = {"attack": [c["conv_id"] for c in atk], "benign": [c["conv_id"] for c in ben]}
    json.dump(cache, open(out_path, "w"))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
