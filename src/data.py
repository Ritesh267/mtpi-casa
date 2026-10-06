"""Loading and splitting. Splits are by SOURCE PROMPT, not by turn and not by
conversation, so the body turns cut from a prompt cannot appear in two
partitions. Turn-level leakage of this kind is the error that inflated the
predecessor study's pipeline figures, and it is designed out here.

One exception, listed in the README under known limitations: lead-in turns and
retrieval filler are drawn by build_bench.py from the whole benign pool before
the split exists, so some of that benign text is shared between partitions."""
import json, hashlib, numpy as np, pandas as pd

def load(path="data/mtpi_bench.jsonl"):
    convs=[json.loads(l) for l in open(path)]
    rows=[]
    for c in convs:
        for i,t in enumerate(c["turns"]):
            rows.append({"conv_id":c["conv_id"],"turn_idx":i,"text":t["text"],
                         "y":t["carries_payload"],"channel":t["channel"],
                         "conv_label":c["label"],"n_turns":c["n_turns"],
                         "delivery":c["delivery"],"payload_turns":c["payload_turns"],
                         "lead_turns":c["lead_turns"],"src":c["source_prompt"]})
    return convs, pd.DataFrame(rows)

def split(df, seed=42, fracs=(0.6,0.2,0.2)):
    src=df["src"].unique()
    h=np.array([int(hashlib.md5(s.encode()).hexdigest()[:8],16) for s in src])
    r=np.random.RandomState(seed).permutation(len(src))
    src=src[r]
    n=len(src); a=int(fracs[0]*n); b=int((fracs[0]+fracs[1])*n)
    part={}
    for s in src[:a]: part[s]="train"
    for s in src[a:b]: part[s]="val"
    for s in src[b:]: part[s]="test"
    df=df.assign(part=df["src"].map(part))
    return df

def conv_split(convs, df):
    m=df.groupby("conv_id")["part"].first().to_dict()
    for c in convs: c["part"]=m.get(c["conv_id"],"train")
    return convs
