"""R5: a transformer encoder trained from scratch on the benchmark corpus.

No pretrained weights are used in the core pipeline (src/). That is a deliberate
constraint of the local-deployment setting this work targets: the defender is
assumed to have the corpus and a consumer machine, not a model-hub download
budget or a vendor account. The later experiments (scripts/ and parts of src2/
and src3/) do use pretrained open-weight models. The architecture is small
enough to train on CPU. The functions below take a `device` argument, so the
same model can be trained on a GPU, but no GPU script for the encoder is
shipped and no reported number comes from one.
"""
from __future__ import annotations
import math, re, time, json, os
import numpy as np, torch, torch.nn as nn
from collections import Counter

torch.manual_seed(42); np.random.seed(42)
_TOK=re.compile(r"[a-z0-9']+|[^\sa-z0-9]")

def tokenize(t:str): return _TOK.findall(str(t).lower())

class Vocab:
    def __init__(self,texts,size=16000,min_count=2):
        c=Counter(w for t in texts for w in tokenize(t))
        keep=[w for w,n in c.most_common(size-4) if n>=min_count]
        self.itos=["<pad>","<unk>","<cls>","<sep>"]+keep
        self.stoi={w:i for i,w in enumerate(self.itos)}
    def __len__(self): return len(self.itos)
    def encode(self,t,maxlen=192):
        ids=[2]+[self.stoi.get(w,1) for w in tokenize(t)][:maxlen-1]
        return ids+[0]*(maxlen-len(ids))

class Encoder(nn.Module):
    def __init__(self,vocab,d=192,layers=4,heads=4,ff=384,maxlen=192,drop=0.1):
        super().__init__()
        self.emb=nn.Embedding(vocab,d,padding_idx=0)
        self.pos=nn.Parameter(torch.zeros(1,maxlen,d)); nn.init.normal_(self.pos,std=0.02)
        L=nn.TransformerEncoderLayer(d,heads,ff,dropout=drop,batch_first=True,
                                     norm_first=True,activation="gelu")
        self.enc=nn.TransformerEncoder(L,layers)
        self.norm=nn.LayerNorm(d); self.drop=nn.Dropout(drop)
        self.head=nn.Linear(d,2)
        self.d=d
    def embed(self,x):
        mask=(x==0)
        h=self.emb(x)*math.sqrt(self.d)+self.pos[:,:x.size(1)]
        h=self.enc(self.drop(h),src_key_padding_mask=mask)
        return self.norm(h[:,0])                 # <cls> pooled
    def forward(self,x):
        z=self.embed(x); return self.head(z),z

def train(texts,labels,vocab=None,epochs=6,bs=32,lr=3e-4,maxlen=192,
          device="cpu",time_budget=None,log=print):
    vocab=vocab or Vocab(texts)
    X=torch.tensor([vocab.encode(t,maxlen) for t in texts],dtype=torch.long)
    y=torch.tensor(np.asarray(labels),dtype=torch.long)
    m=Encoder(len(vocab),maxlen=maxlen).to(device)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=0.01)
    steps=max(1,epochs*math.ceil(len(X)/bs))
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,lr,total_steps=steps,pct_start=0.1)
    lossf=nn.CrossEntropyLoss(label_smoothing=0.05)
    t0=time.time(); done=0
    for ep in range(epochs):
        m.train(); perm=torch.randperm(len(X)); tot=0.0; nb=0; corr=0
        for i in range(0,len(X),bs):
            idx=perm[i:i+bs]; xb,yb=X[idx].to(device),y[idx].to(device)
            logit,_=m(xb); loss=lossf(logit,yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(),1.0)
            opt.step(); sch.step()
            tot+=loss.item(); nb+=1; done+=1
            corr+=(logit.argmax(1)==yb).sum().item()
            if time_budget and time.time()-t0>time_budget:
                log(f"  [budget reached at epoch {ep+1}, step {nb}]"); return m,vocab
        log(f"  epoch {ep+1}/{epochs}  loss {tot/nb:.4f}  train acc {corr/len(X):.4f}  "
            f"({time.time()-t0:.0f}s)")
    return m,vocab

@torch.no_grad()
def embed_all(m,vocab,texts,maxlen=192,bs=64,device="cpu"):
    m.eval(); out=[]
    for i in range(0,len(texts),bs):
        xb=torch.tensor([vocab.encode(t,maxlen) for t in texts[i:i+bs]],dtype=torch.long).to(device)
        out.append(m.embed(xb).cpu().numpy())
    return np.vstack(out)

@torch.no_grad()
def predict_proba(m,vocab,texts,maxlen=192,bs=64,device="cpu"):
    m.eval(); out=[]
    for i in range(0,len(texts),bs):
        xb=torch.tensor([vocab.encode(t,maxlen) for t in texts[i:i+bs]],dtype=torch.long).to(device)
        out.append(torch.softmax(m(xb)[0],dim=1)[:,1].cpu().numpy())
    return np.concatenate(out)
