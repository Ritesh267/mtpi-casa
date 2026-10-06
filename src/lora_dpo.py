"""The internal layer: LoRA adapters trained with Direct Preference Optimisation
on preference pairs the detector generates for itself.

Both pieces are implemented here rather than imported, because the point of the
experiment is the source of the supervision, not the optimiser. LoRA follows Hu
et al. (2022): a frozen base weight W plus a trainable low-rank update BA scaled
by alpha/r. DPO follows Rafailov et al. (2023): the implicit reward is the log-
ratio of policy to reference likelihood, and the loss is the logistic loss on
the difference between chosen and rejected rewards.

The policy here is a small decoder-only model trained from scratch on this
corpus, for the reason given in encoder.py: the core pipeline assumes no
pretrained weights. The study on a pretrained open-weight base (report Section
6.12) is src2/dpo_gpu_study.py, which attaches LoRA with peft and writes out
the same DPO loss. scripts/train_dpo_gpu.py is a separate, untested set-up
through peft and trl; no reported number comes from it.
"""
from __future__ import annotations
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(42)

# --------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base=base
        for p in self.base.parameters(): p.requires_grad=False
        self.r=r; self.scale=alpha/r
        self.A=nn.Parameter(torch.zeros(r, base.in_features))
        self.B=nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop=nn.Dropout(dropout)
    def forward(self,x):
        return self.base(x) + self.drop(x) @ self.A.T @ self.B.T * self.scale

def inject_lora(model, r=8, alpha=16, targets=("q_proj","k_proj","v_proj","o_proj","fc1","fc2")):
    n=0
    for name,mod in list(model.named_modules()):
        for child_name,child in list(mod.named_children()):
            if isinstance(child,nn.Linear) and child_name in targets:
                setattr(mod,child_name,LoRALinear(child,r,alpha)); n+=1
    for p in model.parameters(): p.requires_grad=False
    for m in model.modules():
        if isinstance(m,LoRALinear):
            m.A.requires_grad=True; m.B.requires_grad=True
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total=sum(p.numel() for p in model.parameters())
    return n, trainable, total

# ------------------------------------------------------------ tiny policy LM
class Block(nn.Module):
    def __init__(self,d,h,ff,drop):
        super().__init__()
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d)
        self.q_proj=nn.Linear(d,d); self.k_proj=nn.Linear(d,d)
        self.v_proj=nn.Linear(d,d); self.o_proj=nn.Linear(d,d)
        self.fc1=nn.Linear(d,ff); self.fc2=nn.Linear(ff,d)
        self.h=h; self.dh=d//h; self.drop=nn.Dropout(drop)
    def forward(self,x,mask):
        B,T,D=x.shape; z=self.ln1(x)
        q=self.q_proj(z).view(B,T,self.h,self.dh).transpose(1,2)
        k=self.k_proj(z).view(B,T,self.h,self.dh).transpose(1,2)
        v=self.v_proj(z).view(B,T,self.h,self.dh).transpose(1,2)
        a=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=False)
        a=a.transpose(1,2).reshape(B,T,D)
        x=x+self.drop(self.o_proj(a))
        z=self.ln2(x)
        return x+self.drop(self.fc2(F.gelu(self.fc1(z))))

class PolicyLM(nn.Module):
    def __init__(self,vocab,d=192,layers=4,h=4,ff=384,maxlen=256,drop=0.1):
        super().__init__()
        self.emb=nn.Embedding(vocab,d,padding_idx=0)
        self.pos=nn.Parameter(torch.zeros(1,maxlen,d)); nn.init.normal_(self.pos,std=0.02)
        self.blocks=nn.ModuleList([Block(d,h,ff,drop) for _ in range(layers)])
        self.ln=nn.LayerNorm(d); self.out=nn.Linear(d,vocab,bias=False)
        self.out.weight=self.emb.weight
        self.maxlen=maxlen
    def forward(self,x):
        B,T=x.shape
        causal=torch.tril(torch.ones(T,T,dtype=torch.bool,device=x.device))
        pad=(x!=0).unsqueeze(1).unsqueeze(2)
        mask=causal.unsqueeze(0).unsqueeze(0) & pad
        h=self.emb(x)+self.pos[:,:T]
        for b in self.blocks: h=b(h,mask)
        return self.out(self.ln(h))

# ----------------------------------------------------------------- sequence LL
def seq_logprob(model, ids, prompt_len):
    """Sum log p(response tokens | prompt). Padding excluded."""
    logits=model(ids[:,:-1])
    tgt=ids[:,1:]
    lp=F.log_softmax(logits,dim=-1).gather(-1,tgt.unsqueeze(-1)).squeeze(-1)
    T=tgt.size(1)
    pos=torch.arange(T,device=ids.device).unsqueeze(0)
    m=(pos>=(prompt_len.unsqueeze(1)-1)) & (tgt!=0)
    return (lp*m).sum(1)

def dpo_loss(policy, ref, chosen, rejected, cl, rl, beta=0.1):
    """Rafailov et al. (2023), eq. 7."""
    pc=seq_logprob(policy,chosen,cl); pr=seq_logprob(policy,rejected,rl)
    with torch.no_grad():
        rc=seq_logprob(ref,chosen,cl); rr=seq_logprob(ref,rejected,rl)
    logits=beta*((pc-rc)-(pr-rr))
    loss=-F.logsigmoid(logits).mean()
    acc=(logits>0).float().mean().item()
    margin=(pc-pr).mean().item()
    return loss, acc, margin
