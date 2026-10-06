"""Text representations under comparison, and the engineered feature set.

Six representations. R1-R3 are classical and CPU-cheap; R4 is the engineered
set this line of work started from, extended with conversational structure;
R5 is a transformer encoder trained from scratch on this corpus (src/encoder.py);
R6 combines the learned semantic channel with the engineered one.
"""
from __future__ import annotations
import re, numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------ engineered
ROLE=['you are now','act as','pretend to be','pretend you are','you will now','from now on you',
 'roleplay as','no restrictions','without restrictions','you have no limits','unfiltered',
 'developer mode','dan','dude','void','simulate','you act as','act like','stay in character',
 'your new role','assume the role','take on the persona','respond as','you must act']
OVERRIDE=['ignore previous','ignore all previous','disregard','forget your instructions',
 'forget all previous','ignore instructions','do not follow','bypass','override','new instructions',
 'your new task','ignore your','forget your guidelines','ignore the above','ignore everything above',
 'cancel previous','stop following','your real task','actual instruction','instead of']
LEAK=['system prompt','your instructions','initial prompt','print your','reveal your','repeat the above',
 'what were you told','show me your rules','verbatim']
OBFUS=['base64','rot13','reverse the','spell it','character by character','leetspeak','morse']
COMPLY=['you must','you will','never refuse','always answer','do not refuse','no matter what',
 'without warning','without any warning','regardless of']
FRAME=['hypothetical','fiction','story','novel','screenplay','for research','academic','educational purposes',
 'thought experiment','simulation','game']
_SPECIAL=re.compile(r"[^A-Za-z0-9\s]")
_CODE=re.compile(r"```|</?[a-z]+>|\{\{|\}\}")

FEATNAMES=("n_tokens","log_tokens","avg_word_len","special_ratio","upper_ratio","digit_ratio",
 "role_score","override_score","leak_score","obfus_score","comply_score","frame_score",
 "has_role","has_override","has_leak","has_obfus","imperative_density","second_person_density",
 "code_markers","newline_ratio","max_sentence_len","question_ratio","exclaim_ratio","uniq_ratio")

IMPERATIVES=('answer','respond','reply','write','say','tell','output','print','generate','produce',
 'give','list','explain','describe','translate','summarise','summarize','reveal','show','provide',
 'act','behave','pretend','ignore','disregard','forget','override','bypass','continue','start','stop')

def feats(text:str)->np.ndarray:
    t=str(text); low=t.lower(); w=t.split(); n=max(len(w),1)
    sc=lambda L: sum(1 for p in L if p in low)
    sents=[s for s in re.split(r"(?<=[.!?])\s+",t) if s.strip()]
    toks=re.findall(r"[a-z']+",low)
    return np.array([
      len(w), np.log1p(len(w)),
      np.mean([len(x) for x in w]) if w else 0.0,
      len(_SPECIAL.findall(t))/max(len(t),1),
      sum(c.isupper() for c in t)/max(len(t),1),
      sum(c.isdigit() for c in t)/max(len(t),1),
      sc(ROLE), sc(OVERRIDE), sc(LEAK), sc(OBFUS), sc(COMPLY), sc(FRAME),
      float(sc(ROLE)>0), float(sc(OVERRIDE)>0), float(sc(LEAK)>0), float(sc(OBFUS)>0),
      sum(1 for x in toks if x in IMPERATIVES)/n,
      (low.count(" you ")+low.count("your "))/n,
      float(len(_CODE.findall(t))),
      t.count("\n")/max(len(t),1),
      max((len(s.split()) for s in sents), default=0),
      t.count("?")/n, t.count("!")/n,
      len(set(toks))/max(len(toks),1),
    ],dtype=np.float32)

def feat_matrix(texts): return np.vstack([feats(t) for t in texts])

# ------------------------------------------------------------------ builders
class Rep:
    def __init__(self,name): self.name=name
    def fit(self,texts,y=None): return self
    def transform(self,texts): raise NotImplementedError

class TfidfWord(Rep):
    def __init__(self): super().__init__("R1 tfidf-word")
    def fit(self,texts,y=None):
        self.v=TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=3,max_features=60000,
                               sublinear_tf=True,strip_accents="unicode").fit(texts); return self
    def transform(self,texts): return self.v.transform(texts)

class TfidfChar(Rep):
    def __init__(self): super().__init__("R2 tfidf-char")
    def fit(self,texts,y=None):
        self.v=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=4,max_features=80000,
                               sublinear_tf=True).fit(texts); return self
    def transform(self,texts): return self.v.transform(texts)

class LSA(Rep):
    def __init__(self,k=256): super().__init__(f"R3 lsa-{k}"); self.k=k
    def fit(self,texts,y=None):
        self.w=TfidfWord().fit(texts); self.c=TfidfChar().fit(texts)
        X=sp.hstack([self.w.transform(texts),self.c.transform(texts)]).tocsr()
        self.svd=TruncatedSVD(self.k,random_state=42).fit(X)
        self.sc=StandardScaler().fit(self.svd.transform(X)); return self
    def transform(self,texts):
        X=sp.hstack([self.w.transform(texts),self.c.transform(texts)]).tocsr()
        return self.sc.transform(self.svd.transform(X))

class Engineered(Rep):
    def __init__(self): super().__init__("R4 engineered")
    def fit(self,texts,y=None):
        self.sc=StandardScaler().fit(feat_matrix(texts)); return self
    def transform(self,texts): return self.sc.transform(feat_matrix(texts))

class Hybrid(Rep):
    """R6: learned semantic channel + engineered channel."""
    def __init__(self,sem): super().__init__("R6 hybrid"); self.sem=sem
    def fit(self,texts,y=None):
        self.sem.fit(texts,y); self.eng=Engineered().fit(texts); return self
    def transform(self,texts):
        return np.hstack([np.asarray(self.sem.transform(texts)),self.eng.transform(texts)])
