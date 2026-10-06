"""Stacked turn-level detector.

The conversation layer can only aggregate what the turn layer gives it, so the
turn detector is built as a stack over the three representations that performed
best in the grid rather than as a single model. The stack is fitted on
out-of-fold predictions from the training partition, so the meta-learner never
sees a base model's in-sample score.
"""
import sys, warnings; sys.path.insert(0,"src"); warnings.filterwarnings("ignore")
import numpy as np, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
import reps as R

class StackedTurnDetector:
    def __init__(self, seed=42):
        self.seed=seed
        self.reps=[R.TfidfChar(), R.TfidfWord(), R.LSA(192)]
        self.kinds=["svm","logreg","logreg"]
    def _mk(self,kind):
        if kind=="svm":
            return CalibratedClassifierCV(LinearSVC(C=0.5,random_state=self.seed),cv=3)
        return LogisticRegression(max_iter=3000,C=2.0,random_state=self.seed)
    def fit(self, texts, y):
        texts=list(texts); y=np.asarray(y)
        for r in self.reps: r.fit(texts,y)
        Xs=[r.transform(texts) for r in self.reps]
        oof=np.zeros((len(texts),len(self.reps)))
        skf=StratifiedKFold(5,shuffle=True,random_state=self.seed)
        for tr,te in skf.split(np.zeros(len(y)),y):
            for j,(X,k) in enumerate(zip(Xs,self.kinds)):
                m=self._mk(k).fit(X[tr],y[tr]); oof[te,j]=m.predict_proba(X[te])[:,1]
        self.models=[self._mk(k).fit(X,y) for X,k in zip(Xs,self.kinds)]
        self.meta=LogisticRegression(max_iter=2000).fit(oof,y)
        return self
    def _base_scores(self, texts):
        texts=list(texts)
        return np.column_stack([m.predict_proba(r.transform(texts))[:,1]
                                for m,r in zip(self.models,self.reps)])
    def predict_proba(self, texts):
        p=self.meta.predict_proba(self._base_scores(texts))[:,1]
        return np.column_stack([1-p,p])
