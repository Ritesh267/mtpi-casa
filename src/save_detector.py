"""Bundle the best single grid model with its fitted representation.

Optional. The output, results/detector_for_dpo.pkl, is not read by any other
script: the later steps use the stacked detector that src/eval_casa.py saves
as results/stack.pkl. The script needs results/grid.csv and one of the
results/clf_*.pkl files, both written by src/grid.py; the clf_*.pkl files are
not shipped, so run src/grid.py first."""
import sys,warnings; sys.path.insert(0,"src"); warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, joblib
import data as D, reps as R
grid=pd.read_csv("results/grid.csv"); grid=grid[grid.clf!="TransformerHead"]
b=grid.sort_values("auc",ascending=False).iloc[0]
key=b["rep"].split()[0]
print(f"best turn detector: {b['rep']} + {b['clf']}  AUC {b['auc']:.4f}  thr {b['thr']:.3f}")
convs,df=D.load(); df=D.split(df); tr=df[df.part=="train"]
rep={"R1":R.TfidfWord,"R2":R.TfidfChar,"R3":lambda:R.LSA(256),"R4":R.Engineered}[key]()
rep.fit(tr.text.tolist(),tr.y.values)
bundle=joblib.load(f"results/clf_{key}_{b['clf']}.pkl")
joblib.dump({"clf":bundle["clf"],"thr":bundle["thr"],"rep":rep,
             "name":f"{b['rep']} + {b['clf']}"},"results/detector_for_dpo.pkl")
print("saved results/detector_for_dpo.pkl")
