#!/usr/bin/env python3
"""An alternative GPU set-up for the internal layer, through peft and trl. UNTESTED.

Status. This script was written as a companion to the CPU study in
src/run_dpo.py. It was not run for the report and no reported number comes
from it. The study at scale (report Section 6.12, Table 6.17) is
src2/dpo_gpu_study.py, which loads the base model in float16, attaches LoRA
with peft and writes out its own DPO loop and its own evaluation.

What this script sets up: the three supervision arms against a 4-bit quantised
open-weight base, with rank-8 adapters, gradient accumulation and a
pair-construction rule of the same form as the CPU study. It is not the same
protocol as either study.

Known defects, left as they are. The preference pairs are selected from all
source prompts in the benchmark file, with no held-out partition (src/data.py
is not applied), so an evaluation on benchmark prompts would be in-sample. No
evaluation script is shipped for the adapter it writes.

    pip install torch transformers peft trl bitsandbytes accelerate datasets
    python scripts/train_dpo_gpu.py --base Qwen/Qwen2.5-1.5B-Instruct --arm 3

Hardware, by design and not tested here: a single 16 GB card at rank 8 with a
4-bit base and gradient accumulation.
"""
import argparse, json, os, numpy as np, torch

REFUSALS=["I can't help with that.","I won't follow those instructions.",
          "I'm not able to comply with this request."]
COMPLIANCE=["Sure — here's what you asked for.","Of course, here are the details.",
            "Certainly, this is how it works."]

def build_pairs(prompts, labels, margins, arm, k, seed=42):
    """Arm 1 random selection with ground-truth labels; arm 2 detector-selected
    with ground-truth labels; arm 3 detector-selected with detector labels."""
    rs=np.random.RandomState(seed)
    if arm==1:
        idx=rs.permutation(len(prompts))[:k]
    else:
        idx=np.concatenate([np.where(labels==c)[0][np.argsort(-margins[labels==c])][:k//2]
                            for c in (0,1)])
    rows=[]
    for j,i in enumerate(idx):
        y=labels[i]
        ch = REFUSALS[j%len(REFUSALS)] if y==1 else COMPLIANCE[j%len(COMPLIANCE)]
        rj = COMPLIANCE[j%len(COMPLIANCE)] if y==1 else REFUSALS[j%len(REFUSALS)]
        rows.append({"prompt":prompts[i],"chosen":ch,"rejected":rj})
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--arm",type=int,choices=[1,2,3],required=True)
    ap.add_argument("--pairs",type=int,default=600)
    ap.add_argument("--rank",type=int,default=8)
    ap.add_argument("--alpha",type=int,default=16)
    ap.add_argument("--beta",type=float,default=0.1)
    ap.add_argument("--steps",type=int,default=300)
    ap.add_argument("--bs",type=int,default=2)
    ap.add_argument("--accum",type=int,default=8)
    ap.add_argument("--lr",type=float,default=2e-4)
    ap.add_argument("--four-bit",action="store_true",default=True)
    ap.add_argument("--detector",default="results/stack.pkl")
    ap.add_argument("--bench",default="data/mtpi_bench.jsonl")
    ap.add_argument("--out",default="results/gpu_dpo")
    a=ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset
    import joblib, sys; sys.path.insert(0,"src")

    convs=[json.loads(l) for l in open(a.bench)]
    seen={}
    for c in convs: seen.setdefault(c["source_prompt"],(c["label"],c.get("part","train")))
    prompts=np.array(list(seen.keys()))
    labels=np.array([seen[p][0] for p in prompts])

    det=joblib.load(a.detector)
    p=det.predict_proba(list(prompts))[:,1]
    det_lab=(p>=0.5).astype(int); margin=np.abs(p-0.5)*2
    use_lab = det_lab if a.arm==3 else labels
    print(f"arm {a.arm}: detector agrees with ground truth on "
          f"{(det_lab==labels).mean()*100:.1f}% of {len(prompts)} prompts")

    rows=build_pairs(prompts, use_lab, margin, a.arm, a.pairs)
    ds=Dataset.from_list(rows)

    tok=AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    qc=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,
                          bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True) if a.four_bit else None
    model=AutoModelForCausalLM.from_pretrained(a.base,quantization_config=qc,
                                               dtype=torch.bfloat16,device_map="auto")
    if a.four_bit: model=prepare_model_for_kbit_training(model)
    lcfg=LoraConfig(r=a.rank,lora_alpha=a.alpha,lora_dropout=0.05,bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    model=get_peft_model(model,lcfg); model.print_trainable_parameters()

    cfg=DPOConfig(output_dir=f"{a.out}/arm{a.arm}",per_device_train_batch_size=a.bs,
                  gradient_accumulation_steps=a.accum,learning_rate=a.lr,max_steps=a.steps,
                  beta=a.beta,logging_steps=25,save_strategy="no",bf16=True,
                  report_to=[],remove_unused_columns=False)
    tr=DPOTrainer(model=model,args=cfg,train_dataset=ds,processing_class=tok)
    tr.train()
    model.save_pretrained(f"{a.out}/arm{a.arm}/adapter")
    print(f"adapter written to {a.out}/arm{a.arm}/adapter")
    print("no evaluation script is shipped for this adapter. The refusal scoring and the "
          "capability probe used in the report are in src2/dpo_gpu_study.py: a model aligned "
          "into refusing everything scores well on attack success and is useless.")

if __name__=="__main__": main()
