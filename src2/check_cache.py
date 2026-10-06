"""Correctness check: the cached scorer must reproduce the uncached one."""
import json, sys, numpy as np, torch
sys.path.insert(0, "src2")
import dpo_gpu_study as G
from transformers import AutoModelForCausalLM, AutoTokenizer
base = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(base); lm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16).cuda().eval()
enc = G.Encoder(tok, 320)
P = json.load(open("results2/dpo_pool.json")); prompts = [r["text"] for r in P["test"][:24]]
A = G.score_templates(lm, enc, prompts, G.EVAL_REF + G.EVAL_COMP, "cuda", bs=8)
B = G.score_templates_cached(lm, enc, prompts, G.EVAL_REF + G.EVAL_COMP, "cuda", bs=8)
d = np.abs(A - B)
print(f"max |uncached - cached| = {d.max():.4f}   mean = {d.mean():.4f}   (scores range {A.min():.1f}..{A.max():.1f})")
print("argmax agreement:", float(((A[:, :5].max(1) - A[:, 5:].max(1) > 0) == (B[:, :5].max(1) - B[:, 5:].max(1) > 0)).mean()))
