"""Print the result fields behind the tables of the report, from the shipped result files.

Needs only the Python standard library and can be started from any folder:

    python tools/check_report.py                 # core tables from results_v1_original/
    python tools/check_report.py --run results   # core tables from the later re-run
    python tools/check_report.py --run both      # both, one after the other

Tables 6.1 to 6.11 are printed in the report from results_v1_original/. The folder
results/ holds a later re-run made to rebuild the saved models; its gradient-boosted
rows differ (see the README, "Result folders"). Tables 6.13 to 6.21 have one source
each, in results2/ and results3/.

The script reads JSON only. It does not load a model, the benchmark or a pickle, so it
works on a fresh clone. It exits with status 1 if a result file it expects is missing.
"""
import argparse, glob, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MISSING = []


def J(rel):
    """Load a JSON file given relative to the repository root, or None if it is absent."""
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        MISSING.append(rel)
        print(f"  [missing: {rel}]")
        return None
    with open(p) as f:
        return json.load(f)


def pct(x, d=1):
    return f"{x * 100:.{d}f}"


def core(R):
    print(f"\n================ core pipeline, folder {R}/ ================")
    g = J(f"{R}/grid.json")
    if g:
        rows = [r for r in g if r["clf"] != "TransformerHead"]
        b = max(rows, key=lambda r: r["auc"])
        print(f"Table 6.1   {len(g)} rows. Best single configuration: {b['rep']} + {b['clf']}  "
              f"AUC {b['auc']:.4f}  recall {b['recall']:.3f}  FPR {b['fpr']:.3f}")
    m = J(f"{R}/casa_main.json")
    if m:
        print(f"            stacked turn detector, test AUC {m['turn_auc']:.4f}")
        print("Table 6.2   conversation level, 5% budget")
        for r in m["results"][:7]:
            print(f"  {r['config'][:46]:<46} recall {r['recall']:.3f}  FPR {r['fpr']:.3f}  "
                  f"precision {r['precision']:.3f}  AUC {r['auc']:.4f}")
        c = m["ceiling"]
        print(f"Table 6.3   oracle {c['oracle']:.4f}  stateless {c['stateless']:.4f}  CASA {c['casa']:.4f}  "
              f"recovered {c['recovered_pct']:.1f}% (main partition)")
        print("Table 6.4   recall at budgets 1%, 2%, 5%, 10%, 20%")
        for k, rows in m["operating_curve"].items():
            print(f"  {k:<22} " + "  ".join(f"{r['recall']:.3f}" for r in rows))
        print("Table 6.5   recall by slice (stateless, CASA linear)")
        for s in m["breakdown"]:
            print(f"  {s['slice']:<20} n={s['n']:3d}  {s['stateless_recall']:.3f}  {s['casa_recall']:.3f}   "
                  f"AUC {s['stateless_auc']:.4f}  {s['casa_auc']:.4f}")
        print("Table 6.6   ablation (single run; differences are within run-to-run variation)")
        for r in m["results"][7:]:
            print(f"  {r['config']:<32} AUC {r['auc']:.4f}  change {r['delta_auc']:+.4f}")
        s = m["sprt_cfg"]
        print(f"            accumulator: leak {s['leak']}  alarm {s['alarm']}  priors {s['prior_user']} / {s['prior_retrieved']}")
    a = J(f"{R}/adaptive_final.json")
    if a:
        print("Tables 6.7 and 6.9   detection under each attacker strategy (decision at the end of the conversation)")
        for r in a["rows"]:
            print(f"  {r['strategy']:<10} turns {r['mean_turns']:5.1f}  stateless {r['stateless']:.3f}  "
                  f"CASA-base {r['CASA-base']:.3f}  AT-linear {r['CASA-AT (linear)']:.3f}  AT-gbm {r['CASA-AT (gbm)']:.3f}")
        print("            operating thresholds:", {k: round(v, 5) for k, v in a["thresholds"].items()})
        print("Table 6.10  first row, clean benign:", {k: round(v, 3) for k, v in a["fpr"].items()})
    t = J(f"{R}/adv_train.json")
    if t:
        for s, row in t["fpr_fragmented_benign"].items():
            print(f"  {s:<10} " + "  ".join(f"{k} {v:.3f}" for k, v in row.items() if v is not None))
    z = J(f"{R}/asym_control.json")
    if z:
        print("Table 6.8   attack-only augmentation: detection, and FPR on benign transformed the same way")
        for s in z["detection"]:
            print(f"  {s:<10} {z['detection'][s]:.3f}  {z['fpr'][s]:.3f}")
    d = J(f"{R}/dpo.json")
    if d:
        print("Table 6.11  small CPU policy, one run per arm")
        for r in d["rows"]:
            print(f"  {r['arm']:<26} attacks {r['refuse_attack']:5.1f}  benign {r['refuse_benign']:5.1f}  "
                  f"separation {r['gap']:+6.1f}")


def later():
    print("\n================ later experiments, results2/ ================")
    s = J("results2/asr_live.json")
    if s:
        print("Table 6.13  attack success (%) by served model; benign blocked (%) differs by defence, so the columns are not matched")
        for name, v in s["models"].items():
            print(f"  {name:<38} " + "  ".join(f"{k} {pct(d['asr'], 0)}" for k, d in v["defences"].items()))
        first = next(iter(s["models"].values()))["defences"]
        print("  attacks blocked:  " + "  ".join(f"{k} {pct(d['block_rate'])}" for k, d in first.items()))
        print("  benign blocked:   " + "  ".join(f"{k} {pct(d['benign_block_rate'])}" for k, d in first.items()))
    f = J("results2/casa_fix.json")
    if f:
        print(f"Table 6.14  hardening, training seeds {f.get('seeds')}: mean [95% t-interval], detection %")
        for k, v in f["summary"].items():
            print(f"  {k:<7} " + "  ".join(
                f"{a} {pct(v[a]['mean'])} [{pct(v[a]['lo'])}, {pct(v[a]['hi'])}]" for a in ("none", "dilute", "combined")))
    c = J("results2/composed_eval.json")
    if c:
        print("Table 6.15  composed set (no benign control)")
        for k, v in c.items():
            print(f"  {k:<11} turns {v['mean_turns']:4.1f}  stateless {pct(v['stateless'])}  CASA {pct(v['casa'])}"
                  + (f"  CASA-AT {pct(v['casa_at'])}" if "casa_at" in v else ""))
    x = J("results2/xai_audit.json")
    if x:
        print("Table 6.16  attribution in the deployed linear meta-classifier, top 8 by mean |SHAP|")
        for r in x["features"][:8]:
            print(f"  {r['feature']:<16} {r['group']:<12} coef(sd) {r['coef_std']:+.3f}  perm {r['perm_importance']:+.4f}  "
                  f"|shap| {r['mean_abs_shap']:.4f}")
    S = J("results2/summary.json")
    if S and S.get("dpo_gpu"):
        g = S["dpo_gpu"]
        print(f"Table 6.17  LoRA-DPO on {g['base']}, seeds {g.get('seeds')}: separation in points, mean [95% t-interval]")
        print(f"  A0 (one run)           {g['A0']['sep']:+.1f}")
        for k, v in g.items():
            if isinstance(v, dict) and "sep_mean" in v:
                print(f"  {k:<22} {v['sep_mean']:+.1f} [{v['sep_lo']:+.1f}, {v['sep_hi']:+.1f}]  n={v['n']}  LM loss {v['nll_mean']:.3f}")
    b = J("results2/bootstrap_casa.json")
    if b:
        print("Section 6.12  paired bootstrap, 2,000 resamples of the test partition")
        for k in ("recall_gain", "auc_gain"):
            print(f"  {k:<12} {b[k]['point']:.4f}  95% CI [{b[k]['ci95'][0]:.4f}, {b[k]['ci95'][1]:.4f}]")
    seeds = sorted(glob.glob(os.path.join(ROOT, "results2/multiseed_casa/seed*.json")))
    if seeds:
        print("Section 6.12  further partitions (re-splits of the same corpus)")
        for p in seeds:
            with open(p) as fh:
                m = json.load(fh)
            print(f"  seed {m['seed']}: stateless AUC {m['stateless']['auc']:.4f}  CASA AUC {m['casa_linear']['auc']:.4f}  "
                  f"oracle {m['oracle_auc']:.4f}  recovered {m['recovered_pct']:.1f}%")
    else:
        MISSING.append("results2/multiseed_casa/seed*.json")

    print("\n================ live RAG, online decisions and checks, results3/ ================")
    r = J("results3/rag_retrieval.json")
    if r:
        kb = r["kb"]
        print(f"Table 6.18  knowledge base: {kb['paragraphs']} paragraphs, {kb['chunks']} chunks, {kb['questions']} questions")
        for retr in ("bm25", "dense"):
            print(f"  gold paragraph in top 5, {retr}: {pct(r['gold_recall'][retr]['recall@5'])}")
        for mode in ("targeted", "chunkaware", "hijack"):
            for retr in ("bm25", "dense"):
                v = r["planted"][mode][retr]["k5"]
                print(f"  {mode:<10} {retr:<5} fragment retrieved {pct(v['fragment_retrieved'])}  "
                      f"complete payload delivered {pct(v['session_full_payload_delivered'])}")
    for retr in ("bm25", "dense"):
        d = J(f"results3/rag_detection_{retr}.json")
        if d:
            print(f"Table 6.19  online detection %, {retr}, recalibrated 5% budget [FPR on planted benign]")
            for k in ("stateless_prompt", "stateless_chunk", "casa_prompt_zs", "casa_chunk_zs", "casa_prompt_rag", "casa_chunk_rag"):
                print(f"  {k:<18} " + "  ".join(
                    f"{m} {pct(d['by_mode'][m][k]['recalibrated']['detection'])} "
                    f"[{pct(d['by_mode'][m][k]['recalibrated']['fpr_planted_benign'])}]" for m in d["by_mode"])
                      + f"   clean {pct(d['clean']['test'][k]['recalibrated'])}")
    a = J("results3/rag_asr.json")
    if a:
        print("Table 6.20  served models in the live pipeline: code in the final reply (%), undefended and behind CASA-RAG")
        for m, v in a["models"].items():
            print(f"  {m:<24} answers {pct(v['utility']['with_retrieval'])}  " + "  ".join(
                f"{f_}: {pct(x['undefended'])} / {pct(x['casa_chunk_rag']['asr'])}" for f_, x in v["attacks"].items()))
    o = J("results3/online_eval.json")
    if o:
        print("Table 6.21  detection %, decision at the end (offline) and after every turn (online, recalibrated)")
        for s, row in o["attack"].items():
            print(f"  {s:<10} " + "  ".join(
                f"{d}: {pct(v['offline'])} / {pct(v['online'])}" for d, v in row.items() if isinstance(v, dict)))
        for k, v in o["append"].items():
            c_ = v["attack"]["CASA-base"]
            print(f"  {k} benign turn(s) appended after the attack: CASA offline {pct(c_['offline'])}, online {pct(c_['online'])}")
    x = J("results3/xai_robustness.json")
    if x:
        print("Section 6.11  accumulator: share of mean |SHAP| deployed "
              f"{pct(x['deployed']['group_share']['accumulator'])}%, after standardising "
              f"{pct(x['standardised_refit']['group_share']['accumulator'])}%, AUC change when removed "
              f"{x['removal']['accumulator']['delta']:+.4f}")
    p = J("results3/pace_mechanism.json")
    if p:
        print("Section 6.4   pacing: deployed model, refit without accumulator, refit on standardised features")
        for s in ("pace", "pace-wide"):
            print(f"  {s:<10} {p[s]['deployed']:.3f}  {p[s]['refit_no_accumulator']:.3f}  {p[s]['refit_standardised']:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", choices=["results_v1_original", "results", "both"], default="results_v1_original",
                    help="folder for the core tables 6.1 to 6.11 (default: the original run behind the printed tables)")
    a = ap.parse_args()
    for R in (["results_v1_original", "results"] if a.run == "both" else [a.run]):
        core(R)
    later()
    if MISSING:
        print(f"\n{len(MISSING)} expected result file(s) missing: " + ", ".join(MISSING))
        sys.exit(1)
    print("\nall expected result files were found")


if __name__ == "__main__":
    main()
