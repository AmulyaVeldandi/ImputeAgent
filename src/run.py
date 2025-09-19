import argparse, os, json, yaml, numpy as np, pandas as pd
from pathlib import Path

from .utils.data_io import load_csv, inject_missingness_grid
from .agent.mechanism_detector import MechanismDetector
from .agent.imputer_designer import ImputerDesigner
from .agent.decider import Decider
from .agent.critic import Critic
from .agent.scribe import Scribe
from .model.impute_model import LocalImputer
from .model.sensitivity import run_sensitivity
from .llm.llm_client import LocalLLMClient   # 🔹 changed

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--output", default="results/summary.csv")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--decider_config", default="config/decider.yaml")
    p.add_argument("--llm", choices=["stub", "openai-oss", "bedrock"], default="stub")
    p.add_argument("--sensitivity", choices=["on", "off"], default="on")
    return p.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    with open(args.decider_config) as f: dcfg = yaml.safe_load(f)
    np.random.seed(cfg.get("seed", 42))

    outdir = Path(cfg["output"]["dir"]); outdir.mkdir(parents=True, exist_ok=True)

    df = load_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target {args.target} not in CSV columns.")
    target = args.target

    numeric = [c for c in cfg["data"]["numeric"] if c in df.columns]
    categorical = [c for c in cfg["data"]["categorical"] if c in df.columns]

    # 🔹 instantiate new LLM client
    llm = LocalLLMClient(backend=args.llm, enabled=(args.llm != "stub"))

    mech = MechanismDetector(llm=llm)
    designer = ImputerDesigner(llm=llm)
    decider = Decider(dcfg["decider"], llm=llm)
    critic = Critic(llm_client=llm)   # 🔹 dual critic
    scribe = Scribe()

    rows = []
    best_policy_global = None
    best_score_global = -1e9

    for miss_type, miss_frac, df_missing, mask_df in inject_missingness_grid(
            df, target, numeric, categorical, cfg["missingness"]["types"], cfg["missingness"]["fractions"]):

        mechanism_map = mech.detect(df_missing, target, numeric, categorical)
        candidates = designer.propose_policies(mechanism_map, numeric, categorical)

        imputer = LocalImputer()
        decisions = decider.decide_all(df, df_missing, target, numeric, categorical, mechanism_map, imputer)

        results = []
        for policy in candidates:
            res = imputer.run_policy(
                df_true=df, df_missing=df_missing, mask_df=mask_df,
                target=target, numeric=numeric, categorical=categorical,
                policy=policy, decisions=decisions, downstream=cfg["evaluation"]["downstream_model"]
            )
            eval_pack = critic.evaluate(res, {"policy": policy, "decisions": decisions})
            score = eval_pack["numeric_score"]
            results.append((score, policy, res, eval_pack))

        results.sort(key=lambda x: x[0], reverse=True)
        top_score, top_policy, top_res, top_eval = results[0]

        sens_rows = []
        if args.sensitivity == "on" and miss_type == "MNAR":
            sens_rows = run_sensitivity(
                df_true=df, df_missing=df_missing, mask_df=mask_df,
                target=target, numeric=numeric, categorical=categorical,
                policy=top_policy, decisions=decisions,
                deltas=cfg["evaluation"]["sensitivity_deltas"], imputer=imputer
            )

        rows.append({
            "missing_type": miss_type, "missing_fraction": miss_frac,
            "policy": json.dumps(top_policy),
            "score": top_score, **top_res,
            "critic_eval": json.dumps(top_eval),
            "sensitivity": json.dumps(sens_rows) if sens_rows else "[]"
        })

        if top_score > best_score_global:
            best_score_global, best_policy_global = top_score, top_policy

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output, index=False)
    summary.to_csv(outdir/"summary.csv", index=False)

    imputer = LocalImputer()
    df_missing = df.copy()
    mask_df = df_missing.isna()
    mechanism_map = mech.detect(df_missing, target, numeric, categorical)
    decisions = decider.decide_all(df, df_missing, target, numeric, categorical, mechanism_map, imputer)
    final = imputer.apply_policy_return_imputed(df_missing, target, numeric, categorical, best_policy_global, decisions)
    final.to_csv(outdir/"imputed.csv", index=False)

    report = scribe.render_report(summary, best_policy_global)
    (outdir/"report.md").write_text(report)
    print(f"Done. Wrote:\n- {args.output}\n- {outdir/'imputed.csv'}\n- {outdir/'report.md'}")

if __name__ == "__main__":
    main()
