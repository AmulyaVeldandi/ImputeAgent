import argparse
import os
import json
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from .utils.data_io import load_csv, inject_missingness_grid
from .utils.config_validator import validate_config, validate_decider_config
from .agent.mechanism_detector import MechanismDetector
from .agent.imputer_designer import ImputerDesigner
from .agent.decider import Decider
from .agent.critic import Critic
from .agent.scribe import Scribe
from .model.impute_model import LocalImputer
from .model.sensitivity import run_sensitivity
from .llm.llm_client import LocalLLMClient


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

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Load config files with error handling
    try:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config file: {e}")
        raise

    try:
        with open(args.decider_config) as f:
            dcfg = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Decider config file not found: {args.decider_config}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in decider config: {e}")
        raise

    # Validate configurations
    try:
        validate_config(cfg)
        validate_decider_config(dcfg)
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

    seed = cfg.get("seed", 42)
    np.random.seed(seed)
    logger.info(f"Using random seed: {seed}")

    outdir = Path(cfg["output"]["dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        df = load_csv(args.data)
    except FileNotFoundError:
        logger.error(f"Data file not found: {args.data}")
        raise
    except Exception as e:
        logger.error(f"Failed to load data file: {e}")
        raise
    if args.target not in df.columns:
        raise ValueError(f"Target {args.target} not in CSV columns.")
    target = args.target

    numeric = [c for c in cfg["data"]["numeric"] if c in df.columns]
    categorical = [c for c in cfg["data"]["categorical"] if c in df.columns]

    logger.info(f"Loaded {len(df)} rows, {len(numeric)} numeric and {len(categorical)} categorical columns")

    llm_cfg = cfg.get("llm", {})
    llm_enabled = bool(llm_cfg.get("enabled", True))
    llm = LocalLLMClient(backend=args.llm, enabled=llm_enabled, random_seed=seed)

    mech = MechanismDetector(llm=llm)
    designer = ImputerDesigner(llm=llm)
    decider = Decider(dcfg["decider"], llm=llm)
    critic = Critic(llm_client=llm)
    scribe = Scribe()

    rows = []
    best_policy_global = None
    best_score_global = -1e9
    mechanism_cache = {}
    decision_cache = {}

    def get_mechanism(key, df_missing_local, target_local, numeric_local, categorical_local):
        if key not in mechanism_cache:
            mechanism_cache[key] = mech.detect(df_missing_local, target_local, numeric_local, categorical_local)
        return mechanism_cache[key]

    def get_decisions(key, df_true_local, df_missing_local, target_local, numeric_local,
                      categorical_local, mechanism_map_local, imputer_local):
        if key not in decision_cache:
            decision_cache[key] = decider.decide_all(
                df_true_local,
                df_missing_local,
                target_local,
                numeric_local,
                categorical_local,
                mechanism_map_local,
                imputer_local,
            )
        return decision_cache[key]

    for miss_type, miss_frac, df_missing, mask_df in inject_missingness_grid(
            df, target, numeric, categorical, cfg["missingness"]["types"], cfg["missingness"]["fractions"]):

        cache_key = (miss_type, miss_frac)
        mechanism_map = get_mechanism(cache_key, df_missing, target, numeric, categorical)
        candidates = designer.propose_policies(mechanism_map, numeric, categorical)

        imputer = LocalImputer(llm_client=llm)
        decisions = get_decisions(cache_key, df, df_missing, target, numeric, categorical, mechanism_map, imputer)

        results = []
        for policy in candidates:
            res = imputer.run_policy(
                df_true=df,
                df_missing=df_missing,
                mask_df=mask_df,
                target=target,
                numeric=numeric,
                categorical=categorical,
                policy=policy,
                decisions=decisions,
                downstream=cfg["evaluation"]["downstream_model"]
            )
            eval_pack = critic.evaluate(res, {"policy": policy, "decisions": decisions})
            score = eval_pack["numeric_score"]
            results.append((score, policy, res, eval_pack))

        results.sort(key=lambda x: x[0], reverse=True)
        top_score, top_policy, top_res, top_eval = results[0]

        sens_rows = []
        if args.sensitivity == "on" and miss_type == "MNAR":
            sens_rows = run_sensitivity(
                df_true=df,
                df_missing=df_missing,
                mask_df=mask_df,
                target=target,
                numeric=numeric,
                categorical=categorical,
                policy=top_policy,
                decisions=decisions,
                deltas=cfg["evaluation"]["sensitivity_deltas"],
                imputer=imputer
            )

        rows.append({
            "missing_type": miss_type,
            "missing_fraction": miss_frac,
            "policy": json.dumps(top_policy),
            "score": top_score,
            **top_res,
            "critic_eval": json.dumps(top_eval),
            "sensitivity": json.dumps(sens_rows) if sens_rows else "[]"
        })

        if top_score > best_score_global:
            best_score_global, best_policy_global = top_score, top_policy

    summary = pd.DataFrame(rows)
    try:
        summary.to_csv(args.output, index=False)
        summary.to_csv(outdir / "summary.csv", index=False)
        logger.info(f"Saved summary to {args.output}")
    except Exception as e:
        logger.error(f"Failed to save summary: {e}")
        raise

    imputer = LocalImputer(llm_client=llm)
    df_missing = df.copy()
    mask_df = df_missing.isna()
    base_key = ("base", 0.0)
    mechanism_map = get_mechanism(base_key, df_missing, target, numeric, categorical)
    decisions = get_decisions(base_key, df, df_missing, target, numeric, categorical, mechanism_map, imputer)
    final = imputer.apply_policy_return_imputed(df_missing, target, numeric, categorical, best_policy_global, decisions)

    try:
        final.to_csv(outdir / "imputed.csv", index=False)
        logger.info(f"Saved imputed data to {outdir / 'imputed.csv'}")
    except Exception as e:
        logger.error(f"Failed to save imputed data: {e}")
        raise

    report = scribe.render_report(summary, best_policy_global)
    try:
        (outdir / "report.md").write_text(report)
        logger.info(f"Saved report to {outdir / 'report.md'}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")
        raise

    logger.info(f"Done. Wrote:\n- {args.output}\n- {outdir/'imputed.csv'}\n- {outdir/'report.md'}")


if __name__ == "__main__":
    main()
