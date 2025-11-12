# Impute-Agent

Autonomous pipeline for handling missing data with a blend of classical ML imputers and optional LLM reasoning. Run fully locally with stubbed LLM decisions, or point to local/remote language models when available. AWS hooks (CDK, Lambda, API Gateway, Bedrock, SageMaker) are included but optional.

---

## Highlights
- Agentic workflow: Mechanism detector -> Policy designer -> Decider -> Experimentalist -> Critic -> Scribe.
- Per-column decisions: choose IterativeRF, KNN, Mean, or LLM fills; per-cell overrides supported.
- MNAR sensitivity analysis to stress-test downstream metrics under pattern shifts.
- Local-first defaults: stub LLM backend; switch to local weights or Bedrock when ready.
- AWS-ready CDK stack for serverless deployment.

---

## Repository Layout
```
Impute-Agent/
  README.md
  requirements.txt
  model_download.py        # optional helper for Hugging Face snapshots
  weights.py               # lazy loader for locally stored model artifacts
  config/
    default.yaml           # data columns, missingness grid, evaluation knobs
    decider.yaml           # decider thresholds, weights, and budgets
  data/
    framingham_sample.csv  # sample dataset for local demos
  results/                 # generated summaries (ignored by git)
  models/                  # local model cache (ignored by git)
  src/
    run.py                 # CLI orchestrator
    agent/
      mechanism_detector.py
      imputer_designer.py
      decider.py
      critic.py
      scribe.py
    llm/
      llm_client.py        # stub + local/Bedrock clients
      prompts.py
    model/
      impute_model.py      # IterativeRF/KNN/Mean + LLM override
      sensitivity.py
    utils/
      data_io.py
      metrics.py
      validators.py
  cdk/                     # AWS CDK application
```

`.gitignore` keeps large model artifacts (`models/`) and generated reports under `results/` out of source control.

---

## Local Quickstart

### 1. Install dependencies
```bash
# Basic installation (no LLM)
pip install -r requirements-base.txt

# Or full installation with LLM support
pip install -r requirements.txt
```

### 2. Run the demo
```bash
# Without LLM (classical methods only)
python -m src.run \
  --data data/framingham_sample.csv \
  --target TenYearCHD \
  --llm stub

# With local LLM (optional)
python model_download.py  # Download phi-2 (~5GB) first
python -m src.run \
  --data data/framingham_sample.csv \
  --target TenYearCHD \
  --llm openai-oss
```

Results in: `results/summary.csv`, `results/imputed.csv`, `results/report.md`

---

## Configuration Notes
- `config/default.yaml`: numeric/categorical column lists, missingness grid, evaluation settings, sensitivity deltas, output directory.
- `config/decider.yaml`: thresholds for switching to the LLM, confidence limits, probe size, scoring weights, and resource budget hints.

Override any config path via `--config` or `--decider_config` on the CLI.

---

## Pipeline Overview
1. **MechanismDetector** labels columns MCAR/MAR/MNAR with lightweight heuristics.
2. **ImputerDesigner** proposes candidate policies (per-column method mixes).
3. **Decider** chooses MODEL or LLM per column using stats, cardinality, and missingness context; falls back when the LLM is disabled.
4. **LocalImputer** materialises each policy with Iterative Imputer, KNN, Mean, or LLM fills plus optional MNAR shifts.
5. **Critic** ranks policies using downstream AUC alongside RMSE/accuracy metrics.
6. **Scribe** emits a Markdown report summarising the best policy.

---

## AWS Deployment (Optional)
The `cdk/` package provisions S3 buckets, Lambda, API Gateway, and IAM roles that can call Bedrock or SageMaker.
1. Install Node 18+, AWS CLI, and AWS CDK v2; run `aws configure`.
2. Bootstrap once with `cdk bootstrap`.
3. In `cdk/`, run `npm install` then `cdk deploy` (set context like `bedrockRegion` or `sagemakerEndpointName` in `cdk.json`).
4. Upload CSVs to the raw bucket or call the API endpoint to trigger the pipeline.

Adapt `src/llm/llm_client.py` and the Lambda handler to integrate production LLMs or imputation endpoints.

---

## Git Hygiene
- Model weights in `models/` and results in `results/` are gitignored
- Never commit large model files to the repository

---

## Troubleshooting
- **Target column error**: Fix `--target` or adjust column lists in `config/default.yaml`
- **Memory issues**: Use `--llm stub` for classical methods only
- **Import errors**: Run `pip install -r requirements-base.txt`
- **Model download fails**: Some models need `huggingface-cli login`

---

## License
MIT License. See `LICENSE` for details.

---

## Acknowledgements
Sample dataset based on Framingham heart study derivatives.
