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

`.gitignore` keeps large model artifacts (`models/openai-gpt-oss-20b/`) and generated reports under `results/` out of source control. Regenerate those files locally after pulling if you need fresh outputs.

---

## Local Quickstart

### 1. Environment
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the demo (stubbed LLM)
```bash
python -m src.run \
  --data data/framingham_sample.csv \
  --target TenYearCHD \
  --output results/summary.csv \
  --llm stub \
  --sensitivity on
```
Artifacts land in `results/summary.csv`, `results/imputed.csv`, and `results/report.md` (all ignored by git).

### 3. Optional: use local LLM weights
1. Authenticate with Hugging Face (`huggingface-cli login`).
2. Download weights into `models/openai-gpt-oss-20b/`:
   ```bash
   python model_download.py
   ```
3. Point the CLI to the local model:
   ```bash
   python -m src.run ... --llm openai-oss
   ```
   The loader reads local files only; ensure the snapshot exists before enabling.

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

## Git Hygiene Tips
- If results were previously tracked, run `git rm --cached results/*.csv results/*.md` before committing.
- Keep downloaded model weights under `models/` (ignored) or mount them externally; never commit the 20B snapshot.
- Regenerate local summaries after pulling if you need fresh artifacts for analysis.

---

## Troubleshooting
- `ValueError: Target ... not in CSV columns`: fix `--target` or adjust `config/default.yaml`.
- Long runtimes or memory spikes: ensure `--llm stub` unless the weights are present and hardware can host them.
- `huggingface_hub` errors: confirm network access and authentication before running `model_download.py`.

---

## License
MIT License. See `LICENSE` for details.

---

## Acknowledgements
Sample dataset based on Framingham heart study derivatives.
