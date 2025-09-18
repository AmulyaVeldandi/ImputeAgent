# Impute-Agent 🧠🤖

Impute-Agent is an **AWS-native autonomous agent** for missing-data handling in tabular
clinical/biomedical datasets. It detects missingness mechanisms (MCAR/MAR/MNAR), designs
per-column imputation policies with an LLM-backed Decider, executes pipelines, evaluates
metrics (RMSE/accuracy/AUC), runs MNAR sensitivity, and outputs a clean dataset.

## Local quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.run --data data/framingham_sample.csv --target TenYearCHD --output results/summary.csv --llm on --sensitivity on
```
Artifacts land in `results/` (summary.csv, report.md, imputed.csv).

## AWS hooks
- Swap `src/llm/llm_client.py` for **Bedrock**.
- Swap `src/model/impute_model.py` for **SageMaker** endpoints.
- Use the CDK app in `cdk/` to deploy S3, Lambda, API Gateway, and IAM.
# ImputeAgent
