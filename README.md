# Impute-Agent 🧠🤖  
**An AWS-native autonomous agent for missing-data imputation with LLM reasoning + classical ML.**

Clinical and biomedical datasets often have missing values. That slows research, biases models, and burns analyst time. **Impute-Agent** detects missingness mechanisms (MCAR/MAR/MNAR), designs **per-column** imputation policies with an **LLM-backed Decider**, executes pipelines, evaluates RMSE/accuracy/AUC, runs MNAR sensitivity, and outputs a clean, audit-ready dataset.  
Local run works out of the box. AWS hooks are built in (Bedrock, SageMaker, Lambda, API Gateway, S3) with a CDK app.

---

## ✨ Highlights
- **Agentic workflow**: Planner → Decider → Experimentalist → Critic → Scribe.  
- **LLM for decisions**: Bedrock-ready client (local stub by default).  
- **Per-column strategy**: choose MODEL (IterativeRF/KNN/Mean) vs **LLM imputer**; optional per-cell override.  
- **MNAR sensitivity**: ±δ pattern-mixture shifts with banded metrics.  
- **AWS-ready**: CDK deploys S3 (raw/outputs), Lambda, API Gateway, IAM; Bedrock + SageMaker call sites are stubbed in.

---

## 🧭 Repository Structure
```
Impute-Agent/
├─ README.md
├─ requirements.txt
├─ config/
│  ├─ default.yaml         # data columns, eval knobs, sensitivity deltas
│  └─ decider.yaml         # Decider thresholds & weights
├─ data/
│  └─ framingham_sample.csv
├─ results/                # outputs saved here
├─ src/
│  ├─ run.py               # local orchestrator (CLI)
│  ├─ agent/
│  │  ├─ mechanism_detector.py
│  │  ├─ decider.py
│  │  ├─ imputer_designer.py
│  │  ├─ critic.py
│  │  └─ scribe.py
│  ├─ llm/
│  │  ├─ llm_client.py     # local stub; Bedrock-ready interface
│  │  └─ prompts.py
│  ├─ model/
│  │  ├─ impute_model.py   # IterativeRF/KNN/Mean + LLM override
│  │  └─ sensitivity.py
│  └─ utils/
│     ├─ data_io.py
│     ├─ metrics.py
│     └─ validators.py
└─ cdk/
   ├─ README.md
   ├─ package.json
   ├─ cdk.json
   ├─ tsconfig.json
   ├─ bin/impute-agent.ts
   ├─ lib/impute-agent-stack.ts
   └─ lambda/lambda_handler.py
```

---

## 🧪 Local Quickstart

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Run a demo
```bash
python -m src.run   --data data/framingham_sample.csv   --target TenYearCHD   --output results/summary.csv   --llm on   --sensitivity on
```

### 3) Artifacts
- `results/summary.csv` — metrics per missingness type & fraction.
- `results/imputed.csv` — final imputed dataset with best policy.
- `results/report.md` — human-readable summary (policy + results).

---

## ⚙️ Configuration

### `config/default.yaml`
- `data.numeric`, `data.categorical` — columns to treat as numeric or categorical.
- `missingness.types` — MCAR, MAR, MNAR (simulated for eval).
- `missingness.fractions` — 0.1, 0.2, 0.3.
- `evaluation.sensitivity_deltas` — e.g., `[-0.05, 0.0, 0.05]`.

### `config/decider.yaml`
- `p_mnar_switch` — threshold to consider MNAR strong.
- `llm_conf_threshold`, `validation_min_pass` — safety gates for LLM fills.
- `per_cell_override` — allow per-row LLM overrides.
- `weights` — utility weights for AUC, RMSE, validation, cost, complexity.

---

## 🧠 How the Agent Decides

- **MechanismDetector**: heuristic + logistic probe to label columns MCAR/MAR/MNAR.  
- **ImputerDesigner**: proposes 3–4 candidate policies (IterativeRF, KNN for cats, or LLM for cats; optional MNAR shift).  
- **Decider**: picks **MODEL vs LLM** per column (and optional per cell) using column type, cardinality, mechanism, probes, and thresholds in `decider.yaml`.  
- **Experimentalist**: executes policy, computes imputation errors on masked entries and downstream AUC.  
- **Critic**: selects the best policy with a multi-objective score.  
- **Scribe**: writes a concise report (best policy + metrics table).

**LLM roles (local stub; Bedrock-ready):**  
- *Decider mode*: choose LLM vs MODEL for hard columns.  
- *LLM-imputer*: returns a value using row + dataset constraints; strict validator enforces schema/ranges.

---

## 📊 Outputs & What to Look For
- **avg_rmse** — mean RMSE over masked numeric cells.  
- **avg_acc** — mean accuracy over masked categorical cells.  
- **auc** — downstream classification AUC.  
- **sensitivity** — AUC and errors under ±δ shifts for MNAR columns.

**Win condition for demos:** AUC stays within ≈0.02 of baseline under 20–30% MCAR/MAR; sensitivity bands are narrow under MNAR.

---

## ☁️ AWS Deployment (CDK)

> The CDK app creates: S3 (raw/outputs), Lambda (S3 + API trigger), API Gateway, IAM perms to invoke **Bedrock** and **SageMaker**. You still need to stand up a SageMaker endpoint (or refactor Lambda to run batch / AWS SDK-based imputers).

### 1) Prereqs
- Node 18+, AWS CLI, AWS CDK v2, Python 3.11.
- Configure AWS CLI: `aws configure`.
- Bootstrap once:  
  ```bash
  cdk bootstrap
  ```

### 2) Configure context
Edit `cdk/cdk.json` or pass `-c`:
```json
{
  "bedrockRegion": "us-east-1",
  "sagemakerEndpointName": "impute-agent-endpoint"
}
```

### 3) Deploy
```bash
cd cdk
npm install
cdk deploy
```

**Outputs:**
- `ApiUrl` — call `POST {ApiUrl}/impute`.
- `RawBucketName` — upload CSVs here to auto-trigger the pipeline.
- `OutputsBucketName` — imputed CSVs land here.

### 4) Invoke
- **S3 trigger**: put `file.csv` into `RawBucketName`.  
- **API call**:
```bash
curl -X POST "{ApiUrl}/impute"   -H "Content-Type: application/json"   -d '{"s3_key": "path/in/raw-bucket/file.csv"}'
```
or embed CSV:
```bash
curl -X POST "{ApiUrl}/impute"   -H "Content-Type: application/json"   -d '{"csv": "col1,col2
1,NA
2,3
"}'
```

### 5) Wire real services
- **Bedrock**: replace `src/llm/llm_client.py` with a Bedrock client; update `cdk/lambda/lambda_handler.py::call_bedrock_reasoning` to your model ID + tool protocol.  
- **SageMaker**: replace `src/model/impute_model.py` calls with `sagemaker-runtime:InvokeEndpoint` in Lambda or call your endpoint from the orchestrator.

> Cost tips: use small instances, stop endpoints when idle, consider Batch Transform, set Bedrock call caps.

---

## 🔐 Security & Safety
- No PHI in sample data.  
- LLM never fills identifier columns.  
- Validators enforce numeric ranges and categorical vocab.  
- Buckets are private + SSE-S3 by default (CDK stack).  
- Least-privilege IAM recommended (tighten ARNs after prototyping).

---

## 🧩 Troubleshooting
- **`Target ... not in CSV columns`**: fix `--target` or update `config/default.yaml`.  
- **AUC is `nan`**: target must be binary integers (0/1) and not all one class.  
- **LLM off**: run with `--llm off` for a baseline.  
- **CDK destroy blocked**: empty the buckets then `cdk destroy`.

---

## 🛣️ Roadmap (nice-to-have)
- Add **SoftImpute/GAIN/VAE** as extra tools.  
- Bedrock **AgentCore** loop with explicit tool schemas for `propose_value`.  
- Add **Amazon Q** interface for “Explain what changed and why”.  
- Per-column **uncertainty visualizations** and calibration plots.

---

## 📝 License
MIT.

---

## 💬 Acknowledgements
Framingham-style sample data is open dataset.

---

### One-liner for the hackathon submission
> “Impute-Agent is an AWS-native autonomous agent that decides *how* to impute missing values using Bedrock reasoning and SageMaker models, verifies with a critic, quantifies MNAR risk with sensitivity bands, and returns a clean, audit-ready dataset via API.”
