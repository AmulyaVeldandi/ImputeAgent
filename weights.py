from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent / "models" / "openai-gpt-oss-20b"

if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f"Local model directory not found at {MODEL_DIR}. "
        "Download the model artifacts manually and place them in this folder."
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True,
)
