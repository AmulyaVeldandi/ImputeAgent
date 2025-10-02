import random
import json
import re
from pathlib import Path

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

from src.llm.prompts import DECIDER_PROMPT, IMPUTER_PROMPT, CRITIC_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "openai-gpt-oss-20b"


class LocalLLMClient:
    _MODEL_CACHE = {}

    def __init__(self, backend="stub", model_name=None, enabled=True):
        self.enabled = enabled
        self.backend = backend
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.client = None

        if backend == "openai-oss" and enabled:
            if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
                raise RuntimeError("openai-oss backend requires transformers and torch. Install optional dependencies or use --llm stub")

            model_dir = Path(self.model_name) if self.model_name else DEFAULT_MODEL_DIR
            if not model_dir.exists():
                raise FileNotFoundError(
                    f"Local model directory not found at {model_dir}. "
                    "Place the manually downloaded weights/tokenizer files there or pass model_name to LocalLLMClient."
                )
            if not model_dir.is_dir():
                raise NotADirectoryError(f"Expected a directory for the model path, got {model_dir}")

            resolved_model = str(model_dir)
            cache_key = (backend, resolved_model)
            cached = LocalLLMClient._MODEL_CACHE.get(cache_key)
            if cached:
                self.tokenizer, self.model = cached
                print(f"[LLM] Reusing cached model: {resolved_model}")
            else:
                print(f"[LLM] Loading OpenAI OSS model from local path: {resolved_model}")
                tokenizer = AutoTokenizer.from_pretrained(resolved_model, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(
                    resolved_model,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    local_files_only=True
                )
                LocalLLMClient._MODEL_CACHE[cache_key] = (tokenizer, model)
                self.tokenizer, self.model = tokenizer, model
            self.model_name = resolved_model

        elif backend == "bedrock" and enabled:
            # TODO: AWS Bedrock integration with boto3
            self.client = None

    def _generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Helper: generate text from LLM backend"""
        if self.backend == "openai-oss":
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if self.backend == "bedrock":
            raise NotImplementedError("Bedrock integration needs boto3 setup.")

        return ""

    def _safe_json_parse(self, text: str, fallback: dict) -> dict:
        """Extract JSON object returned by the LLM; otherwise surface raw text."""
        candidates = []
        if text:
            matches = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
            if matches:
                candidates.extend(matches)
        for chunk in reversed(candidates):
            try:
                return json.loads(chunk)
            except Exception:
                continue
        cleaned = (text or "").strip()
        if cleaned:
            return {"raw_comment": cleaned}
        return fallback

    def decide(self, payload: dict) -> dict:
        """Decide which imputation strategy to use"""
        if not self.enabled or self.backend == "stub":
            if payload.get("col_type") == "categorical" and payload.get("cardinality", 0) >= 10:
                return {"decision": "LLM", "confidence": 0.75, "why": "high-card categorical (stub)"}
            return {"decision": "MODEL", "confidence": 0.8, "why": "default (stub)"}

        if self.backend == "openai-oss":
            col_type = payload.get("col_type", "unknown")
            stats = payload.get("column_stats", {})
            cardinality = payload.get("cardinality", 0)

            prompt = DECIDER_PROMPT + f"\nColumn type: {col_type}\nCardinality: {cardinality}\nStats: {stats}\n"
            text = self._generate(prompt)
            return self._safe_json_parse(
                text,
                {"decision": "MODEL", "confidence": 0.5, "why": "fallback"}
            )

    def impute(self, payload: dict) -> dict:
        """Suggest a plausible replacement value for missing data"""
        if not self.enabled or self.backend == "stub":
            stats = payload.get("column_stats", {})
            if "min" in stats and "max" in stats:
                v = 0.5 * (stats["min"] + stats["max"])
                v += random.uniform(-0.05, 0.05) * (stats["max"] - stats["min"])
                return {"value": round(v, 2), "confidence": 0.75, "justification": "centered value (stub)"}
            av = stats.get("allowed_values", ["0"])
            return {"value": av[0], "confidence": 0.8, "justification": "most frequent (stub)"}

        if self.backend == "openai-oss":
            stats = payload.get("column_stats", {})
            col_type = payload.get("column_type") or payload.get("col_type", "unknown")

            prompt = IMPUTER_PROMPT + f"\nColumn type: {col_type}\nStats: {stats}\n"
            text = self._generate(prompt)
            return self._safe_json_parse(
                text,
                {"value": None, "confidence": 0.5, "justification": "fallback"}
            )

    def critique(self, payload: dict) -> dict:
        """Evaluate a decision or imputation using an LLM Critic"""
        if not self.enabled or self.backend == "stub":
            return {
                "plausible": True,
                "confidence": 0.7,
                "risk": "low",
                "comment": "stub critic - assumes reasonable"
            }

        if self.backend == "openai-oss":
            prompt = CRITIC_PROMPT + f"\nDecision: {payload}\n"
            text = self._generate(prompt, max_new_tokens=128)
            return self._safe_json_parse(
                text,
                {"plausible": True, "confidence": 0.5, "risk": "medium", "comment": "fallback"}
            )
