# import json, random

# def get_llm_client(enabled: bool):
#     return LocalLLMClient(enabled=enabled)

# class LocalLLMClient:
#     def __init__(self, enabled=True): self.enabled = enabled

#     def decide(self, payload: dict) -> dict:
#         if not self.enabled:
#             return {"decision":"MODEL","confidence":0.8,"why":"LLM disabled"}
#         if payload.get("col_type")=="categorical" and payload.get("cardinality",0)>=10:
#             return {"decision":"LLM","confidence":0.75,"why":"high-card categorical"}
#         return {"decision":"MODEL","confidence":0.8,"why":"default"}

#     def impute(self, payload: dict) -> dict:
#         if not self.enabled:
#             return {"value": payload.get("column_stats",{}).get("mode","0"), "confidence":0.51, "justification":"stub"}
#         stats = payload.get("column_stats",{})
#         if "min" in stats and "max" in stats:
#             v = 0.5*(stats["min"]+stats["max"])
#             v += random.uniform(-0.05,0.05)*(stats["max"]-stats["min"])
#             return {"value": round(v,2), "confidence":0.75, "justification":"centered value"}
#         av = stats.get("allowed_values",["0"])
#         return {"value": av[0], "confidence":0.8, "justification":"most frequent"}

import random
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.llm.prompts import DECIDER_PROMPT, IMPUTER_PROMPT, CRITIC_PROMPT

class LocalLLMClient:
    def __init__(self, backend="stub", model_name=None, enabled=True):
        self.enabled = enabled
        self.backend = backend
        self.model_name = model_name or "openai/gpt-oss-20b"

        if backend == "openai-oss" and enabled:
            print(f"🔹 Loading OpenAI OSS model locally: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

        elif backend == "bedrock" and enabled:
            # TODO: AWS Bedrock integration with boto3
            self.client = None

    def _generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Helper: generate text from LLM backend"""
        if self.backend == "openai-oss":
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        elif self.backend == "bedrock":
            raise NotImplementedError("Bedrock integration needs boto3 setup.")

        else:
            return ""

    def _safe_json_parse(self, text: str, fallback: dict) -> dict:
        """Try to parse JSON output safely, fallback if needed"""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception:
            pass
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
            col_type = payload.get("col_type", "unknown")

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