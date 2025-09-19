from src.llm.llm_client import LocalLLMClient

class Critic:
    def __init__(self, llm_client: LocalLLMClient = None):
        self.llm_client = llm_client

    def score_numeric(self, res: dict) -> float:
        auc = float(res.get("auc", 0.0))
        acc = float(res.get("avg_acc", 0.0))
        rmse = float(res.get("avg_rmse", 0.0) or 0.0)
        return 2.0 * auc + 1.0 * acc - 0.02 * rmse

    def score_llm(self, decision_payload: dict) -> dict:
        if self.llm_client:
            return self.llm_client.critique(decision_payload)
        return {
            "plausible": True,
            "confidence": 0.6,
            "risk": "medium",
            "comment": "LLM critic not enabled"
        }

    def evaluate(self, numeric_res: dict, decision_payload: dict) -> dict:
        return {
            "numeric_score": self.score_numeric(numeric_res),
            "llm_score": self.score_llm(decision_payload)
        }
