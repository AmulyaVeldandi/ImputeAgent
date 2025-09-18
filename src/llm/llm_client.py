import json, random

def get_llm_client(enabled: bool):
    return LocalLLMClient(enabled=enabled)

class LocalLLMClient:
    def __init__(self, enabled=True): self.enabled = enabled

    def decide(self, payload: dict) -> dict:
        if not self.enabled:
            return {"decision":"MODEL","confidence":0.8,"why":"LLM disabled"}
        if payload.get("col_type")=="categorical" and payload.get("cardinality",0)>=10:
            return {"decision":"LLM","confidence":0.75,"why":"high-card categorical"}
        return {"decision":"MODEL","confidence":0.8,"why":"default"}

    def impute(self, payload: dict) -> dict:
        if not self.enabled:
            return {"value": payload.get("column_stats",{}).get("mode","0"), "confidence":0.51, "justification":"stub"}
        stats = payload.get("column_stats",{})
        if "min" in stats and "max" in stats:
            v = 0.5*(stats["min"]+stats["max"])
            v += random.uniform(-0.05,0.05)*(stats["max"]-stats["min"])
            return {"value": round(v,2), "confidence":0.75, "justification":"centered value"}
        av = stats.get("allowed_values",["0"])
        return {"value": av[0], "confidence":0.8, "justification":"most frequent"}
