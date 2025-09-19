from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np, pandas as pd

@dataclass
class Decision:
    mode: str
    confidence: float
    reason: str
    overrides: Dict[str, Any]

class Decider:
    def __init__(self, cfg: Dict, llm=None):
        self.cfg = cfg
        self.llm = llm

    # def decide_column(self, col_name: str, col_type: str, cardinality: int, p_mnar: float) -> Decision:
    #     model_first = (col_type in {"numeric","binary"}) and (p_mnar < self.cfg["p_mnar_switch"])
    #     mode = "MODEL" if model_first else ("LLM" if col_type=="categorical" and cardinality>=10 else "MODEL")
    #     conf = 0.75 if mode=="MODEL" else 0.7
    #     reason = f"default policy col_type={col_type}, card={cardinality}, p_mnar={p_mnar:.2f}"
    #     return Decision(mode=mode, confidence=conf, reason=reason, overrides={"cells":[]})
    def decide_column(self, col_name, col_type, cardinality, p_mnar, missing_frac):
        # default to MODEL
        mode = "MODEL"
        conf = 0.75
        reason = f"default: {col_type}, card={cardinality}, p_mnar={p_mnar:.2f}, frac={missing_frac:.2f}"

        if col_type == "numeric":
            if p_mnar >= 0.5 or missing_frac >= 0.3:
                mode, conf, reason = "LLM", 0.8, "numeric + high MNAR/missing → use LLM"
        elif col_type == "binary":
            if p_mnar >= 0.5 or (cardinality == 2 and missing_frac >= 0.2):
                mode, conf, reason = "LLM", 0.7, "binary rare/biased → use LLM"
        elif col_type == "categorical":
            if cardinality >= 10 or p_mnar >= 0.5:
                mode, conf, reason = "LLM", 0.8, "categorical high-card/MNAR → use LLM"
            else:
                mode, conf, reason = "MODEL", 0.75, "categorical low-card/MCAR → model good enough"

        return Decision(mode=mode, confidence=conf, reason=reason, overrides={"cells":[]})

    def decide_all(self, df_true: pd.DataFrame, df_missing: pd.DataFrame, target: str,
                numeric: List[str], categorical: List[str], mechanism_map: Dict[str,str], imputer) -> Dict[str,Decision]:
        decisions = {}
        for c in numeric+categorical:
            col_type = "numeric" if c in numeric else ("categorical" if c in categorical else "binary")
            card = int(df_true[c].nunique(dropna=True))
            p_mnar = 1.0 if mechanism_map.get(c)=="MNAR" else (0.6 if mechanism_map.get(c)=="MAR" else 0.3)
            missing_frac = df_missing[c].isna().mean()   # 🔹 add this line
            decisions[c] = self.decide_column(c, col_type, card, p_mnar, missing_frac)
        return decisions

