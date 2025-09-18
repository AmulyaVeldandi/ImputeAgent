import numpy as np, pandas as pd
from typing import Dict, List
from sklearn.linear_model import LogisticRegression

class MechanismDetector:
    def __init__(self, llm=None):
        self.llm = llm

    def detect(self, df: pd.DataFrame, target: str, numeric: List[str], categorical: List[str]) -> Dict[str,str]:
        mech = {}
        feats = [c for c in df.columns if c != target]
        for c in feats:
            y = df[c].isna().astype(int).values
            X = df.drop(columns=[c, target]).copy()
            X = pd.get_dummies(X, drop_first=True)
            if y.sum()==0 or y.sum()==len(y):
                mech[c] = "MCAR"
                continue
            try:
                lr = LogisticRegression(max_iter=1000, solver="liblinear")
                lr.fit(X.fillna(0), y)
                score = lr.score(X.fillna(0), y)
            except Exception:
                score = 0.5
            skew = abs(pd.to_numeric(df[c], errors="coerce")).dropna().skew() if pd.api.types.is_numeric_dtype(df[c]) else 0.0
            if score < 0.55:
                mech[c] = "MCAR"
            elif skew > 1.0:
                mech[c] = "MNAR"
            else:
                mech[c] = "MAR"
        return mech
