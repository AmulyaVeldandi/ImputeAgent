import pandas as pd
from typing import Any, Dict

def validate_cell(value: Any, constraints: Dict) -> bool:
    if value is None: return False
    if "allowed_values" in constraints:
        return str(value) in set(map(str, constraints["allowed_values"]))
    if "min" in constraints and "max" in constraints:
        try:
            v = float(value)
            return constraints["min"] <= v <= constraints["max"]
        except Exception:
            return False
    return True

def column_constraints(df: pd.DataFrame, col: str):
    s = df[col].dropna()
    if s.empty:
        return {}
    if pd.api.types.is_numeric_dtype(s):
        return {"min": float(s.min()), "max": float(s.max())}
    else:
        return {"allowed_values": sorted(map(str, s.unique().tolist()))}
