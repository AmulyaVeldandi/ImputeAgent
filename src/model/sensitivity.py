from typing import List, Dict
import pandas as pd, numpy as np
from .impute_model import LocalImputer
from ..utils.metrics import downstream_auc, imputation_errors

def run_sensitivity(df_true, df_missing, mask_df, target, numeric, categorical, policy, decisions, deltas, imputer: LocalImputer):
    rows = []
    base = imputer.apply_policy_return_imputed(df_missing, target, numeric, categorical, policy, decisions)
    for d in deltas:
        shifted = base.copy()
        for c in numeric:
            std = float(pd.to_numeric(df_true[c], errors="coerce").dropna().std() or 0.0)
            if std>0: shifted[c] = shifted[c].astype(float) + d*std
        auc = downstream_auc(shifted, target, numeric, categorical)
        errs = imputation_errors(df_true, shifted, mask_df, numeric, categorical)
        rows.append({"delta": d, "auc": auc, **errs})
    return rows
