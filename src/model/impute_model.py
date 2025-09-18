import pandas as pd, numpy as np
from typing import Dict, List
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from ..utils.validators import validate_cell, column_constraints
from ..utils.metrics import imputation_errors, downstream_auc

class LocalImputer:
    def _fit_transform(self, df_missing, numeric, categorical, method: str):
        out = df_missing.copy()
        if method == "Mean":
            if numeric: out[numeric] = SimpleImputer(strategy="mean").fit_transform(out[numeric])
            if categorical: out[categorical] = SimpleImputer(strategy="most_frequent").fit_transform(out[categorical])
        elif method == "KNN":
            if numeric: out[numeric] = KNNImputer(n_neighbors=5).fit_transform(out[numeric])
            if categorical: out[categorical] = SimpleImputer(strategy="most_frequent").fit_transform(out[categorical])
        else:  # IterativeRF
            if numeric:
                it = IterativeImputer(
                    estimator=RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                    max_iter=10, random_state=42
                )
                out[numeric] = it.fit_transform(out[numeric])
            if categorical:
                out[categorical] = SimpleImputer(strategy="most_frequent").fit_transform(out[categorical])
        return out

    def _llm_fill(self, df_missing, full_df, target, numeric, categorical, columns: List[str], llm_client):
        out = df_missing.copy()
        for c in columns:
            cons = column_constraints(full_df, c)
            idx = out[c].index[out[c].isna()]
            for i in idx:
                row_ctx = out.loc[i, [col for col in out.columns if col != c]].to_dict()
                stats = cons
                payload = {"column": c, "row_context": row_ctx, "column_stats": stats, "constraints": cons}
                resp = llm_client.impute(payload)
                val = resp.get("value")
                if validate_cell(val, cons):
                    out.at[i, c] = val
                else:
                    if c in numeric:
                        out.at[i, c] = out[c].astype(float).mean()
                    else:
                        out.at[i, c] = out[c].mode(dropna=True).iloc[0] if out[c].dropna().size else 0
        return out

    def run_policy(self, df_true, df_missing, mask_df, target, numeric, categorical, policy: Dict, decisions: Dict, downstream="logistic"):
        per_col = policy["per_column"]
        mnar_shift = float(policy.get("mnar_shift", 0.0))
        model_method = "IterativeRF"
        model_missing = df_missing.copy()
        model_filled = self._fit_transform(model_missing, numeric, categorical, method=model_method)

        from ..llm.llm_client import get_llm_client
        llm_client = get_llm_client(enabled=True)

        use_llm_cols = [c for c,d in decisions.items() if d.mode=="LLM"]
        if use_llm_cols:
            model_filled = self._llm_fill(model_filled, df_true, target, numeric, categorical, use_llm_cols, llm_client)

        if abs(mnar_shift) > 1e-9:
            for c in numeric:
                std = float(pd.to_numeric(df_true[c], errors="coerce").dropna().std() or 0.0)
                if std>0:
                    model_filled[c] = model_filled[c].astype(float) + mnar_shift*std

        errs = imputation_errors(df_true, model_filled, mask_df, numeric, categorical)
        auc = downstream_auc(pd.concat([model_filled.drop(columns=[]), df_true[target]], axis=1), target, numeric, categorical)
        return {"auc": auc, **errs}

    def apply_policy_return_imputed(self, df_missing, target, numeric, categorical, policy, decisions):
        filled = self._fit_transform(df_missing, numeric, categorical, method="IterativeRF")
        from ..llm.llm_client import get_llm_client
        llm_client = get_llm_client(enabled=True)
        use_llm_cols = [c for c,d in decisions.items() if d.mode=="LLM"]
        if use_llm_cols:
            filled = self._llm_fill(filled, df_missing, target, numeric, categorical, use_llm_cols, llm_client)
        return filled
