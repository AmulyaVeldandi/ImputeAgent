import pandas as pd
from typing import Dict, List
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor

from src.agent.decider import Decision
from ..utils.validators import validate_cell, column_constraints
from ..utils.metrics import imputation_errors, downstream_auc


class LocalImputer:
    def __init__(self, llm_client=None):
        # Pass in the LLM client from run.py so backend choice is respected
        self.llm_client = llm_client
        self._llm_cache: Dict = {}

    def _canonical_method(self, name: str) -> str:
        if not name:
            return "IterativeRF"
        key = str(name).strip().upper()
        if key == "LLM":
            return "LLM"
        if key == "KNN":
            return "KNN"
        if key == "MEAN":
            return "Mean"
        return "IterativeRF"

    def _resolve_methods(self, numeric: List[str], categorical: List[str],
                         policy: Dict, decisions: Dict) -> Dict[str, str]:
        per_policy = policy.get("per_column", {}) if policy else {}
        resolved: Dict[str, str] = {}
        for col in numeric + categorical:
            base_choice = per_policy.get(col, "IterativeRF")
            decision_obj = decisions.get(col) if decisions else None
            decision_mode = None
            overrides = None

            if isinstance(decision_obj, Decision):
                decision_mode = decision_obj.mode
                overrides = decision_obj.overrides
            elif isinstance(decision_obj, dict):
                decision_mode = decision_obj.get("decision")
                overrides = decision_obj.get("overrides")

            method_override = None
            if isinstance(overrides, dict):
                method_override = overrides.get("method")

            choice = base_choice
            if method_override:
                choice = method_override
            elif decision_mode and str(decision_mode).strip().upper() not in {"", "MODEL"}:
                choice = decision_mode

            resolved[col] = self._canonical_method(choice)
        return resolved

    def _impute_numeric_block(self, df: pd.DataFrame, cols: List[str], method: str) -> pd.DataFrame:
        if not cols:
            return pd.DataFrame(index=df.index, columns=cols)
        if method == "Mean":
            imputer = SimpleImputer(strategy="mean")
        elif method == "KNN":
            imputer = KNNImputer(n_neighbors=5)
        else:  # IterativeRF fallback
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                max_iter=10,
                random_state=42,
            )
        transformed = imputer.fit_transform(df[cols])
        return pd.DataFrame(transformed, index=df.index, columns=cols)

    def _impute_categorical_block(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        if not cols:
            return pd.DataFrame(index=df.index, columns=cols)
        imputer = SimpleImputer(strategy="most_frequent")
        transformed = imputer.fit_transform(df[cols])
        return pd.DataFrame(transformed, index=df.index, columns=cols)

    def _llm_fill(self, df_current: pd.DataFrame, reference_df: pd.DataFrame,
                  mask_df: pd.DataFrame, numeric: List[str], categorical: List[str],
                  columns: List[str], decisions: Dict) -> pd.DataFrame:
        out = df_current.copy()
        for col in columns:
            cons = column_constraints(reference_df, col)
            source_mask = mask_df[col].fillna(False) if mask_df is not None else out[col].isna()
            indices = source_mask[source_mask].index
            decision_obj = decisions.get(col) if decisions else None
            decision_ctx = {}
            if isinstance(decision_obj, Decision):
                decision_ctx = {
                    "mode": decision_obj.mode,
                    "confidence": decision_obj.confidence,
                    "reason": decision_obj.reason,
                    "overrides": decision_obj.overrides,
                }
            elif isinstance(decision_obj, dict):
                decision_ctx = {
                    "mode": decision_obj.get("decision"),
                    "confidence": decision_obj.get("confidence"),
                    "reason": decision_obj.get("why") or decision_obj.get("reason"),
                    "overrides": decision_obj.get("overrides"),
                }

            for idx in indices:
                existing = out.at[idx, col]
                row_ctx = reference_df.loc[idx, [c for c in reference_df.columns if c != col]].to_dict()
                cache_key = (col, idx, tuple(sorted(row_ctx.items())))
                cached_value = self._llm_cache.get(cache_key)
                if cached_value is not None:
                    out.at[idx, col] = cached_value
                    continue

                payload = {
                    "column": col,
                    "column_type": "categorical" if col in categorical else ("numeric" if col in numeric else "other"),
                    "row_context": row_ctx,
                    "column_stats": cons,
                    "constraints": cons,
                    "decision": decision_ctx,
                }
                if "allowed_values" in cons:
                    payload["allowed_values"] = cons["allowed_values"]
                if "min" in cons and "max" in cons:
                    payload["numeric_range"] = {"min": cons["min"], "max": cons["max"]}

                response = self.llm_client.impute(payload) if self.llm_client else {"value": None}
                value = response.get("value") if isinstance(response, dict) else None

                if value is not None and pd.api.types.is_numeric_dtype(reference_df[col]):
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        pass

                if validate_cell(value, cons):
                    out.at[idx, col] = value
                    self._llm_cache[cache_key] = value
                else:
                    print(f"[WARN] LLM produced invalid value for {col} at index {idx}: {response}")
                    if "allowed_values" in cons and cons["allowed_values"]:
                        fallback = cons["allowed_values"][0]
                    elif "min" in cons and "max" in cons:
                        fallback = 0.5 * (float(cons["min"]) + float(cons["max"]))
                    else:
                        fallback = existing
                    out.at[idx, col] = fallback
                    self._llm_cache[cache_key] = fallback
        return out

    def _materialize_policy(self, df_source: pd.DataFrame, reference_df: pd.DataFrame,
                             mask_df: pd.DataFrame, numeric: List[str], categorical: List[str],
                             policy: Dict, decisions: Dict) -> pd.DataFrame:
        resolved = self._resolve_methods(numeric, categorical, policy, decisions)
        filled = df_source.copy()

        numeric_methods = {resolved.get(col, "IterativeRF") for col in numeric}
        for method in sorted(numeric_methods):
            if method == "LLM":
                continue
            cols = [c for c in numeric if resolved.get(c, "IterativeRF") == method]
            block = self._impute_numeric_block(filled, cols, method)
            for col in cols:
                filled[col] = block[col]

        categorical_methods = {resolved.get(col, "IterativeRF") for col in categorical}
        for method in sorted(categorical_methods):
            if method == "LLM":
                continue
            cols = [c for c in categorical if resolved.get(c, "IterativeRF") == method]
            block = self._impute_categorical_block(filled, cols)
            for col in cols:
                filled[col] = block[col]

        llm_columns = [c for c, choice in resolved.items() if choice == "LLM"]
        if self.llm_client and llm_columns:
            filled = self._llm_fill(filled, reference_df, mask_df, numeric, categorical, llm_columns, decisions)

        return filled

    def run_policy(self, df_true, df_missing, mask_df, target, numeric, categorical,
                   policy: Dict, decisions: Dict, downstream="logistic"):
        reference = df_true if df_true is not None else df_missing
        filled = self._materialize_policy(df_missing.copy(), reference, mask_df, numeric, categorical, policy, decisions)

        mnar_shift = float(policy.get("mnar_shift", 0.0)) if policy else 0.0
        if abs(mnar_shift) > 1e-9:
            for col in numeric:
                std = float(pd.to_numeric(df_true[col], errors="coerce").dropna().std() or 0.0) if df_true is not None else 0.0
                if std > 0:
                    filled[col] = pd.to_numeric(filled[col], errors="coerce") + mnar_shift * std

        nan_counts = filled.isna().sum()
        if nan_counts.any():
            remaining = {k: int(v) for k, v in nan_counts[nan_counts > 0].items()}
            print(f"[WARN] NaNs remain before downstream model: {remaining}")

        errs = imputation_errors(df_true, filled, mask_df, numeric, categorical)
        auc = downstream_auc(filled, target, numeric, categorical)
        return {"auc": auc, **errs}

    def apply_policy_return_imputed(self, df_missing, target, numeric, categorical, policy, decisions):
        mask_df = df_missing.isna()
        filled = self._materialize_policy(df_missing.copy(), df_missing, mask_df, numeric, categorical, policy, decisions)
        return filled
