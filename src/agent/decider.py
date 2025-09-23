from dataclasses import dataclass
from typing import Dict, Any, List
import pandas as pd

from src.utils.validators import column_constraints


@dataclass
class Decision:
    mode: str
    confidence: float
    reason: str
    overrides: Dict[str, Any]


class Decider:
    """Choose imputation strategies per column, optionally delegating to an LLM."""

    def __init__(self, cfg: Dict, llm=None):
        self.cfg = cfg or {}
        self.llm = llm
        self._fallback_conf = float(self.cfg.get("default_confidence", 0.75))

    def _baseline_decision(self, col_name: str, col_type: str, cardinality: int,
                           p_mnar: float, missing_frac: float) -> Decision:
        """Heuristic fallback mirroring the original behaviour."""
        mode = "MODEL"
        conf = self._fallback_conf
        reason = (f"default: {col_type}, card={cardinality}, "
                  f"p_mnar={p_mnar:.2f}, frac={missing_frac:.2f}")

        if col_type == "numeric":
            if p_mnar >= 0.5 or missing_frac >= 0.3:
                mode, conf, reason = "LLM", 0.8, "numeric high-MNAR/missing -> use LLM"
        elif col_type == "binary":
            if p_mnar >= 0.5 or (cardinality == 2 and missing_frac >= 0.2):
                mode, conf, reason = "LLM", 0.7, "binary biased -> use LLM"
        elif col_type == "categorical":
            if cardinality >= 10 or p_mnar >= 0.5:
                mode, conf, reason = "LLM", 0.8, "categorical high-card/MNAR -> use LLM"
            else:
                mode, conf, reason = "MODEL", 0.75, "categorical low-card/MCAR -> model ok"

        return Decision(mode=mode, confidence=conf, reason=reason, overrides={"cells": []})

    def _build_payload(self, col_name: str, col_type: str, cardinality: int,
                       p_mnar: float, missing_frac: float, mechanism: str,
                       df_true: pd.DataFrame, df_missing: pd.DataFrame) -> Dict[str, Any]:
        stats = column_constraints(df_true, col_name)
        probe_size = int(self.cfg.get("llm_probe_size", 20))
        observed = df_missing[col_name].dropna().tolist()[:probe_size]
        sample = df_true[col_name].dropna().tolist()[:probe_size]
        return {
            "column": col_name,
            "col_type": col_type,
            "cardinality": cardinality,
            "mechanism": mechanism,
            "p_mnar": p_mnar,
            "missing_fraction": missing_frac,
            "column_stats": stats,
            "observed_samples": observed,
            "reference_samples": sample,
        }

    def decide_column(self, col_name: str, col_type: str, cardinality: int, p_mnar: float,
                      missing_frac: float, mechanism: str,
                      df_true: pd.DataFrame, df_missing: pd.DataFrame) -> Decision:
        baseline = self._baseline_decision(col_name, col_type, cardinality, p_mnar, missing_frac)

        if not self.llm or getattr(self.llm, "enabled", False) is False:
            return baseline

        payload = self._build_payload(col_name, col_type, cardinality, p_mnar, missing_frac,
                                      mechanism, df_true, df_missing)
        try:
            llm_response = self.llm.decide(payload) or {}
        except Exception as exc:  # defensive
            reason = f"LLM decide failed ({exc}); fallback"
            return Decision(mode=baseline.mode,
                            confidence=baseline.confidence,
                            reason=f"{baseline.reason} | {reason}",
                            overrides=baseline.overrides)

        mode = str(llm_response.get("decision", baseline.mode or "MODEL")).upper()
        allowed_modes = {"MODEL", "LLM", "KNN", "MEAN", "ITERATIVERF"}
        if mode not in allowed_modes:
            mode = baseline.mode

        confidence = llm_response.get("confidence")
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = baseline.confidence

        reason = llm_response.get("why") or llm_response.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            reason = baseline.reason + " | llm:fallback"

        overrides = llm_response.get("overrides")
        merged_overrides = baseline.overrides
        if isinstance(overrides, dict):
            merged_overrides = {**baseline.overrides, **overrides}

        return Decision(mode=mode, confidence=confidence, reason=reason, overrides=merged_overrides)

    def decide_all(self, df_true: pd.DataFrame, df_missing: pd.DataFrame, target: str,
                   numeric: List[str], categorical: List[str],
                   mechanism_map: Dict[str, str], imputer) -> Dict[str, Decision]:
        decisions: Dict[str, Decision] = {}
        for c in numeric + categorical:
            col_type = "numeric" if c in numeric else ("categorical" if c in categorical else "binary")
            card = int(df_true[c].nunique(dropna=True))
            mech = mechanism_map.get(c, "MCAR")
            p_mnar = 1.0 if mech == "MNAR" else (0.6 if mech == "MAR" else 0.3)
            missing_frac = df_missing[c].isna().mean()
            decisions[c] = self.decide_column(
                col_name=c,
                col_type=col_type,
                cardinality=card,
                p_mnar=p_mnar,
                missing_frac=missing_frac,
                mechanism=mech,
                df_true=df_true,
                df_missing=df_missing,
            )
        return decisions
