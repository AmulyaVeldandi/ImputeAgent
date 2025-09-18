import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

def imputation_errors(true_df, imputed_df, mask_df, numeric, categorical):
    out = {}
    for c in numeric:
        m = mask_df[c].fillna(False).values
        if m.any():
            err = float(np.sqrt(np.mean((true_df.loc[m,c].astype(float) - imputed_df.loc[m,c].astype(float))**2)))
            out[f"rmse_{c}"] = err
    for c in categorical:
        m = mask_df[c].fillna(False).values
        if m.any():
            acc = float(np.mean(true_df.loc[m,c].astype(str).values == imputed_df.loc[m,c].astype(str).values))
            out[f"acc_{c}"] = acc
    rmse_vals = [v for k,v in out.items() if k.startswith("rmse_")]
    acc_vals = [v for k,v in out.items() if k.startswith("acc_")]
    out["avg_rmse"] = float(np.mean(rmse_vals)) if rmse_vals else np.nan
    out["avg_acc"] = float(np.mean(acc_vals)) if acc_vals else np.nan
    return out

def downstream_auc(df, target, numeric, categorical):
    X = df[numeric+categorical].copy()
    y = df[target].astype(int).values
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numeric),
    ])
    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:,1]
    auc = roc_auc_score(y, proba)
    return float(auc)
