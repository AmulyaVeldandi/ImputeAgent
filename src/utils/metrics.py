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
    X = df.drop(columns=[target])
    y = df[target]

    # Ensure y is a 1D array/series
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 1:
            y = y.iloc[:, 0]
        else:
            raise ValueError(f"Target column '{target}' returned multiple columns: {y.columns.tolist()}")

    pipe = Pipeline([
        ("pre", ColumnTransformer([
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
        ])),
        ("clf", LogisticRegression(max_iter=500))
    ])
    pipe.fit(X, y)
    y_pred = pipe.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_pred)

