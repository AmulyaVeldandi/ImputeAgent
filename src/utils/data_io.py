import pandas as pd, numpy as np
from typing import Iterator, Tuple, List

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, na_values=["NA","NaN",""])

def inject_missingness_grid(
    df: pd.DataFrame,
    target: str,
    numeric: List[str],
    categorical: List[str],
    types: List[str],
    fracs: List[float],
) -> Iterator[Tuple[str, float, pd.DataFrame, pd.DataFrame]]:
    rng = np.random.RandomState(42)
    feats = [c for c in df.columns if c != target]
    for t in types:
        for frac in fracs:
            out = df.copy()
            mask_all = pd.DataFrame(False, index=df.index, columns=feats)
            if t == "MCAR":
                for c in feats:
                    m = rng.rand(len(df)) < frac
                    out.loc[m, c] = np.nan
                    mask_all.loc[m, c] = True
            elif t == "MAR":
                driver = df[numeric].fillna(df[numeric].median()).sum(axis=1)
                p = (driver - driver.min())/(driver.max()-driver.min()+1e-9)
                for c in feats:
                    m = rng.rand(len(df)) < (frac*(0.5+0.5*p))
                    out.loc[m, c] = np.nan
                    mask_all.loc[m, c] = True
            else:
                for c in feats:
                    x = df[c]
                    if pd.api.types.is_numeric_dtype(x):
                        q = x.quantile(0.7)
                        candidates = x.index[x > q].tolist()
                    else:
                        candidates = x.dropna().index.tolist()
                    rng.shuffle(candidates)
                    k = int(frac*len(df))
                    idx = candidates[:k]
                    out.loc[idx, c] = np.nan
                    mask_all.loc[idx, c] = True
            yield t, frac, out, mask_all
