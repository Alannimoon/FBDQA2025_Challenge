# preprocess.py
import numpy as np
import pandas as pd
from feature_cols import SIZE_COLS

def time_to_seconds(t: pd.Series) -> pd.Series:
    parts = t.astype(str).str.split(":", expand=True)
    hh = parts[0].astype(int)
    mm = parts[1].astype(int)
    ss = parts[2].astype(int)
    return hh * 3600 + mm * 60 + ss

def exp_log_transform_sizes(x: np.ndarray) -> np.ndarray:
    """
    原文：exp(100*x)-1
    这里做一个温和clip，防止极端值溢出（实测一般不会影响）
    """
    x = np.clip(x, -0.5, 0.5)  # 防溢出
    return np.expm1(100.0 * x)

def preprocess_exp_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    - size列：exp(100*x)-1
    - amount_delta：log1p
    - 报价类不变
    - time转成 time_sec + time_interval
    """
    out = df.copy()

    sec = time_to_seconds(out["time"])
    out["time_sec"] = sec.astype(np.int32)
    out["time_interval"] = (sec // 1800).astype(np.int16)

    out["amount_delta"] = np.log1p(out["amount_delta"].astype(np.float64)).astype(np.float32)

    for c in SIZE_COLS:
        out[c] = exp_log_transform_sizes(out[c].astype(np.float64).to_numpy()).astype(np.float32)

    # 清理
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out
