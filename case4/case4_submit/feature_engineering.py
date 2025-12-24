# feature_engineering.py
import numpy as np
import pandas as pd

BASE_FEATURES = [
    "date","time","sym","n_close","amount_delta","n_midprice",
    "n_bid1","n_bsize1","n_bid2","n_bsize2","n_bid3","n_bsize3","n_bid4","n_bsize4","n_bid5","n_bsize5",
    "n_ask1","n_asize1","n_ask2","n_asize2","n_ask3","n_asize3","n_ask4","n_asize4","n_ask5","n_asize5",
]

def _time_to_seconds(t: pd.Series) -> pd.Series:
    parts = t.astype(str).str.split(":", expand=True)
    hh = parts[0].astype(int)
    mm = parts[1].astype(int)
    ss = parts[2].astype(int)
    return hh * 3600 + mm * 60 + ss

def add_zou_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    对“一个文件的一整段”做一次特征工程（训练用）；
    推理时对“100行窗口”做一次同样的特征工程（推理用）。
    """
    df = df_raw.copy()
    eps = 1e-12

    # time bucket
    sec = _time_to_seconds(df["time"])
    df["time_sec"] = sec
    df["time_interval"] = (sec // 1800).astype(np.int16)

    # amount log1p
    df["amount"] = np.log1p(df["amount_delta"].astype(float))

    # L1~L5 features
    for i in range(1, 6):
        bid = df[f"n_bid{i}"].astype(float)
        ask = df[f"n_ask{i}"].astype(float)

        bsz_raw = df[f"n_bsize{i}"].astype(float)
        asz_raw = df[f"n_asize{i}"].astype(float)

        # overwrite with log1p size
        df[f"n_bsize{i}"] = np.log1p(bsz_raw)
        df[f"n_asize{i}"] = np.log1p(asz_raw)

        bsz = df[f"n_bsize{i}"].astype(float)
        asz = df[f"n_asize{i}"].astype(float)
        denom = (bsz + asz).replace(0.0, np.nan)

        df[f"spread_{i}"] = ask - bid
        df[f"mid_price{i}"] = (ask + bid) / 2.0
        df[f"relative_bid_density{i}"] = bsz / (denom + eps)
        df[f"relative_ask_density{i}"] = asz / (denom + eps)
        df[f"weighted_ab{i}"] = (bid * asz + ask * bsz) / (denom + eps)

    # vol rel diff
    b1, a1 = df["n_bsize1"].astype(float), df["n_asize1"].astype(float)
    df["vol1_rel_diff"] = (b1 - a1) / (b1 + a1 + eps)

    b3 = df["n_bsize1"] + df["n_bsize2"] + df["n_bsize3"]
    a3 = df["n_asize1"] + df["n_asize2"] + df["n_asize3"]
    df["vol3_rel_diff"] = (b3 - a3) / (b3 + a3 + eps)

    b5 = df["n_bsize1"] + df["n_bsize2"] + df["n_bsize3"] + df["n_bsize4"] + df["n_bsize5"]
    a5 = df["n_asize1"] + df["n_asize2"] + df["n_asize3"] + df["n_asize4"] + df["n_asize5"]
    df["vol5_rel_diff"] = (b5 - a5) / (b5 + a5 + eps)

    # diffs
    df["close_delta1"] = df["n_close"].astype(float).diff()
    for i in range(1, 6):
        df[f"n_bid{i}_delta1"] = df[f"n_bid{i}"].astype(float).diff()
        df[f"n_ask{i}_delta1"] = df[f"n_ask{i}"].astype(float).diff()
        df[f"mid_price_delta{i}"] = df[f"mid_price{i}"].astype(float).diff()
        df[f"n_bsize{i}_diff"] = df[f"n_bsize{i}"].astype(float).diff()
        df[f"n_asize{i}_diff"] = df[f"n_asize{i}"].astype(float).diff()

    # rolling mean/std/vs_mean
    w = 10
    df = df.copy() 
    df["close_mean"] = df["n_close"].astype(float).rolling(window=w).mean()
    df["close_std"]  = df["n_close"].astype(float).rolling(window=w).std()
    df["close_vs_mean"] = df["n_close"].astype(float) / (df["close_mean"] + eps)

    for i in range(1, 6):
        for col in [f"n_bid{i}", f"n_ask{i}", f"n_bsize{i}", f"n_asize{i}", f"mid_price{i}"]:
            s = df[col].astype(float)
            m = s.rolling(window=w).mean()
            sd = s.rolling(window=w).std()
            df[f"{col}_mean"] = m
            df[f"{col}_std"] = sd
            df[f"{col}_vs_mean"] = s / (m + eps)

    # clean
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

def infer_numeric_cols(df_feat: pd.DataFrame) -> list[str]:
    # 固定数值列顺序（训练端保存，推理端加载）
    return df_feat.select_dtypes(include=[np.number]).columns.tolist()

def window_to_vector_with_cols(df_window_raw: pd.DataFrame, num_cols: list[str]) -> np.ndarray:
    df_feat = add_zou_features(df_window_raw)

    # 缺列补0（极少发生，但要防御）
    for c in num_cols:
        if c not in df_feat.columns:
            df_feat[c] = 0.0

    mat = df_feat[num_cols].to_numpy(dtype=np.float32)  # (100, F)
    return mat.reshape(-1)  # (100*F,)
