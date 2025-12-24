# build_cache_label10.py
import os, json
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset import list_snapshot_files  # 你case5里已有
from feature_cols import BASE_FEATURES
from preprocess import preprocess_exp_log

LABEL_COL = "label_10"

def build_cache(data_dir: str, cache_dir: str, label_col: str = LABEL_COL):
    os.makedirs(cache_dir, exist_ok=True)

    files = list_snapshot_files(data_dir)
    if not files:
        raise RuntimeError("No snapshot files found under data_dir.")

    # 用第一个文件确定 feature_cols（数值列顺序必须固定）
    first = pd.read_csv(files[0])
    base0 = first[BASE_FEATURES].copy()
    feat0 = preprocess_exp_log(base0)
    feature_cols = [c for c in feat0.columns if c != "time"]  # time字符串不要

    meta = {
        "label": label_col,
        "feature_cols": feature_cols,
        "preprocess": "exp_log (size: expm1(100*x), amount_delta: log1p, add time_sec/time_interval)",
        "base_features": BASE_FEATURES,
        "n_features": len(feature_cols),
    }
    with open(os.path.join(cache_dir, "cache_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 逐文件缓存
    for fp in tqdm(files, desc="Caching", unit="file"):
        name = os.path.basename(fp).replace(".csv", "")
        x_path = os.path.join(cache_dir, f"{name}.X.npy")
        y_path = os.path.join(cache_dir, f"{name}.y.npy")

        # 已缓存跳过
        if os.path.exists(x_path) and os.path.exists(y_path):
            continue

        df = pd.read_csv(fp)
        missing = [c for c in BASE_FEATURES + [label_col] if c not in df.columns]
        if missing:
            raise ValueError(f"{fp} missing columns: {missing}")

        base = df[BASE_FEATURES].copy()
        y = df[label_col].to_numpy()
        y = np.nan_to_num(y, nan=1.0).astype(np.int64)  # nan -> 1

        feat = preprocess_exp_log(base)
        # 对齐列
        for c in feature_cols:
            if c not in feat.columns:
                feat[c] = 0.0
        X = feat[feature_cols].to_numpy(dtype=np.float32)

        np.save(x_path, X)
        np.save(y_path, y.astype(np.int8))  # 0/1/2 用 int8 足够

    print(f"[OK] cache saved in: {cache_dir}")
    print(f"[OK] meta: {os.path.join(cache_dir, 'cache_meta.json')}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--label_col", default=LABEL_COL)
    args = ap.parse_args()
    build_cache(args.data_dir, args.cache_dir, args.label_col)
