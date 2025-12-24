# build_cache_case4_label10.py
import os, glob, json, re
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from feature_engineering import BASE_FEATURES, add_zou_features, infer_numeric_cols

LABEL_COL = "label_10"
_pat = re.compile(r"snapshot_sym(\d+)_date(\d+)_(am|pm)\.csv$")

def list_csv_files(data_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "snapshot_sym*_date*_*.csv")))
    return files

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def csv_to_cached_arrays(csv_path: str, num_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)

    missing = [c for c in BASE_FEATURES + [LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    base = df[BASE_FEATURES].copy()
    y = df[LABEL_COL].to_numpy()
    y = np.nan_to_num(y, nan=1.0).astype(np.int8)  # 0/1/2

    feat = add_zou_features(base)  # 你case4的特征工程（会fillna(0)）
    for c in num_cols:
        if c not in feat.columns:
            feat[c] = 0.0
    X = feat[num_cols].to_numpy(dtype=np.float32)  # (T,F)

    return X, y

def main(
    data_dir: str,
    cache_dir: str,
    overwrite: bool = False,
):
    ensure_dir(cache_dir)

    files = list_csv_files(data_dir)
    if not files:
        raise RuntimeError(f"No csv files found under: {data_dir}")

    # 1) 用第一个文件推断 num_cols（数值列顺序固定！）
    df0 = pd.read_csv(files[0])
    base0 = df0[BASE_FEATURES].copy()
    feat0 = add_zou_features(base0)
    num_cols = infer_numeric_cols(feat0)

    meta = {
        "label": LABEL_COL,
        "window": 100,
        "num_cols": num_cols,
        "n_features": len(num_cols),
        "source": "case4 add_zou_features() cached per-file as (T,F) float32 + y int8",
    }
    with open(os.path.join(cache_dir, "cache_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 2) 逐文件缓存
    for fp in tqdm(files, desc="Caching", unit="file"):
        name = os.path.basename(fp).replace(".csv", "")
        x_path = os.path.join(cache_dir, f"{name}.X.npy")
        y_path = os.path.join(cache_dir, f"{name}.y.npy")

        if (not overwrite) and os.path.exists(x_path) and os.path.exists(y_path):
            continue

        X, y = csv_to_cached_arrays(fp, num_cols)
        np.save(x_path, X)
        np.save(y_path, y)

    print(f"[OK] cache_dir = {cache_dir}")
    print(f"[OK] meta      = {os.path.join(cache_dir, 'cache_meta.json')}")
    print(f"[OK] files     = {len(files)}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    main(args.data_dir, args.cache_dir, overwrite=args.overwrite)
