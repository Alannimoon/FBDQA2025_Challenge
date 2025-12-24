# train_label10_xgb_stream_cached.py
import os, glob, json, re
from typing import List, Tuple, Dict, Iterator

import numpy as np
import xgboost as xgb
from tqdm import tqdm

LABEL_COL = "label_10"
WINDOW = 100

_pat = re.compile(r"snapshot_sym(\d+)_date(\d+)_(am|pm)\.csv$")

def parse_sym_date_sess_from_name(name: str) -> Tuple[int, int, int]:
    # name like snapshot_sym0_date12_am
    m = _pat.search(name + ".csv") if not name.endswith(".csv") else _pat.search(name)
    if not m:
        return (10**9, 10**9, 0)
    sym = int(m.group(1))
    date = int(m.group(2))
    sess = 0 if m.group(3) == "am" else 1
    return sym, date, sess

def list_cached_pairs(cache_dir: str) -> List[Tuple[str, str, str]]:
    # returns (stem, X_path, y_path)
    Xs = sorted(glob.glob(os.path.join(cache_dir, "snapshot_sym*_date*_*.X.npy")))
    pairs = []
    for xp in Xs:
        stem = os.path.basename(xp).replace(".X.npy", "")
        yp = os.path.join(cache_dir, f"{stem}.y.npy")
        if os.path.exists(yp):
            pairs.append((stem, xp, yp))
    pairs.sort(key=lambda t: parse_sym_date_sess_from_name(t[0]))
    return pairs

def split_by_sym_date(stems: List[str], val_ratio: float = 0.2) -> Tuple[List[str], List[str]]:
    by_sym: Dict[int, Dict[int, List[str]]] = {}
    for s in stems:
        sym, date, _ = parse_sym_date_sess_from_name(s)
        by_sym.setdefault(sym, {}).setdefault(date, []).append(s)

    train, val = [], []
    for sym, by_date in by_sym.items():
        dates = sorted(by_date.keys())
        k = max(1, int(round(len(dates) * val_ratio)))
        val_dates = set(dates[-k:])
        for d in dates:
            (val if d in val_dates else train).extend(by_date[d])

    train.sort(key=parse_sym_date_sess_from_name)
    val.sort(key=parse_sym_date_sess_from_name)
    return train, val

def iter_batches_from_cache(
    cache_dir: str,
    stems: List[str],
    batch_size: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    X_buf, y_buf = [], []

    for stem in tqdm(stems, desc="Reading cached", unit="file"):
        X = np.load(os.path.join(cache_dir, f"{stem}.X.npy"), mmap_mode="r")  # (T,F)
        y = np.load(os.path.join(cache_dir, f"{stem}.y.npy"), mmap_mode="r")  # (T,)
        T = int(y.shape[0])
        if T < WINDOW:
            continue

        # window flatten
        for t in range(WINDOW - 1, T):
            X_buf.append(np.asarray(X[t-(WINDOW-1):t+1], dtype=np.float32).reshape(-1))
            y_buf.append(int(y[t]))

            if len(X_buf) >= batch_size:
                yield np.stack(X_buf, axis=0), np.asarray(y_buf, dtype=np.int32)
                X_buf, y_buf = [], []

    if X_buf:
        yield np.stack(X_buf, axis=0), np.asarray(y_buf, dtype=np.int32)

class NumpyBatchIter(xgb.core.DataIter):
    def __init__(self, cache_dir: str, stems: List[str], batch_size: int):
        super().__init__()
        self.cache_dir = cache_dir
        self.stems = stems
        self.batch_size = batch_size
        self._it = None

    def reset(self):
        self._it = iter_batches_from_cache(self.cache_dir, self.stems, self.batch_size)

    def next(self, input_data):
        try:
            X, y = next(self._it)
        except StopIteration:
            return 0
        input_data(data=X, label=y)
        return 1

def main(
    cache_dir: str,
    batch_size: int = 512,
    out_model: str = "xgb_label10_cached.json",
    out_meta: str = "meta_label10_cached.json",
    nthread: int = 32,
    max_bin: int = 64,
    verbose_eval: int = 1,
):
    meta_path = os.path.join(cache_dir, "cache_meta.json")
    if not os.path.exists(meta_path):
        raise RuntimeError(f"Missing {meta_path}. Run build_cache_case4_label10.py first.")
    with open(meta_path, "r", encoding="utf-8") as f:
        cache_meta = json.load(f)

    pairs = list_cached_pairs(cache_dir)
    if not pairs:
        raise RuntimeError(f"No cached *.X.npy found under: {cache_dir}")

    stems = [p[0] for p in pairs]
    train_stems, val_stems = split_by_sym_date(stems, val_ratio=0.2)
    print(f"[Cache] total={len(stems)} train={len(train_stems)} val={len(val_stems)}")

    # 写训练meta
    meta = {
        "label": LABEL_COL,
        "window": WINDOW,
        "num_cols": cache_meta["num_cols"],
        "n_features": cache_meta["n_features"],
        "batch_size_train": batch_size,
        "max_bin": max_bin,
        "split": "by sym, last 20% dates as val",
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    train_iter = NumpyBatchIter(cache_dir, train_stems, batch_size=batch_size)
    val_iter   = NumpyBatchIter(cache_dir, val_stems,   batch_size=batch_size)

    # 关键：max_bin 一致；val 必须 ref=dtrain
    dtrain = xgb.QuantileDMatrix(train_iter, max_bin=max_bin)
    dval   = xgb.QuantileDMatrix(val_iter, ref=dtrain, max_bin=max_bin)

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "nthread": nthread,
        "max_bin": max_bin,   # ✅ 必须
    }

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=verbose_eval,
    )

    bst.save_model(out_model)
    print(f"[OK] Saved model -> {out_model}")
    print(f"[OK] Saved meta  -> {out_meta}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--out_model", default="xgb_label10_cached.json")
    ap.add_argument("--out_meta", default="meta_label10_cached.json")
    ap.add_argument("--nthread", type=int, default=32)
    ap.add_argument("--max_bin", type=int, default=64)
    ap.add_argument("--verbose_eval", type=int, default=10)
    args = ap.parse_args()

    main(
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        out_model=args.out_model,
        out_meta=args.out_meta,
        nthread=args.nthread,
        max_bin=args.max_bin,
        verbose_eval=args.verbose_eval,
    )
