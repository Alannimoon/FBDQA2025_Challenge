# dataset.py
import os, glob, re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from feature_cols import BASE_FEATURES
from preprocess import preprocess_exp_log

_pat = re.compile(r"snapshot_sym(\d+)_date(\d+)_(am|pm)\.csv$")

def parse_sym_date_sess(path: str) -> Tuple[int,int,int]:
    name = os.path.basename(path)
    m = _pat.search(name)
    if not m:
        return (10**9, 10**9, 0)
    sym = int(m.group(1))
    date = int(m.group(2))
    sess = 0 if m.group(3) == "am" else 1
    return sym, date, sess

def list_snapshot_files(data_dir: str) -> List[str]:
    files = glob.glob(os.path.join(data_dir, "snapshot_sym*_date*_*.csv"))
    files.sort(key=parse_sym_date_sess)
    return files

def split_files_by_date(files: List[str], test_start: int, test_end: int, val_days: int = 10):
    """
    时间切分（贴近原文）：
    - test: [test_start, test_end] 这些日期（含 am/pm 全部文件）
    - val : test_start 前面的 val_days 天
    - train: 剩下更早的
    """
    # 收集每个sym的日期集合（按sym内时间切）
    by_sym_dates: Dict[int, List[int]] = {}
    by_sym_by_date: Dict[int, Dict[int, List[str]]] = {}

    for fp in files:
        sym, date, _ = parse_sym_date_sess(fp)
        by_sym_by_date.setdefault(sym, {}).setdefault(date, []).append(fp)

    train, val, test = [], [], []

    for sym, by_date in by_sym_by_date.items():
        dates = sorted(by_date.keys())
        # test日期固定
        test_dates = [d for d in dates if test_start <= d <= test_end]
        # val日期：test_start之前的最后val_days天
        prev_dates = [d for d in dates if d < test_start]
        val_dates = prev_dates[-val_days:] if len(prev_dates) >= 1 else []

        for d in dates:
            fps = by_date[d]
            if d in test_dates:
                test.extend(fps)
            elif d in val_dates:
                val.extend(fps)
            else:
                # 只用早于test_start的作为train，避免未来信息
                if d < test_start:
                    train.extend(fps)

    train.sort(key=parse_sym_date_sess)
    val.sort(key=parse_sym_date_sess)
    test.sort(key=parse_sym_date_sess)
    return train, val, test

@dataclass
class FileCacheItem:
    X: np.ndarray  # (T, F)
    y: np.ndarray  # (T,)
    feature_cols: List[str]

class SnapshotWindowDataset(Dataset):
    def __init__(
        self,
        files: List[str],
        label_col: str = "label_10",
        window: int = 100,
        cache_files: int = 8,
    ):
        self.files = files
        self.label_col = label_col
        self.window = window
        self.cache_files = cache_files

        # index: list of (file_idx, t_end)
        self.index: List[Tuple[int,int]] = []

        # LRU cache
        self._cache: Dict[int, FileCacheItem] = {}
        self._lru: List[int] = []

        # 先用第一个文件确定特征列顺序（固定训练/推理一致）
        if len(self.files) == 0:
            raise ValueError("No files provided to dataset.")

        first = self._load_file(0)
        self.feature_cols = first.feature_cols
        self.n_features = len(self.feature_cols)

        # 构建全体样本索引（只存索引，不存X）
        for fi in range(len(self.files)):
            item = self._load_file(fi)
            T = item.X.shape[0]
            if T < self.window:
                continue
            # t_end 从 window-1 到 T-1，y用 y[t_end]
            for t in range(self.window - 1, T):
                self.index.append((fi, t))

        # 建完索引后，清空cache，避免占内存（训练时再懒加载）
        self._cache.clear()
        self._lru.clear()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        fi, t = self.index[idx]
        item = self._load_file(fi)
        Xw = item.X[t - (self.window - 1): t + 1]  # (window, F)
        y = int(item.y[t])
        return Xw.astype(np.float32), y

    def _touch_lru(self, fi: int):
        if fi in self._lru:
            self._lru.remove(fi)
        self._lru.append(fi)
        # evict
        while len(self._lru) > self.cache_files:
            old = self._lru.pop(0)
            if old in self._cache:
                del self._cache[old]

    def _load_file(self, fi: int) -> FileCacheItem:
        if fi in self._cache:
            self._touch_lru(fi)
            return self._cache[fi]

        fp = self.files[fi]
        df = pd.read_csv(fp)

        # 只保留需要列
        need = BASE_FEATURES + [self.label_col]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"{fp} missing columns: {missing}")

        base = df[BASE_FEATURES].copy()
        y = df[self.label_col].to_numpy()
        y = np.nan_to_num(y, nan=1.0).astype(np.int64)

        # 预处理（exp_log）
        base2 = preprocess_exp_log(base)

        # 选择数值特征列：date/sym/time_sec/time_interval + 价格/量等
        # 注意：原始time字符串不要
        num_cols = [c for c in base2.columns if c != "time"]
        # 固定列顺序：第一次见到的顺序作为标准
        if len(self._cache) == 0 and len(self._lru) == 0 and fi == 0:
            feature_cols = num_cols
        else:
            # 若已经确定了feature_cols（dataset构造时），就对齐
            feature_cols = getattr(self, "feature_cols", num_cols)

        for c in feature_cols:
            if c not in base2.columns:
                base2[c] = 0.0
        X = base2[feature_cols].to_numpy(dtype=np.float32)

        item = FileCacheItem(X=X, y=y, feature_cols=feature_cols)
        self._cache[fi] = item
        self._touch_lru(fi)
        return item
