# dataset_cached.py
import os, json, bisect
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

@dataclass
class CacheItem:
    X: np.memmap
    y: np.memmap

class CachedWindowDataset(Dataset):
    def __init__(self, cache_dir: str, files: List[str], window: int = 100, lru_size: int = 16):
        """
        files: 仍然用原来的 snapshot 文件列表（用于确定哪些属于 train/val/test），
               但实际读取的是 cache_dir 下对应的 .X.npy/.y.npy
        """
        self.cache_dir = cache_dir
        self.window = window
        self.lru_size = lru_size

        meta_path = os.path.join(cache_dir, "cache_meta.json")
        if not os.path.exists(meta_path):
            raise RuntimeError(f"cache_meta.json not found: {meta_path}. Please run build_cache_label10.py first.")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.feature_cols = meta["feature_cols"]
        self.n_features = meta["n_features"]

        # 映射 snapshot.csv -> cached npy path
        self.cache_pairs: List[Tuple[str,str]] = []
        for fp in files:
            name = os.path.basename(fp).replace(".csv", "")
            x_path = os.path.join(cache_dir, f"{name}.X.npy")
            y_path = os.path.join(cache_dir, f"{name}.y.npy")
            if os.path.exists(x_path) and os.path.exists(y_path):
                self.cache_pairs.append((x_path, y_path))
            else:
                # 缓存缺失，跳过或报错都行；这里选择报错更安全
                raise RuntimeError(f"Missing cache for {fp} -> {x_path} / {y_path}")

        # 计算每个文件贡献的样本数，并做前缀和
        self.counts = []
        for x_path, y_path in tqdm(self.cache_pairs, desc="Scan cache lens", unit="file"):
            y = np.load(y_path, mmap_mode="r")
            T = int(y.shape[0])
            c = max(0, T - window + 1)
            self.counts.append(c)
        self.prefix = np.cumsum([0] + self.counts)  # prefix[i] = 前 i 个文件样本总数
        self.total = int(self.prefix[-1])

        # LRU cache for open memmaps
        self._lru: List[int] = []
        self._open: Dict[int, CacheItem] = {}

    def __len__(self):
        return self.total

    def _get_file(self, fi: int) -> CacheItem:
        if fi in self._open:
            # touch lru
            if fi in self._lru:
                self._lru.remove(fi)
            self._lru.append(fi)
            return self._open[fi]

        x_path, y_path = self.cache_pairs[fi]
        X = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")

        item = CacheItem(X=X, y=y)
        self._open[fi] = item
        self._lru.append(fi)

        while len(self._lru) > self.lru_size:
            old = self._lru.pop(0)
            if old in self._open:
                del self._open[old]
        return item

    def __getitem__(self, idx: int):
        # 定位到文件 fi
        fi = bisect.bisect_right(self.prefix, idx) - 1
        base = int(self.prefix[fi])
        offset = idx - base  # 在该文件内的第 offset 个窗口
        t_end = (self.window - 1) + offset

        item = self._get_file(fi)
        Xw = item.X[t_end - (self.window - 1): t_end + 1]  # (window, F)
        Xw = np.array(Xw, dtype=np.float32, copy=True)  
        y = int(item.y[t_end])
        return np.asarray(Xw, dtype=np.float32), y
