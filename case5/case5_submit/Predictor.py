# Predictor.py
import os, sys, json
from typing import List
import numpy as np
import pandas as pd
import torch

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.append(curr_dir)

from model import iTransformerClassifier
from preprocess import preprocess_exp_log
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
BASE_FEATURES = [
    "date","time","sym","n_close","amount_delta","n_midprice",
    "n_bid1","n_bsize1","n_bid2","n_bsize2","n_bid3","n_bsize3","n_bid4","n_bsize4","n_bid5","n_bsize5",
    "n_ask1","n_asize1","n_ask2","n_asize2","n_ask3","n_asize3","n_ask4","n_asize4","n_ask5","n_asize5",
]

class Predictor:
    def __init__(self, model_path: str = "model.pth", config_path: str = "config.json"):
        abs_model = os.path.join(curr_dir, model_path)
        abs_cfg   = os.path.join(curr_dir, config_path)

        cfg = json.load(open(abs_cfg, "r", encoding="utf-8"))
        cols_path = os.path.join(curr_dir, "feature_cols_label10.txt")
        with open(cols_path, "r", encoding="utf-8") as f:
            self.feature_cols = [ln.strip() for ln in f if ln.strip()]
        self.window = 100
        self.device = torch.device("cpu")

        # 必须与训练一致
        self.model = iTransformerClassifier(
            n_vars=len(self.feature_cols),
            seq_len=self.window,
            d_model=64,
            n_layers=2,
            n_heads=4,
        ).to(self.device)

        state = torch.load(abs_model, map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def _df_to_window_tensor(self, df_raw: pd.DataFrame) -> torch.Tensor:
        df = df_raw.copy().reset_index(drop=True)

        # 取最后 window 行（平台每个样本一般就是100行，但保险起见）
        if len(df) >= self.window:
            df = df.tail(self.window).reset_index(drop=True)
        else:
            # 不足100行：前面补0行
            pad = pd.DataFrame([{c: 0 for c in df.columns}] * (self.window - len(df)))
            df = pd.concat([pad, df], ignore_index=True)

        # 只保留基础字段，缺列补0
        for c in BASE_FEATURES:
            if c not in df.columns:
                df[c] = 0
        df = df[BASE_FEATURES]

        # 与训练一致的预处理（会新增 time_sec/time_interval，并变换 size/amount）
        df = preprocess_exp_log(df)
        df = df.drop(columns=["time"], errors="ignore")

        # 按训练保存的 feature_cols 取列，缺的补0
        for c in self.feature_cols:
            if c not in df.columns:
                df[c] = 0.0
        X = df[self.feature_cols].to_numpy(dtype=np.float32)  # (100, F)

        # 变成 (1, 100, F)
        return torch.from_numpy(np.ascontiguousarray(X)).unsqueeze(0)

    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        out: List[List[int]] = []
        with torch.no_grad():
            for df in x:
                X = self._df_to_window_tensor(df).to(self.device)
                logits = self.model(X)                 # (1,3)
                pred = int(torch.argmax(logits, dim=1).item())
                out.append([pred])
        return out
