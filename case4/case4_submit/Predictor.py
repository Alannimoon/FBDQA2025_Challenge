# Predictor.py
import os, sys, json
from typing import List

import numpy as np
import pandas as pd
import xgboost as xgb

# 1) 确保能 import 同目录 .py
curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from feature_engineering import BASE_FEATURES, window_to_vector_with_cols


def get_predict(prob: np.ndarray, ratio=(0.95, 0.95)) -> np.ndarray:
    p = np.array(prob, dtype=np.float32)
    p[:, 0] *= float(ratio[0])
    p[:, 2] *= float(ratio[1])
    return np.argmax(p, axis=1)


class Predictor:
    def __init__(self):
        self.label_names = ["label_10"]
        self.ratio = (0.95, 0.95)

        # 2) 用绝对路径读文件（平台 cwd 不可靠）
        meta_path  = os.path.join(curr_dir, "meta.json")
        model_path = os.path.join(curr_dir, "model.pth")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.num_cols = meta["num_cols"]

        self.bst = xgb.Booster()
        self.bst.load_model(model_path)

    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        X_vec = []
        for df in x:
            df2 = df.copy()

            # 缺列补 0（别 raise，平台数据偶尔字段类型/缺失会坑你）
            for c in BASE_FEATURES:
                if c not in df2.columns:
                    df2[c] = 0.0
            df2 = df2[BASE_FEATURES]

            X_vec.append(window_to_vector_with_cols(df2, self.num_cols))

        X = np.stack(X_vec, axis=0).astype(np.float32)
        prob = self.bst.predict(xgb.DMatrix(X))
        pred = get_predict(prob, self.ratio)
        return [[int(p)] for p in pred]
