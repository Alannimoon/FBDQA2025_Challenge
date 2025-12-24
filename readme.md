# Readme
本项目为清华大学2025秋《金融大数据与量化分析》的大作业比赛仓库。
## 训练与提交流程简述
### Case4（XGBoost）
1. 构建缓存
将每个 csv 预处理成 numpy 缓存，避免训练阶段反复读取与重复特征工程。

```
python build_cache_case4_label10.py \
  --data_dir ../data \
  --cache_dir ../cache_case4_label10 \
  2>&1 | tee cache_build.lo
```

输出在 ../cache_case4_label10/：
- snapshot_*.X.npy：每个文件的特征矩阵 (T, F)
- snapshot_*.y.npy：对应的 label_10 序列 (T,)
- cache_meta.json：缓存的全局信息（特征维度等）

这些都用不到

2. 使用缓存进行流式训练

从 .X.npy/.y.npy 以流式方式构建 QuantileDMatrix 训练 XGBoost，降低内存压力

```
python train_label10_xgb_stream_cached.py \
  --cache_dir ../cache_case4_label10 \
  --batch_size 512 \
  --max_bin 64 \
  --nthread 32 \
  --verbose_eval 10 \
  2>&1 | tee xgb_cached.log
```

输出在当前目录：

- xgb_label10_cached.json：训练好的 XGBoost Booster 模型

- meta_label10_cached.json：训练配置与数据维度说明

停止条件：训练最多 num_boost_round=5000，但实际会在验证集指标连续 100 轮无提升时触发 early stop 并停止

3. 打包提交到评测平台

- Predictor.py 需要重新做

- feature_engineering.py 推理特征工程，与训练一致

- model.pth 由 xgb_label10_cached.json 复制得到

- meta.json 由 meta_label10_cached.json 复制得到，且包含 Predictor 需要字段

- config.json

- requirements.txt


注意：Predictor.py 内应使用相对自身文件的绝对路径读取 model.pth/meta.json，并确保能 import 同目录的 feature_engineering.py（常用做法是把当前目录加入 sys.path）

344
0.53538
0.370547
0.491639
74.258800
0.000635
0.000271
label_10