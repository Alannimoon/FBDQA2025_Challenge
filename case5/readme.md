## 训练与提交流程简述

### Case5（iTransformer）

| 操作系统 | Ubuntu                        |
| -------- | ----------------------------- |
| 开发环境 | GT-Ubuntu22.04-CMD-V3.2       |
| 资源类型 | RTX5090                       |
| 资源配置 | 14核 120GB内存 1卡 * 32GB显存 |
| 系统云盘 | 高效云盘 50GB                 |
| 数据云盘 | 高效云盘 200GB                |
| 带宽     | 按流量 25Mbps                 |
| IP地址   | 36.103.236.222                |
| 用户名   | ubuntu                        |
| 密码     | uRDwyMM.5nxo4CKn              |
| 端口     | 22                            |

**1. 构建缓存**

避免每次训练都重新扫描/解析 csv，训练时按样本窗口（window=100）快速取数据。

```
python build_cache_label10.py \
  --data_dir ../data \
  --cache_dir ../cache_label10 \
  2>&1 | tee cache_build.log
```

会在 `../cache_label10/` 下缓存文件

**2. 训练 iTransformer 分类器**

读取缓存，训练 iTransformer，并在验证集上 early stopping，保存 best checkpoint

```
python train_itransformer_label10.py \
  --data_dir ../data \
  --cache_dir ../cache_label10 \
  --batch_size 256 \
  --num_workers 4 \
  2>&1 | tee train.log
```

输出

- `out_label10_itransformer/best.pt`：验证集最优模型
- `out_label10_itransformer/meta.json`：训练配置与特征列等信息
- `out_label10_itransformer/train_log.jsonl`：训练过程日志（每 epoch 指标）

**3. 导出提交包**

提交文件夹中包含平台推理所需的最小集合：

- `case5_submit/Predictor.py`：平台入口类 `Predictor`
- `case5_submit/model.py`：模型结构（与训练一致）
- `case5_submit/model.pth`：最终权重（从 `best.pt` 提取/转换得到）
- `case5_submit/preprocess.py`：与训练一致的特征预处理

**4. 评测效果**

仅评测了 label_10

| 准确率   | 召回率   | F0.5分数 | 累计收益率 | 单次收益率 | 模型评分 |
| -------- | -------- | -------- | ---------- | ---------- | -------- |
| 0.454349 | 0.448393 | 0.453144 | 73.101100  | 0.000438   | 0.000007 |
