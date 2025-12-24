## 训练与提交流程简述
### Case4（XGBoost）

| 操作系统 | Ubuntu                  |
| -------- | ----------------------- |
| 开发环境 | GT-Ubuntu22.04-CMD-V3.0 |
| 资源类型 | 纯CPU                   |
| 资源配置 | 64核 128GB内存          |
| 系统云盘 | 高效云盘 80GB           |
| 数据云盘 | 高效云盘 200GB          |
| 带宽     | 按流量 25Mbps           |
| IP地址   | 36.103.199.242          |
| 用户名   | ubuntu                  |
| 密码     | uh5L.pnLnTNhfyzo        |
| 端口     | 22                      |

**1. 构建缓存**
将每个 csv 预处理成 numpy 缓存，避免训练阶段反复读取与重复特征工程。

```
python build_cache_case4_label10.py \
  --data_dir ../data \
  --cache_dir ../cache_case4_label10 \
  2>&1 | tee cache_build.lo
```

会在 ../cache_case4_label10/ 输出一些 npy, json 文件，不太用管 

**2. 使用缓存进行流式训练**

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

**3. 打包提交到评测平台**

- Predictor.py 需要重新做

- feature_engineering.py 推理特征工程，与训练一致

- model.pth 由 xgb_label10_cached.json 复制得到

- meta.json 由 meta_label10_cached.json 复制得到，且包含 Predictor 需要字段

- config.json

- requirements.txt

注意：Predictor.py 内应使用相对自身文件的绝对路径读取 model.pth/meta.json，并确保能 import 同目录的 feature_engineering.py（常用做法是把当前目录加入 sys.path）

**4. 评测效果**

仅评测了 label_10

| 准确率  | 召回率   | F0.5分数 | 累计收益率 | 单次收益率 | 模型评分 |
| ------- | -------- | -------- | ---------- | ---------- | -------- |
| 0.53538 | 0.370547 | 0.491639 | 74.258800  | 0.000635   | 0.000271 |

和平台其他的提交相比，目前的版本累计收益率还不错，F0.5分数相当好，但单次收益率很不理想，最好的目前有 0.002549，可能和准确率挂钩？我看有些评测准确率大于0.6，他们的单次收益率都不差