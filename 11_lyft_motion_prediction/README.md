# Lyft Motion Prediction for Autonomous Vehicles

竞赛：https://www.kaggle.com/competitions/lyft-motion-prediction-autonomous-vehicles  
任务：预测周围交通参与者未来 50 帧 (5s) 轨迹；指标为多模态负对数似然（最多 3 个假设）。  
提交：Kernels-only，`submission.csv`。

## 方法

官方 [l5kit](https://github.com/lyft/l5kit) 栅格化 BEV（语义地图 + agent 历史）→ **ResNet18** 回归 50×(x,y) 位移。

## 环境

推荐用仓库的 `kaggle` conda 环境（已可跑，需设置 protobuf 纯 Python 实现）：

```bash
conda activate kaggle
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

也可使用专用 `lyft` 环境（Python 3.9 + 官方 l5kit 依赖）。

## 用法

```bash
# 下载 ≈18.3GB → 11_lyft_motion_prediction/data/
python 11_lyft_motion_prediction/download.py

# 训练 + 写 submission.csv（20k steps，约数小时）
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python -u 11_lyft_motion_prediction/solve.py

# 或一键
python 11_lyft_motion_prediction/run_all.py
```

配置：`agent_motion_config.yaml`（默认 `max_num_steps: 20000`）。

产物：`resnet18_motion.pt`、`submission.csv`。

## Kaggle 提交

1. 上传权重 Dataset（`resnet18_motion.pt` + 可选 config）
2. Notebook 挂竞赛数据 + 权重，Internet Off，GPU
3. 使用 `kaggle_infer.ipynb` → Save & Run All → Submit
