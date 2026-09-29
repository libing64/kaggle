# Kaggle 提交说明（Cassava）

本地训练权重：`09_leaf_disease_cls/efficientnet_b0.pt`（v3，384 + TTA 验证约 **0.813**）  
推理 Notebook：`09_leaf_disease_cls/kaggle_infer.ipynb`（384 + 4-way TTA）

## 1. 上传权重为 Dataset

1. 打开 https://www.kaggle.com/datasets → 已有 Dataset 则 **New Version**，否则 **New Dataset**
2. 上传最新 `efficientnet_b0.pt`（约 16MB）
3. 标题例如 `cassava-efficientnet-b0`
4. 设为 Private → Create / 发布新版本

## 2. 新建竞赛 Notebook

1. 打开 https://www.kaggle.com/competitions/cassava-leaf-disease-classification/code
2. **New Notebook**
3. 右侧：
   - Accelerator = **GPU**
   - Internet = **Off**
   - **Add Input（两个都要加）**：
     1. Competitions → `cassava-leaf-disease-classification`
     2. Datasets → `cassava-efficientnet-b0`（你的权重）
4. 把更新后的 `kaggle_infer.ipynb` 复制进去

如果日志里出现 `test_dir ... exists False`，说明只挂了权重 Dataset，还没挂竞赛数据。挂上后路径通常是：

- `/kaggle/input/cassava-leaf-disease-classification/test_images`
- 或 `/kaggle/input/competitions/cassava-leaf-disease-classification/test_images`

Notebook 已改为自动搜索这两种路径。

## 3. 提交

1. **Save Version** → Save & Run All
2. 跑完后在 Output 确认有 `submission.csv`
3. **Submit to Competition**

本地只有 1 张测试图，线上会换成完整测试集；Notebook 里按 `test_images/*.jpg` 预测即可。
