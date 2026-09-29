# Kaggle 提交说明（Cassava）

本地训练权重：`09_leaf_disease_cls/efficientnet_b0.pt`  
推理 Notebook：`09_leaf_disease_cls/kaggle_infer.ipynb`

## 1. 上传权重为 Dataset

1. 打开 https://www.kaggle.com/datasets → **New Dataset**
2. 上传 `efficientnet_b0.pt`（约 16MB）
3. 标题例如 `cassava-efficientnet-b0`
4. 设为 Private → Create

## 2. 新建竞赛 Notebook

1. 打开 https://www.kaggle.com/competitions/cassava-leaf-disease-classification/code
2. **New Notebook**
3. 右侧：
   - Accelerator = **GPU**
   - Internet = **Off**
   - Add Input：竞赛数据 + 上一步权重 Dataset
4. 把 `kaggle_infer.ipynb` 的单元格复制进去，或直接上传该 notebook 文件

## 3. 提交

1. **Save Version** → Save & Run All
2. 跑完后在 Output 确认有 `submission.csv`
3. **Submit to Competition**

本地只有 1 张测试图，线上会换成完整测试集；Notebook 里按 `test_images/*.jpg` 预测即可。
