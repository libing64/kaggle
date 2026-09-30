# iMet Collection 2019 — FGVC6

竞赛：https://www.kaggle.com/competitions/imet-2019-fgvc6  
任务：多标签属性标注（1103 类），指标 **mean F2**。  
提交方式：Kernels-only（可本地训练，Notebook 推理）。

## 方法：知识蒸馏

1. **Teacher**：`tf_efficientnet_b3.ns_jft_in1k`，BCEWithLogits + 硬标签  
2. **Student**：`tf_efficientnet_b0.ns_jft_in1k`  
   \[
   L = \alpha\,\mathrm{BCE}(z_s, y) + (1-\alpha)\,T^2\,\mathrm{BCE}(z_s/T,\ \sigma(z_t/T))
   \]
3. 验证集搜索 F2 阈值，写出 `submission.csv`

## 用法

```bash
conda activate kaggle
python 10_iMet_Collection/download.py   # ~22.6GB → 10_iMet_Collection/data/
python 10_iMet_Collection/solve.py      # teacher 再 student
# 或
python 10_iMet_Collection/run_all.py
```

产物：`teacher_b3.pt`、`student_b0.pt`、`submission.csv`、`meta.json`
