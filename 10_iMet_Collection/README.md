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

## Kaggle 提交（Kernels-only）

公开下载的 `test/` / `sample_submission.csv` 只有 **7443** 行；评分器要求 **38801** 行（私有测试集）。

若 Submit 重跑日志里仍是 `rows 7443` / `n_submit=7443`，说明 **当前 Late Submission 没有挂载私有 test 图片**（2019 kernels-only 竞赛的常见现状）：没有那 ~3.1 万张图和对应 id，无法靠改阈值或“凑行数”通过评测。

请先用更新后的 `kaggle_infer.ipynb` 再 Submit 一次，把日志里这段贴出来：

- `non-train png folders ...`
- `sample_submission rows ...`
- `selected id source ... n_submit=...`

若 `n_submit` 仍是 7443，这条竞赛线基本无法再晚交打分；只能把公开 7443 当本地验证，或换仍提供完整测试集的竞赛。
