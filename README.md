下面按“从易到难、覆盖分类/回归/图像/NLP/时序/不平衡”挑 10 个 Kaggle 练手项目，基本都能在 Kaggle 搜到，适合从零跑通完整机器学习流程。

一、10 个推荐项目

# 项目（Kaggle 搜索名） 类型 练的核心能力 建议 baseline / 指标

1 Titanic – ML from Disaster 二分类 缺失值、类别编码、特征工程、交叉验证 逻辑回归/随机森林；Accuracy。Kaggle 官方最友好入门赛

2 House Prices – Advanced Regression Techniques（Ames） 回归 79 特征、偏态目标、正则化、特征选择 Ridge/Lasso/RandomForest/XGBoost；RMSE（对数）。适合练特征工程

3 Digit Recognizer（MNIST） 图像多分类 张量预处理、归一化、MLP→CNN Softmax CNN；Accuracy。CV 入门首选

4 Spaceship Titanic 二分类/结构化 混合类型、字符串解析、管道化预处理 梯度提升/RF；Accuracy。科幻版 Titanic，比原版多特征处理

5 Natural Language Processing with Disaster Tweets 文本分类 分词、TF-IDF、朴素贝叶斯→Transformer Logistic+TF-IDF / 小型 BERT；F1。NLP 入门赛

6 Store Sales – Time Series Forecasting 时序预测 季节性、滞后特征、滚动统计、树模型/时序模型 LightGBM+时间特征 / Prophet；SMAPE/NRMSE。Kaggle 官方时序入门

7 SMS Spam Collection（数据集，非竞赛） 文本二分类 文本清洗、n-gram、类别不均衡初步 TF-IDF+Logistic；Precision/Recall/F1

8 Credit Card Fraud Detection（mlg-ulb 数据集） 极度不平衡分类 下采样/SMOTE/类别权重、PR 曲线、阈值 逻辑回归/XGB+class_weight；PR-AUC、Recall

9 CIFAR-10（数据集/自制训练） 图像 10 类 CNN、数据增强、BN、学习率、混淆矩阵 小 ResNet/CNN；Accuracy。比 MNIST 进一阶

10 IMDb 50K Movie Reviews 情感分类 词袋→词向量→LSTM/DistilBERT TF-IDF+LR 打底，再上 Transformer；Accuracy/F1

备选替换：不想要图像可换 Iris（纯分类概念）、Bike Sharing Demand（回归+时序）、Telco Churn（客户流失，练业务分类和不平衡）。

二、本地环境 `kaggle`

已创建 conda 环境，覆盖下载/上传、表格模型、图像 CNN、NLP Transformer、时序、不平衡分类。本机 RTX 5060 Ti 使用 PyTorch 2.14 + CUDA 13.0（含 sm_120）。

```bash
conda activate kaggle
```

Jupyter 内核名：`Python (kaggle)`。凭证已就位：`~/.kaggle/kaggle.json`。

常用命令：

```bash
# 竞赛数据（对应上面 1–6、部分 7–10）
kaggle competitions download -c titanic -p data/titanic
kaggle competitions download -c house-prices-advanced-regression-techniques -p data/house-prices
kaggle competitions download -c digit-recognizer -p data/digit-recognizer
kaggle competitions download -c spaceship-titanic -p data/spaceship-titanic
kaggle competitions download -c nlp-getting-started -p data/nlp-getting-started
kaggle competitions download -c store-sales-time-series-forecasting -p data/store-sales

# 提交
kaggle competitions submit -c titanic -f submissions/titanic.csv -m "baseline"

# 数据集（SMS Spam / Credit Card Fraud / IMDb 等）
# kaggle datasets download -d uciml/sms-spam-collection-dataset -p data/sms-spam
```

复现环境（新机器）：

```bash
conda create -n kaggle python=3.12 pip -y
conda activate kaggle
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m ipykernel install --user --name kaggle --display-name "Python (kaggle)"
```

目录：`data/` 下载数据，`notebooks/` 实验，`models/` 权重，`submissions/` 提交文件（均已 gitignore）。
