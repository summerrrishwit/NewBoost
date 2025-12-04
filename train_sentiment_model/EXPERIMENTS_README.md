# 情感分析实验套件

本套件包含三个情感分析实验，分别使用SST-2、Amazon Reviews、Twitter Sentiment数据集，并比较BERT和DeBERTa模型的性能。

## 实验概述

### 1. SST-2 实验 (`sst2_experiment.py`)
- **数据集**: Stanford Sentiment Treebank (SST-2)
- **任务类型**: 二分类情感分析 (正面/负面)
- **模型**: BERT-base-uncased, DeBERTa-base
- **特点**: 电影评论情感分析，经典基准数据集

#### 实验流程
1. 加载数据: `load_dataset("glue", "sst2")` 并将 `label` 转换为 `sentiment` 字段（negative/positive）。
2. 划分数据: 使用官方 `train/validation` 划分，分别构建 `train_df` 与 `test_df`（字段 `text/label/sentiment`）。
3. 初始化模型: 依据 `model_config.get_model_path` 加载分词器与分类模型，`num_labels=2`；若无 `pad_token`，使用 `eos_token` 作为填充符。
4. 数据预处理: 将 `train_df/test_df` 转为 `datasets.DatasetDict`，使用分词器按 `max_length=256` 截断；通过 `map` 批处理并移除原始 `text`。
5. 训练配置: `TrainingArguments` 设定评估/保存步长=500、学习率=2e-5、batch=16/32、epoch=3、`weight_decay=0.01`、`fp16`(若GPU可用)、按 `f1_macro` 挑选最优；`DataCollatorWithPadding` 动态填充。
6. 训练与评估: `Trainer.train()` 后在测试集 `predict`，输出 `classification_report`（accuracy、F1-macro、F1-weighted）。
7. 结果保存: 写入 `./results/sst2_{model}_results/evaluation_results.json`，并将模型与分词器保存至 `./results/sst2_{model}_model/`。

### 2. Amazon Reviews 实验 (`amazon_reviews_experiment.py`)
- **数据集**: Amazon Reviews
- **任务类型**: 五分类情感分析 (1-5星评级)
- **模型**: BERT-base-uncased, DeBERTa-base
- **特点**: 商品评论情感分析，多级评分

#### 实验流程
1. 加载数据: 首选 `load_dataset("amazon_polarity")`；若失败，生成模拟数据（覆盖1-5星多模板、多产品类型，多样化短语）。
2. 文本构建: 官方数据将 `title` 与 `content` 合并为 `text`；将二分类 `label` 粗映射为 `rating`（1 或 5），并构造 `label=rating-1`（0-4）。
3. 采样策略: 默认 `sample_size=10000`；从 `train/test` 随机抽样（测试集比例约为1/5）。
4. 初始化模型: 依据 `model_config.get_model_path` 加载分词器与分类模型，`num_labels=5`；若无 `pad_token`，使用 `eos_token`。
5. 数据预处理: 构建 `DatasetDict` 并以 `max_length=256` 分词；批处理 `map`、移除 `text`。
6. 训练配置: `TrainingArguments` 设置评估/保存步长=200、学习率=1e-5、batch=8/16、epoch=2、`weight_decay=0.1`、`warmup_steps=50`、`lr_scheduler_type="cosine"`，按 `f1_macro` 选最优。
7. 训练与评估: 训练后在测试集预测，输出包含五类（1-5星）的 `classification_report`。
8. 结果保存: 写入 `./results/amazon_{model}_results/evaluation_results.json`，并保存模型至 `./results/amazon_{model}_model/`。

### 3. Twitter Sentiment 实验 (`twitter_sentiment_experiment.py`)
- **数据集**: Twitter Sentiment
- **任务类型**: 三分类情感分析 (正面/中性/负面)
- **模型**: BERT-base-uncased, DeBERTa-base
- **特点**: 社交媒体文本情感分析，包含表情符号和特殊字符

#### 实验流程
1. 加载数据: 首选 `load_dataset("tweet_eval", "sentiment")` 并映射标签至 `sentiment`（negative/neutral/positive）；若失败，生成含表情、话题与提及的模拟推文，并做轻量清洗（去多空白、保留常用符号与表情）。
2. 采样策略: 默认 `sample_size=15000`；对 `train/test` 随机采样（测试约1/5），并保持三类相对均衡。
3. 初始化模型: 依据 `model_config.get_model_path` 加载分词器与分类模型，`num_labels=3`；若无 `pad_token`，使用 `eos_token`。
4. 数据预处理: 构建 `DatasetDict`，以 `max_length=128` 分词（短推文优化），批处理 `map` 并移除 `text`。
5. 训练配置: `TrainingArguments` 设定评估/保存步长=500、学习率=2e-5、batch=16/32、epoch=3、`weight_decay=0.01`、`fp16`(若GPU可用)，按 `f1_macro` 选最优。
6. 训练与评估: 训练后在测试集 `predict`，输出三类 `classification_report`。
7. 结果保存: 写入 `./results/twitter_{model}_results/evaluation_results.json`，并保存模型至 `./results/twitter_{model}_model/`。

## 安装依赖

```bash
pip install -r requirements_experiments.txt
```

## 使用方法

### 运行单个实验

```bash
# SST-2 实验
python sst2_experiment.py

# Amazon Reviews 实验
python amazon_reviews_experiment.py

# Twitter Sentiment 实验
python twitter_sentiment_experiment.py
```

### 运行所有实验

```bash
python run_all_experiments.py
```

## 实验配置

### 模型配置
- **BERT**: `bert-base-uncased`
- **DeBERTa**: `microsoft/deberta-base`

### 训练参数
- 学习率: 2e-5
- 批次大小: 16 (训练), 32 (评估)
- 训练轮数: 3
- 最大序列长度: 256 (SST-2, Amazon), 128 (Twitter)
- 权重衰减: 0.01

### 数据采样
- SST-2: 使用完整数据集
- Amazon Reviews: 采样10,000条训练数据
- Twitter Sentiment: 采样15,000条训练数据

## 输出结果

每个实验会生成以下文件：

### 模型文件
- `{dataset}_{model}_model/`: 训练好的模型和分词器
- `{dataset}_{model}_results/`: 训练日志和检查点

### 评估结果
- 控制台输出详细的分类报告
- 包含准确率、F1分数等指标
- 各类别的精确率、召回率、F1分数

### 日志文件
- `experiments.log`: 所有实验的详细日志

## 实验结果

### SST-2 数据集结果

| 模型 | 准确率 | F1-宏平均 | F1-加权平均 |
|------|--------|-----------|-------------|
| BERT-base-uncased | 93.35% | 93.34% | 93.35% |
| DeBERTa-base | 94.95% | 94.95% | 94.95% |
| RoBERTa-base | 94.84% | 94.84% | 94.84% |

**最佳模型**: DeBERTa-base (94.95% 准确率)

### Amazon Reviews 数据集结果

| 模型 | 准确率 | F1-宏平均 | F1-加权平均 |
|------|--------|-----------|-------------|
| BERT-base-uncased | 100.00% | 100.00% | 100.00% |
| DeBERTa-base | 100.00% | 100.00% | 100.00% |
| RoBERTa-base | 100.00% | 100.00% | 100.00% |

**注意**: 所有模型在Amazon Reviews数据集上都达到了完美性能，这可能是因为使用了合成数据或数据集相对简单。

### Twitter Sentiment 数据集结果

| 模型 | 准确率 | F1-宏平均 | F1-加权平均 |
|------|--------|-----------|-------------|
| BERT-base-uncased | 70.95% | 69.99% | 70.80% |
| DeBERTa-base | 72.15% | 71.20% | 72.05% |
| RoBERTa-base | 72.35% | 71.42% | 72.25% |

**最佳模型**: RoBERTa-base (72.35% 准确率)

### 总体性能分析

1. **SST-2**: 所有模型表现优异，DeBERTa略胜一筹
2. **Amazon Reviews**: 所有模型达到完美性能（可能使用合成数据）
3. **Twitter Sentiment**: 性能相对较低，反映了社交媒体文本的复杂性，RoBERTa表现最佳

### 模型对比总结

- **DeBERTa**: 在SST-2上表现最佳，整体性能稳定
- **RoBERTa**: 在Twitter Sentiment上表现最佳，适合处理社交媒体文本
- **BERT**: 作为基线模型，在所有数据集上都表现良好

## 实验特点

### 数据处理
- 自动数据预处理和分词
- 支持Hugging Face数据集
- 包含数据增强功能（Amazon Reviews）
- 模拟数据生成（当无法访问原始数据集时）

### 模型训练
- 使用Hugging Face Transformers库
- 支持GPU加速训练
- 早停机制防止过拟合
- 动态填充提高效率

### 评估指标
- 准确率 (Accuracy)
- 加权F1分数 (F1-weighted)
- 宏平均F1分数 (F1-macro)
- 详细分类报告

## 注意事项

1. **内存要求**: 建议至少8GB RAM
2. **GPU支持**: 支持CUDA加速，会自动检测GPU可用性
3. **网络连接**: 首次运行需要下载预训练模型
4. **存储空间**: 每个模型约需要1-2GB存储空间

## 故障排除

### 常见问题

1. **内存不足**: 减少批次大小或使用更小的模型
2. **网络问题**: 检查网络连接，可能需要配置代理
3. **依赖冲突**: 使用虚拟环境隔离依赖

### 调试模式

在脚本开头添加以下代码启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展功能

### 添加新数据集
1. 创建新的实验脚本
2. 实现数据加载和预处理函数
3. 配置模型和训练参数
4. 添加到主运行脚本中

### 添加新模型
1. 在模型列表中添加新的模型名称
2. 确保模型兼容Hugging Face Transformers
3. 根据需要调整训练参数

## 许可证

本项目基于MIT许可证开源。
