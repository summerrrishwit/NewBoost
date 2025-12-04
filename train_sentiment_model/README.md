## SentimentLens：多数据集情感分析实验

本项目用于对多种情感分析数据集（Twitter、SST‑2、Amazon Reviews 等）在不同预训练模型（BERT、DeBERTa、RoBERTa）和不同微调策略（全参数微调、LoRA、Prefix Tuning 等）下进行系统对比实验，并统一保存评估结果用于后续分析与可视化。

更详细的实验设计、指标说明和结果对比，请参考 `EXPERIMENTS_README.md`。

---

### 项目结构

#### 核心模块

- **`experiment_utils.py`**: 实验工具和基础训练器类，包含：
  - **工具函数**：
    - `set_global_seed()`: 统一设置随机种子
    - `save_evaluation_results()`: 统一保存评估结果
    - `EvalMeta`: 评估元信息数据类
  - **基础训练器类**（`BaseSentimentTrainer`），包含所有公共的训练逻辑：
    - GPU检测和设置
    - 模型加载和初始化
    - 数据预处理和分词
    - 训练参数配置
    - 训练流程管理
    - 最佳模型自动保存
    - 评估指标计算

- **`gpu_utils.py`**: GPU工具模块：
  - **`get_available_gpus()`**: 检测所有可用的GPU设备
  - **`setup_multi_gpu()`**: 设置多GPU训练（DataParallel）

#### 实验脚本

- **`amazon_experiment.py`**: Amazon Reviews 二分类情感分析实验（基于 `amazon_polarity`）。
  - 继承 `experiment_utils.BaseSentimentTrainer`
  - 实现 `load_data()` 和 `get_target_names()`
  - 覆盖 `get_training_args()` 以使用Amazon特定的训练参数
  - 覆盖 `evaluate()` 以使用 `experiment_utils.save_evaluation_results`

- **`twitter_sentiment_experiment.py`**: TweetEval Sentiment 三分类实验（negative/neutral/positive）。
  - 继承 `experiment_utils.BaseSentimentTrainer`
  - 实现 `load_data()` 和 `get_target_names()`
  - 覆盖 `tokenize_data()` 以使用较短的max_length（128，适合Twitter推文）

- **`sst2_experiment.py`**: GLUE SST‑2 二分类实验（negative/positive）。
  - 继承 `experiment_utils.BaseSentimentTrainer`
  - 实现 `load_data()` 和 `get_target_names()`

#### 工具模块

- **`run_all_experiments.py`**: （可选）统一入口脚本，用于批量运行多模型、多数据集实验。
- **`experiment_utils.py`**: 公共工具函数：
  - **`set_global_seed`**: 统一设置随机种子，保证实验可复现。
  - **`save_evaluation_results`**: 统一保存评估结果（accuracy、macro/weighted F1、完整 `classification_report`、混淆矩阵、预测/真实标签等）。
- **`model_config.py`**: 管理本地模型路径及可用模型列表（`get_model_path` / `get_available_models`）。
- **`results/`**: 各实验运行完后的结果目录，包含 JSON 评估文件和保存的模型。
- **`EXPERIMENTS_README.md`**: 详细实验说明文档（中文），包括设计思路、实验设置和部分结果总结。

#### 代码结构优势

通过继承 `experiment_utils.BaseSentimentTrainer`，三个实验脚本的代码量大幅减少，主要优势：

1. **代码复用**：公共逻辑（GPU设置、模型加载、训练流程等）只需维护一份
2. **易于扩展**：新增数据集实验只需继承基类并实现 `load_data()` 和 `get_target_names()`
3. **统一接口**：所有实验脚本使用相同的接口，便于批量运行和对比
4. **易于维护**：修改公共逻辑时只需更新基类，所有实验自动受益

---

### 环境依赖

建议使用 Python 3.9+，主要依赖见 `requirements.txt`，核心包括：

- `transformers`
- `datasets`
- `torch`
- `scikit-learn`
- `pandas`
- `numpy`
- `peft`（用于 LoRA / Prefix Tuning 等参数高效微调）

安装依赖示例：

```bash
pip install -r requirements.txt
```

如果你新增了 PEFT 相关实验，请确保：

```bash
pip install peft
```

---

### 数据集说明

- **Amazon Reviews (binary)**  
  - 脚本：`amazon_experiment.py`  
  - 数据源：`amazon_polarity`（Hugging Face Datasets）  
  - 标签：`0 = negative`, `1 = positive`

- **Twitter Sentiment (3‑class)**  
  - 脚本：`twitter_sentiment_experiment.py`  
  - 数据源：`tweet_eval`, 子任务 `"sentiment"`  
  - 标签：`0 = negative`, `1 = neutral`, `2 = positive`

- **SST‑2 (binary)**  
  - 脚本：`sst2_experiment.py`  
  - 数据源：`glue`, 子任务 `"sst2"`  
  - 标签：`0 = negative`, `1 = positive`

所有脚本都会在内部自动调用 `datasets.load_dataset(...)` 下载数据集，无需手动预处理。

---

### 微调方式（Full / LoRA / Prefix Tuning）

以 `amazon_experiment.py` 为例，每个训练器都支持多种微调方式（不同文件可能略有差异，但整体思路一致）：

- **`finetune_method="full"`**: 全参数微调（默认方式），直接对 Transformer 全部参数 + 分类头进行更新。
- **`finetune_method="lora"`**: 使用 LoRA 进行参数高效微调，只在部分注意力层引入低秩矩阵，其他参数冻结。
- **`finetune_method="prefix"`**: 使用 Prefix Tuning，在输入前注入若干可学习 token，而不直接修改原模型参数。

在训练器中一般通过构造函数指定，例如（以 Amazon 为例）：

```python
trainer = AmazonReviewsTrainer(
    model_name="bert-base-uncased",
    num_labels=2,
    finetune_method="lora",  # "full" / "lora" / "prefix"
)
```

对应的 `setup_model` 内部会根据 `finetune_method` 选择：

- 直接 `AutoModelForSequenceClassification.from_pretrained(...)`（full）
- 或用 `LoraConfig` / `PrefixTuningConfig` + `get_peft_model(...)` 包装成 PEFT 模型。

---

### GPU检测与多GPU训练

训练脚本会自动检测并利用所有可用的GPU设备：

- **自动GPU检测**：训练开始时会自动检测系统中所有可用的GPU，并输出每个GPU的名称和显存信息
- **多GPU支持**：如果检测到多个GPU，会自动使用 `torch.nn.DataParallel` 进行数据并行训练
- **单GPU训练**：单个GPU时直接使用该GPU进行训练
- **CPU回退**：无GPU时自动回退到CPU训练（batch size会自动减小以适应CPU）

**GPU使用示例输出**：
```
检测到 2 个GPU设备:
  GPU 0: NVIDIA GeForce RTX 3090 (24.00 GB)
  GPU 1: NVIDIA GeForce RTX 3090 (24.00 GB)
使用 2 个GPU进行训练 (DataParallel)
```

**性能优化**：
- 多GPU环境下，总batch size = `per_device_batch_size` × GPU数量
- 根据GPU数量自动调整 `dataloader_num_workers` 以优化数据加载速度
- 多GPU时自动启用FP16混合精度训练以加速训练

### Checkpoint保存与最佳模型

训练过程中采用智能的checkpoint保存策略：

- **训练过程checkpoint**：
  - 按步数保存checkpoint（`save_steps`，通常为200-500步）
  - 保留最近3个checkpoint（`save_total_limit=3`），避免磁盘空间浪费
  - 所有checkpoint保存在 `*_results/` 目录下

- **最佳模型自动保存**：
  - 训练过程中持续监控验证集指标（`metric_for_best_model="f1_macro"`）
  - 训练结束后自动加载最佳模型（`load_best_model_at_end=True`）
  - **自动保存最佳模型到独立目录**：训练完成后，最佳模型会自动保存到 `*_best_model/` 目录
  - 最佳模型目录包含：
    - 完整的模型权重（`pytorch_model.bin` 或 `model.safetensors`）
    - Tokenizer文件（`tokenizer.json`, `tokenizer_config.json`, `vocab.json`等）
    - 模型配置文件（`config.json`）
    - 训练参数配置（`training_args.bin`）

**模型保存目录结构**：
```
results/
├── twitter_bert-base-uncased_results/          # 训练checkpoint目录
│   ├── checkpoint-500/                          # 训练过程中的checkpoint
│   ├── checkpoint-1000/
│   ├── checkpoint-1500/                         # 最佳模型checkpoint（如果是最佳）
│   └── logs/                                    # 训练日志
├── twitter_bert-base-uncased_best_model/       # 最佳模型目录（自动保存）
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer.json
│   └── ...
└── twitter_bert-base-uncased_model/            # 最终模型目录（训练结束时的状态）
    ├── pytorch_model.bin
    └── ...
```

**使用最佳模型进行推理**：
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 加载最佳模型
model_path = "./results/twitter_bert-base-uncased_best_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
```

### 统一评估与结果文件

所有实验的 `evaluate` 方法都推荐调用 `experiment_utils.save_evaluation_results` 来保存结果，它会输出一个标准化的 `evaluation_results.json`，其中包含：

- **`meta`**: 模型名、数据集名、随机种子、额外配置（如 `num_labels`、微调方式等）。
- **`test_accuracy`**, **`test_f1_macro`**, **`test_f1_weighted`**。
- 完整的 `classification_report`（来自 `sklearn.metrics.classification_report(output_dict=True)`）。
- **`confusion_matrix`**: 混淆矩阵（二维列表形式）。
- **`predictions`** 与 **`true_labels`**：每条样本的预测标签和真实标签。

每个实验会使用类似的目录结构保存结果，例如（以 Amazon 为例）：

- `results/amazon_bert-base-uncased_results/evaluation_results.json`
- `results/amazon_bert-base-uncased_best_model/`（最佳模型目录）
- `results/amazon_bert-base-uncased_model/`（最终模型目录）

---

### 运行示例

#### 1. 运行单个数据集实验

- **Amazon Reviews 二分类**：

```bash
python amazon_experiment.py
```

- **Twitter Sentiment 三分类**：

```bash
python twitter_sentiment_experiment.py
```

- **SST‑2 二分类**：

```bash
python sst2_experiment.py
```

如果你在脚本中增加了 `finetune_method` 参数循环（例如对 full / lora / prefix 三种方式分别实验），脚本会自动依次运行，并在 `results/` 目录下生成多个子目录。

#### 2. 统一运行多个实验

如果你希望一键跑完所有模型与数据集组合，可以在 `run_all_experiments.py` 中配置好要跑的：

- 模型列表（BERT / DeBERTa / RoBERTa）
- 数据集列表（Twitter / SST‑2 / Amazon）
- 微调方式列表（full / lora / prefix）

然后直接运行：

```bash
python run_all_experiments.py
```

具体组合与实验设计请参考 `EXPERIMENTS_README.md`。

---

### 随机种子与复现性

为保证实验可复现，建议在每个脚本的主入口调用 `set_global_seed(seed)`，并在保存结果时将 `seed` 写入 `EvalMeta`：

```python
from experiment_utils import set_global_seed

if __name__ == "__main__":
    set_global_seed(42)
    # 之后再创建 Trainer、加载数据、开始训练
```

同时，`TrainingArguments` 中也可以设置相同的 `seed`，进一步增强复现性。

---

### 更多说明

- 若要扩展新的情感数据集（例如 Yelp / IMDB 等），推荐流程是：
  1. 新建一个 `xxx_experiment.py`，仿照 Amazon / Twitter / SST‑2 的结构；
  2. 在其中实现对应的 `load_data` 逻辑（使用 `datasets.load_dataset`），并统一返回包含 `text` 和 `label` 的 DataFrame；
  3. 继续复用现有的 `tokenize_data` / `train` / `evaluate` / `save_model` / `save_evaluation_results` 逻辑。
- 对于更详细的实验方案和结果分析，请阅读 `EXPERIMENTS_README.md`，那里可以记录你后续补充的图表、表格和结论。
