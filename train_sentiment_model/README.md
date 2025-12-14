## SentimentLens：多数据集情感分析实验

本项目用于对多种情感分析数据集（Twitter、SST‑2、Amazon Reviews 等）在不同预训练模型（BERT、DeBERTa、RoBERTa）和不同微调策略（全参数微调、LoRA、Prefix Tuning 等）下进行系统对比实验，并统一保存评估结果用于后续分析与可视化。

更详细的实验设计、指标说明和结果对比，请参考 `EXPERIMENTS_README.md`。

---

### 项目结构

#### 核心模块

- **`experiment_utils/`**: 公共实验工具包（模块化拆分）：
  - `trainer.py`: `BaseSentimentTrainer` 基类，封装训练/评估通用逻辑
  - `data.py`: `set_global_seed` 与本地数据加载 `load_local_dataset`
  - `results.py`: 评估结果结构与 `save_evaluation_results`
  - `peft_utils.py`: LoRA / Prefix Tuning 配置工具
  - `runtime.py`: TensorBoard 启动、parquet 检查等运行时工具
  - `cli.py`: 实验脚本通用命令行参数解析器与解析函数
  - `__init__.py`: 统一出口，保持 `from experiment_utils import ...` 的用法

- **`gpu_utils.py`**: GPU工具模块：
  - **`get_available_gpus()`**: 检测所有可用的GPU设备
  - **`setup_multi_gpu()`**: 设置多GPU训练（DataParallel）

#### 实验脚本

- **`amazon_experiment.py`**: Amazon Reviews 二分类情感分析实验（基于 `amazon_polarity`）。
  - 继承 `experiment_utils.BaseSentimentTrainer`
  - 实现 `load_data()` 和 `get_target_names()`
  - 支持从本地 parquet 文件或 Hugging Face 加载数据
  - 支持 full/lora/prefix 三种微调方式
  - 覆盖 `get_training_args()` 以使用Amazon特定的训练参数
  - 覆盖 `evaluate()` 以使用 `experiment_utils.save_evaluation_results`

- **`sst2_experiment.py`**: GLUE SST‑2 二分类实验（negative/positive）。
  - 继承 `experiment_utils.BaseSentimentTrainer`
  - 实现 `load_data()` 和 `get_target_names()`
  - 支持从本地 parquet 文件或 Hugging Face 加载数据
  - 支持 full/lora/prefix 三种微调方式
  - 支持 TensorBoard 可视化训练过程

- **`twitter_experiment.py`**: TweetEval Sentiment 三分类实验（negative/neutral/positive）
  - 继承 `experiment_utils.BaseSentimentTrainer`
  - 实现 `load_data()` 和 `get_target_names()`
  - 支持从本地 parquet 文件或 Hugging Face 加载数据
  - 支持 full/lora/prefix 三种微调方式
  - 覆盖 `tokenize_data()` 以使用较短的max_length（128，适合Twitter推文）
  - 支持 TensorBoard 可视化训练过程

#### 工具模块

- **`run_all_experiments.py`**: （可选）统一入口脚本，用于批量运行多模型、多数据集实验。
- **`experiment_utils/`**: 模块化的实验工具包（数据加载、结果保存、PEFT 配置、通用 CLI、基础训练器、TensorBoard 与 parquet 检查等）。
- **`model_config.py`**: 管理本地模型路径及可用模型列表（`get_model_path` / `get_available_models`）。
- **`results/`**: 各实验运行完后的结果目录，包含 JSON 评估文件和保存的模型。
- **`EXPERIMENTS_README.md`**: 详细实验说明文档（中文），包括设计思路、实验设置和部分结果总结。


---

安装依赖示例：

```bash
pip install -r requirements.txt
```
---

#### 数据源支持

所有实验脚本支持两种数据加载方式（按优先级）：

1. **本地 Parquet 文件**（优先）：从本地 `dataset/` 目录加载 parquet 格式的数据集
2. **Hugging Face数据集**（回退）：如果本地文件不存在或有问题，自动从Hugging Face下载数据集


**数据集目录结构**：
```
train_sentiment_model/
├── dataset/
│   ├── glue/
│   │   └── sst2/
│   │       ├── train-*.parquet
│   │       ├── validation-*.parquet
│   │       └── test-*.parquet
│   └── amazon_polarity/
│       └── amazon_polarity/
│           ├── train-*.parquet
│           └── test-*.parquet
```

**自动加载机制**：
- 脚本会自动检查 `dataset/` 目录下是否存在对应的 parquet 文件
- 如果存在且格式正确，优先使用本地文件（速度更快）
- 如果本地文件不存在、是 Git LFS 指针文件或格式错误，自动回退到 Hugging Face

**Git LFS 支持**：
- 如果 parquet 文件使用 Git LFS 管理，脚本会自动检测指针文件
- 遇到指针文件时会提示运行 `git lfs pull` 下载实际文件
- 错误信息会明确指出需要下载的文件路径

**使用示例**：
```python
# 默认优先使用本地 parquet 文件
trainer_obj = SentimentTrainer(model_name, num_labels=2)
train_df, test_df = trainer_obj.load_data(use_local_parquet=True)

# 强制从 Hugging Face 加载
train_df, test_df = trainer_obj.load_data(use_local_parquet=False)
```

**优势**：
- 本地 parquet 文件加载速度更快，无需网络连接
- 自动回退机制保证即使本地文件不可用也能正常运行
- 支持 Git LFS 管理大型数据集文件

---

### 微调方式（Full / LoRA / Prefix Tuning）

所有实验脚本都支持三种微调方式，通过 `finetune_method` 参数指定：

- **`finetune_method="full"`**（默认）：全参数微调，直接对 Transformer 全部参数 + 分类头进行更新。
- **`finetune_method="lora"`**：使用 LoRA 进行参数高效微调，只在部分注意力层引入低秩矩阵，其他参数冻结。
- **`finetune_method="prefix"`**：使用 Prefix Tuning，在输入前注入若干可学习 token，而不直接修改原模型参数。

**使用示例**：

```python
# 全参数微调（默认）
trainer = AmazonReviewsTrainer(model_name="bert-base-uncased", num_labels=2)

# LoRA 微调
trainer = AmazonReviewsTrainer(
    model_name="bert-base-uncased",
    num_labels=2,
    finetune_method="lora"
)

# Prefix Tuning
trainer = AmazonReviewsTrainer(
    model_name="bert-base-uncased",
    num_labels=2,
    finetune_method="prefix"
)
```

**LoRA 自定义参数**：

```python
trainer = AmazonReviewsTrainer(model_name="bert-base-uncased", num_labels=2, finetune_method="lora")
trainer.setup_model(
    move_to_gpu=False,
    r=8,                    # LoRA 秩（默认8）
    lora_alpha=16,          # LoRA alpha（默认16）
    lora_dropout=0.1,       # LoRA dropout（默认0.1）
    target_modules=None     # 目标模块，None时自动检测
)
```

**Prefix Tuning 自定义参数**：

```python
trainer = AmazonReviewsTrainer(model_name="bert-base-uncased", num_labels=2, finetune_method="prefix")
trainer.setup_model(
    move_to_gpu=False,
    num_virtual_tokens=20,  # 虚拟token数量（默认20）
    prefix_projection=False # 是否使用投影层（默认False）
)
```

**实现细节**：

- 微调功能在 `experiment_utils/peft_utils.py` 中实现，包含 `setup_lora_model()` 和 `setup_prefix_tuning_model()` 函数
- `BaseSentimentTrainer.setup_model()` 方法会根据 `finetune_method` 自动应用相应的配置
- 模型保存时会自动识别 PEFT 模型并使用相应的保存方法
- 所有三个实验（SST-2、Amazon Reviews、Twitter Sentiment）都支持微调功能

**运行实验**：

所有实验脚本支持通过命令行参数选择微调方法和模型：

**运行所有方法和模型（默认）**：
```bash
python amazon_experiment.py   # 运行所有模型（BERT/DeBERTa/RoBERTa）和所有方法（full/lora/prefix）
python sst2_experiment.py     # 运行所有模型和所有方法
python twitter_experiment.py  # 运行所有模型和所有方法
```

**只运行指定的微调方法**：
```bash
# 只运行 LoRA 微调
python amazon_experiment.py --methods lora

# 运行多个指定方法
python amazon_experiment.py --methods full lora

# 只运行全参数微调
python sst2_experiment.py --methods full
```

**只运行指定的模型**：
```bash
# 只运行 BERT 模型
python amazon_experiment.py --models bert

# 运行多个指定模型
python amazon_experiment.py --models bert roberta

# 组合使用：只运行 BERT 模型的 LoRA 微调
python amazon_experiment.py --models bert --methods lora
```

**TensorBoard 可视化**：
```bash
# 使用默认设置（自动启动 TensorBoard，端口 6006）
python twitter_experiment.py

# 指定 TensorBoard 端口
python twitter_experiment.py --tb-port 6007

# 禁用 TensorBoard
python twitter_experiment.py --no-tensorboard
```

TensorBoard 会自动记录训练过程中的各种指标：
- 训练损失 (train/loss)
- 验证损失 (eval/loss)
- 准确率 (eval/accuracy)
- F1 分数 (eval/f1_macro, eval/f1_weighted)
- 学习率 (train/learning_rate)
- 训练步数 (train/epoch)

运行实验后，访问 `http://localhost:6006`（或指定的端口）查看实时训练可视化。

**查看帮助信息**：
```bash
python amazon_experiment.py --help
python sst2_experiment.py --help
python twitter_experiment.py --help
```

**结果保存**：

结果会保存在不同的目录中，例如：
- `results/amazon_bert-base-uncased_results/`（full）
- `results/amazon_bert-base-uncased_lora_results/`（lora）
- `results/amazon_bert-base-uncased_prefix_results/`（prefix）

**命令行参数说明**：
- `--methods`: 选择微调方法，可选值：`full`、`lora`、`prefix`、`all`（默认）
- `--models`: 选择模型，可选值：`bert`、`deberta`、`roberta`、`all`（默认）
- `--tensorboard`: 启动 TensorBoard 可视化（默认启用）
- `--tb-port`: 指定 TensorBoard 端口（默认 6006）
- `--no-tensorboard`: 禁用 TensorBoard 可视化
- 可以同时指定多个方法和模型，例如：`--methods full lora --models bert roberta`

---

### GPU检测与多GPU训练

训练脚本会自动检测并利用所有可用的GPU设备：

- **自动GPU检测**：训练开始时会自动检测系统中所有可用的GPU，并输出每个GPU的名称和显存信息
- **多GPU支持**：如果检测到多个GPU，会自动使用 `torch.nn.DataParallel` 进行数据并行训练
- **单GPU训练**：单个GPU时直接使用该GPU进行训练
- **CPU回退**：无GPU时自动回退到CPU训练（batch size会自动减小以适应CPU）


**性能优化**：
- 多GPU环境下，总batch size = `per_device_batch_size` × GPU数量
- 根据GPU数量自动调整 `dataloader_num_workers` 以优化数据加载速度
- 多GPU时自动启用FP16混合精度训练以加速训练
- **CUDA multiprocessing 优化**：模型在 tokenize 完成后再移到 GPU，避免多进程 tokenize 时的 CUDA 错误

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

**默认运行（所有模型和所有方法）**：

- **Amazon Reviews 二分类**：
```bash
python amazon_experiment.py
# 运行所有模型（BERT/DeBERTa/RoBERTa）和所有方法（full/lora/prefix）
```

- **Twitter Sentiment 三分类**：
```bash
python twitter_experiment.py
# 运行所有模型和所有方法
```

- **SST‑2 二分类**：
```bash
python sst2_experiment.py
# 运行所有模型和所有方法
```

**指定微调方法**：
```bash
# 只运行全参数微调
python amazon_experiment.py --methods full

# 只运行 LoRA 微调
python sst2_experiment.py --methods lora

# 运行多个方法
python twitter_experiment.py --methods full lora
```

**指定模型**：
```bash
# 只运行 BERT 模型
python amazon_experiment.py --models bert

# 运行多个模型
python sst2_experiment.py --models bert roberta
```

**组合使用**：
```bash
# 只运行 BERT 模型的 LoRA 微调
python amazon_experiment.py --models bert --methods lora

# 运行 BERT 和 RoBERTa 的全参数和 LoRA 微调
python sst2_experiment.py --models bert roberta --methods full lora
```

**查看帮助信息**：
```bash
python amazon_experiment.py --help
python sst2_experiment.py --help
python twitter_experiment.py --help
```

#### 2. 统一运行多个实验

如果你希望一键跑完所有模型与数据集组合，可以在 `run_all_experiments.py` 中配置好要跑的：

- 模型列表（BERT / DeBERTa / RoBERTa）
- 数据集列表（Twitter / SST‑2 / Amazon）
- 微调方式列表（full / lora / prefix）

**注意**：每个实验脚本也可以通过命令行参数灵活选择，无需修改代码。

**注意**：每个实验脚本也可以通过命令行参数灵活选择，无需修改代码。

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
#### 其他常见问题

- **CUDA multiprocessing 错误**：代码已优化，模型会在 tokenize 完成后再移到 GPU，避免此问题。
- **内存不足**：减少 batch size 或使用更小的模型。
- **Git LFS 指针文件**：运行 `git lfs pull` 下载实际数据文件。

---

### 更多说明

- 若要扩展新的情感数据集（例如 Yelp / IMDB 等），推荐流程是：
  1. 新建一个 `xxx_experiment.py`，仿照 Amazon / SST‑2 的结构；
  2. 在其中实现对应的 `load_data` 逻辑，支持从本地 parquet 文件或 Hugging Face 加载数据；
  3. 继续复用现有的 `tokenize_data` / `train` / `evaluate` / `save_model` / `save_evaluation_results` 逻辑。
  
**注意**：训练流程已优化，模型会在 tokenize 完成后再移到 GPU，避免 CUDA multiprocessing 错误。
- 对于更详细的实验方案和结果分析，请阅读 `EXPERIMENTS_README.md`，那里可以记录你后续补充的图表、表格和结论。
