# NewsBoost

NewsBoost 是一款基于 Streamlit 的新闻智能分析应用，聚合 Google News RSS 数据并叠加情感分析、摘要生成、关键词统计与可视化，帮助分析者快速洞察舆情脉络与媒体叙事。系统采用模块化架构，具备缓存机制与多格式导出能力，可在投研、竞品洞察、危机响应等场景中快速落地。

---

## ✨ 功能亮点

- **多源收集**：通过 `NewsDataCollector` 动态拼接 Google News RSS（按查询词、区域、分类）并完成清洗、去重与时间排序。
- **双路情感分析**：优先使用 Hugging Face `cardiffnlp/twitter-roberta-base-sentiment-latest`，失败自动回落 VADER（compound ≥ 0.05 判为 positive，≤ -0.05 判为 negative），并缓存单条结果。
- **抽象式摘要**：`TextSummarizer` 结合 BART / DistilBART，对标题集合生成精炼摘要。
- **关键词与词云**：正则分词、停用词过滤与词频统计，支持词云配色与排除词自定义。
- **交互式可视化**：Plotly 情感分布、WordCloud 词云，以及按情感着色的标题卡片、摘要、Top 来源指标等。
- **导出与集成**：`DataExporter` 提供 CSV / JSON / WordCloud PNG，便于报告或下游分析。

---

## ⚙️ 环境与安装

- Python ≥ 3.8
- 依赖定义于 `requirements.txt`
- 首次运行需联网下载 Transformer 模型

```bash
git clone https://github.com/Sol-so-special/NewsBoost
cd NewsBoost
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧭 使用指南

1. 启动应用：`streamlit run app.py`
2. 侧边栏配置：
   - Query / Category / Region（US, UK, CA, AU, NG, IN, DE, FR）
   - `max_articles`（默认 50）、关键词过滤、词云排除词、色盘
3. 点击 `Analyze News` 触发数据抓取与分析流程：
   - 收集 → 过滤 → 情感分析 → 指标卡片 → 可视化 → 摘要/关键词 → 数据表 → 导出
4. 在标签页中查看：
   - Overview：情感上色的标题卡片与外链
   - Visualizations：情感分布柱状图 + 词云
   - Summary：摘要 + Top 关键词
   - Data：可选列的数据表
   - Export：CSV / JSON / WordCloud 下载

典型场景：
- **投研/宏观**：追踪行业或主题的情感波动，结合 Top 来源辨识信息来源结构。
- **竞品监测**：通过关键词过滤聚焦竞品名称/事件，观察舆论正负面趋势。
- **危机响应**：实时关注品牌相关负面情绪飙升并导出报告材料。

---

## 🏗️ 架构与模块

```
NewsBoost/
├── app.py                # Streamlit 前端 & 交互逻辑
├── news_boost/
│   ├── collector.py      # NewsDataCollector：RSS 构建与抓取
│   ├── analyzer.py       # SentimentAnalyzer：Transformer + VADER
│   ├── summarizer.py     # TextSummarizer：BART/DistilBART 摘要
│   ├── utils.py          # 缓存、批量分析、关键词工具
│   ├── visualizer.py     # DataVisualizer：Plotly + WordCloud
│   └── exporter.py       # DataExporter：CSV / JSON / PNG
├── requirements.txt
├── introduce.md          # 深度介绍（本文档来源之一）
└── README.md
```

**数据流**
```
参数配置 → NewsDataCollector → SentimentAnalyzer → TextSummarizer
          → Keyword Analysis → DataVisualizer → DataExporter
```

- `collector.py`：`get_google_news_url` 构建 RSS，`scrape_rss_feed` 抓取并清洗标题/时间/来源，采用 `@st.cache_data(ttl=300)` 缓存 5 分钟。
- `analyzer.py`：`load_models` 通过 `pipeline` 加载 Transformer，失败回退 VADER；`SentimentAnalyzer.analyze_text` 支持缓存与阈值映射。
- `summarizer.py`：`load_summarizer` 依次尝试 BART 与 DistilBART，`TextSummarizer.summarize_headlines` 对标题去重、随机打乱后裁剪至 ~1000 字符再摘要，并缓存结果。
- `utils.py`：`get_analyzers` 统一初始化各核心对象；`generate_keyword_analysis` 使用正则 `\b\w+\b` 分词、排除≤3字符与停用词，输出 Top 20 关键词。
- `visualizer.py`：固定顺序配色的情感柱状图、可配置色盘的 WordCloud（宽 800x400、最多 100 词、随机种子 42）。
- `exporter.py`：CSV/JSON 字节导出与词云 PNG 转换，便于 Streamlit 下载。

---

## 🧠 算法与实现细节

- **情感分析**：HF pipeline 直接给出标签；若转为 VADER，则按 compound 阈值映射并以 `@st.cache_data` 记忆同一标题的结果，显著减少重复计算。
- **摘要生成**：在输入层去重/乱序标题，避免摘要偏向最新资讯；短于 3 条标题或模型不可用时会返回友好提示。
- **关键词统计**：可传入 `exclude_words` 手动剔除品牌词；统计结果不仅用于词云，还在 Summary Tab 中按频次列出，方便快速检阅主题。
- **来源与指标**：Overview Tab 对每条新闻打上情感标签颜色，Summary Tab 统计 Top Source 及正负比，便于监控来源结构。

---

## 🔧 配置与性能

- `STREAMLIT_CACHE_DIR`：自定义 Streamlit 缓存目录（可选）
- `TRANSFORMERS_CACHE`：自定义模型缓存路径（可选）
- 主要缓存策略：
  - RSS 数据：`@st.cache_data(ttl=300)`
  - 模型加载：`@st.cache_resource`
  - 情感与摘要：`@st.cache_data` 按输入签名复用
- 性能建议：
  - 模型下载慢或内存紧张时，可直接依赖 VADER 或降低 `max_articles`
  - 服务器部署可通过 `TRANSFORMERS_CACHE` 指向本地 SSD、配置反向代理提升并发

---

## 📤 导出与集成

- CSV / JSON：适配 BI、数据仓库或 Notebook
- WordCloud PNG：用于报告或演示材料
- 可在 `DataExporter` 基础上拓展 Parquet / Excel / 数据库落盘

---

## 🎯 模型训练功能（train_sentiment_model/）

项目包含完整的情感分析模型训练模块，支持在多个数据集上训练和评估模型。

### 训练特性

- **自动GPU检测与多GPU支持**：
  - 自动检测系统中所有可用的GPU设备
  - 多GPU环境下使用 `DataParallel` 进行并行训练
  - 单GPU时自动使用该GPU，无GPU时回退到CPU训练
  - 根据GPU数量自动调整batch size和dataloader workers数量

- **Checkpoint保存策略**：
  - 训练过程中按步数保存checkpoint（`save_steps`）
  - 保留最近3个checkpoint（`save_total_limit=3`），避免磁盘空间浪费
  - 训练结束后自动加载最佳模型（`load_best_model_at_end=True`）
  - **自动保存最佳模型**：训练完成后，最佳模型会自动保存到独立的 `*_best_model` 目录
  - 最佳模型目录包含模型权重、tokenizer和训练参数配置

- **支持的数据集**：
  - **SST-2**：二分类情感分析（negative/positive）
  - **Amazon Reviews**：二分类情感分析（negative/positive）
  - **twitter sentiment**：二分类情感分析（negative/neutral/positive）

- **支持的模型**：
  - BERT (`bert-base-uncased`)
  - DeBERTa (`microsoft/deberta-base`)
  - RoBERTa (`roberta-base`)

### 训练示例

```bash
# 运行单个数据集实验
cd train_sentiment_model
python sst2_experiment.py
python amazon_experiment.py

# 或运行所有实验
python run_all_experiments.py
```

**数据加载**：
- 优先从本地 `dataset/` 目录加载 parquet 格式的数据集（速度更快）
- 如果本地文件不存在或有问题，自动回退到 Hugging Face 下载
- 支持 Git LFS 管理的大型数据集文件

### GPU使用说明

训练脚本会自动检测并利用所有可用GPU：

- **多GPU训练**：如果检测到多个GPU，会自动使用 `torch.nn.DataParallel` 进行数据并行训练
- **单GPU训练**：单个GPU时直接使用该GPU
- **CPU训练**：无GPU时自动回退到CPU（batch size会自动减小）

训练开始时会输出GPU信息：
```
检测到 2 个GPU设备:
  GPU 0: NVIDIA GeForce RTX 3090 (24.00 GB)
  GPU 1: NVIDIA GeForce RTX 3090 (24.00 GB)
使用 2 个GPU进行训练 (DataParallel)
```

### 模型保存说明

训练完成后，模型会保存在以下位置：

1. **训练checkpoint目录** (`*_results/`)：
   - 包含训练过程中的checkpoint文件
   - 保留最近3个checkpoint（包含最佳模型checkpoint）

2. **最佳模型目录** (`*_best_model/`)：
   - 自动保存训练过程中表现最好的模型
   - 包含完整的模型权重、tokenizer和配置
   - 可直接用于推理部署

3. **最终模型目录** (`*_model/`)：
   - 保存训练结束时的最终模型状态

### 评估结果

每个实验的评估结果（accuracy、F1-score、分类报告等）会保存在对应的 `results/` 子目录中的 `evaluation_results.json` 文件中。

---

## 🔄 可扩展性

- 数据源：接入更多 RSS / API（注意使用条款）
- 模型：替换、微调自定义情感或摘要模型
- 分析：加入实体识别、主题聚类、同义词归并、停用词表
- 可视化：添加时间序列、地域分布、来源地图等
- 部署：Docker 化、私有化、企业鉴权与审计

---

## 🛡️ 健壮性与错误处理

- RSS 抓取异常会通过 `st.error` 提示并返回空列表，避免页面崩溃。
- 模型加载失败会触发 `st.warning`，同时降级至 VADER 或跳过摘要模块。
- 当过滤结果为空、摘要输入过短或词云词表不足时，界面会给出可操作的提示语句，引导调参。
- 所有网络请求使用共享 `requests.Session` 与 UA 设置，可在 `collector.py` 中扩展重试和节流策略。

