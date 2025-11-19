## 项目介绍（introduce.md）

### 1. 概览
NewsBoost 是一款基于 Streamlit 的新闻智能分析应用，聚合 RSS 新闻数据，结合情感分析、摘要生成、关键词统计与可视化，帮助使用者快速洞察媒体叙事与舆情趋势。系统采用模块化架构，内置缓存与导出能力，适合在业务分析、投研、竞品与危机响应等场景中快速落地。

- **核心能力**：新闻收集、情感分析（Transformer + VADER 兜底）、摘要生成（BART/DistilBART）、关键词分析、可视化（Plotly/WordCloud）、数据导出（CSV/JSON/PNG）。
- **前端框架**：Streamlit（`app.py`）。
- **核心包**：`news_boost/` 包含 `collector.py`、`analyzer.py`、`summarizer.py`、`visualizer.py`、`utils.py`、`exporter.py`。

---

### 2. 架构与数据流

```
用户参数（查询/区域/分类/过滤/可视化）
        │
        ▼
NewsDataCollector（`collector.py`）
  - 生成 Google News RSS URL
  - 抓取并清洗条目（时间排序、标题清洗、来源抽取）
        │  输出：articles（title, link, published, summary, source）
        ▼
SentimentAnalyzer（`analyzer.py`）
  - 优先使用 Hugging Face `cardiffnlp/twitter-roberta-base-sentiment-latest`
  - 失败则回退至 VADER（基于 compound 阈值映射）
        │  输出：sentiment_label ∈ {positive, neutral, negative}
        ▼
TextSummarizer（`summarizer.py`）
  - 优先使用 `facebook/bart-large-cnn`
  - 回退 `sshleifer/distilbart-cnn-12-6`
        │  输出：新闻标题集合的抽象式摘要
        ▼
关键词分析（`utils.py::generate_keyword_analysis`）
  - 正则分词、长度过滤、停用词/排除词过滤、词频统计（Top-N）
        │
        ▼
可视化（`visualizer.py`）
  - 情感分布柱状图（Plotly，固定顺序/配色）
  - 词云（WordCloud，可配置色盘/排除词）
        │
        ▼
导出（`exporter.py`）
  - CSV / JSON / WordCloud PNG 下载
```

数据在 `app.py` 中以交互式流程组织：侧边栏配置 → 抓取 → 过滤 → 情感分析 → 指标与图表 → 摘要与关键词 → 数据表 → 导出。

---

### 3. 运行环境与安装

- Python ≥ 3.8
- 依赖：见根目录 `requirements.txt`
- 首次运行会自动下载 Transformer 模型（需网络）

安装与启动（示例）：
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

### 4. 配置与环境变量

- `STREAMLIT_CACHE_DIR`：可选，自定义 Streamlit 缓存目录
- `TRANSFORMERS_CACHE`：可选，自定义 Hugging Face 模型缓存路径

性能相关可在代码中调整：
- `@st.cache_data(ttl=300)`：RSS 抓取结果缓存 5 分钟（`collector.py`）
- 模型加载使用 `@st.cache_resource`，复用会话级资源（`analyzer.py`、`summarizer.py`、`utils.py`）

---

### 5. 模块与关键函数

#### 5.1 `news_boost/collector.py`
- 类 `NewsDataCollector`
  - `get_google_news_url(query, region='US', language='en', category=None)`：生成 Google News RSS URL。优先 `query`，否则按 `category` 或默认头条。
  - `scrape_rss_feed(url, max_articles=50)`：抓取并解析 RSS，清洗标题（去除 ` - Source` 尾缀）、解析发布时间并按时间降序；返回裁剪后的文章列表。
  - `collect_news_data(query=None, region='US', category=None, max_articles=50)`：主入口，拼 URL 并调用抓取函数。
用户输入 query / category
        ↓
get_google_news_url() 生成 RSS 链接
        ↓
feedparser 读取 RSS XML 数据
        ↓
提取标题、摘要、发布时间
        ↓
清洗标题、格式化时间
        ↓
按时间排序（最新在前）
        ↓
返回前 50 条新闻结果
        ↓
Streamlit 缓存（5 分钟）

实现要点：
- 使用 `feedparser` 解析 RSS；
- `dateutil.parser` 解析发布时间的兜底；
- `requests.Session` 设置 UA 与持久连接；
- Streamlit 缓存减少重复请求。

#### 5.2 `news_boost/analyzer.py`
- 函数 `load_models()`：
  - 首选 HF `pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")`；
  - 失败时警告并回退 `VADER`；返回（分析器, 是否 HF）。
- 类 `SentimentAnalyzer`
  - `analyze_text(text)`：带缓存的单条文本分析；
    - 若 HF 可用，直接调用 `pipeline` 得到标签；
    - 否则以 VADER 的 `compound` 分数映射为 `positive`/`neutral`/`negative`（阈值：≥0.05、(−0.05, 0.05)、≤−0.05）。

实现要点：
- `@st.cache_resource` 缓存模型加载；`@st.cache_data` 缓存每条文本分析结果；
- VADER 作可靠兜底，保证在无 GPU/无模型下载时仍可用。

#### 5.3 `news_boost/summarizer.py`
- 函数 `load_summarizer()`：
  - 首选 `facebook/bart-large-cnn`；失败回退 `sshleifer/distilbart-cnn-12-6`；两者均失败则标记不可用并提示原因。
- 类 `TextSummarizer`
  - `summarize_headlines(headlines)`：合并部分标题（上限约 1000 字符），生成抽象式摘要；输入过短或模型不可用返回提示。

实现要点：
- 对输入长度进行限制以适配模型 token 限制；
- 随机打乱后拼接，增加摘要覆盖的多样性；
- `@st.cache_data` 缓存按输入集合稳定复用结果。

#### 5.4 `news_boost/utils.py`
- `get_analyzers()`：统一初始化 `collector`、`analyzer`、`summarizer`、`visualizer` 并缓存为资源。
- `process_sentiment_analysis(titles)`：对标题列表做批量情感分析并缓存。
- `generate_keyword_analysis(titles, exclude_words)`：
  - 文本小写与正则分词；
  - 过滤长度≤3与排除词；
  - 统计词频并返回 Top 20 `(keyword, frequency)`。

#### 5.5 `news_boost/visualizer.py`
- 类 `DataVisualizer`
  - `plot_sentiment_distribution(sentiments)`：固定顺序（Positive/Neutral/Negative）与指定配色（#00aa00/#ffffff/#ff4444）的柱状图（Plotly）。
  - `create_wordcloud(text_data, exclude_words, colormap='viridis')`：生成可配置词云，支持排除词、连字符、去纯数字等清洗策略。

实现要点：
- 使用 `seaborn-v0_8` 样式；
- 词云宽 800×400、最多 100 词，`random_state=42` 保持可重复性。

#### 5.6 `news_boost/exporter.py`
- 类 `DataExporter`
  - `to_csv(df)`：导出 CSV 字节串；
  - `to_json(df)`：记录级 JSON（ISO 日期格式）；
  - `wordcloud_to_png(wordcloud)`：将 WordCloud 转 PNG 字节，供下载按钮使用。

---

### 6. 前端交互与界面（`app.py`）

- 侧边栏参数：
  - 查询词、区域（US/UK/CA/AU/NG/IN/DE/FR）、分类（General/Business/Technology/Health/Science/Sports）、最大文章数、关键词过滤、词云排除词、色盘等。
- 主流程按钮：`Analyze News` 触发采集与分析；
- 指标卡片：总数、正/中/负占比、正负比、Top 来源；
- 标签页：
  - Overview：按情感着色的标题卡片（链接可点击跳转原文）；
  - Visualizations：情感分布柱状图 + 词云；
  - Summary：基于标题集合的摘要 + Top 关键词表；
  - Data：可选择列的 DataFrame 视图；
  - Export：CSV/JSON/WordCloud PNG 下载。

---

### 7. 算法与实现细节

- **情感分析**：
  - 首选 HF RoBERTa 微调模型，标签直接来自 `pipeline`；
  - 失败兜底 VADER：根据 `compound` 分数阈值映射到三分类；
  - 通过 `@st.cache_data` 改善重复分析时的延迟。

- **摘要生成**：
  - 抽象式摘要（BART/DistilBART）；
  - 合并标题时控制最大字符数，避免超过模型输入限制；
  - 对输入过短与异常场景做了防御性处理。

- **关键词分析**：
  - 正则 `\b\w+\b` 分词，小写化、长度与排除词过滤；
  - `Counter.most_common(20)` 输出 Top 20；
  - 词云在可视化层进一步做清洗（移除标点、纯数字、短词等）。

---

### 8. 性能与缓存策略

- RSS 抓取与数据处理：`@st.cache_data(ttl=300)`，5 分钟内相同请求命中缓存；
- 模型与可视化组件：`@st.cache_resource`/`@st.cache_data`，跨交互复用；
- 前端上提供进度条与懒加载（如摘要生成在 Tab 内触发）。

建议：
- 大量数据或低内存环境可降低 `max_articles`、禁用摘要或更换小模型；
- 可将 `TRANSFORMERS_CACHE` 指向更快的本地盘；
- 服务器化部署时适当提升进程/会话并发能力（如通过反向代理 + 会话隔离）。

---

### 9. 导出与集成

- CSV/JSON：便于与 BI / 数据仓库 / Notebook 集成；
- 词云 PNG：用于报告或汇报材料；
- 可在 `DataExporter` 基础上扩展 Parquet、Excel、数据库落盘等能力。

---

### 10. 错误处理与健壮性

- RSS 抓取异常：UI 以 `st.error` 显示错误并返回空列表；
- 模型下载失败：以 `st.warning` 提示并 fallback；
- 输入校验：空结果、过滤后无数据、摘要输入过短等均有提示；
- 网络与速率：默认使用 Session 与 UA；可按需加入节流与重试策略。

---

### 11. 测试与示例

- `tests/` 目录包含基础用例与演示脚本，可用于本地校验环境与依赖；
- 结合 `README.md` 的使用章节可快速复现实验流程。

---

### 12. 典型使用场景

- 金融与投研：追踪市场/行业/宏观关键词的情感与热度波动；
- 竞品与品牌：监测竞品动态与品牌舆情摘要；
- 危机响应：出现负面情感飙升时，快速聚合要点并导出汇报。

---

### 13. 可定制与扩展建议

- 数据源：扩展至更多 RSS 或 API（需遵守各源条款）；
- 模型：替换/微调更适配领域的情感模型与摘要模型；
- 规则：加入自定义停用词表、同义词归并、实体识别与主题聚类；
- 可视化：增加时间序列、来源分布、地域地图等；
- 部署：Docker 化、私有化部署、企业级鉴权与审计。


