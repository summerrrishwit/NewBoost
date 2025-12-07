#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置管理模块
提供应用和训练相关的所有配置
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


@dataclass
class ModelConfig:
    """模型相关配置"""
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    summarizer_model: str = "facebook/bart-large-cnn"
    fallback_summarizer: str = "sshleifer/distilbart-cnn-12-6"
    
    # 本地模型路径（如果使用本地模型）
    local_sentiment_model: Optional[str] = None
    local_summarizer_model: Optional[str] = None
    local_fallback_summarizer: Optional[str] = None


@dataclass
class CacheConfig:
    """缓存相关配置"""
    cache_ttl: int = 300  # 5分钟
    enable_cache: bool = True
    cache_max_entries: int = 1000


@dataclass
class PathsConfig:
    """路径相关配置"""
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    models_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    
    def __post_init__(self):
        """初始化路径"""
        self.models_dir = self.base_dir / "model"
        self.results_dir = self.base_dir / "train_sentiment_model" / "results"
        self.logs_dir = self.base_dir / "logs"
        self.data_dir = self.base_dir / "data"
        
        # 确保目录存在
        for dir_path in [self.models_dir, self.results_dir, self.logs_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingConfig:
    """训练相关配置"""
    training_models: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("bert-base-uncased", "BERT"),
        ("microsoft/deberta-base", "DeBERTa"),
        ("roberta-base", "RoBERTa")
    ])
    default_batch_size: int = 16
    default_learning_rate: float = 2e-5
    default_epochs: int = 3
    default_max_length: int = 256
    save_total_limit: int = 1  # 只保存一个checkpoint


@dataclass
class AppConfig:
    """应用主配置"""
    # 新闻收集配置
    default_region: str = "US"
    default_language: str = "en"
    default_max_articles: int = 50
    default_category: Optional[str] = None
    
    # RSS配置
    rss_timeout: int = 10  # 秒
    rss_max_retries: int = 3
    
    # 日志配置
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # UI配置
    page_title: str = "NewsBoost"
    page_icon: str = "🌐"
    layout: str = "wide"
    
    # 可视化配置
    wordcloud_max_words: int = 150
    wordcloud_width: int = 1000
    wordcloud_height: int = 500
    default_colormap: str = "viridis"
    
    # 导出配置
    export_formats: List[str] = field(default_factory=lambda: ["csv", "json", "png", "txt"])


@dataclass
class Config:
    """总配置类"""
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    app: AppConfig = field(default_factory=AppConfig)
    
    def __post_init__(self):
        """从环境变量加载配置"""
        # 模型配置
        if os.getenv("SENTIMENT_MODEL"):
            self.model.sentiment_model = os.getenv("SENTIMENT_MODEL")
        if os.getenv("SUMMARIZER_MODEL"):
            self.model.summarizer_model = os.getenv("SUMMARIZER_MODEL")
        if os.getenv("LOCAL_SENTIMENT_MODEL"):
            self.model.local_sentiment_model = os.getenv("LOCAL_SENTIMENT_MODEL")
        if os.getenv("LOCAL_SUMMARIZER_MODEL"):
            self.model.local_summarizer_model = os.getenv("LOCAL_SUMMARIZER_MODEL")
        
        # 缓存配置
        if os.getenv("CACHE_TTL"):
            self.cache.cache_ttl = int(os.getenv("CACHE_TTL"))
        if os.getenv("ENABLE_CACHE"):
            self.cache.enable_cache = os.getenv("ENABLE_CACHE").lower() == "true"
        
        # 应用配置
        if os.getenv("LOG_LEVEL"):
            self.app.log_level = os.getenv("LOG_LEVEL")
        if os.getenv("RSS_TIMEOUT"):
            self.app.rss_timeout = int(os.getenv("RSS_TIMEOUT"))


# 全局配置实例
_config: Optional[Config] = None


def get_config() -> Config:
    """获取配置实例（单例模式）"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config() -> Config:
    """重新加载配置"""
    global _config
    _config = Config()
    return _config
