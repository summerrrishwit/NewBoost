#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载与随机性控制相关的通用工具。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_local_dataset(
    train_path: str,
    test_path: Optional[str] = None,
    text_column: str = "text",
    label_column: str = "label",
    label_mapping: Optional[Dict[str, int]] = None,
    file_format: str = "auto",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    从本地 csv/json/jsonl 文件加载数据集，并返回标准化后的训练/测试 DataFrame。
    """
    if file_format == "auto":
        train_ext = Path(train_path).suffix.lower()
        if train_ext == ".csv":
            file_format = "csv"
        elif train_ext in [".json", ".jsonl"]:
            file_format = "json"
        else:
            raise ValueError("无法自动检测文件格式，请指定 file_format 参数")

    logger.info(f"从本地文件加载训练集: {train_path} (格式: {file_format})")
    if file_format == "csv":
        train_df = pd.read_csv(train_path)
    elif file_format == "json":
        if Path(train_path).suffix.lower() == ".jsonl":
            train_data = []
            with open(train_path, "r", encoding="utf-8") as f:
                for line in f:
                    train_data.append(json.loads(line.strip()))
            train_df = pd.DataFrame(train_data)
        else:
            with open(train_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                train_df = pd.DataFrame(data)
            elif isinstance(data, dict) and "data" in data:
                train_df = pd.DataFrame(data["data"])
            else:
                raise ValueError("JSON文件格式不正确，应为列表或包含'data'键的字典")
    else:
        raise ValueError(f"不支持的文件格式: {file_format}")

    if text_column not in train_df.columns:
        raise ValueError(f"训练集中未找到文本列: {text_column}。可用列: {train_df.columns.tolist()}")
    if label_column not in train_df.columns:
        raise ValueError(f"训练集中未找到标签列: {label_column}。可用列: {train_df.columns.tolist()}")

    if label_mapping:
        train_df["label"] = train_df[label_column].map(label_mapping)
        if train_df["label"].isna().any():
            missing_labels = train_df[train_df["label"].isna()][label_column].unique()
            raise ValueError(f"训练集中存在未映射的标签: {missing_labels}")
    else:
        if train_df[label_column].dtype == "object":
            unique_labels = sorted(train_df[label_column].unique())
            label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            logger.info(f"自动创建标签映射: {label_mapping}")
            train_df["label"] = train_df[label_column].map(label_mapping)
        else:
            train_df["label"] = train_df[label_column]

    train_df = train_df.rename(columns={text_column: "text"})

    if test_path:
        logger.info(f"从本地文件加载测试集: {test_path} (格式: {file_format})")
        if file_format == "csv":
            test_df = pd.read_csv(test_path)
        elif file_format == "json":
            if Path(test_path).suffix.lower() == ".jsonl":
                test_data = []
                with open(test_path, "r", encoding="utf-8") as f:
                    for line in f:
                        test_data.append(json.loads(line.strip()))
                test_df = pd.DataFrame(test_data)
            else:
                with open(test_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    test_df = pd.DataFrame(data)
                elif isinstance(data, dict) and "data" in data:
                    test_df = pd.DataFrame(data["data"])
                else:
                    raise ValueError("JSON文件格式不正确")

        if text_column not in test_df.columns:
            raise ValueError(f"测试集中未找到文本列: {text_column}")
        if label_column not in test_df.columns:
            raise ValueError(f"测试集中未找到标签列: {label_column}")

        if label_mapping:
            test_df["label"] = test_df[label_column].map(label_mapping)
        else:
            if test_df[label_column].dtype == "object":
                test_df["label"] = test_df[label_column].map(label_mapping)
            else:
                test_df["label"] = test_df[label_column]

        test_df = test_df.rename(columns={text_column: "text"})
    else:
        logger.info("未提供测试集，从训练集中分割（80%训练，20%测试）")
        train_df, test_df = train_test_split(
            train_df, test_size=0.2, random_state=42, stratify=train_df["label"]
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    train_df = train_df[["text", "label"]].copy()
    test_df = test_df[["text", "label"]].copy()

    logger.info(f"训练集大小: {len(train_df)}")
    logger.info(f"测试集大小: {len(test_df)}")
    logger.info(f"标签分布:\n{train_df['label'].value_counts().sort_index()}")

    return train_df, test_df


def set_global_seed(seed: int) -> None:
    """统一设置随机种子，保证实验可复现。"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
