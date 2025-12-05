#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用实验工具函数和基础训练器类：
- 全局随机种子设置
- 评估结果统一保存
- 本地数据集加载（支持CSV、JSON、JSONL格式）
- 基础训练器类（BaseSentimentTrainer）
"""

import os
import json
import shutil
import logging
import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from model_config import get_model_path
from gpu_utils import get_available_gpus, setup_multi_gpu

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==================== 实验工具函数 ====================

def load_local_dataset(train_path, test_path=None, text_column='text', label_column='label', 
                       label_mapping=None, file_format='auto'):
    """
    从本地文件加载数据集
    
    Args:
        train_path: 训练集文件路径（CSV或JSON）
        test_path: 测试集文件路径（CSV或JSON），如果为None则从train_path中分割
        text_column: 文本列名（默认'text'）
        label_column: 标签列名（默认'label'）
        label_mapping: 标签映射字典，例如 {'positive': 1, 'negative': 0} 或 {'negative': 0, 'neutral': 1, 'positive': 2}
        file_format: 文件格式，'auto'（自动检测）、'csv'或'json'
        
    Returns:
        train_df, test_df: 训练集和测试集的DataFrame
    """
    import json
    from pathlib import Path
    
    # 自动检测文件格式
    if file_format == 'auto':
        train_ext = Path(train_path).suffix.lower()
        if train_ext == '.csv':
            file_format = 'csv'
        elif train_ext in ['.json', '.jsonl']:
            file_format = 'json'
        else:
            raise ValueError(f"无法自动检测文件格式，请指定file_format参数。支持格式: .csv, .json, .jsonl")
    
    # 读取训练集
    logger.info(f"从本地文件加载训练集: {train_path} (格式: {file_format})")
    if file_format == 'csv':
        train_df = pd.read_csv(train_path)
    elif file_format == 'json':
        if Path(train_path).suffix.lower() == '.jsonl':
            # JSONL格式：每行一个JSON对象
            train_data = []
            with open(train_path, 'r', encoding='utf-8') as f:
                for line in f:
                    train_data.append(json.loads(line.strip()))
            train_df = pd.DataFrame(train_data)
        else:
            # 标准JSON格式
            with open(train_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                train_df = pd.DataFrame(data)
            elif isinstance(data, dict) and 'data' in data:
                train_df = pd.DataFrame(data['data'])
            else:
                raise ValueError("JSON文件格式不正确，应为列表或包含'data'键的字典")
    else:
        raise ValueError(f"不支持的文件格式: {file_format}")
    
    # 检查必需的列
    if text_column not in train_df.columns:
        raise ValueError(f"训练集中未找到文本列: {text_column}。可用列: {train_df.columns.tolist()}")
    if label_column not in train_df.columns:
        raise ValueError(f"训练集中未找到标签列: {label_column}。可用列: {train_df.columns.tolist()}")
    
    # 处理标签映射
    if label_mapping:
        # 将标签映射为数字
        train_df['label'] = train_df[label_column].map(label_mapping)
        if train_df['label'].isna().any():
            missing_labels = train_df[train_df['label'].isna()][label_column].unique()
            raise ValueError(f"训练集中存在未映射的标签: {missing_labels}")
    else:
        # 如果没有提供映射，尝试将标签转换为数字
        if train_df[label_column].dtype == 'object':
            unique_labels = sorted(train_df[label_column].unique())
            label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            logger.info(f"自动创建标签映射: {label_mapping}")
            train_df['label'] = train_df[label_column].map(label_mapping)
        else:
            train_df['label'] = train_df[label_column]
    
    # 重命名文本列
    train_df = train_df.rename(columns={text_column: 'text'})
    
    # 读取测试集
    if test_path:
        logger.info(f"从本地文件加载测试集: {test_path} (格式: {file_format})")
        if file_format == 'csv':
            test_df = pd.read_csv(test_path)
        elif file_format == 'json':
            if Path(test_path).suffix.lower() == '.jsonl':
                test_data = []
                with open(test_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        test_data.append(json.loads(line.strip()))
                test_df = pd.DataFrame(test_data)
            else:
                with open(test_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    test_df = pd.DataFrame(data)
                elif isinstance(data, dict) and 'data' in data:
                    test_df = pd.DataFrame(data['data'])
                else:
                    raise ValueError("JSON文件格式不正确")
        
        # 检查测试集的列
        if text_column not in test_df.columns:
            raise ValueError(f"测试集中未找到文本列: {text_column}")
        if label_column not in test_df.columns:
            raise ValueError(f"测试集中未找到标签列: {label_column}")
        
        # 应用相同的标签映射
        if label_mapping:
            test_df['label'] = test_df[label_column].map(label_mapping)
        else:
            if test_df[label_column].dtype == 'object':
                test_df['label'] = test_df[label_column].map(label_mapping)
            else:
                test_df['label'] = test_df[label_column]
        
        test_df = test_df.rename(columns={text_column: 'text'})
    else:
        # 如果没有提供测试集，从训练集中分割（80/20）
        logger.info("未提供测试集，从训练集中分割（80%训练，20%测试）")
        train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
    
    # 确保只保留需要的列
    train_df = train_df[['text', 'label']].copy()
    test_df = test_df[['text', 'label']].copy()
    
    logger.info(f"训练集大小: {len(train_df)}")
    logger.info(f"测试集大小: {len(test_df)}")
    logger.info(f"标签分布:\n{train_df['label'].value_counts().sort_index()}")
    
    return train_df, test_df


@dataclass
class EvalMeta:
    """评估元信息数据类"""
    model_name: str
    dataset: str
    seed: int
    extra_config: Optional[Dict[str, Any]] = None


def set_global_seed(seed: int) -> None:
    """统一设置随机种子，保证实验可复现。"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_evaluation_results(
    result_dir: str,
    meta: EvalMeta,
    report: Dict[str, Any],
    y_true: Iterable[int],
    y_pred: Iterable[int],
) -> None:
    """
    统一保存评估结果到 JSON 文件，并附加混淆矩阵等信息。

    Parameters
    ----------
    result_dir : str
        结果保存目录。
    meta : EvalMeta
        实验的元信息（模型名、数据集名、seed、额外配置等）。
    report : dict
        sklearn classification_report(output_dict=True) 的结果。
    y_true : Iterable[int]
        测试集真实标签。
    y_pred : Iterable[int]
        模型预测标签。
    """
    os.makedirs(result_dir, exist_ok=True)

    y_true_arr = np.array(list(y_true))
    y_pred_arr = np.array(list(y_pred))

    # 混淆矩阵
    cm = confusion_matrix(y_true_arr, y_pred_arr).tolist()

    eval_results = {
        "meta": asdict(meta),
        "timestamp": datetime.now().isoformat(),
        "test_accuracy": report.get("accuracy"),
        "test_f1_macro": report.get("macro avg", {}).get("f1-score"),
        "test_f1_weighted": report.get("weighted avg", {}).get("f1-score"),
        "classification_report": report,
        "confusion_matrix": cm,
        "predictions": y_pred_arr.tolist(),
        "true_labels": y_true_arr.tolist(),
    }

    out_path = os.path.join(result_dir, "evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)


# ==================== 基础训练器类 ====================

class BaseSentimentTrainer(ABC):
    """情感分析训练器基类"""
    
    def __init__(self, model_name, num_labels, dataset_name):
        """
        初始化训练器
        
        Args:
            model_name: 模型名称
            num_labels: 标签数量
            dataset_name: 数据集名称（用于日志和保存路径）
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.dataset_name = dataset_name
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.gpu_ids = []
    
    @abstractmethod
    def load_data(self, **kwargs):
        """
        加载数据集（子类必须实现）
        
        Returns:
            train_df, test_df: 训练集和测试集的DataFrame
        """
        pass
    
    def setup_model(self):
        """设置模型和分词器"""
        model_path = get_model_path(self.model_name)
        logger.info(f"正在加载模型: {self.model_name} -> {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=self.num_labels
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 检测并设置GPU
        self.gpu_ids = get_available_gpus()
        self.model = setup_multi_gpu(self.model, self.gpu_ids)
    
    def tokenize_data(self, train_df, test_df, max_length=256):
        """
        数据预处理和分词
        
        Args:
            train_df: 训练集DataFrame
            test_df: 测试集DataFrame
            max_length: 最大序列长度
            
        Returns:
            dataset: 处理后的DatasetDict
        """
        logger.info("正在进行数据预处理...")
        
        def tokenize(batch):
            return self.tokenizer(
                batch["text"],
                padding=False,
                truncation=True,
                max_length=max_length
            )
        
        train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
        test_dataset = Dataset.from_pandas(test_df[["text", "label"]])
        dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
        
        dataset = dataset.map(
            tokenize,
            batched=True,
            batch_size=32,
            num_proc=4,
            remove_columns=["text"]
        )
        
        return dataset
    
    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        
        acc = accuracy_score(labels, preds)
        f1_weighted = f1_score(labels, preds, average="weighted")
        f1_macro = f1_score(labels, preds, average="macro")
        
        return {
            "accuracy": acc,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro
        }
    
    def get_training_args(self, output_dir, **kwargs):
        """
        获取训练参数配置
        
        Args:
            output_dir: 输出目录
            **kwargs: 可选的训练参数覆盖
            
        Returns:
            TrainingArguments对象
        """
        # 根据GPU数量调整batch size
        num_gpus = len(self.gpu_ids) if self.gpu_ids else 0
        
        # 默认参数
        default_args = {
            "per_device_train_batch_size": 16 if num_gpus > 0 else 8,
            "per_device_eval_batch_size": 32 if num_gpus > 0 else 16,
            "learning_rate": 2e-5,
            "num_train_epochs": 3,
            "eval_steps": 500,
            "save_steps": 500,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "logging_steps": 100,
        }
        
        # 允许子类通过kwargs覆盖默认参数
        default_args.update(kwargs)
        
        if num_gpus > 0:
            fp16 = True
            dataloader_num_workers = min(4, num_gpus * 2)
        else:
            fp16 = False
            dataloader_num_workers = 2
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="steps",
            eval_steps=default_args["eval_steps"],
            save_strategy="steps",
            save_steps=default_args["save_steps"],
            learning_rate=default_args["learning_rate"],
            per_device_train_batch_size=default_args["per_device_train_batch_size"],
            per_device_eval_batch_size=default_args["per_device_eval_batch_size"],
            num_train_epochs=default_args["num_train_epochs"],
            weight_decay=default_args["weight_decay"],
            logging_dir=f"{output_dir}/logs",
            logging_steps=default_args["logging_steps"],
            report_to="none",
            disable_tqdm=False,
            fp16=fp16,
            dataloader_num_workers=dataloader_num_workers,
            load_best_model_at_end=True,  # 训练结束后自动加载最佳模型
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=3,  # 保存最近3个checkpoint（包含最佳模型）
            warmup_steps=default_args["warmup_steps"],
            # 多GPU相关设置
            ddp_find_unused_parameters=False if num_gpus > 1 else None,
        )
        
        return training_args
    
    def train(self, dataset, output_dir, **training_kwargs):
        """
        训练模型
        
        Args:
            dataset: 处理后的数据集
            output_dir: 输出目录
            **training_kwargs: 传递给get_training_args的额外参数
            
        Returns:
            trainer: Trainer对象
        """
        logger.info("开始训练模型...")
        
        training_args = self.get_training_args(output_dir, **training_kwargs)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        trainer.train()
        
        # 训练结束后，最佳模型已通过 load_best_model_at_end=True 自动加载
        # 保存最佳模型到单独目录
        best_model_dir = output_dir.replace("_results", "_best_model")
        logger.info(f"保存最佳模型到: {best_model_dir}")
        
        # 如果模型使用了DataParallel，需要获取原始模型
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(best_model_dir)
        self.tokenizer.save_pretrained(best_model_dir)
        
        # 同时保存训练状态和最佳checkpoint信息
        best_checkpoint_dir = trainer.state.best_model_checkpoint if hasattr(trainer.state, 'best_model_checkpoint') else None
        if best_checkpoint_dir and os.path.exists(best_checkpoint_dir):
            logger.info(f"最佳checkpoint路径: {best_checkpoint_dir}")
            # 复制最佳checkpoint的training_args.bin到最佳模型目录
            training_args_path = os.path.join(best_checkpoint_dir, "training_args.bin")
            if os.path.exists(training_args_path):
                shutil.copy2(training_args_path, os.path.join(best_model_dir, "training_args.bin"))
        
        logger.info(f"最佳模型已保存到: {best_model_dir}")
        
        return trainer
    
    @abstractmethod
    def get_target_names(self):
        """
        获取分类标签名称（子类必须实现）
        
        Returns:
            list: 标签名称列表
        """
        pass
    
    def evaluate(self, trainer, dataset, model_name, seed=42):
        """
        评估模型
        
        Args:
            trainer: Trainer对象
            dataset: 数据集
            model_name: 模型名称
            seed: 随机种子
            
        Returns:
            predictions: 预测结果
        """
        logger.info("正在评估模型...")
        
        predictions = trainer.predict(dataset["test"])
        preds = np.argmax(predictions.predictions, axis=1)
        
        test_labels = dataset["test"]["label"]
        target_names = self.get_target_names()
        
        report = classification_report(
            test_labels,
            preds,
            target_names=target_names,
            zero_division=0,
            output_dict=True
        )
        
        print(f"\n=== {self.dataset_name} 测试集评估结果 ===")
        print(classification_report(
            test_labels,
            preds,
            target_names=target_names,
            zero_division=0
        ))
        
        result_dir = f"./results/{self.dataset_name.lower().replace(' ', '_')}_{model_name.replace('/', '_')}_results"
        os.makedirs(result_dir, exist_ok=True)
        
        eval_results = {
            "model_name": model_name,
            "dataset": self.dataset_name,
            "timestamp": datetime.now().isoformat(),
            "test_accuracy": report['accuracy'],
            "test_f1_macro": report['macro avg']['f1-score'],
            "test_f1_weighted": report['weighted avg']['f1-score'],
            "classification_report": report,
            "predictions": preds.tolist(),
            "true_labels": test_labels
        }
        
        with open(f"{result_dir}/evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"评估结果已保存到: {result_dir}/evaluation_results.json")
        
        return predictions
    
    def save_model(self, output_dir):
        """
        保存模型
        
        Args:
            output_dir: 输出目录
        """
        logger.info(f"正在保存模型到 {output_dir}")
        # 如果模型使用了DataParallel，需要获取原始模型
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("模型保存完成!")
