#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用实验工具函数和基础训练器类：
- 全局随机种子设置
- 评估结果统一保存
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

try:
    from peft import (
        LoraConfig,
        PrefixTuningConfig,
        TaskType,
        get_peft_model,
        PeftModel
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT库未安装，LoRA和Prefix Tuning功能不可用")


# ==================== 实验工具函数 ====================

def load_local_dataset(train_path, test_path=None, text_column='text', label_column='label', 
                       label_mapping=None, file_format='auto'):
    import json
    from pathlib import Path
    
    if file_format == 'auto':
        train_ext = Path(train_path).suffix.lower()
        if train_ext == '.csv':
            file_format = 'csv'
        elif train_ext in ['.json', '.jsonl']:
            file_format = 'json'
        else:
            raise ValueError(f"无法自动检测文件格式，请指定file_format参数。支持格式: .csv, .json, .jsonl")
    
    logger.info(f"从本地文件加载训练集: {train_path} (格式: {file_format})")
    if file_format == 'csv':
        train_df = pd.read_csv(train_path)
    elif file_format == 'json':
        if Path(train_path).suffix.lower() == '.jsonl':
            train_data = []
            with open(train_path, 'r', encoding='utf-8') as f:
                for line in f:
                    train_data.append(json.loads(line.strip()))
            train_df = pd.DataFrame(train_data)
        else:
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
    
    if text_column not in train_df.columns:
        raise ValueError(f"训练集中未找到文本列: {text_column}。可用列: {train_df.columns.tolist()}")
    if label_column not in train_df.columns:
        raise ValueError(f"训练集中未找到标签列: {label_column}。可用列: {train_df.columns.tolist()}")
    
    if label_mapping:
        train_df['label'] = train_df[label_column].map(label_mapping)
        if train_df['label'].isna().any():
            missing_labels = train_df[train_df['label'].isna()][label_column].unique()
            raise ValueError(f"训练集中存在未映射的标签: {missing_labels}")
    else:
        if train_df[label_column].dtype == 'object':
            unique_labels = sorted(train_df[label_column].unique())
            label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            logger.info(f"自动创建标签映射: {label_mapping}")
            train_df['label'] = train_df[label_column].map(label_mapping)
        else:
            train_df['label'] = train_df[label_column]
    
    train_df = train_df.rename(columns={text_column: 'text'})
    
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


def setup_lora_model(model, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=None):
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT库未安装，无法使用LoRA功能。请运行: pip install peft")
    
    if target_modules is None:
        model_type = model.config.model_type if hasattr(model.config, 'model_type') else None
        if model_type in ['bert', 'roberta', 'deberta']:
            target_modules = ["query", "value"]
        else:
            target_modules = ["query", "value"]
            logger.warning(f"未知模型类型 {model_type}，使用默认目标模块: {target_modules}")
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    logger.info(f"LoRA配置: r={r}, alpha={lora_alpha}, dropout={lora_dropout}, target_modules={target_modules}")
    logger.info(f"可训练参数数量: {model.num_parameters(only_trainable=True):,}")
    logger.info(f"总参数数量: {model.num_parameters():,}")
    
    return model


def setup_prefix_tuning_model(model, num_virtual_tokens=20, prefix_projection=False):
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT库未安装，无法使用Prefix Tuning功能。请运行: pip install peft")
    
    # 获取模型配置
    if hasattr(model.config, 'hidden_size'):
        hidden_size = model.config.hidden_size
    elif hasattr(model.config, 'd_model'):
        hidden_size = model.config.d_model
    else:
        hidden_size = 768  # 默认值
        logger.warning(f"无法检测模型hidden_size，使用默认值: {hidden_size}")
    
    prefix_config = PrefixTuningConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=num_virtual_tokens,
        encoder_hidden_size=hidden_size if prefix_projection else None,
    )
    
    # 由于 PEFT 0.13.2 的 bug，get_peft_model 和 PeftModel.__init__ 会传递不支持的 low_cpu_mem_usage 参数
    # 我们使用 workaround：monkey patch add_adapter 方法来忽略这个参数
    try:
        # 尝试使用 get_peft_model（如果库已修复）
        model = get_peft_model(model, prefix_config)
    except TypeError as e:
        if "low_cpu_mem_usage" in str(e):
            # 使用 workaround：monkey patch add_adapter 来忽略 low_cpu_mem_usage 参数
            logger.warning("检测到 PEFT 库的 low_cpu_mem_usage bug，使用 workaround...")
            import peft.peft_model
            original_add_adapter = peft.peft_model.PeftModel.add_adapter
            
            def patched_add_adapter(self, adapter_name, peft_config, **kwargs):
                # 移除 low_cpu_mem_usage 参数（如果存在）
                kwargs.pop('low_cpu_mem_usage', None)
                # 调用原始方法，不传递 low_cpu_mem_usage
                return original_add_adapter(self, adapter_name, peft_config)
            
            # 临时替换方法
            peft.peft_model.PeftModel.add_adapter = patched_add_adapter
            try:
                model = PeftModel(model, prefix_config, adapter_name="default")
            finally:
                # 恢复原方法
                peft.peft_model.PeftModel.add_adapter = original_add_adapter
        else:
            raise
    
    logger.info(f"Prefix Tuning配置: num_virtual_tokens={num_virtual_tokens}, prefix_projection={prefix_projection}")
    logger.info(f"可训练参数数量: {model.num_parameters(only_trainable=True):,}")
    logger.info(f"总参数数量: {model.num_parameters():,}")
    
    return model


# ==================== 基础训练器类 ====================

class BaseSentimentTrainer(ABC):
    """情感分析训练器基类"""
    
    def __init__(self, model_name, num_labels, dataset_name, finetune_method="full"):
        """
        初始化训练器
        
        Args:
            model_name: 模型名称
            num_labels: 标签数量
            dataset_name: 数据集名称（用于日志和保存路径）
            finetune_method: 微调方法，可选 "full"（全参数微调）、"lora"（LoRA微调）、"prefix"（Prefix Tuning）
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.dataset_name = dataset_name
        self.finetune_method = finetune_method
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.gpu_ids = []
        self._model_on_gpu = False
    
    @abstractmethod
    def load_data(self, **kwargs):
        pass
    
    def setup_model(self, move_to_gpu=True, **kwargs):
        """
        设置模型和分词器
        
        Args:
            move_to_gpu: 是否立即将模型移动到GPU（默认True）
            **kwargs: 额外的模型设置参数
                - 对于LoRA: r, lora_alpha, lora_dropout, target_modules
                - 对于Prefix Tuning: num_virtual_tokens, prefix_projection
        """
        model_path = get_model_path(self.model_name)
        logger.info(f"正在加载模型: {self.model_name} -> {model_path}")
        logger.info(f"微调方法: {self.finetune_method}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=self.num_labels
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.finetune_method == "lora":
            r = kwargs.get("r", 8)
            lora_alpha = kwargs.get("lora_alpha", 16)
            lora_dropout = kwargs.get("lora_dropout", 0.1)
            target_modules = kwargs.get("target_modules", None)
            self.model = setup_lora_model(
                self.model,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules
            )
        elif self.finetune_method == "prefix":
            num_virtual_tokens = kwargs.get("num_virtual_tokens", 20)
            prefix_projection = kwargs.get("prefix_projection", False)
            self.model = setup_prefix_tuning_model(
                self.model,
                num_virtual_tokens=num_virtual_tokens,
                prefix_projection=prefix_projection
            )
        elif self.finetune_method != "full":
            raise ValueError(f"未知的微调方法: {self.finetune_method}。支持的方法: 'full', 'lora', 'prefix'")
        
        self.gpu_ids = get_available_gpus()
        
        if move_to_gpu:
            self.move_model_to_gpu()
        else:
            logger.info("模型将保持在CPU上，稍后可通过 move_model_to_gpu() 移动到GPU")
    
    def move_model_to_gpu(self):
        """将模型移动到GPU"""
        if self.model is None:
            raise ValueError("模型尚未初始化，请先调用 setup_model()")
        
        if self._model_on_gpu:
            logger.info("模型已在GPU上")
            return
        
        if len(self.gpu_ids) > 0:
            logger.info(f"将模型移动到GPU: {self.gpu_ids}")
            self.model = setup_multi_gpu(self.model, self.gpu_ids)
            self._model_on_gpu = True
        else:
            logger.warning("未检测到GPU，模型将保持在CPU上")
    
    def tokenize_data(self, train_df, test_df, max_length=256):
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
        num_gpus = len(self.gpu_ids) if self.gpu_ids else 0
        
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
            report_to="tensorboard",
            disable_tqdm=False,
            fp16=fp16,
            dataloader_num_workers=dataloader_num_workers,
            load_best_model_at_end=True,  # 训练结束后自动加载最佳模型
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=3,  # 保存最近3个checkpoint（包含最佳模型）
            warmup_steps=default_args["warmup_steps"],
            ddp_find_unused_parameters=False if num_gpus > 1 else None,
        )
        
        return training_args
    
    def train(self, dataset, output_dir, **training_kwargs):
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
        
        # 检查是否是PEFT模型
        if PEFT_AVAILABLE and hasattr(model_to_save, 'save_pretrained'):
            # PEFT模型会自动处理保存
            model_to_save.save_pretrained(best_model_dir)
        else:
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
        pass
    
    def evaluate(self, trainer, dataset, model_name, seed=42):
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
        logger.info(f"正在保存模型到 {output_dir}")
        # 如果模型使用了DataParallel，需要获取原始模型
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 检查是否是PEFT模型
        if PEFT_AVAILABLE and hasattr(model_to_save, 'save_pretrained'):
            # PEFT模型会自动处理保存
            model_to_save.save_pretrained(output_dir)
        else:
            model_to_save.save_pretrained(output_dir)
        
        self.tokenizer.save_pretrained(output_dir)
        logger.info("模型保存完成!")
