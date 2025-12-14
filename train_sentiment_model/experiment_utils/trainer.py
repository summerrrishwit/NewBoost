#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础训练器实现，供各数据集的训练脚本继承使用。
"""

import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Iterable

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from model_config import get_model_path
from gpu_utils import get_available_gpus, setup_multi_gpu
from .peft_utils import PEFT_AVAILABLE, setup_lora_model, setup_prefix_tuning_model

logger = logging.getLogger(__name__)


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
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=self.num_labels)

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
                target_modules=target_modules,
            )
        elif self.finetune_method == "prefix":
            num_virtual_tokens = kwargs.get("num_virtual_tokens", 20)
            prefix_projection = kwargs.get("prefix_projection", False)
            self.model = setup_prefix_tuning_model(
                self.model,
                num_virtual_tokens=num_virtual_tokens,
                prefix_projection=prefix_projection,
            )
        elif self.finetune_method != "full":
            raise ValueError(f"未知的微调方法: {self.finetune_method}。支持的方法: 'full', 'lora', 'prefix'")

        self.gpu_ids = get_available_gpus()

        if move_to_gpu:
            self.move_model_to_gpu()
        else:
            logger.info("模型将保持在CPU上，稍后可通过 move_model_to_gpu() 移动到GPU")

    def move_model_to_gpu(self):
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
            return self.tokenizer(batch["text"], padding=False, truncation=True, max_length=max_length)

        train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
        test_dataset = Dataset.from_pandas(test_df[["text", "label"]])
        dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

        dataset = dataset.map(tokenize, batched=True, batch_size=32, num_proc=4, remove_columns=["text"])

        return dataset

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)

        acc = accuracy_score(labels, preds)
        f1_weighted = f1_score(labels, preds, average="weighted")
        f1_macro = f1_score(labels, preds, average="macro")

        return {"accuracy": acc, "f1_weighted": f1_weighted, "f1_macro": f1_macro}

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
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=3,
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

        best_model_dir = output_dir.replace("_results", "_best_model")
        logger.info(f"保存最佳模型到: {best_model_dir}")

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model

        if PEFT_AVAILABLE and hasattr(model_to_save, "save_pretrained"):
            model_to_save.save_pretrained(best_model_dir)
        else:
            model_to_save.save_pretrained(best_model_dir)

        self.tokenizer.save_pretrained(best_model_dir)

        best_checkpoint_dir = trainer.state.best_model_checkpoint if hasattr(trainer.state, "best_model_checkpoint") else None
        if best_checkpoint_dir and os.path.exists(best_checkpoint_dir):
            logger.info(f"最佳checkpoint路径: {best_checkpoint_dir}")
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
            output_dict=True,
        )

        print(f"\n=== {self.dataset_name} 测试集评估结果 ===")
        print(
            classification_report(
                test_labels,
                preds,
                target_names=target_names,
                zero_division=0,
            )
        )

        result_dir = f"./results/{self.dataset_name.lower().replace(' ', '_')}_{model_name.replace('/', '_')}_results"
        os.makedirs(result_dir, exist_ok=True)

        eval_results = {
            "model_name": model_name,
            "dataset": self.dataset_name,
            "timestamp": datetime.now().isoformat(),
            "test_accuracy": report["accuracy"],
            "test_f1_macro": report["macro avg"]["f1-score"],
            "test_f1_weighted": report["weighted avg"]["f1-score"],
            "classification_report": report,
            "predictions": preds.tolist(),
            "true_labels": test_labels,
        }

        with open(f"{result_dir}/evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)

        logger.info(f"评估结果已保存到: {result_dir}/evaluation_results.json")

        return predictions

    def save_model(self, output_dir):
        logger.info(f"正在保存模型到 {output_dir}")
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model

        if PEFT_AVAILABLE and hasattr(model_to_save, "save_pretrained"):
            model_to_save.save_pretrained(output_dir)
        else:
            model_to_save.save_pretrained(output_dir)

        self.tokenizer.save_pretrained(output_dir)
        logger.info("模型保存完成!")
