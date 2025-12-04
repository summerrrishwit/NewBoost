#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Amazon Reviews 情感分析实验
使用BERT、DeBERTa、RoBERTa模型进行二分类情感分析
"""

import os
import logging
import random
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from experiment_utils import BaseSentimentTrainer
from model_config import get_available_models
from experiment_utils import EvalMeta, save_evaluation_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class AmazonReviewsTrainer(BaseSentimentTrainer):
    """Amazon Reviews情感分析训练器（真实二分类：负面 / 正面）"""
    
    def __init__(self, model_name, num_labels=2):
        super().__init__(model_name, num_labels, "Amazon Reviews")
        self.label_encoder = None
        
    def load_data(self, sample_size=10000):
        """加载Amazon Reviews数据集"""
        logger.info("正在加载Amazon Reviews数据集 (amazon_polarity)...")
        dataset = load_dataset("amazon_polarity")

        def combine_text(example):
            example["text"] = example["title"] + " " + example["content"]
            return example

        dataset = dataset.map(combine_text)

        if sample_size and len(dataset["train"]) > sample_size:
            train_indices = random.sample(range(len(dataset["train"])), sample_size)
            # 测试集按 1/5 的比例采样
            test_sample_size = max(1, sample_size // 5)
            test_indices = random.sample(range(len(dataset["test"])), test_sample_size)

            train_data = dataset["train"].select(train_indices)
            test_data = dataset["test"].select(test_indices)
        else:
            train_data = dataset["train"]
            test_data = dataset["test"]

        # 转换为 DataFrame
        # amazon_polarity 原始标签：0 = negative, 1 = positive
        train_df = pd.DataFrame(
            {
                "text": train_data["text"],
                "rating": [lbl + 1 for lbl in train_data["label"]],
                "label": train_data["label"],
            }
        )

        test_df = pd.DataFrame(
            {
                "text": test_data["text"],
                "rating": [lbl + 1 for lbl in test_data["label"]],
                "label": test_data["label"],
            }
        )

        # 创建标签编码器（这里主要是保持接口一致，标签为 0/1）
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([0, 1])

        logger.info(f"训练集大小: {len(train_df)}")
        logger.info(f"测试集大小: {len(test_df)}")
        logger.info(f"评级分布 (1=负面, 2=正面):\n{train_df['rating'].value_counts().sort_index()}")

        return train_df, test_df
    
    def get_training_args(self, output_dir, **kwargs):
        """获取训练参数配置（Amazon使用不同的参数）"""
        # 调用基类方法，但覆盖Amazon特定的参数
        training_kwargs = {
            "per_device_train_batch_size": 8 if len(self.gpu_ids) > 0 else 4,
            "per_device_eval_batch_size": 16 if len(self.gpu_ids) > 0 else 8,
            "learning_rate": 1e-5,  # 降低学习率
            "num_train_epochs": 2,  # 减少训练轮数
            "eval_steps": 200,  # 更频繁的评估
            "save_steps": 200,
            "warmup_steps": 50,  # 减少预热步数
            "weight_decay": 0.1,  # 增加权重衰减
            "logging_steps": 50,
            "lr_scheduler_type": "cosine",  # 使用余弦学习率调度
        }
        training_kwargs.update(kwargs)
        return super().get_training_args(output_dir, **training_kwargs)
    
    def get_target_names(self):
        """获取分类标签名称"""
        return ['negative', 'positive']
    
    def evaluate(self, trainer, dataset, model_name, seed=42):
        """评估模型（使用experiment_utils保存结果）"""
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
        
        print(f"\n=== {self.dataset_name} 测试集评估结果（二分类：negative / positive） ===")
        print(
            classification_report(
                test_labels,
                preds,
                target_names=target_names,
                zero_division=0,
            )
        )
        
        result_dir = f"./results/amazon_{model_name.replace('/', '_')}_results"
        meta = EvalMeta(
            model_name=model_name,
            dataset=self.dataset_name,
            seed=seed,
            extra_config={"num_labels": self.num_labels},
        )
        save_evaluation_results(
            result_dir=result_dir,
            meta=meta,
            report=report,
            y_true=test_labels,
            y_pred=preds,
        )
        logger.info(f"评估结果已保存到: {result_dir}/evaluation_results.json")
        
        return predictions


def run_amazon_experiment(model_name, model_display_name):
    """运行Amazon Reviews二分类实验"""
    print(f"\n{'='*60}")
    print(f"开始 {model_display_name} 在 Amazon Reviews 数据集上的实验")
    print(f"{'='*60}")
    
    trainer_obj = AmazonReviewsTrainer(model_name, num_labels=2)
    
    train_df, test_df = trainer_obj.load_data(sample_size=10000)
    
    trainer_obj.setup_model()
    dataset = trainer_obj.tokenize_data(train_df, test_df)
    
    # 训练并自动保存最佳模型
    results_dir = f"./results/amazon_{model_name.replace('/', '_')}_results"
    trainer = trainer_obj.train(dataset, results_dir)
    
    # 评估最佳模型
    predictions = trainer_obj.evaluate(trainer, dataset, model_name)
    
    # 注意：最佳模型已在train()方法中自动保存到 *_best_model 目录
    # 这里保留原有的save_model调用以保持兼容性（保存最终状态）
    final_model_dir = f"./results/amazon_{model_name.replace('/', '_')}_model"
    trainer_obj.save_model(final_model_dir)
    
    print(f"\n{model_display_name} 在 Amazon Reviews 数据集上的实验完成!")
    print(f"最佳模型已保存到: {results_dir.replace('_results', '_best_model')}")
    return trainer_obj


if __name__ == "__main__":
    available_models = get_available_models()
    print(f"可用的本地模型: {available_models}")
    
    models = [
        ("bert-base-uncased", "BERT"),
        ("microsoft/deberta-base", "DeBERTa"),
        ("roberta-base", "RoBERTa")
    ]
    
    results = {}
    for model_name, display_name in models:
        try:
            result = run_amazon_experiment(model_name, display_name)
            results[display_name] = result
        except Exception as e:
            print(f"实验 {display_name} 失败: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print("所有 Amazon Reviews 实验完成!")
    print(f"{'='*60}")
