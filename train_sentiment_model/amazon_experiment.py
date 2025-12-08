#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Amazon Reviews 情感分析实验
使用BERT、DeBERTa、RoBERTa模型进行二分类情感分析
"""

import os
import logging
import random
import glob
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from experiment_utils import BaseSentimentTrainer, EvalMeta, save_evaluation_results
from model_config import get_available_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")


def _is_git_lfs_pointer(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            return first_line == "version https://git-lfs.github.com/spec/v1"
    except:
        return False


def _check_and_get_parquet_files(data_dir, split_name):
    pattern = os.path.join(data_dir, f"{split_name}-*.parquet")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"未找到 {split_name} 的 parquet 文件: {pattern}")

    for file_path in files:
        if _is_git_lfs_pointer(file_path):
            raise ValueError(
                f"文件 {file_path} 是 Git LFS 指针文件，不是实际的 parquet 文件。\n"
                f"请先运行 'git lfs pull' 或手动下载实际的 parquet 文件。"
            )
    
    return files


class AmazonReviewsTrainer(BaseSentimentTrainer):
    def __init__(self, model_name, num_labels=2, finetune_method="full"):
        super().__init__(model_name, num_labels, "Amazon Reviews", finetune_method=finetune_method)
        self.label_encoder = None
        
    def load_data(self, sample_size=10000, use_local_parquet=True):
        # 尝试从本地 parquet 文件加载
        ap_dir = os.path.join(DATASET_DIR, "amazon_polarity", "amazon_polarity")
        use_local = False
        
        if use_local_parquet and os.path.exists(ap_dir):
            try:
                logger.info("尝试从本地 parquet 文件加载Amazon Reviews数据集...")
                train_files = _check_and_get_parquet_files(ap_dir, "train")
                test_files = _check_and_get_parquet_files(ap_dir, "test")

                data_files = {
                    "train": train_files,
                    "test": test_files,
                }

                dataset = load_dataset("parquet", data_files=data_files)
                use_local = True
                logger.info("✓ 从本地 parquet 文件加载数据集成功")
            except (ValueError, FileNotFoundError) as e:
                # Git LFS 指针文件或文件不存在，使用 HuggingFace
                logger.warning(f"本地 parquet 文件不可用: {str(e)}")
                logger.info("→ 从 HuggingFace 加载数据集...")
            except Exception as e:
                # 其他错误，也尝试从 HuggingFace 加载
                error_msg = str(e)
                if "Parquet magic bytes" in error_msg or "not a parquet file" in error_msg:
                    logger.warning(f"本地 parquet 文件格式错误: {error_msg}")
                    logger.info("→ 从 HuggingFace 加载数据集...")
                else:
                    raise
        
        if not use_local:
            logger.info("从Hugging Face加载Amazon Reviews数据集 (amazon_polarity)...")
            dataset = load_dataset("amazon_polarity")

        def combine_text(example):
            if "text" not in example:
                if "title" in example and "content" in example:
                    example["text"] = example["title"] + " " + example["content"]
                elif "content" in example:
                    example["text"] = example["content"]
            return example

        dataset = dataset.map(combine_text)

        if sample_size and len(dataset["train"]) > sample_size:
            train_indices = random.sample(range(len(dataset["train"])), sample_size)
            test_sample_size = max(1, sample_size // 5)
            test_indices = random.sample(range(len(dataset["test"])), test_sample_size)

            train_data = dataset["train"].select(train_indices)
            test_data = dataset["test"].select(test_indices)
        else:
            train_data = dataset["train"]
            test_data = dataset["test"]

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

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([0, 1])

        logger.info(f"训练集大小: {len(train_df)}")
        logger.info(f"测试集大小: {len(test_df)}")
        logger.info(f"评级分布 (1=负面, 2=正面):\n{train_df['rating'].value_counts().sort_index()}")

        return train_df, test_df
    
    def get_training_args(self, output_dir, **kwargs):
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
        return ['negative', 'positive']
    
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


def run_amazon_experiment(model_name, model_display_name, finetune_method="full"):
    print(f"\n{'='*60}")
    print(f"开始 {model_display_name} 在 Amazon Reviews 数据集上的实验")
    print(f"微调方法: {finetune_method}")
    print(f"{'='*60}")
    
    trainer_obj = AmazonReviewsTrainer(model_name, num_labels=2, finetune_method=finetune_method)
    
    train_df, test_df = trainer_obj.load_data(sample_size=10000)
    
    trainer_obj.setup_model(move_to_gpu=False)
    dataset = trainer_obj.tokenize_data(train_df, test_df)
    trainer_obj.move_model_to_gpu()
    
    # 在结果目录中包含微调方法
    method_suffix = f"_{finetune_method}" if finetune_method != "full" else ""
    results_dir = f"./results/amazon_{model_name.replace('/', '_')}{method_suffix}_results"
    trainer = trainer_obj.train(dataset, results_dir)
    
    predictions = trainer_obj.evaluate(trainer, dataset, model_name)
    
    # 这里保留原有的save_model调用以保持兼容性（保存最终状态）
    final_model_dir = f"./results/amazon_{model_name.replace('/', '_')}{method_suffix}_model"
    trainer_obj.save_model(final_model_dir)
    
    print(f"\n{model_display_name} 在 Amazon Reviews 数据集上的实验完成!")
    print(f"最佳模型已保存到: {results_dir.replace('_results', '_best_model')}")
    return trainer_obj


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="运行 Amazon Reviews 情感分析实验")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=["full", "lora", "prefix", "all"],
        default=["all"],
        help="选择微调方法: 'full'（全参数微调）、'lora'（LoRA微调）、'prefix'（Prefix Tuning）、'all'（所有方法，默认）"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["bert", "deberta", "roberta", "all"],
        default=["all"],
        help="选择模型: 'bert'、'deberta'、'roberta'、'all'（所有模型，默认）"
    )
    
    args = parser.parse_args()
    
    available_models = get_available_models()
    print(f"可用的本地模型: {available_models}")
    
    # 解析模型选择
    model_map = {
        "bert": ("bert-base-uncased", "BERT"),
        "deberta": ("deberta-base", "DeBERTa"),
        "roberta": ("roberta-base", "RoBERTa")
    }
    
    if "all" in args.models:
        models = list(model_map.values())
    else:
        models = [model_map[m] for m in args.models if m in model_map]
    
    # 解析微调方法选择
    if "all" in args.methods:
        finetune_methods = ["full", "lora", "prefix"]
    else:
        finetune_methods = args.methods
    
    print(f"\n选择的模型: {[m[1] for m in models]}")
    print(f"选择的微调方法: {finetune_methods}")
    print(f"{'='*60}\n")
    
    results = {}
    for model_name, display_name in models:
        for finetune_method in finetune_methods:
            try:
                result = run_amazon_experiment(model_name, display_name, finetune_method=finetune_method)
                results[f"{display_name}_{finetune_method}"] = result
            except Exception as e:
                print(f"实验 {display_name} ({finetune_method}) 失败: {str(e)}")
                continue
    
    print(f"\n{'='*60}")
    print("所有 Amazon Reviews 实验完成!")
    print(f"{'='*60}")
