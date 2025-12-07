#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SST-2 (Stanford Sentiment Treebank) 情感分析实验
使用BERT、DeBERTa、RoBERTa模型进行二分类情感分析
"""

import os
import logging
import glob
import pandas as pd
from datasets import load_dataset
from experiment_utils import BaseSentimentTrainer
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
    
    # 检查是否是 Git LFS 指针文件
    for file_path in files:
        if _is_git_lfs_pointer(file_path):
            raise ValueError(
                f"文件 {file_path} 是 Git LFS 指针文件，不是实际的 parquet 文件。\n"
                f"请先运行 'git lfs pull' 或手动下载实际的 parquet 文件。"
            )
    
    return files


class SentimentTrainer(BaseSentimentTrainer):
    def __init__(self, model_name, num_labels=2, finetune_method="full"):
        super().__init__(model_name, num_labels, "SST-2", finetune_method=finetune_method)
        
    def load_data(self, use_local_parquet=True):
        sst2_dir = os.path.join(DATASET_DIR, "glue", "sst2")
        use_local = False
        
        if use_local_parquet and os.path.exists(sst2_dir):
            try:
                logger.info("尝试从本地 parquet 文件加载SST-2数据集...")
                train_files = _check_and_get_parquet_files(sst2_dir, "train")
                validation_files = _check_and_get_parquet_files(sst2_dir, "validation")
                test_files = _check_and_get_parquet_files(sst2_dir, "test")

                data_files = {
                    "train": train_files,
                    "validation": validation_files,
                    "test": test_files,
                }

                dataset = load_dataset("parquet", data_files=data_files)
                use_local = True
                logger.info("✓ 从本地 parquet 文件加载数据集成功")
            except (ValueError, FileNotFoundError) as e:
                logger.warning(f"本地 parquet 文件不可用: {str(e)}")
                logger.info("→ 从 HuggingFace 加载数据集...")
            except Exception as e:
                error_msg = str(e)
                if "Parquet magic bytes" in error_msg or "not a parquet file" in error_msg:
                    logger.warning(f"本地 parquet 文件格式错误: {error_msg}")
                    logger.info("→ 从 HuggingFace 加载数据集...")
                else:
                    raise
        
        if not use_local:
            logger.info("从Hugging Face加载SST-2数据集...")
            dataset = load_dataset("glue", "sst2")
        
        def convert_labels(example):
            example['sentiment'] = 'positive' if example['label'] == 1 else 'negative'
            return example
        
        dataset = dataset.map(convert_labels)
        
        train_df = pd.DataFrame({
            'text': dataset['train']['sentence'],
            'label': dataset['train']['label'],
            'sentiment': dataset['train']['sentiment']
        })
        
        test_df = pd.DataFrame({
            'text': dataset['validation']['sentence'],
            'label': dataset['validation']['label'],
            'sentiment': dataset['validation']['sentiment']
        })
        
        logger.info(f"训练集大小: {len(train_df)}")
        logger.info(f"测试集大小: {len(test_df)}")
        logger.info(f"类别分布:\n{train_df['sentiment'].value_counts()}")
        
        return train_df, test_df
    
    def get_target_names(self):
        return ['negative', 'positive']


def run_sst2_experiment(model_name, model_display_name, finetune_method="full"):
    print(f"\n{'='*60}")
    print(f"开始 {model_display_name} 在 SST-2 数据集上的实验")
    print(f"微调方法: {finetune_method}")
    print(f"{'='*60}")
    
    trainer_obj = SentimentTrainer(model_name, num_labels=2, finetune_method=finetune_method)
    train_df, test_df = trainer_obj.load_data()
    
    # 先加载模型但不移到GPU（避免tokenize时的CUDA multiprocessing错误）
    trainer_obj.setup_model(move_to_gpu=False)
    dataset = trainer_obj.tokenize_data(train_df, test_df)
    # tokenize完成后再将模型移到GPU
    trainer_obj.move_model_to_gpu()
    
    # 在结果目录中包含微调方法
    method_suffix = f"_{finetune_method}" if finetune_method != "full" else ""
    results_dir = f"./results/sst2_{model_name.replace('/', '_')}{method_suffix}_results"
    trainer = trainer_obj.train(dataset, results_dir)
    
    predictions = trainer_obj.evaluate(trainer, dataset, model_name)
    
    final_model_dir = f"./results/sst2_{model_name.replace('/', '_')}{method_suffix}_model"
    trainer_obj.save_model(final_model_dir)
    
    print(f"\n{model_display_name} 在 SST-2 数据集上的实验完成!")
    print(f"最佳模型已保存到: {results_dir.replace('_results', '_best_model')}")
    return trainer_obj


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="运行 SST-2 情感分析实验")
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
        "deberta": ("microsoft/deberta-base", "DeBERTa"),
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
                result = run_sst2_experiment(model_name, display_name, finetune_method=finetune_method)
                results[f"{display_name}_{finetune_method}"] = result
            except Exception as e:
                print(f"实验 {display_name} ({finetune_method}) 失败: {str(e)}")
                continue
    
    print(f"\n{'='*60}")
    print("所有 SST-2 实验完成!")
    print(f"{'='*60}")