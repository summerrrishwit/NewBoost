#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Twitter Sentiment 情感分析实验
使用BERT、DeBERTa、RoBERTa模型进行三分类情感分析（正面、负面、中性）
"""

import os
import logging
import random
import glob
import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
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


class TwitterSentimentTrainer(BaseSentimentTrainer):
    def __init__(self, model_name, num_labels=3):
        super().__init__(model_name, num_labels, "Twitter Sentiment")
        self.label_encoder = None
        
    def load_data(self, sample_size=15000, use_local_parquet=True):
        # 尝试从本地 parquet 文件加载
        sentiment_dir = os.path.join(DATASET_DIR, "tweet_eval", "sentiment")
        use_local = False
        
        if use_local_parquet and os.path.exists(sentiment_dir):
            try:
                logger.info("尝试从本地 parquet 文件加载Twitter Sentiment数据集...")
                train_files = _check_and_get_parquet_files(sentiment_dir, "train")
                validation_files = _check_and_get_parquet_files(sentiment_dir, "validation")
                test_files = _check_and_get_parquet_files(sentiment_dir, "test")

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
            logger.info("从Hugging Face加载Twitter Sentiment数据集...")
            dataset = load_dataset("tweet_eval", "sentiment")
        
        def convert_labels(example):
            sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            example['sentiment'] = sentiment_map[example['label']]
            return example
        
        dataset = dataset.map(convert_labels)
        
        if sample_size and len(dataset['train']) > sample_size:
            train_indices = random.sample(range(len(dataset['train'])), sample_size)
            test_indices = random.sample(range(len(dataset['test'])), sample_size // 5)
            
            train_data = dataset['train'].select(train_indices)
            test_data = dataset['test'].select(test_indices)
        else:
            train_data = dataset['train']
            test_data = dataset['test']
        
        train_df = pd.DataFrame({
            'text': train_data['text'],
            'label': train_data['label'],
            'sentiment': train_data['sentiment']
        })
        
        test_df = pd.DataFrame({
            'text': test_data['text'],
            'label': test_data['label'],
            'sentiment': test_data['sentiment']
        })
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['negative', 'neutral', 'positive'])
        
        logger.info(f"训练集大小: {len(train_df)}")
        logger.info(f"测试集大小: {len(test_df)}")
        logger.info(f"情感分布:\n{train_df['sentiment'].value_counts()}")
        
        return train_df, test_df
    
    def tokenize_data(self, train_df, test_df, max_length=128):
        return super().tokenize_data(train_df, test_df, max_length=max_length)
    
    def get_target_names(self):
        return ['negative', 'neutral', 'positive']


def run_twitter_experiment(model_name, model_display_name):
    print(f"\n{'='*60}")
    print(f"开始 {model_display_name} 在 Twitter Sentiment 数据集上的实验")
    print(f"{'='*60}")
    
    trainer_obj = TwitterSentimentTrainer(model_name, num_labels=3)
    train_df, test_df = trainer_obj.load_data(sample_size=15000)
    
    trainer_obj.setup_model(move_to_gpu=False)
    dataset = trainer_obj.tokenize_data(train_df, test_df)
    trainer_obj.move_model_to_gpu()
    
    results_dir = f"./results/twitter_{model_name.replace('/', '_')}_results"
    trainer = trainer_obj.train(dataset, results_dir)
    
    predictions = trainer_obj.evaluate(trainer, dataset, model_name)
    
    # 这里保留原有的save_model调用以保持兼容性（保存最终状态）
    final_model_dir = f"./results/twitter_{model_name.replace('/', '_')}_model"
    trainer_obj.save_model(final_model_dir)
    
    print(f"\n{model_display_name} 在 Twitter Sentiment 数据集上的实验完成!")
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
            result = run_twitter_experiment(model_name, display_name)
            results[display_name] = result
        except Exception as e:
            print(f"实验 {display_name} 失败: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print("所有 Twitter Sentiment 实验完成!")
    print(f"{'='*60}")
