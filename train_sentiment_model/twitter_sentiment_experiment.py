#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Twitter Sentiment 情感分析实验
使用BERT、DeBERTa、RoBERTa模型进行三分类情感分析（正面、负面、中性）
"""

import os
import logging
import random
import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from experiment_utils import BaseSentimentTrainer, load_local_dataset
from model_config import get_available_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TwitterSentimentTrainer(BaseSentimentTrainer):
    """Twitter情感分析训练器"""
    
    def __init__(self, model_name, num_labels=3):
        super().__init__(model_name, num_labels, "Twitter Sentiment")
        self.label_encoder = None
        
    def load_data(self, sample_size=15000, train_path=None, test_path=None, 
                  text_column='text', label_column='label', file_format='auto'):
        """
        加载Twitter Sentiment数据集
        
        Args:
            sample_size: 从Hugging Face数据集采样的大小（仅当使用HF数据集时）
            train_path: 本地训练集文件路径（CSV或JSON），如果提供则从本地读取
            test_path: 本地测试集文件路径（CSV或JSON），如果提供则从本地读取
            text_column: 本地文件的文本列名（默认'text'）
            label_column: 本地文件的标签列名（默认'label'）
            file_format: 本地文件格式，'auto'（自动检测）、'csv'或'json'
        
        Returns:
            train_df, test_df: 训练集和测试集的DataFrame
        """
        # 如果提供了本地文件路径，从本地读取
        if train_path:
            logger.info("从本地文件加载Twitter Sentiment数据集...")
            # Twitter三分类标签映射
            label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
            train_df, test_df = load_local_dataset(
                train_path=train_path,
                test_path=test_path,
                text_column=text_column,
                label_column=label_column,
                label_mapping=label_mapping,
                file_format=file_format
            )
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(['negative', 'neutral', 'positive'])
            return train_df, test_df
        
        # 否则从Hugging Face加载
        logger.info("从Hugging Face加载Twitter Sentiment数据集...")
        dataset = load_dataset("tweet_eval", "sentiment")
        
        def convert_labels(example):
            sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            example['sentiment'] = sentiment_map[example['label']]
            return example
        
        dataset = dataset.map(convert_labels)
        
        # 采样数据（如果数据集太大）
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
        """数据预处理和分词（Twitter推文通常较短，使用max_length=128）"""
        return super().tokenize_data(train_df, test_df, max_length=max_length)
    
    def get_target_names(self):
        """获取分类标签名称"""
        return ['negative', 'neutral', 'positive']


def run_twitter_experiment(model_name, model_display_name):
    """运行Twitter Sentiment实验"""
    print(f"\n{'='*60}")
    print(f"开始 {model_display_name} 在 Twitter Sentiment 数据集上的实验")
    print(f"{'='*60}")
    
    trainer_obj = TwitterSentimentTrainer(model_name, num_labels=3)
    train_df, test_df = trainer_obj.load_data(sample_size=15000)
    trainer_obj.setup_model()
    dataset = trainer_obj.tokenize_data(train_df, test_df)
    
    # 训练并自动保存最佳模型
    results_dir = f"./results/twitter_{model_name.replace('/', '_')}_results"
    trainer = trainer_obj.train(dataset, results_dir)
    
    # 评估最佳模型
    predictions = trainer_obj.evaluate(trainer, dataset, model_name)
    
    # 注意：最佳模型已在train()方法中自动保存到 *_best_model 目录
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
