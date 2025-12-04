#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SST-2 (Stanford Sentiment Treebank) 情感分析实验
使用BERT、DeBERTa、RoBERTa模型进行二分类情感分析
"""

import os
import logging
import pandas as pd
from datasets import load_dataset
from experiment_utils import BaseSentimentTrainer
from model_config import get_available_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SentimentTrainer(BaseSentimentTrainer):
    """SST-2情感分析训练器"""
    
    def __init__(self, model_name, num_labels=2):
        super().__init__(model_name, num_labels, "SST-2")
        
    def load_data(self):
        """加载SST-2数据集"""
        logger.info("正在加载SST-2数据集...")
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
        """获取分类标签名称"""
        return ['negative', 'positive']


def run_sst2_experiment(model_name, model_display_name):
    """运行SST-2实验"""
    print(f"\n{'='*60}")
    print(f"开始 {model_display_name} 在 SST-2 数据集上的实验")
    print(f"{'='*60}")
    
    trainer_obj = SentimentTrainer(model_name, num_labels=2)
    train_df, test_df = trainer_obj.load_data()
    trainer_obj.setup_model()
    
    dataset = trainer_obj.tokenize_data(train_df, test_df)
    
    # 训练并自动保存最佳模型
    results_dir = f"./results/sst2_{model_name.replace('/', '_')}_results"
    trainer = trainer_obj.train(dataset, results_dir)
    
    # 评估最佳模型
    predictions = trainer_obj.evaluate(trainer, dataset, model_name)
    
    # 注意：最佳模型已在train()方法中自动保存到 *_best_model 目录
    # 这里保留原有的save_model调用以保持兼容性（保存最终状态）
    final_model_dir = f"./results/sst2_{model_name.replace('/', '_')}_model"
    trainer_obj.save_model(final_model_dir)
    
    print(f"\n{model_display_name} 在 SST-2 数据集上的实验完成!")
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
            result = run_sst2_experiment(model_name, display_name)
            results[display_name] = result
        except Exception as e:
            print(f"实验 {display_name} 失败: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print("所有 SST-2 实验完成!")
    print(f"{'='*60}")
