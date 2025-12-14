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
from experiment_utils import (
    BaseSentimentTrainer,
    build_common_arg_parser,
    check_and_get_parquet_files,
    parse_selected_methods,
    parse_selected_models,
    start_tensorboard,
    tensorboard_should_start,
)
from model_config import get_available_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")


class TwitterSentimentTrainer(BaseSentimentTrainer):
    def __init__(self, model_name, num_labels=3, finetune_method="full"):
        super().__init__(model_name, num_labels, "Twitter Sentiment", finetune_method=finetune_method)
        self.label_encoder = None
        
    def load_data(self, sample_size=15000, use_local_parquet=True):
        # 尝试从本地 parquet 文件加载
        sentiment_dir = os.path.join(DATASET_DIR, "tweet_eval", "sentiment")
        use_local = False
        
        if use_local_parquet and os.path.exists(sentiment_dir):
            try:
                logger.info("尝试从本地 parquet 文件加载Twitter Sentiment数据集...")
                train_files = check_and_get_parquet_files(sentiment_dir, "train")
                validation_files = check_and_get_parquet_files(sentiment_dir, "validation")
                test_files = check_and_get_parquet_files(sentiment_dir, "test")

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


def run_twitter_experiment(model_name, model_display_name, finetune_method="full", start_tb=True, tb_port=6006):
    print(f"\n{'='*60}")
    print(f"开始 {model_display_name} 在 Twitter Sentiment 数据集上的实验")
    print(f"微调方法: {finetune_method}")
    print(f"{'='*60}")
    
    trainer_obj = TwitterSentimentTrainer(model_name, num_labels=3, finetune_method=finetune_method)
    train_df, test_df = trainer_obj.load_data(sample_size=15000)
    
    trainer_obj.setup_model(move_to_gpu=False)
    dataset = trainer_obj.tokenize_data(train_df, test_df)
    trainer_obj.move_model_to_gpu()
    
    # 在结果目录中包含微调方法
    method_suffix = f"_{finetune_method}" if finetune_method != "full" else ""
    results_dir = f"./results/twitter_{model_name.replace('/', '_')}{method_suffix}_results"
    
    # 启动 TensorBoard（如果启用）
    tb_process = None
    if start_tb:
        # TensorBoard 日志目录是 results_dir 下的 logs 子目录
        tb_log_dir = os.path.join(results_dir, "logs")
        tb_process = start_tensorboard(tb_log_dir, tb_port)
    
    trainer = trainer_obj.train(dataset, results_dir)
    
    predictions = trainer_obj.evaluate(trainer, dataset, model_name)
    
    # 这里保留原有的save_model调用以保持兼容性（保存最终状态）
    final_model_dir = f"./results/twitter_{model_name.replace('/', '_')}{method_suffix}_model"
    trainer_obj.save_model(final_model_dir)
    
    print(f"\n{model_display_name} 在 Twitter Sentiment 数据集上的实验完成!")
    print(f"最佳模型已保存到: {results_dir.replace('_results', '_best_model')}")
    if tb_process:
        print(f"TensorBoard 仍在运行，访问 http://localhost:{tb_port} 查看训练可视化")
    
    return trainer_obj


if __name__ == "__main__":
    parser = build_common_arg_parser("运行 Twitter Sentiment 情感分析实验")
    args = parser.parse_args()
    
    start_tb = tensorboard_should_start(args)
    
    available_models = get_available_models()
    print(f"可用的本地模型: {available_models}")
    
    # 解析模型选择
    model_map = {
        "bert": ("bert-base-uncased", "BERT"),
        "deberta": ("deberta-base", "DeBERTa"),
        "roberta": ("roberta-base", "RoBERTa")
    }
    
    models = parse_selected_models(args.models, model_map)
    finetune_methods = parse_selected_methods(args.methods)
    
    print(f"\n选择的模型: {[m[1] for m in models]}")
    print(f"选择的微调方法: {finetune_methods}")
    print(f"{'='*60}\n")
    
    # 如果启用 TensorBoard，为所有实验启动一个全局 TensorBoard（监控所有结果目录）
    tb_process = None
    if start_tb:
        results_base_dir = "./results"
        # 如果目录不存在，创建它（TensorBoard 可以监控空目录，后续会写入日志）
        if not os.path.exists(results_base_dir):
            os.makedirs(results_base_dir, exist_ok=True)
            logger.info(f"创建结果目录: {results_base_dir}")
        logger.info(f"启动全局 TensorBoard，监控目录: {results_base_dir}")
        tb_process = start_tensorboard(results_base_dir, args.tb_port)
        if tb_process:
            print(f"\n{'='*60}")
            print(f"TensorBoard 已启动!")
            print(f"访问地址: http://localhost:{args.tb_port}")
            print(f"监控目录: {results_base_dir}")
            print(f"{'='*60}\n")
        else:
            logger.warning("TensorBoard 启动失败，请检查端口是否被占用或 TensorBoard 是否已安装")
    
    results = {}
    for model_name, display_name in models:
        for finetune_method in finetune_methods:
            try:
                # 对于每个实验，不单独启动 TensorBoard（使用全局的）
                result = run_twitter_experiment(
                    model_name, 
                    display_name, 
                    finetune_method=finetune_method,
                    start_tb=False  # 使用全局 TensorBoard
                )
                results[f"{display_name}_{finetune_method}"] = result
            except Exception as e:
                print(f"实验 {display_name} ({finetune_method}) 失败: {str(e)}")
                continue
    
    print(f"\n{'='*60}")
    print("所有 Twitter Sentiment 实验完成!")
    if tb_process:
        print(f"TensorBoard 仍在运行，访问 http://localhost:{args.tb_port} 查看所有实验的可视化")
    print(f"{'='*60}")
