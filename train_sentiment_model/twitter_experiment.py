#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Twitter Sentiment 情感分析实验
使用BERT、DeBERTa、RoBERTa模型进行三分类情感分析（正面、负面、中性）
"""

import os
import sys
import logging
import random
import glob
import subprocess
import threading
import time
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


def start_tensorboard(log_dir, port=6006):
    """在后台启动 TensorBoard 服务器"""
    try:
        # 检查 tensorboard 是否已安装
        import tensorboard
        logger.info(f"启动 TensorBoard，日志目录: {log_dir}, 端口: {port}")
        logger.info(f"TensorBoard 访问地址: http://localhost:{port}")
        
        # 启动 tensorboard 进程
        cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", log_dir, "--port", str(port)]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 等待一下确保启动成功
        time.sleep(2)
        if process.poll() is None:
            logger.info(f"✓ TensorBoard 已启动在端口 {port}")
            return process
        else:
            logger.warning("TensorBoard 启动失败，可能端口已被占用")
            return None
    except ImportError:
        logger.warning("TensorBoard 未安装，跳过可视化")
        return None
    except Exception as e:
        logger.warning(f"启动 TensorBoard 时出错: {e}")
        return None


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
    import argparse
    
    parser = argparse.ArgumentParser(description="运行 Twitter Sentiment 情感分析实验")
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
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=True,
        help="启动 TensorBoard 可视化（默认启用）"
    )
    parser.add_argument(
        "--tb-port",
        type=int,
        default=6006,
        help="TensorBoard 端口（默认 6006）"
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="禁用 TensorBoard 可视化"
    )
    
    args = parser.parse_args()
    
    # 处理 tensorboard 参数
    start_tb = args.tensorboard and not args.no_tensorboard
    
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
    
    # 如果启用 TensorBoard，为所有实验启动一个全局 TensorBoard（监控所有结果目录）
    tb_process = None
    if start_tb:
        results_base_dir = "./results"
        if os.path.exists(results_base_dir):
            logger.info(f"启动全局 TensorBoard，监控目录: {results_base_dir}")
            tb_process = start_tensorboard(results_base_dir, args.tb_port)
            if tb_process:
                print(f"\n{'='*60}")
                print(f"TensorBoard 已启动!")
                print(f"访问地址: http://localhost:{args.tb_port}")
                print(f"监控目录: {results_base_dir}")
                print(f"{'='*60}\n")
    
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
