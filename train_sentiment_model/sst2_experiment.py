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


class SentimentTrainer(BaseSentimentTrainer):
    def __init__(self, model_name, num_labels=2, finetune_method="full"):
        super().__init__(model_name, num_labels, "SST-2", finetune_method=finetune_method)
        
    def load_data(self, use_local_parquet=True, return_dataset=False):
        """
        加载SST-2数据集
        
        Args:
            use_local_parquet: 是否优先使用本地parquet文件
            return_dataset: 如果为True，返回Dataset对象而不是DataFrame（用于优化，避免多次map）
        
        Returns:
            如果 return_dataset=False: (train_df, test_df)
            如果 return_dataset=True: DatasetDict with 'train' and 'validation' splits
        """
        sst2_dir = os.path.join(DATASET_DIR, "glue", "sst2")
        use_local = False
        
        if use_local_parquet and os.path.exists(sst2_dir):
            try:
                logger.info("尝试从本地 parquet 文件加载SST-2数据集...")
                train_files = check_and_get_parquet_files(sst2_dir, "train")
                validation_files = check_and_get_parquet_files(sst2_dir, "validation")
                test_files = check_and_get_parquet_files(sst2_dir, "test")

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
        
        # 重命名 'sentence' 列为 'text' 以便统一处理
        if 'sentence' in dataset['train'].column_names:
            dataset = dataset.rename_column('sentence', 'text')
        
        # 如果返回Dataset，不进行map操作（将在tokenize_data中一次性完成）
        if return_dataset:
            logger.info(f"训练集大小: {len(dataset['train'])}")
            logger.info(f"测试集大小: {len(dataset['validation'])}")
            return dataset
        
        # 否则，进行标签转换并返回DataFrame（保持向后兼容）
        def convert_labels(example):
            example['sentiment'] = 'positive' if example['label'] == 1 else 'negative'
            return example
        
        dataset = dataset.map(convert_labels)
        
        train_df = pd.DataFrame({
            'text': dataset['train']['text'],
            'label': dataset['train']['label'],
            'sentiment': dataset['train']['sentiment']
        })
        
        test_df = pd.DataFrame({
            'text': dataset['validation']['text'],
            'label': dataset['validation']['label'],
            'sentiment': dataset['validation']['sentiment']
        })
        
        logger.info(f"训练集大小: {len(train_df)}")
        logger.info(f"测试集大小: {len(test_df)}")
        logger.info(f"类别分布:\n{train_df['sentiment'].value_counts()}")
        
        return train_df, test_df
    
    def get_target_names(self):
        return ['negative', 'positive']
    
    def tokenize_data(self, train_df=None, test_df=None, dataset=None, max_length=256):
        """
        对数据进行tokenization
        
        Args:
            train_df: 训练集DataFrame（向后兼容）
            test_df: 测试集DataFrame（向后兼容）
            dataset: DatasetDict对象（优化路径，避免多次map）
            max_length: 最大序列长度
        
        Returns:
            tokenized DatasetDict
        """
        from datasets import DatasetDict, Dataset
        
        # 如果提供了dataset，直接使用（优化路径：合并所有转换）
        if dataset is not None:
            logger.info("使用优化路径：一次性完成标签转换和tokenization...")
            
            # 重命名validation为test以保持一致性（在map之前）
            if 'validation' in dataset and 'test' not in dataset:
                dataset['test'] = dataset['validation']
            
            def convert_and_tokenize(batch):
                # 合并标签转换和tokenization
                batch['sentiment'] = ['positive' if lbl == 1 else 'negative' for lbl in batch['label']]
                tokenized = self.tokenizer(batch["text"], padding=False, truncation=True, max_length=max_length)
                return tokenized
            
            dataset = dataset.map(
                convert_and_tokenize, 
                batched=True, 
                batch_size=32, 
                num_proc=4, 
                remove_columns=["text"]
            )
            return dataset
        
        # 否则使用原有逻辑（向后兼容）
        return super().tokenize_data(train_df, test_df, max_length=max_length)


def run_sst2_experiment(model_name, model_display_name, finetune_method="full", start_tb=True, tb_port=6006):
    print(f"\n{'='*60}")
    print(f"开始 {model_display_name} 在 SST-2 数据集上的实验")
    print(f"微调方法: {finetune_method}")
    print(f"{'='*60}")
    
    trainer_obj = SentimentTrainer(model_name, num_labels=2, finetune_method=finetune_method)
    
    # 先加载模型但不移到GPU（避免tokenize时的CUDA multiprocessing错误）
    trainer_obj.setup_model(move_to_gpu=False)
    
    # 优化：直接加载Dataset并一次性完成所有转换，避免多次map
    raw_dataset = trainer_obj.load_data(return_dataset=True)
    dataset = trainer_obj.tokenize_data(dataset=raw_dataset)
    
    # tokenize完成后再将模型移到GPU
    trainer_obj.move_model_to_gpu()
    
    # 在结果目录中包含微调方法
    method_suffix = f"_{finetune_method}" if finetune_method != "full" else ""
    results_dir = f"./results/sst2_{model_name.replace('/', '_')}{method_suffix}_results"
    
    # 启动 TensorBoard（如果启用）
    tb_process = None
    if start_tb:
        # TensorBoard 日志目录是 results_dir 下的 logs 子目录
        tb_log_dir = os.path.join(results_dir, "logs")
        tb_process = start_tensorboard(tb_log_dir, tb_port)
    
    trainer = trainer_obj.train(dataset, results_dir)
    
    predictions = trainer_obj.evaluate(trainer, dataset, model_name)
    
    final_model_dir = f"./results/sst2_{model_name.replace('/', '_')}{method_suffix}_model"
    trainer_obj.save_model(final_model_dir)
    
    print(f"\n{model_display_name} 在 SST-2 数据集上的实验完成!")
    print(f"最佳模型已保存到: {results_dir.replace('_results', '_best_model')}")
    if tb_process:
        print(f"TensorBoard 仍在运行，访问 http://localhost:{tb_port} 查看训练可视化")
    
    return trainer_obj


if __name__ == "__main__":
    parser = build_common_arg_parser("运行 SST-2 情感分析实验")
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
                result = run_sst2_experiment(
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
    print("所有 SST-2 实验完成!")
    if tb_process:
        print(f"TensorBoard 仍在运行，访问 http://localhost:{args.tb_port} 查看所有实验的可视化")
    print(f"{'='*60}")
