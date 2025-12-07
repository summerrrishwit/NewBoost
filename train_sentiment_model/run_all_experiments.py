#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Callable, Any, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_config import get_available_models
from config import get_config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_experiment_suite(
    experiment_func: Callable,
    models: List[Tuple[str, str]],
    dataset_name: str,
    finetune_methods: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    通用实验运行函数，消除代码重复
    
    Args:
        experiment_func: 实验函数，接受 (model_name, display_name, finetune_method) 参数
        models: 模型列表，格式为 [(model_name, display_name), ...]
        dataset_name: 数据集名称，用于日志输出
        finetune_methods: 微调方法列表，如果为 None 则只运行默认方法
        
    Returns:
        实验结果字典
    """
    logger.info(f"开始 {dataset_name} 实验...")
    results = {}
    
    try:
        # 确定要运行的微调方法
        if finetune_methods is None:
            finetune_methods = [None]  # 只运行默认方法
        
        for finetune_method in finetune_methods:
            method_suffix = f" ({finetune_method})" if finetune_method else ""
            
            for model_name, display_name in models:
                try:
                    logger.info(
                        f"开始 {display_name}{method_suffix} 在 {dataset_name} 数据集上的实验"
                    )
                    
                    # 调用实验函数
                    if finetune_method:
                        result = experiment_func(model_name, display_name, finetune_method=finetune_method)
                        result_key = f"{display_name}_{finetune_method}"
                    else:
                        result = experiment_func(model_name, display_name)
                        result_key = display_name
                    
                    results[result_key] = result
                    logger.info(
                        f"{display_name}{method_suffix} 在 {dataset_name} 数据集上的实验完成"
                    )
                    
                except Exception as e:
                    logger.error(
                        f"{dataset_name} 实验 {display_name}{method_suffix} 失败: {str(e)}",
                        exc_info=True
                    )
                    continue
        
        logger.info(f"{dataset_name} 所有实验完成!")
        return results
        
    except Exception as e:
        logger.error(f"{dataset_name} 实验失败: {str(e)}", exc_info=True)
        return {}


def run_sst2_experiments(finetune_methods: Optional[List[str]] = None) -> Dict[str, Any]:
    """运行 SST-2 实验"""
    from sst2_experiment import run_sst2_experiment
    
    config = get_config()
    models = config.training.training_models
    
    return run_experiment_suite(
        experiment_func=run_sst2_experiment,
        models=models,
        dataset_name="SST-2",
        finetune_methods=finetune_methods
    )


def run_twitter_experiments(finetune_methods: Optional[List[str]] = None) -> Dict[str, Any]:
    """运行 Twitter 实验"""
    from twitter_experiment import run_twitter_experiment
    
    config = get_config()
    models = config.training.training_models
    
    return run_experiment_suite(
        experiment_func=run_twitter_experiment,
        models=models,
        dataset_name="Twitter Sentiment",
        finetune_methods=finetune_methods
    )


def run_amazon_experiments(finetune_methods: Optional[List[str]] = None) -> Dict[str, Any]:
    """运行 Amazon 实验"""
    from amazon_experiment import run_amazon_experiment
    
    config = get_config()
    models = config.training.training_models
    
    return run_experiment_suite(
        experiment_func=run_amazon_experiment,
        models=models,
        dataset_name="Amazon Reviews",
        finetune_methods=finetune_methods
    )

def create_summary_report(all_results: Dict[str, Any], finetune_methods: Optional[List[str]] = None):
    """
    创建实验总结报告
    
    Args:
        all_results: 所有实验结果
        finetune_methods: 使用的微调方法列表
    """
    logger.info("创建实验总结报告...")
    
    import json
    
    config = get_config()
    
    summary = {
        "experiment_timestamp": datetime.now().isoformat(),
        "datasets": ["SST-2", "Twitter Sentiment", "Amazon Reviews"],
        "models": [display_name for _, display_name in config.training.training_models],
        "finetune_methods": finetune_methods or ["full"],
        "results_directory": str(config.paths.results_dir),
        "checkpoint_strategy": "只保存一个checkpoint (save_total_limit=1)",
        "evaluation_results": "每个实验的评估结果保存在对应的results子目录中",
        "experiment_summary": {
            dataset: {
                "total_experiments": len(results),
                "successful": sum(1 for r in results.values() if r is not None),
                "failed": sum(1 for r in results.values() if r is None)
            }
            for dataset, results in all_results.items()
        }
    }
    
    summary_path = config.paths.results_dir / "experiment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"实验总结报告已保存到: {summary_path}")


def main(finetune_methods: Optional[List[str]] = None):
    """
    主函数，运行所有实验
    
    Args:
        finetune_methods: 要运行的微调方法列表，如果为 None 则只运行默认方法
    """
    config = get_config()
    
    print("=" * 80)
    print("开始运行所有情感分析实验")
    print("=" * 80)
    
    available_models = get_available_models()
    print(f"可用的本地模型: {available_models}")
    
    if finetune_methods:
        print(f"微调方法: {', '.join(finetune_methods)}")
    else:
        print("微调方法: 默认（full）")
    
    # 确保结果目录存在
    config.paths.results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # 运行所有实验
    experiments = [
        ("SST-2", run_sst2_experiments),
        ("Twitter Sentiment", run_twitter_experiments),
        ("Amazon Reviews", run_amazon_experiments)
    ]
    
    for dataset_name, experiment_func in experiments:
        print("\n" + "=" * 60)
        print(f"开始 {dataset_name} 实验")
        print("=" * 60)
        
        try:
            results = experiment_func(finetune_methods=finetune_methods)
            all_results[dataset_name] = results
        except Exception as e:
            logger.error(f"{dataset_name} 实验组失败: {str(e)}", exc_info=True)
            all_results[dataset_name] = {}
    
    # 创建总结报告
    create_summary_report(all_results, finetune_methods)
    
    print("\n" + "=" * 80)
    print("所有实验完成!")
    print("=" * 80)
    print(f"结果保存在 {config.paths.results_dir} 目录中")
    print("每个实验的详细评估结果保存在对应的子目录中")
    print("=" * 80)

if __name__ == "__main__":
    main()