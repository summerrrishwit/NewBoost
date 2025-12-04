#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行所有情感分析实验
包括SST-2、Twitter、Amazon数据集上的BERT、DeBERTa、RoBERTa模型对比
"""

import os
import sys
import logging
from datetime import datetime
from model_config import get_available_models

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

def run_sst2_experiments():
    """运行SST-2实验"""
    logger.info("开始SST-2实验...")
    
    try:
        from sst2_experiment import run_sst2_experiment
        
        # 定义模型列表
        models = [
            ("bert-base-uncased", "BERT"),
            ("microsoft/deberta-base", "DeBERTa"),
            ("roberta-base", "RoBERTa")
        ]
        
        results = {}
        for model_name, display_name in models:
            try:
                logger.info(f"开始 {display_name} 在 SST-2 数据集上的实验")
                result = run_sst2_experiment(model_name, display_name)
                results[display_name] = result
                logger.info(f"{display_name} 在 SST-2 数据集上的实验完成")
            except Exception as e:
                logger.error(f"SST-2 实验 {display_name} 失败: {str(e)}")
                continue
        
        logger.info("SST-2 所有实验完成!")
        return results
        
    except Exception as e:
        logger.error(f"SST-2 实验失败: {str(e)}")
        return {}

def run_twitter_experiments():
    """运行Twitter实验"""
    logger.info("开始Twitter实验...")
    
    try:
        from twitter_sentiment_experiment import run_twitter_experiment
        
        # 定义模型列表
        models = [
            ("bert-base-uncased", "BERT"),
            ("microsoft/deberta-base", "DeBERTa"),
            ("roberta-base", "RoBERTa")
        ]
        
        results = {}
        for model_name, display_name in models:
            try:
                logger.info(f"开始 {display_name} 在 Twitter 数据集上的实验")
                result = run_twitter_experiment(model_name, display_name)
                results[display_name] = result
                logger.info(f"{display_name} 在 Twitter 数据集上的实验完成")
            except Exception as e:
                logger.error(f"Twitter 实验 {display_name} 失败: {str(e)}")
                continue
        
        logger.info("Twitter 所有实验完成!")
        return results
        
    except Exception as e:
        logger.error(f"Twitter 实验失败: {str(e)}")
        return {}

def run_amazon_experiments():
    """运行Amazon实验"""
    logger.info("开始Amazon实验...")
    
    try:
        from amazon_reviews_experiment import run_amazon_experiment
        
        # 定义模型列表
        models = [
            ("bert-base-uncased", "BERT"),
            ("microsoft/deberta-base", "DeBERTa"),
            ("roberta-base", "RoBERTa")
        ]
        
        results = {}
        for model_name, display_name in models:
            try:
                logger.info(f"开始 {display_name} 在 Amazon 数据集上的实验")
                result = run_amazon_experiment(model_name, display_name)
                results[display_name] = result
                logger.info(f"{display_name} 在 Amazon 数据集上的实验完成")
            except Exception as e:
                logger.error(f"Amazon 实验 {display_name} 失败: {str(e)}")
                continue
        
        logger.info("Amazon 所有实验完成!")
        return results
        
    except Exception as e:
        logger.error(f"Amazon 实验失败: {str(e)}")
        return {}

def create_summary_report():
    """创建实验总结报告"""
    logger.info("创建实验总结报告...")
    
    import json
    from datetime import datetime
    
    # 收集所有结果
    summary = {
        "experiment_timestamp": datetime.now().isoformat(),
        "datasets": ["SST-2", "Twitter Sentiment", "Amazon Reviews"],
        "models": ["BERT", "DeBERTa", "RoBERTa"],
        "results_directory": "./results/",
        "checkpoint_strategy": "只保存一个checkpoint (save_total_limit=1)",
        "evaluation_results": "每个实验的评估结果保存在对应的results子目录中"
    }
    
    # 保存总结报告
    with open("./results/experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info("实验总结报告已保存到: ./results/experiment_summary.json")

def main():
    """主函数"""
    print("=" * 80)
    print("开始运行所有情感分析实验")
    print("=" * 80)
    
    # 检查可用的本地模型
    available_models = get_available_models()
    print(f"可用的本地模型: {available_models}")
    
    # 创建results目录
    os.makedirs("./results", exist_ok=True)
    
    # 运行所有实验
    all_results = {}
    
    # SST-2实验
    print("\n" + "=" * 60)
    print("开始 SST-2 实验")
    print("=" * 60)
    sst2_results = run_sst2_experiments()
    all_results["SST-2"] = sst2_results
    
    # Twitter实验
    print("\n" + "=" * 60)
    print("开始 Twitter 实验")
    print("=" * 60)
    twitter_results = run_twitter_experiments()
    all_results["Twitter"] = twitter_results
    
    # Amazon实验
    print("\n" + "=" * 60)
    print("开始 Amazon 实验")
    print("=" * 60)
    amazon_results = run_amazon_experiments()
    all_results["Amazon"] = amazon_results
    
    # 创建总结报告
    create_summary_report()
    
    print("\n" + "=" * 80)
    print("所有实验完成!")
    print("=" * 80)
    print("结果保存在 ./results/ 目录中")
    print("每个实验的详细评估结果保存在对应的子目录中")
    print("=" * 80)

if __name__ == "__main__":
    main()