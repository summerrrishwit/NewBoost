#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验脚本的统一命令行参数与解析工具。
"""

import argparse
from typing import Dict, List, Tuple


def build_common_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=["full", "lora", "prefix", "all"],
        default=["all"],
        help="选择微调方法: 'full'（全参数微调）、'lora'（LoRA微调）、'prefix'（Prefix Tuning）、'all'（所有方法，默认）",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["bert", "deberta", "roberta", "all"],
        default=["all"],
        help="选择模型: 'bert'、'deberta'、'roberta'、'all'（所有模型，默认）",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="启动 TensorBoard 可视化（默认启用，除非指定 --no-tensorboard）",
    )
    parser.add_argument(
        "--tb-port",
        type=int,
        default=6006,
        help="TensorBoard 端口（默认 6006）",
    )
    parser.add_argument("--no-tensorboard", action="store_true", help="禁用 TensorBoard 可视化")
    return parser


def parse_selected_models(raw_models: List[str], model_map: Dict[str, Tuple[str, str]]) -> List[Tuple[str, str]]:
    if "all" in raw_models:
        return list(model_map.values())
    return [model_map[m] for m in raw_models if m in model_map]


def parse_selected_methods(raw_methods: List[str]) -> List[str]:
    if "all" in raw_methods:
        return ["full", "lora", "prefix"]
    return raw_methods


def tensorboard_should_start(args) -> bool:
    """综合 tensorboard 和 no-tensorboard 参数，返回是否应启动 TensorBoard。
    
    默认启用 TensorBoard，除非：
    1. 明确指定了 --no-tensorboard，或
    2. 既没有指定 --tensorboard 也没有指定 --tb-port（但这种情况不应该发生，因为 --tb-port 有默认值）
    
    如果用户提供了 --tb-port，说明他们想要使用 TensorBoard，应该启动。
    """
    if args.no_tensorboard:
        return False
    return True
