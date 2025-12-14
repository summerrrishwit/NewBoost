#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验工具包入口，统一导出常用的训练与实验辅助函数。
"""

import os

# 关闭 tokenizer 并行警告，保持与原实现一致
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .data import load_local_dataset, set_global_seed
from .results import EvalMeta, save_evaluation_results
from .peft_utils import PEFT_AVAILABLE, setup_lora_model, setup_prefix_tuning_model
from .trainer import BaseSentimentTrainer
from .runtime import (
    is_git_lfs_pointer,
    check_and_get_parquet_files,
    start_tensorboard,
)
from .cli import (
    build_common_arg_parser,
    parse_selected_methods,
    parse_selected_models,
    tensorboard_should_start,
)

__all__ = [
    "BaseSentimentTrainer",
    "EvalMeta",
    "PEFT_AVAILABLE",
    "check_and_get_parquet_files",
    "build_common_arg_parser",
    "is_git_lfs_pointer",
    "load_local_dataset",
    "parse_selected_methods",
    "parse_selected_models",
    "save_evaluation_results",
    "set_global_seed",
    "setup_lora_model",
    "setup_prefix_tuning_model",
    "start_tensorboard",
    "tensorboard_should_start",
]
