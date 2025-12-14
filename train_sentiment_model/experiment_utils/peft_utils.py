#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
与参数高效微调相关的工具函数。
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from peft import (
        LoraConfig,
        PrefixTuningConfig,
        TaskType,
        get_peft_model,
        PeftModel,
    )

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT库未安装，LoRA和Prefix Tuning功能不可用")


def _get_available_modules(model, candidate_modules):
    """检查模型中实际存在的模块名称
    
    PEFT 库匹配模块时使用模块名称的最后一部分（类名），
    例如 'deberta.encoder.layer.0.attention.self.query_proj' 会被匹配为 'query_proj'
    """
    available_modules = []
    # 获取所有模块名称的最后一部分（类名）
    module_class_names = set()
    for name, module in model.named_modules():
        if name:  # 跳过根模块
            class_name = name.split('.')[-1]
            module_class_names.add(class_name)
    
    # 检查候选模块是否存在于模型中
    for candidate in candidate_modules:
        if candidate in module_class_names:
            available_modules.append(candidate)
    
    return available_modules


def setup_lora_model(model, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1, target_modules=None):
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT库未安装，无法使用LoRA功能。请运行: pip install peft")

    if target_modules is None:
        model_type = model.config.model_type if hasattr(model.config, "model_type") else None
        
        # 根据模型类型选择候选模块
        if model_type and "deberta" in model_type.lower():
            # DeBERTa 系列：v1 可能使用 in_proj，v2/v3 使用 query_proj/value_proj
            # 按优先级排序：先尝试 v2/v3 的格式，再尝试 v1 的格式
            candidate_modules = ["query_proj", "value_proj", "in_proj", "query", "value"]
        elif model_type in ["bert", "roberta"]:
            # BERT 和 RoBERTa 使用 query 和 value
            candidate_modules = ["query", "value"]
        else:
            # 默认尝试所有可能的模块名称
            candidate_modules = ["query", "value", "query_proj", "value_proj", "in_proj"]
            logger.warning(f"未知模型类型 {model_type}，将尝试自动检测目标模块")
        
        # 自动检测实际存在的模块
        available_modules = _get_available_modules(model, candidate_modules)
        
        if available_modules:
            # 优先选择 query_proj/value_proj 或 query/value 的组合
            if "query_proj" in available_modules and "value_proj" in available_modules:
                target_modules = ["query_proj", "value_proj"]
            elif "query" in available_modules and "value" in available_modules:
                target_modules = ["query", "value"]
            elif "in_proj" in available_modules:
                target_modules = ["in_proj"]
            else:
                # 如果都没有，使用找到的第一个模块
                target_modules = available_modules[:2] if len(available_modules) >= 2 else available_modules
            logger.info(f"自动检测到目标模块: {target_modules}")
        else:
            # 如果自动检测失败，打印所有模块名称以便调试
            all_modules = [name.split('.')[-1] for name, _ in model.named_modules() if len(name.split('.')) > 0]
            unique_modules = sorted(set(all_modules))
            logger.error(f"无法找到合适的目标模块。模型类型: {model_type}")
            logger.error(f"可用的模块名称示例: {unique_modules[:20]}...")
            raise ValueError(
                f"无法为模型类型 '{model_type}' 自动检测目标模块。"
                f"请手动指定 target_modules 参数。"
            )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    try:
        model = get_peft_model(model, lora_config)
    except ValueError as e:
        if "not found in the base model" in str(e) or "Target modules" in str(e):
            # 如果目标模块不存在，提供更详细的错误信息
            all_modules = [name.split('.')[-1] for name, _ in model.named_modules() if len(name.split('.')) > 0]
            unique_modules = sorted(set(all_modules))
            logger.error(f"指定的目标模块 {target_modules} 在模型中不存在")
            logger.error(f"模型类型: {model.config.model_type if hasattr(model.config, 'model_type') else 'unknown'}")
            logger.error(f"可用的模块名称（前30个）: {unique_modules[:30]}")
            raise ValueError(
                f"目标模块 {target_modules} 在模型中不存在。\n"
                f"模型类型: {model.config.model_type if hasattr(model.config, 'model_type') else 'unknown'}\n"
                f"请检查可用的模块名称并手动指定 target_modules 参数。"
            ) from e
        else:
            raise

    logger.info(f"LoRA配置: r={r}, alpha={lora_alpha}, dropout={lora_dropout}, target_modules={target_modules}")
    logger.info(f"可训练参数数量: {model.num_parameters(only_trainable=True):,}")
    logger.info(f"总参数数量: {model.num_parameters():,}")

    return model


def setup_prefix_tuning_model(model, num_virtual_tokens: int = 20, prefix_projection: bool = False):
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT库未安装，无法使用Prefix Tuning功能。请运行: pip install peft")

    if hasattr(model.config, "hidden_size"):
        hidden_size = model.config.hidden_size
    elif hasattr(model.config, "d_model"):
        hidden_size = model.config.d_model
    else:
        hidden_size = 768
        logger.warning(f"无法检测模型hidden_size，使用默认值: {hidden_size}")

    prefix_config = PrefixTuningConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=num_virtual_tokens,
        encoder_hidden_size=hidden_size if prefix_projection else None,
    )

    try:
        model = get_peft_model(model, prefix_config)
    except TypeError as e:
        if "low_cpu_mem_usage" in str(e):
            logger.warning("检测到 PEFT 库的 low_cpu_mem_usage bug，使用 workaround...")
            import peft.peft_model

            original_add_adapter = peft.peft_model.PeftModel.add_adapter

            def patched_add_adapter(self, adapter_name, peft_config, **kwargs):
                kwargs.pop("low_cpu_mem_usage", None)
                return original_add_adapter(self, adapter_name, peft_config)

            peft.peft_model.PeftModel.add_adapter = patched_add_adapter
            try:
                model = PeftModel(model, prefix_config, adapter_name="default")
            finally:
                peft.peft_model.PeftModel.add_adapter = original_add_adapter
        else:
            raise

    logger.info(f"Prefix Tuning配置: num_virtual_tokens={num_virtual_tokens}, prefix_projection={prefix_projection}")
    logger.info(f"可训练参数数量: {model.num_parameters(only_trainable=True):,}")
    logger.info(f"总参数数量: {model.num_parameters():,}")

    return model
