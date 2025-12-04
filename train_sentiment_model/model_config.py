#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型配置管理
统一管理本地预下载模型的路径配置
"""

import os
from pathlib import Path

class ModelConfig:
    """模型配置管理类"""
    
    def __init__(self, base_models_dir="./models"):
        """
        初始化模型配置
        
        Args:
            base_models_dir: 模型根目录路径
        """
        self.base_models_dir = Path(base_models_dir)
        self.model_paths = {
            "bert-base-uncased": self.base_models_dir / "bert-base-uncased",
            "microsoft/deberta-base": self.base_models_dir / "deberta-base" / "deberta-base",
            "deberta-base": self.base_models_dir / "deberta-base" / "deberta-base",  # 别名
            "roberta-base": self.base_models_dir / "roberta-base"
        }
        
        # 验证模型路径是否存在
        self._validate_model_paths()
    
    def _validate_model_paths(self):
        """验证模型路径是否存在"""
        for model_name, model_path in self.model_paths.items():
            if not model_path.exists():
                print(f"警告: 模型路径不存在: {model_path}")
                print(f"请确保模型已下载到 {model_path}")
            else:
                print(f"✓ 找到模型: {model_name} -> {model_path}")
    
    def get_model_path(self, model_name):
        """
        获取模型路径
        Args:
            model_name: 模型名称
        Returns:
            str: 模型路径
        """
        if model_name in self.model_paths:
            model_path = self.model_paths[model_name]
            if model_path.exists():
                return str(model_path)
            else:
                print(f"警告: 模型路径不存在，将使用在线模型: {model_name}")
                return model_name
        else:
            print(f"警告: 未知模型名称，将使用在线模型: {model_name}")
            return model_name
    
    def get_available_models(self):
        """获取可用的本地模型列表"""
        available = []
        for model_name, model_path in self.model_paths.items():
            if model_path.exists():
                available.append(model_name)
        return available
    
    def is_model_available_locally(self, model_name):
        """检查模型是否在本地可用"""
        if model_name in self.model_paths:
            return self.model_paths[model_name].exists()
        return False

# 全局模型配置实例
model_config = ModelConfig()

def get_model_path(model_name):
    """便捷函数：获取模型路径"""
    return model_config.get_model_path(model_name)

def get_available_models():
    """便捷函数：获取可用模型列表"""
    return model_config.get_available_models()

def is_model_available_locally(model_name):
    """便捷函数：检查模型是否本地可用"""
    return model_config.is_model_available_locally(model_name)
