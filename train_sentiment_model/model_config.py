#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

class ModelConfig:
    def __init__(self, base_models_dir="./models"):
        self.base_models_dir = Path(base_models_dir)
        self.model_paths = {
            "bert-base-uncased": self.base_models_dir / "bert-base-uncased",
            "deberta-base": self.base_models_dir / "deberta-base" ,
            "roberta-base": self.base_models_dir / "roberta-base"
        }
        
        self._validate_model_paths()
    
    def _validate_model_paths(self):
        for model_name, model_path in self.model_paths.items():
            if not model_path.exists():
                print(f"警告: 模型路径不存在: {model_path}")
                print(f"请确保模型已下载到 {model_path}")
            else:
                print(f"✓ 找到模型: {model_name} -> {model_path}")
    
    def get_model_path(self, model_name):
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
        available = []
        for model_name, model_path in self.model_paths.items():
            if model_path.exists():
                available.append(model_name)
        return available
    
    def is_model_available_locally(self, model_name):
        if model_name in self.model_paths:
            return self.model_paths[model_name].exists()
        return False

model_config = ModelConfig()

def get_model_path(model_name):
    return model_config.get_model_path(model_name)

def get_available_models():
    return model_config.get_available_models()

def is_model_available_locally(model_name):
    return model_config.is_model_available_locally(model_name)
