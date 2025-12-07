#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import logging

logger = logging.getLogger(__name__)


def get_available_gpus():
    if not torch.cuda.is_available():
        logger.warning("CUDA不可用，将使用CPU训练")
        return []
    
    num_gpus = torch.cuda.device_count()
    gpu_ids = list(range(num_gpus))
    
    logger.info(f"检测到 {num_gpus} 个GPU设备:")
    for gpu_id in gpu_ids:
        gpu_name = torch.cuda.get_device_name(gpu_id)
        gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        logger.info(f"  GPU {gpu_id}: {gpu_name} ({gpu_memory:.2f} GB)")
    
    return gpu_ids


def setup_multi_gpu(model, gpu_ids):
    if len(gpu_ids) > 1:
        logger.info(f"使用 {len(gpu_ids)} 个GPU进行训练 (DataParallel)")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        torch.cuda.set_device(gpu_ids[0])
    elif len(gpu_ids) == 1:
        logger.info(f"使用单个GPU进行训练 (GPU {gpu_ids[0]})")
        model = model.to(f"cuda:{gpu_ids[0]}")
        torch.cuda.set_device(gpu_ids[0])
    else:
        logger.info("使用CPU进行训练")
    
    return model




