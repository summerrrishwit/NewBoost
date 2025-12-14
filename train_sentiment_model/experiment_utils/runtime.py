#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行时辅助函数：检查数据文件、启动 TensorBoard 等。
"""

import glob
import logging
import os
import subprocess
import sys
import time
from typing import List

logger = logging.getLogger(__name__)


def is_git_lfs_pointer(file_path: str) -> bool:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            return first_line == "version https://git-lfs.github.com/spec/v1"
    except Exception:
        return False


def check_and_get_parquet_files(data_dir: str, split_name: str) -> List[str]:
    pattern = os.path.join(data_dir, f"{split_name}-*.parquet")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"未找到 {split_name} 的 parquet 文件: {pattern}")

    for file_path in files:
        if is_git_lfs_pointer(file_path):
            raise ValueError(
                f"文件 {file_path} 是 Git LFS 指针文件，不是实际的 parquet 文件。\n"
                f"请先运行 'git lfs pull' 或手动下载实际的 parquet 文件。"
            )

    return files


def start_tensorboard(log_dir: str, port: int = 6006):
    try:
        import tensorboard  # noqa: F401

        logger.info(f"启动 TensorBoard，日志目录: {log_dir}, 端口: {port}")
        logger.info(f"TensorBoard 访问地址: http://localhost:{port}")

        cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", log_dir, "--port", str(port)]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        time.sleep(2)
        if process.poll() is None:
            logger.info(f"✓ TensorBoard 已启动在端口 {port}")
            return process

        logger.warning("TensorBoard 启动失败，可能端口已被占用")
        return None
    except ImportError:
        logger.warning("TensorBoard 未安装，跳过可视化")
        return None
    except Exception as exc:
        logger.warning(f"启动 TensorBoard 时出错: {exc}")
        return None
