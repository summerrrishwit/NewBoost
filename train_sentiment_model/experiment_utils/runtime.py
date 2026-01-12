#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行时辅助函数：检查数据文件、启动 TensorBoard 等。
"""

import glob
import logging
import os
import socket
import subprocess
import sys
import time
from typing import List, Optional

logger = logging.getLogger(__name__)


def is_git_lfs_pointer(file_path: str) -> bool:
    """检查文件是否为 Git LFS 指针文件。"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            return first_line == "version https://git-lfs.github.com/spec/v1"
    except Exception:
        return False


def check_and_get_parquet_files(data_dir: str, split_name: str) -> List[str]:
    """在给定目录下查找 parquet 文件并校验是否为真实文件。"""
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


def is_port_in_use(port: int) -> bool:
    """检查端口是否被占用。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except OSError:
            return True


def find_process_using_port(port: int) -> Optional[int]:
    """查找占用指定端口的进程 PID。"""
    try:
        # 读取 /proc/net/tcp 文件
        with open('/proc/net/tcp', 'r') as f:
            lines = f.readlines()
        
        # 将端口转换为十六进制（小端序）
        port_hex = format(port, '04X')
        
        for line in lines[1:]:  # 跳过标题行
            parts = line.split()
            if len(parts) >= 2:
                local_addr = parts[1]
                inode = parts[9] if len(parts) > 9 else None
                
                # 检查端口是否匹配 (格式: IP:PORT，PORT是十六进制小端序)
                # 例如 6006 (0x1776) 在文件中显示为 7617
                port_reversed = port_hex[2:4] + port_hex[0:2]  # 小端序
                if port_reversed in local_addr.upper() or port_hex in local_addr.upper():
                    # 查找使用该 inode 的进程
                    if inode:
                        for pid_dir in os.listdir('/proc'):
                            if pid_dir.isdigit():
                                try:
                                    fd_dir = f'/proc/{pid_dir}/fd'
                                    if os.path.exists(fd_dir):
                                        for fd in os.listdir(fd_dir):
                                            fd_path = f'{fd_dir}/{fd}'
                                            try:
                                                link = os.readlink(fd_path)
                                                if f'socket:[{inode}]' in link:
                                                    return int(pid_dir)
                                            except (OSError, ValueError):
                                                pass
                                except (OSError, PermissionError):
                                    pass
    except (FileNotFoundError, PermissionError, ValueError) as e:
        logger.debug(f"查找端口占用进程时出错: {e}")
    
    return None


def kill_process_on_port(port: int) -> bool:
    """终止占用指定端口的进程。"""
    pid = find_process_using_port(port)
    if pid:
        try:
            # 尝试优雅终止
            os.kill(pid, 15)  # SIGTERM
            time.sleep(1)
            # 检查进程是否还存在
            try:
                os.kill(pid, 0)  # 检查进程是否存在
                # 如果还存在，强制终止
                os.kill(pid, 9)  # SIGKILL
                logger.info(f"已强制终止占用端口 {port} 的进程 (PID: {pid})")
            except ProcessLookupError:
                logger.info(f"已终止占用端口 {port} 的进程 (PID: {pid})")
            return True
        except (ProcessLookupError, PermissionError) as e:
            logger.warning(f"无法终止进程 {pid}: {e}")
            return False
    return False


def start_tensorboard(log_dir: str, port: int = 6006):
    """在后台启动 TensorBoard 服务器。
    
    如果端口被占用，会自动尝试终止占用端口的进程后重新启动。
    """
    try:
        import tensorboard  # noqa: F401

        logger.info(f"启动 TensorBoard，日志目录: {log_dir}, 端口: {port}")
        logger.info(f"TensorBoard 访问地址: http://localhost:{port}")

        # 检查端口是否被占用
        if is_port_in_use(port):
            logger.warning(f"端口 {port} 已被占用，尝试终止占用该端口的进程...")
            if kill_process_on_port(port):
                time.sleep(1)  # 等待端口释放
            else:
                logger.warning(f"无法释放端口 {port}，TensorBoard 可能启动失败")

        cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", log_dir, "--port", str(port)]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        time.sleep(2)
        if process.poll() is None:
            logger.info(f"✓ TensorBoard 已启动在端口 {port}")
            return process

        # 如果启动失败，读取错误信息
        try:
            stderr_output = process.stderr.read()
            if stderr_output:
                logger.debug(f"TensorBoard 启动错误: {stderr_output}")
        except Exception:
            pass

        logger.warning("TensorBoard 启动失败，可能端口已被占用或 TensorBoard 配置有误")
        return None
    except ImportError:
        logger.warning("TensorBoard 未安装，跳过可视化")
        return None
    except Exception as exc:
        logger.warning(f"启动 TensorBoard 时出错: {exc}")
        return None
