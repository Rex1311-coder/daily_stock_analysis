# -*- coding: utf-8 -*-
"""
环境引导初始化模块
负责：.env 加载、代理配置、日志初始化
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Set
from dotenv import dotenv_values

logger = logging.getLogger(__name__)

# 记录进程启动时的环境变量，用于区分"用户设置"和".env 注入"
_INITIAL_PROCESS_ENV: Dict[str, str] = {}
# 记录当前由 .env 管理的键
_RUNTIME_ENV_FILE_KEYS: Set[str] = set()
# 防止重复初始化
_bootstrapped: bool = False


def _get_env_path() -> Path:
    """获取 .env 文件路径"""
    env_file = os.getenv("ENV_FILE")
    if env_file:
        return Path(env_file)
    return Path(__file__).resolve().parent.parent / ".env"


def _read_env_values() -> Optional[Dict[str, str]]:
    """读取 .env 文件内容"""
    env_path = _get_env_path()
    if not env_path.exists():
        return {}
    
    try:
        values = dotenv_values(env_path)
    except Exception as exc:
        logger.warning("读取配置文件 %s 失败: %s", env_path, exc)
        return None
    
    return {
        str(key): "" if value is None else str(value)
        for key, value in values.items()
        if key is not None
    }


def _apply_proxy_settings():
    """应用代理配置"""
    if os.getenv("GITHUB_ACTIONS") == "true":
        return  # CI 环境跳过代理
    
    if os.getenv("USE_PROXY", "false").lower() != "true":
        return
    
    proxy_host = os.getenv("PROXY_HOST", "127.0.0.1")
    proxy_port = os.getenv("PROXY_PORT", "10809")
    proxy_url = f"http://{proxy_host}:{proxy_port}"
    os.environ["http_proxy"] = proxy_url
    os.environ["https_proxy"] = proxy_url
    logger.info("已启用代理: %s", proxy_url)


def setup_environment() -> None:
    """
    初始化环境（幂等操作）
    
    执行顺序：
    1. 加载 .env 文件
    2. 应用代理设置
    """
    global _bootstrapped, _INITIAL_PROCESS_ENV, _RUNTIME_ENV_FILE_KEYS
    
    if _bootstrapped:
        return
    
    # 保存初始环境变量快照
    _INITIAL_PROCESS_ENV = dict(os.environ)
    
    # 加载 .env
    from src.config import setup_env
    setup_env()
    
    # 记录由 .env 注入的键
    env_values = _read_env_values()
    if env_values is not None:
        _RUNTIME_ENV_FILE_KEYS = {
            key for key in env_values
            if key not in _INITIAL_PROCESS_ENV
        }
    
    # 应用代理
    _apply_proxy_settings()
    
    _bootstrapped = True
    logger.debug("环境初始化完成")


def reload_env_values() -> None:
    """重新加载 .env 配置，但保留用户显式设置的环境变量"""
    global _RUNTIME_ENV_FILE_KEYS
    
    latest_values = _read_env_values()
    if latest_values is None:
        return
    
    managed_keys = {
        key for key in latest_values
        if key not in _INITIAL_PROCESS_ENV
    }
    
    # 移除不再存在的键
    for key in _RUNTIME_ENV_FILE_KEYS - managed_keys:
        os.environ.pop(key, None)
    
    # 更新现有键
    for key in managed_keys:
        os.environ[key] = latest_values[key]
    
    _RUNTIME_ENV_FILE_KEYS = managed_keys
    logger.debug("环境变量已重新加载")


def setup_bootstrap_logging(debug: bool = False) -> None:
    """
    初始化早期日志（仅输出到 stderr）
    在完整日志系统就绪前使用
    """
    level = logging.DEBUG if debug else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    
    # 避免重复添加 handler
    if not any(
        isinstance(h, logging.StreamHandler) 
        and getattr(h, "stream", None) is sys.stderr
        for h in root.handlers
    ):
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
        )
        root.addHandler(handler)
