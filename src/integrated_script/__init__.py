#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Script Package

一个模块化的数据处理整合脚本项目，提供多种YOLO数据集处理功能。

Modules:
    config: 配置管理和异常定义
    core: 核心基础类和工具函数
    processors: 各种数据处理器
    ui: 用户界面和交互

Example:
    >>> from integrated_script import IntegratedProcessor
    >>> processor = IntegratedProcessor()
    >>> processor.run()
"""

__version__ = "1.0.0"
__author__ = "Integrated Script Team"
__email__ = "team@example.com"
__license__ = "MIT"

# 导入主要类和函数
from .core.base import BaseProcessor
from .config.exceptions import (
    ProcessingError,
    PathError,
    FileProcessingError,
    ConfigurationError,
)
from .config.settings import ConfigManager

# 定义公共API
__all__ = [
    "BaseProcessor",
    "ProcessingError",
    "PathError",
    "FileProcessingError",
    "ConfigurationError",
    "ConfigManager",
    "__version__",
]

# 包级别的配置
DEFAULT_CONFIG = {
    "version": __version__,
    "debug": False,
    "log_level": "INFO",
}