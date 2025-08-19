#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Module

配置管理模块，包含设置管理和异常定义。
"""

from .exceptions import (
    ProcessingError,
    PathError,
    FileProcessingError,
    ConfigurationError,
)
from .settings import ConfigManager

__all__ = [
    "ProcessingError",
    "PathError", 
    "FileProcessingError",
    "ConfigurationError",
    "ConfigManager",
]