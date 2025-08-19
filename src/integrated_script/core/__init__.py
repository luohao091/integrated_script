#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Module

核心模块，包含基础类、工具函数和通用功能。
"""

from .base import BaseProcessor
from .logging_config import get_logger, setup_logging
from .progress import ProgressManager, progress_context
from .utils import (
    copy_file_safe,
    create_directory,
    delete_file_safe,
    get_file_list,
    move_file_safe,
    safe_file_operation,
    validate_path,
)

__all__ = [
    "BaseProcessor",
    "safe_file_operation",
    "validate_path",
    "get_file_list",
    "create_directory",
    "copy_file_safe",
    "move_file_safe",
    "delete_file_safe",
    "ProgressManager",
    "progress_context",
    "setup_logging",
    "get_logger",
]
