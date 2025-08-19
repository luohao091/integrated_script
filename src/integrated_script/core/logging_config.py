#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
logging_config.py

日志配置模块

提供统一的日志配置和管理功能。
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器

    为不同级别的日志添加颜色显示，支持Windows兼容性。
    """

    # ANSI颜色代码
    COLORS = {
        "DEBUG": "\033[36m",  # 青色
        "INFO": "\033[32m",  # 绿色
        "WARNING": "\033[33m",  # 黄色
        "ERROR": "\033[31m",  # 红色
        "CRITICAL": "\033[35m",  # 紫色
        "RESET": "\033[0m",  # 重置
    }

    # 简化的级别标识符（用于不支持颜色的终端）
    LEVEL_SYMBOLS = {
        "DEBUG": "[DEBUG]",
        "INFO": "[INFO]",
        "WARNING": "[WARN]",
        "ERROR": "[ERROR]",
        "CRITICAL": "[CRIT]",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_colors = self._should_use_colors()

    def _should_use_colors(self) -> bool:
        """检测是否应该使用颜色"""
        try:
            from .windows_compat import check_color_support

            return check_color_support()
        except ImportError:
            # 如果无法导入兼容性模块，使用简单检查
            if os.environ.get("NO_COLOR"):
                return False
            if os.environ.get("FORCE_COLOR"):
                return True
            return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    def format(self, record):
        """格式化日志记录"""
        # 获取原始格式化结果
        log_message = super().format(record)

        if self.use_colors:
            # 添加颜色
            color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
            reset = self.COLORS["RESET"]
            return f"{color}{log_message}{reset}"
        else:
            # 不使用颜色，但添加级别标识符以便区分
            symbol = self.LEVEL_SYMBOLS.get(record.levelname, "[LOG]")
            # 替换级别名称为带符号的版本
            if " - " + record.levelname + " - " in log_message:
                log_message = log_message.replace(
                    " - " + record.levelname + " - ", f" - {symbol} - "
                )
            return log_message


class LogManager:
    """日志管理器

    提供统一的日志配置和管理功能。

    Attributes:
        log_dir (Path): 日志目录
        log_level (str): 日志级别
        loggers (Dict): 已创建的日志记录器
    """

    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        """初始化日志管理器

        Args:
            log_dir: 日志目录
            log_level: 日志级别
        """
        self.log_dir = Path(log_dir)
        self.log_level = log_level.upper()
        self.loggers: Dict[str, logging.Logger] = {}

        # 创建日志目录
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 配置根日志记录器
        self._setup_root_logger()

    def _setup_root_logger(self) -> None:
        """配置根日志记录器"""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level))

        # 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # 添加控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.log_level))

        # 控制台格式（支持颜色和Windows兼容性）
        console_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        console_formatter = ColoredFormatter(console_format)
        console_handler.setFormatter(console_formatter)

        root_logger.addHandler(console_handler)

        # 添加文件处理器
        log_file = (
            self.log_dir / f"integrated_script_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"  # 10MB
        )
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别

        # 文件格式（无颜色）
        file_format = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        file_formatter = logging.Formatter(file_format)
        file_handler.setFormatter(file_formatter)

        root_logger.addHandler(file_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """获取指定名称的日志记录器

        Args:
            name: 日志记录器名称

        Returns:
            logging.Logger: 日志记录器实例
        """
        if name not in self.loggers:
            logger = logging.getLogger(name)
            logger.setLevel(getattr(logging, self.log_level))
            self.loggers[name] = logger

        return self.loggers[name]

    def create_module_logger(
        self, module_name: str, log_file: Optional[str] = None
    ) -> logging.Logger:
        """为模块创建专用日志记录器

        Args:
            module_name: 模块名称
            log_file: 专用日志文件名（可选）

        Returns:
            logging.Logger: 模块日志记录器
        """
        logger = self.get_logger(module_name)

        if log_file:
            # 添加模块专用文件处理器
            module_log_file = self.log_dir / log_file
            module_handler = logging.handlers.RotatingFileHandler(
                module_log_file,
                maxBytes=5 * 1024 * 1024,  # 5MB
                backupCount=3,
                encoding="utf-8",
            )
            module_handler.setLevel(logging.DEBUG)

            # 模块文件格式
            module_format = (
                "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            )
            module_formatter = logging.Formatter(module_format)
            module_handler.setFormatter(module_formatter)

            logger.addHandler(module_handler)

        return logger

    def set_level(self, level: str) -> None:
        """设置日志级别

        Args:
            level: 日志级别
        """
        self.log_level = level.upper()
        log_level_int = getattr(logging, self.log_level)

        # 更新根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level_int)

        # 更新所有现有的日志记录器（包括不在self.loggers中的）
        for name, logger in logging.Logger.manager.loggerDict.items():
            if isinstance(logger, logging.Logger):
                logger.setLevel(log_level_int)

        # 更新self.loggers中的日志记录器
        for logger in self.loggers.values():
            logger.setLevel(log_level_int)

        # 更新控制台处理器级别
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                handler.setLevel(log_level_int)

    def add_error_file_handler(self, error_log_file: str = "errors.log") -> None:
        """添加错误日志文件处理器

        Args:
            error_log_file: 错误日志文件名
        """
        error_file = self.log_dir / error_log_file
        error_handler = logging.handlers.RotatingFileHandler(
            error_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"  # 5MB
        )
        error_handler.setLevel(logging.ERROR)

        # 错误日志格式
        error_format = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s\n%(exc_info)s"
        error_formatter = logging.Formatter(error_format)
        error_handler.setFormatter(error_formatter)

        # 添加到根日志记录器
        root_logger = logging.getLogger()
        root_logger.addHandler(error_handler)

    def cleanup_old_logs(self, days: int = 30) -> None:
        """清理旧日志文件

        Args:
            days: 保留天数
        """
        import time

        cutoff_time = time.time() - (days * 24 * 60 * 60)

        for log_file in self.log_dir.glob("*.log*"):
            try:
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    print(f"删除旧日志文件: {log_file}")
            except Exception as e:
                print(f"删除日志文件失败 {log_file}: {str(e)}")

    def get_log_stats(self) -> Dict[str, Any]:
        """获取日志统计信息

        Returns:
            Dict[str, Any]: 日志统计信息
        """
        stats = {
            "log_dir": str(self.log_dir),
            "log_level": self.log_level,
            "loggers_count": len(self.loggers),
            "log_files": [],
            "total_size": 0,
        }

        for log_file in self.log_dir.glob("*.log*"):
            try:
                file_size = log_file.stat().st_size
                stats["log_files"].append(
                    {
                        "name": log_file.name,
                        "size": file_size,
                        "modified": datetime.fromtimestamp(
                            log_file.stat().st_mtime
                        ).isoformat(),
                    }
                )
                stats["total_size"] += file_size
            except Exception:
                pass

        return stats


# 全局日志管理器实例
_log_manager: Optional[LogManager] = None


def setup_logging(
    log_dir: str = "logs", log_level: str = "INFO", enable_error_file: bool = True
) -> LogManager:
    """设置日志系统

    Args:
        log_dir: 日志目录
        log_level: 日志级别
        enable_error_file: 是否启用错误日志文件

    Returns:
        LogManager: 日志管理器实例
    """
    global _log_manager

    _log_manager = LogManager(log_dir, log_level)

    if enable_error_file:
        _log_manager.add_error_file_handler()

    return _log_manager


def get_logger(name: str = None) -> logging.Logger:
    """获取日志记录器

    Args:
        name: 日志记录器名称，默认使用调用模块名

    Returns:
        logging.Logger: 日志记录器实例
    """
    global _log_manager

    if _log_manager is None:
        _log_manager = setup_logging()

    if name is None:
        # 自动获取调用模块名
        import inspect

        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "unknown")

    return _log_manager.get_logger(name)


def set_log_level(level: str) -> None:
    """设置全局日志级别

    Args:
        level: 日志级别
    """
    global _log_manager

    if _log_manager is None:
        _log_manager = setup_logging()

    _log_manager.set_level(level)


def get_log_manager() -> Optional[LogManager]:
    """获取日志管理器实例

    Returns:
        LogManager: 日志管理器实例或None
    """
    return _log_manager


# 日志装饰器
def log_function_call(
    logger: Optional[logging.Logger] = None,
    log_args: bool = False,
    log_result: bool = False,
):
    """函数调用日志装饰器

    Args:
        logger: 日志记录器
        log_args: 是否记录参数
        log_result: 是否记录返回值
    """

    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"

            # 记录函数调用开始
            if log_args:
                logger.debug(
                    f"调用函数 {func_name}，参数: args={args}, kwargs={kwargs}"
                )
            else:
                logger.debug(f"调用函数 {func_name}")

            try:
                result = func(*args, **kwargs)

                # 记录函数调用成功
                if log_result:
                    logger.debug(f"函数 {func_name} 执行成功，返回值: {result}")
                else:
                    logger.debug(f"函数 {func_name} 执行成功")

                return result

            except Exception as e:
                # 记录函数调用失败
                logger.error(f"函数 {func_name} 执行失败: {str(e)}", exc_info=True)
                raise

        return wrapper

    return decorator


# 性能监控装饰器
def log_performance(logger: Optional[logging.Logger] = None, threshold: float = 1.0):
    """性能监控装饰器

    Args:
        logger: 日志记录器
        threshold: 性能警告阈值（秒）
    """

    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        def wrapper(*args, **kwargs):
            import time

            func_name = f"{func.__module__}.{func.__name__}"
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time

                if elapsed_time > threshold:
                    logger.warning(
                        f"函数 {func_name} 执行时间较长: {elapsed_time:.2f}秒"
                    )
                else:
                    logger.debug(f"函数 {func_name} 执行时间: {elapsed_time:.2f}秒")

                return result

            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(
                    f"函数 {func_name} 执行失败 (耗时 {elapsed_time:.2f}秒): {str(e)}"
                )
                raise

        return wrapper

    return decorator
