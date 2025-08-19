#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base.py

基础处理器类定义

提供所有处理器的基础功能和通用接口。
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..config.exceptions import PathError, ProcessingError
from ..config.settings import ConfigManager
from .logging_config import get_logger


class BaseProcessor(ABC):
    """基础处理器抽象类

    所有处理器的基类，提供通用功能和接口规范。

    Attributes:
        config (ConfigManager): 配置管理器
        logger (logging.Logger): 日志记录器
        name (str): 处理器名称
    """

    def __init__(self, config: Optional[ConfigManager] = None, name: str = None):
        """初始化基础处理器

        Args:
            config: 配置管理器实例
            name: 处理器名称
        """
        self.config = config or ConfigManager()
        self.name = name or self.__class__.__name__
        self.logger = get_logger(self.name)

        # 初始化状态
        self._initialized = False
        self._processing = False
        self._results = {}

        # 执行初始化
        self._initialize()

    def _initialize(self) -> None:
        """内部初始化方法"""
        try:
            self.logger.info(f"初始化处理器: {self.name}")

            # 验证配置
            self.config.validate()

            # 创建必要的目录
            self._create_directories()

            # 调用子类初始化
            self.initialize()

            self._initialized = True
            self.logger.debug(f"处理器 {self.name} 初始化完成")

        except Exception as e:
            self.logger.error(f"处理器 {self.name} 初始化失败: {str(e)}")
            raise ProcessingError(
                f"处理器初始化失败: {str(e)}", context={"processor": self.name}
            )

    def _create_directories(self) -> None:
        """创建必要的目录"""
        dirs_to_create = [
            self.config.get("paths.temp_dir", "temp"),
            self.config.get("paths.log_dir", "logs"),
        ]

        for dir_path in dirs_to_create:
            if dir_path:
                try:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                    self.logger.debug(f"创建目录: {dir_path}")
                except Exception as e:
                    self.logger.warning(f"创建目录失败 {dir_path}: {str(e)}")

    @abstractmethod
    def initialize(self) -> None:
        """子类特定的初始化逻辑

        子类必须实现此方法来执行特定的初始化操作。
        """
        pass

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """主要处理方法

        子类必须实现此方法来执行主要的处理逻辑。

        Returns:
            处理结果
        """
        pass

    def validate_input(self, *args, **kwargs) -> bool:
        """验证输入参数

        子类可以重写此方法来实现特定的输入验证。

        Returns:
            bool: 验证是否通过
        """
        return True

    def cleanup(self) -> None:
        """清理资源

        子类可以重写此方法来实现特定的清理逻辑。
        """
        self.logger.debug(f"清理处理器资源: {self.name}")

    def run(self, *args, **kwargs) -> Any:
        """运行处理器

        执行完整的处理流程，包括验证、处理和清理。

        Returns:
            处理结果
        """
        if not self._initialized:
            raise ProcessingError("处理器未初始化", context={"processor": self.name})

        if self._processing:
            raise ProcessingError("处理器正在运行中", context={"processor": self.name})

        try:
            self._processing = True
            self.logger.info(f"开始运行处理器: {self.name}")

            # 验证输入
            if not self.validate_input(*args, **kwargs):
                raise ProcessingError("输入验证失败", context={"processor": self.name})

            # 执行处理
            result = self.process(*args, **kwargs)

            # 保存结果
            self._results = {
                "success": True,
                "result": result,
                "processor": self.name,
                "timestamp": self._get_timestamp(),
            }

            self.logger.info(f"处理器 {self.name} 运行完成")
            return result

        except Exception as e:
            self.logger.error(f"处理器 {self.name} 运行失败: {str(e)}")
            self._results = {
                "success": False,
                "error": str(e),
                "processor": self.name,
                "timestamp": self._get_timestamp(),
            }
            raise

        finally:
            self._processing = False
            try:
                self.cleanup()
            except Exception as e:
                self.logger.warning(f"清理资源时出错: {str(e)}")

    def get_results(self) -> Dict[str, Any]:
        """获取处理结果

        Returns:
            dict: 处理结果字典
        """
        return self._results.copy()

    def is_initialized(self) -> bool:
        """检查是否已初始化

        Returns:
            bool: 是否已初始化
        """
        return self._initialized

    def is_processing(self) -> bool:
        """检查是否正在处理

        Returns:
            bool: 是否正在处理
        """
        return self._processing

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值
        """
        return self.config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """设置配置值

        Args:
            key: 配置键
            value: 配置值
        """
        self.config.set(key, value)

    def validate_path(
        self,
        path: Union[str, Path],
        must_exist: bool = True,
        must_be_dir: bool = False,
        must_be_file: bool = False,
    ) -> Path:
        """验证路径

        Args:
            path: 路径
            must_exist: 是否必须存在
            must_be_dir: 是否必须是目录
            must_be_file: 是否必须是文件

        Returns:
            Path: 验证后的路径对象

        Raises:
            PathError: 路径验证失败
        """
        path_obj = Path(path)

        if must_exist and not path_obj.exists():
            raise PathError(f"路径不存在: {path}", path=str(path))

        if must_be_dir and path_obj.exists() and not path_obj.is_dir():
            raise PathError(f"路径不是目录: {path}", path=str(path))

        if must_be_file and path_obj.exists() and not path_obj.is_file():
            raise PathError(f"路径不是文件: {path}", path=str(path))

        return path_obj

    def get_file_list(
        self,
        directory: Union[str, Path],
        extensions: Optional[List[str]] = None,
        recursive: bool = False,
    ) -> List[Path]:
        """获取文件列表

        Args:
            directory: 目录路径
            extensions: 文件扩展名列表
            recursive: 是否递归搜索

        Returns:
            List[Path]: 文件路径列表
        """
        dir_path = self.validate_path(directory, must_exist=True, must_be_dir=True)

        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        files = []
        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                if extensions is None or file_path.suffix.lower() in [
                    ext.lower() for ext in extensions
                ]:
                    files.append(file_path)

        return sorted(files)

    def _get_timestamp(self) -> str:
        """获取当前时间戳

        Returns:
            str: ISO格式的时间戳
        """
        from datetime import datetime

        return datetime.now().isoformat()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, initialized={self._initialized})"

    def __repr__(self) -> str:
        return self.__str__()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()
        return False
