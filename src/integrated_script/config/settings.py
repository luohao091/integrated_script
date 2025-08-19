#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
settings.py

配置管理模块

提供配置文件的读取、写入、验证和管理功能。
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

from .exceptions import ConfigurationError, FileProcessingError


class ConfigManager:
    """配置管理器
    
    负责管理应用程序的配置信息，支持JSON和YAML格式。
    
    Attributes:
        config_file (Path): 配置文件路径
        config_data (dict): 配置数据
        auto_save (bool): 是否自动保存
    """
    
    DEFAULT_CONFIG = {
        "version": "1.0.0",
        "debug": False,
        "log_level": "INFO",
        "paths": {
            "input_dir": "",
            "output_dir": "",
            "temp_dir": "temp",
            "log_dir": "logs"
        },
        "processing": {
            "batch_size": 100,
            "max_workers": 4,
            "timeout": 300,
            "retry_count": 3
        },
        "ui": {
            "language": "zh_CN",
            "theme": "default",
            "show_progress": True
        },
        "yolo": {
            "image_formats": [".jpg", ".jpeg", ".png", ".bmp"],
            "label_format": ".txt",
            "classes_file": "classes.txt",
            "validate_on_load": True
        },
        "image_processing": {
            "default_output_format": "jpg",
            "jpeg_quality": 95,
            "png_compression": 6,
            "webp_quality": 90,
            "quality_analysis": {
                "custom_levels": [
                    {"name": "超高清 4K", "threshold": [3840, 2160]},
                    {"name": "超高清 2K", "threshold": [2560, 1440]},
                    {"name": "全高清", "threshold": [1920, 1080]},
                    {"name": "高清", "threshold": [1280, 720]},
                    {"name": "标清", "threshold": [720, 480]}
                ]
            },
            "resize": {
                "maintain_aspect_ratio": True,
                "interpolation": "LANCZOS",
                "default_size": [640, 640]
            },
            "auto_orient": True,
            "strip_metadata": False,
            "parallel_processing": True,
            "chunk_size": 50
        }
    }
    
    def __init__(self, config_file: Union[str, Path] = None, auto_save: bool = True):
        """初始化配置管理器
        
        Args:
            config_file: 配置文件路径，默认为当前目录下的config.json
            auto_save: 是否自动保存配置更改
        """
        if config_file is None:
            config_file = Path.cwd() / "config.json"
        
        self.config_file = Path(config_file)
        self.auto_save = auto_save
        self.config_data = self.DEFAULT_CONFIG.copy()
        self._last_modified = None
        
        # 创建配置目录
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        self.load()
    
    def load(self) -> None:
        """从文件加载配置
        
        Raises:
            ConfigurationError: 配置文件格式错误
            FileProcessingError: 文件读取错误
        """
        if not self.config_file.exists():
            # 如果配置文件不存在，创建默认配置
            self.save()
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                if self.config_file.suffix.lower() == '.yaml' or self.config_file.suffix.lower() == '.yml':
                    loaded_config = yaml.safe_load(f)
                else:
                    loaded_config = json.load(f)
            
            if not isinstance(loaded_config, dict):
                raise ConfigurationError(
                    "配置文件格式错误：根对象必须是字典",
                    config_file=str(self.config_file)
                )
            
            # 合并配置（保留默认值）
            self._merge_config(loaded_config)
            self._last_modified = self.config_file.stat().st_mtime
            
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ConfigurationError(
                f"配置文件解析错误: {str(e)}",
                config_file=str(self.config_file)
            )
        except Exception as e:
            raise FileProcessingError(
                f"读取配置文件失败: {str(e)}",
                file_path=str(self.config_file),
                operation="read"
            )
    
    def save(self) -> None:
        """保存配置到文件
        
        Raises:
            FileProcessingError: 文件写入错误
        """
        try:
            # 添加元数据
            save_data = self.config_data.copy()
            save_data["_metadata"] = {
                "last_updated": datetime.now().isoformat(),
                "version": self.config_data.get("version", "1.0.0")
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                if self.config_file.suffix.lower() == '.yaml' or self.config_file.suffix.lower() == '.yml':
                    yaml.dump(save_data, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
                else:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            self._last_modified = self.config_file.stat().st_mtime
            
        except Exception as e:
            raise FileProcessingError(
                f"保存配置文件失败: {str(e)}",
                file_path=str(self.config_file),
                operation="write"
            )
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键，支持点号分隔的嵌套键（如 'paths.input_dir'）
            default: 默认值
            
        Returns:
            配置值
            
        Example:
            >>> config.get('paths.input_dir')
            >>> config.get('processing.batch_size', 100)
        """
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值
        
        Args:
            key: 配置键，支持点号分隔的嵌套键
            value: 配置值
            
        Example:
            >>> config.set('paths.input_dir', '/path/to/input')
            >>> config.set('processing.batch_size', 200)
        """
        keys = key.split('.')
        current = self.config_data
        
        # 创建嵌套结构
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
        
        if self.auto_save:
            self.save()
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """批量更新配置
        
        Args:
            config_dict: 配置字典
        """
        self._merge_config(config_dict)
        
        if self.auto_save:
            self.save()
    
    def reset(self) -> None:
        """重置为默认配置"""
        # 尝试从default_config.yaml加载默认配置
        default_config_path = Path(__file__).parent / "default_config.yaml"
        if default_config_path.exists():
            try:
                with open(default_config_path, 'r', encoding='utf-8') as f:
                    import yaml
                    default_config = yaml.safe_load(f)
                    # 将嵌套的配置展平为我们的格式
                    self.config_data = self._flatten_yaml_config(default_config)
            except Exception:
                # 如果加载失败，使用内置默认配置
                self.config_data = self.DEFAULT_CONFIG.copy()
        else:
            self.config_data = self.DEFAULT_CONFIG.copy()
        
        if self.auto_save:
            self.save()
    
    def _flatten_yaml_config(self, yaml_config: Dict[str, Any]) -> Dict[str, Any]:
        """将YAML配置转换为扁平格式"""
        flattened = self.DEFAULT_CONFIG.copy()
        
        # 映射YAML配置到我们的格式
        if 'app' in yaml_config:
            app_config = yaml_config['app']
            if 'version' in app_config:
                flattened['version'] = app_config['version']
            if 'debug' in app_config:
                flattened['debug'] = app_config['debug']
        
        if 'logging' in yaml_config:
            logging_config = yaml_config['logging']
            if 'level' in logging_config:
                flattened['log_level'] = logging_config['level']
        
        if 'file_processing' in yaml_config:
            file_config = yaml_config['file_processing']
            if 'batch_size' in file_config:
                flattened['processing']['batch_size'] = file_config['batch_size']
            if 'default_output_dir' in file_config:
                flattened['paths']['output_dir'] = file_config['default_output_dir']
        
        if 'yolo' in yaml_config:
            yolo_config = yaml_config['yolo']
            if 'supported_image_formats' in yolo_config:
                flattened['yolo']['image_formats'] = yolo_config['supported_image_formats']
            if 'label_format' in yolo_config:
                flattened['yolo']['label_format'] = yolo_config['label_format']
            if 'classes_file' in yolo_config:
                flattened['yolo']['classes_file'] = yolo_config['classes_file']
        
        return flattened
    
    def validate(self) -> bool:
        """验证配置的有效性
        
        Returns:
            bool: 配置是否有效
            
        Raises:
            ConfigurationError: 配置验证失败
        """
        required_keys = ['version', 'paths', 'processing', 'ui', 'yolo']
        
        for key in required_keys:
            if key not in self.config_data:
                raise ConfigurationError(
                    f"缺少必需的配置项: {key}",
                    config_key=key,
                    config_file=str(self.config_file)
                )
        
        # 验证路径配置
        paths = self.config_data.get('paths', {})
        for path_key, path_value in paths.items():
            if path_value and not isinstance(path_value, str):
                raise ConfigurationError(
                    f"路径配置必须是字符串: {path_key}",
                    config_key=f"paths.{path_key}"
                )
        
        # 验证处理配置
        processing = self.config_data.get('processing', {})
        numeric_keys = ['batch_size', 'max_workers', 'timeout', 'retry_count']
        for key in numeric_keys:
            if key in processing and not isinstance(processing[key], int):
                raise ConfigurationError(
                    f"处理配置必须是整数: {key}",
                    config_key=f"processing.{key}"
                )
        
        return True
    
    def reload_if_changed(self) -> bool:
        """如果文件已更改则重新加载
        
        Returns:
            bool: 是否重新加载了配置
        """
        if not self.config_file.exists():
            return False
        
        current_mtime = self.config_file.stat().st_mtime
        if self._last_modified is None or current_mtime > self._last_modified:
            self.load()
            return True
        
        return False
    
    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """合并配置字典
        
        Args:
            new_config: 新的配置字典
        """
        def merge_dict(base: dict, update: dict) -> dict:
            """递归合并字典"""
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
            return base
        
        merge_dict(self.config_data, new_config)
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有配置
        
        Returns:
            dict: 配置字典的副本
        """
        return self.config_data.copy()
    
    def export(self, export_file: Union[str, Path], format_type: str = 'json') -> None:
        """导出配置到文件
        
        Args:
            export_file: 导出文件路径
            format_type: 导出格式 ('json' 或 'yaml')
        """
        export_path = Path(export_file)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                if format_type.lower() == 'yaml':
                    yaml.dump(self.config_data, f, default_flow_style=False,
                             allow_unicode=True, indent=2)
                else:
                    json.dump(self.config_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise FileProcessingError(
                f"导出配置失败: {str(e)}",
                file_path=str(export_path),
                operation="export"
            )
    
    def __str__(self) -> str:
        return f"ConfigManager(file={self.config_file}, auto_save={self.auto_save})"
    
    def __repr__(self) -> str:
        return self.__str__()