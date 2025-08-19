#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file_processor.py

文件处理器

提供文件复制、移动、删除、重命名等基本文件操作功能。
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable

from ..core.base import BaseProcessor
from ..core.utils import (
    validate_path, get_file_list, create_directory,
    safe_file_operation, get_unique_filename, format_file_size,
    copy_file_safe, move_file_safe, delete_file_safe
)
from ..core.progress import progress_context, process_with_progress
from ..config.exceptions import ProcessingError, FileProcessingError


class FileProcessor(BaseProcessor):
    """文件处理器
    
    提供文件复制、移动、删除、重命名等基本文件操作功能。
    
    Attributes:
        operation_count (int): 操作计数
        total_size_processed (int): 已处理的总文件大小
    """
    
    def __init__(self, **kwargs):
        """初始化文件处理器"""
        super().__init__(name="FileProcessor", **kwargs)
        
        self.operation_count = 0
        self.total_size_processed = 0
    
    def initialize(self) -> None:
        """初始化处理器"""
        self.logger.info("文件处理器初始化完成")
        self.operation_count = 0
        self.total_size_processed = 0
    
    def process(self, *args, **kwargs) -> Any:
        """主要处理方法（由子方法实现具体功能）"""
        raise NotImplementedError("请使用具体的处理方法")
    
    def copy_files(self, source_dir: str, target_dir: str,
                   file_patterns: Optional[List[str]] = None,
                   recursive: bool = False,
                   overwrite: bool = False,
                   preserve_structure: bool = True) -> Dict[str, Any]:
        """批量复制文件
        
        Args:
            source_dir: 源目录
            target_dir: 目标目录
            file_patterns: 文件模式列表（如 ['*.txt', '*.py']）
            recursive: 是否递归处理
            overwrite: 是否覆盖已存在的文件
            preserve_structure: 是否保持目录结构
            
        Returns:
            Dict[str, Any]: 复制结果
        """
        try:
            source_path = validate_path(source_dir, must_exist=True, must_be_dir=True)
            target_path = validate_path(target_dir, must_exist=False)
            
            create_directory(target_path)
            
            self.logger.info(f"开始复制文件: {source_path} -> {target_path}")
            self.logger.info(f"递归: {recursive}, 覆盖: {overwrite}, 保持结构: {preserve_structure}")
            
            # 获取文件列表
            if file_patterns:
                files_to_copy = []
                for pattern in file_patterns:
                    if recursive:
                        files_to_copy.extend(source_path.rglob(pattern))
                    else:
                        files_to_copy.extend(source_path.glob(pattern))
                # 去重并过滤文件
                files_to_copy = [f for f in set(files_to_copy) if f.is_file()]
            else:
                # 获取所有文件
                files_to_copy = get_file_list(source_path, extensions=None, recursive=recursive)
            
            result = {
                "success": True,
                "source_dir": str(source_path),
                "target_dir": str(target_path),
                "copied_files": [],
                "failed_files": [],
                "skipped_files": [],
                "statistics": {
                    "total_files": len(files_to_copy),
                    "copied_count": 0,
                    "failed_count": 0,
                    "skipped_count": 0,
                    "total_size": 0
                }
            }
            
            # 复制文件
            def copy_single_file(file_path: Path) -> Dict[str, Any]:
                try:
                    file_size = file_path.stat().st_size
                    
                    # 计算目标路径
                    if preserve_structure and recursive:
                        rel_path = file_path.relative_to(source_path)
                        target_file = target_path / rel_path
                        create_directory(target_file.parent)
                    else:
                        target_file = target_path / file_path.name
                    
                    # 检查是否需要覆盖
                    if target_file.exists() and not overwrite:
                        # 生成唯一文件名
                        target_file = get_unique_filename(target_file.parent, target_file.name)
                    
                    # 复制文件
                    if target_file.exists() and not overwrite:
                        return {
                            "success": False,
                            "action": "skipped",
                            "source_file": str(file_path),
                            "target_file": str(target_file),
                            "reason": "文件已存在且不允许覆盖",
                            "file_size": file_size
                        }
                    
                    copy_file_safe(file_path, target_file)
                    
                    return {
                        "success": True,
                        "action": "copied",
                        "source_file": str(file_path),
                        "target_file": str(target_file),
                        "file_size": file_size
                    }
                    
                except Exception as e:
                    return {
                        "success": False,
                        "action": "failed",
                        "source_file": str(file_path),
                        "error": str(e),
                        "file_size": 0
                    }
            
            # 批量处理
            copy_results = process_with_progress(
                files_to_copy,
                copy_single_file,
                "复制文件"
            )
            
            # 统计结果
            for copy_result in copy_results:
                if copy_result:
                    if copy_result["success"] and copy_result["action"] == "copied":
                        result["copied_files"].append(copy_result)
                        result["statistics"]["copied_count"] += 1
                        result["statistics"]["total_size"] += copy_result["file_size"]
                    elif copy_result["action"] == "skipped":
                        result["skipped_files"].append(copy_result)
                        result["statistics"]["skipped_count"] += 1
                    else:
                        result["failed_files"].append(copy_result)
                        result["statistics"]["failed_count"] += 1
                        result["success"] = False
            
            # 格式化大小信息
            result["statistics"]["total_size_formatted"] = format_file_size(
                result["statistics"]["total_size"]
            )
            
            self.operation_count += result["statistics"]["copied_count"]
            self.total_size_processed += result["statistics"]["total_size"]
            
            self.logger.info(f"文件复制完成: {result['statistics']}")
            return result
            
        except Exception as e:
            raise ProcessingError(f"文件复制失败: {str(e)}")
    
    def move_files(self, source_dir: str, target_dir: str,
                   file_patterns: Optional[List[str]] = None,
                   recursive: bool = False,
                   overwrite: bool = False,
                   preserve_structure: bool = True) -> Dict[str, Any]:
        """批量移动文件
        
        Args:
            source_dir: 源目录
            target_dir: 目标目录
            file_patterns: 文件模式列表
            recursive: 是否递归处理
            overwrite: 是否覆盖已存在的文件
            preserve_structure: 是否保持目录结构
            
        Returns:
            Dict[str, Any]: 移动结果
        """
        try:
            source_path = validate_path(source_dir, must_exist=True, must_be_dir=True)
            target_path = validate_path(target_dir, must_exist=False)
            
            create_directory(target_path)
            
            self.logger.info(f"开始移动文件: {source_path} -> {target_path}")
            
            # 获取文件列表
            if file_patterns:
                files_to_move = []
                for pattern in file_patterns:
                    if recursive:
                        files_to_move.extend(source_path.rglob(pattern))
                    else:
                        files_to_move.extend(source_path.glob(pattern))
                files_to_move = [f for f in set(files_to_move) if f.is_file()]
            else:
                files_to_move = get_file_list(source_path, extensions=None, recursive=recursive)
            
            result = {
                "success": True,
                "source_dir": str(source_path),
                "target_dir": str(target_path),
                "moved_files": [],
                "failed_files": [],
                "skipped_files": [],
                "statistics": {
                    "total_files": len(files_to_move),
                    "moved_count": 0,
                    "failed_count": 0,
                    "skipped_count": 0,
                    "total_size": 0
                }
            }
            
            # 移动文件
            def move_single_file(file_path: Path) -> Dict[str, Any]:
                try:
                    file_size = file_path.stat().st_size
                    
                    # 计算目标路径
                    if preserve_structure and recursive:
                        rel_path = file_path.relative_to(source_path)
                        target_file = target_path / rel_path
                        create_directory(target_file.parent)
                    else:
                        target_file = target_path / file_path.name
                    
                    # 检查是否需要覆盖
                    if target_file.exists() and not overwrite:
                        target_file = get_unique_filename(target_file.parent, target_file.name)
                    
                    # 移动文件
                    if target_file.exists() and not overwrite:
                        return {
                            "success": False,
                            "action": "skipped",
                            "source_file": str(file_path),
                            "target_file": str(target_file),
                            "reason": "文件已存在且不允许覆盖",
                            "file_size": file_size
                        }
                    
                    move_file_safe(file_path, target_file)
                    
                    return {
                        "success": True,
                        "action": "moved",
                        "source_file": str(file_path),
                        "target_file": str(target_file),
                        "file_size": file_size
                    }
                    
                except Exception as e:
                    return {
                        "success": False,
                        "action": "failed",
                        "source_file": str(file_path),
                        "error": str(e),
                        "file_size": 0
                    }
            
            # 批量处理
            move_results = process_with_progress(
                files_to_move,
                move_single_file,
                "移动文件"
            )
            
            # 统计结果
            for move_result in move_results:
                if move_result:
                    if move_result["success"] and move_result["action"] == "moved":
                        result["moved_files"].append(move_result)
                        result["statistics"]["moved_count"] += 1
                        result["statistics"]["total_size"] += move_result["file_size"]
                    elif move_result["action"] == "skipped":
                        result["skipped_files"].append(move_result)
                        result["statistics"]["skipped_count"] += 1
                    else:
                        result["failed_files"].append(move_result)
                        result["statistics"]["failed_count"] += 1
                        result["success"] = False
            
            # 格式化大小信息
            result["statistics"]["total_size_formatted"] = format_file_size(
                result["statistics"]["total_size"]
            )
            
            self.operation_count += result["statistics"]["moved_count"]
            self.total_size_processed += result["statistics"]["total_size"]
            
            self.logger.info(f"文件移动完成: {result['statistics']}")
            return result
            
        except Exception as e:
            raise ProcessingError(f"文件移动失败: {str(e)}")
    
    def delete_files(self, target_dir: str,
                     file_patterns: Optional[List[str]] = None,
                     recursive: bool = False,
                     confirm_callback: Optional[Callable[[List[Path]], bool]] = None) -> Dict[str, Any]:
        """批量删除文件
        
        Args:
            target_dir: 目标目录
            file_patterns: 文件模式列表
            recursive: 是否递归处理
            confirm_callback: 确认回调函数
            
        Returns:
            Dict[str, Any]: 删除结果
        """
        try:
            target_path = validate_path(target_dir, must_exist=True, must_be_dir=True)
            
            self.logger.info(f"开始删除文件: {target_path}")
            
            # 获取文件列表
            if file_patterns:
                files_to_delete = []
                for pattern in file_patterns:
                    if recursive:
                        files_to_delete.extend(target_path.rglob(pattern))
                    else:
                        files_to_delete.extend(target_path.glob(pattern))
                files_to_delete = [f for f in set(files_to_delete) if f.is_file()]
            else:
                files_to_delete = get_file_list(target_path, extensions=None, recursive=recursive)
            
            # 确认删除
            if confirm_callback and not confirm_callback(files_to_delete):
                return {
                    "success": False,
                    "message": "用户取消删除操作",
                    "statistics": {"total_files": len(files_to_delete)}
                }
            
            result = {
                "success": True,
                "target_dir": str(target_path),
                "deleted_files": [],
                "failed_files": [],
                "statistics": {
                    "total_files": len(files_to_delete),
                    "deleted_count": 0,
                    "failed_count": 0,
                    "total_size": 0
                }
            }
            
            # 删除文件
            def delete_single_file(file_path: Path) -> Dict[str, Any]:
                try:
                    file_size = file_path.stat().st_size
                    
                    delete_file_safe(file_path)
                    
                    return {
                        "success": True,
                        "deleted_file": str(file_path),
                        "file_size": file_size
                    }
                    
                except Exception as e:
                    return {
                        "success": False,
                        "failed_file": str(file_path),
                        "error": str(e),
                        "file_size": 0
                    }
            
            # 批量处理
            delete_results = process_with_progress(
                files_to_delete,
                delete_single_file,
                "删除文件"
            )
            
            # 统计结果
            for delete_result in delete_results:
                if delete_result:
                    if delete_result["success"]:
                        result["deleted_files"].append(delete_result)
                        result["statistics"]["deleted_count"] += 1
                        result["statistics"]["total_size"] += delete_result["file_size"]
                    else:
                        result["failed_files"].append(delete_result)
                        result["statistics"]["failed_count"] += 1
                        result["success"] = False
            
            # 格式化大小信息
            result["statistics"]["total_size_formatted"] = format_file_size(
                result["statistics"]["total_size"]
            )
            
            self.operation_count += result["statistics"]["deleted_count"]
            self.total_size_processed += result["statistics"]["total_size"]
            
            self.logger.info(f"文件删除完成: {result['statistics']}")
            return result
            
        except Exception as e:
            raise ProcessingError(f"文件删除失败: {str(e)}")
    
    def rename_files(self, target_dir: str,
                     rename_pattern: str,
                     file_patterns: Optional[List[str]] = None,
                     recursive: bool = False,
                     preview_only: bool = False) -> Dict[str, Any]:
        """批量重命名文件
        
        Args:
            target_dir: 目标目录
            rename_pattern: 重命名模式（支持 {name}, {ext}, {index} 等占位符）
            file_patterns: 文件模式列表
            recursive: 是否递归处理
            preview_only: 仅预览，不实际重命名
            
        Returns:
            Dict[str, Any]: 重命名结果
        """
        try:
            target_path = validate_path(target_dir, must_exist=True, must_be_dir=True)
            
            self.logger.info(f"开始重命名文件: {target_path}")
            self.logger.info(f"重命名模式: {rename_pattern}")
            
            # 获取文件列表
            if file_patterns:
                files_to_rename = []
                for pattern in file_patterns:
                    if recursive:
                        files_to_rename.extend(target_path.rglob(pattern))
                    else:
                        files_to_rename.extend(target_path.glob(pattern))
                files_to_rename = [f for f in set(files_to_rename) if f.is_file()]
            else:
                files_to_rename = get_file_list(target_path, extensions=None, recursive=recursive)
            
            # 按名称排序以确保一致的索引
            files_to_rename.sort(key=lambda x: x.name)
            
            result = {
                "success": True,
                "target_dir": str(target_path),
                "rename_pattern": rename_pattern,
                "preview_only": preview_only,
                "renamed_files": [],
                "failed_files": [],
                "statistics": {
                    "total_files": len(files_to_rename),
                    "renamed_count": 0,
                    "failed_count": 0
                }
            }
            
            # 重命名文件
            def rename_single_file(file_info: tuple) -> Dict[str, Any]:
                file_path, index = file_info
                try:
                    # 生成新文件名
                    new_name = rename_pattern.format(
                        name=file_path.stem,
                        ext=file_path.suffix,
                        index=index + 1,
                        index0=index
                    )
                    
                    new_path = file_path.parent / new_name
                    
                    # 确保新文件名唯一
                    if new_path.exists() and new_path != file_path:
                        new_path = get_unique_filename(new_path.parent, new_path.name)
                    
                    if preview_only:
                        return {
                            "success": True,
                            "action": "preview",
                            "old_name": str(file_path),
                            "new_name": str(new_path)
                        }
                    
                    # 实际重命名
                    if new_path != file_path:
                        file_path.rename(new_path)
                    
                    return {
                        "success": True,
                        "action": "renamed",
                        "old_name": str(file_path),
                        "new_name": str(new_path)
                    }
                    
                except Exception as e:
                    return {
                        "success": False,
                        "action": "failed",
                        "old_name": str(file_path),
                        "error": str(e)
                    }
            
            # 准备文件和索引
            files_with_index = [(f, i) for i, f in enumerate(files_to_rename)]
            
            # 批量处理
            rename_results = process_with_progress(
                files_with_index,
                rename_single_file,
                "重命名文件" if not preview_only else "预览重命名"
            )
            
            # 统计结果
            for rename_result in rename_results:
                if rename_result:
                    if rename_result["success"]:
                        result["renamed_files"].append(rename_result)
                        if not preview_only:
                            result["statistics"]["renamed_count"] += 1
                    else:
                        result["failed_files"].append(rename_result)
                        result["statistics"]["failed_count"] += 1
                        result["success"] = False
            
            if not preview_only:
                self.operation_count += result["statistics"]["renamed_count"]
            
            action = "预览" if preview_only else "重命名"
            self.logger.info(f"文件{action}完成: {result['statistics']}")
            return result
            
        except Exception as e:
            raise ProcessingError(f"文件重命名失败: {str(e)}")
    
    def rename_files_with_temp(self, target_dir: str,
                              rename_pattern: str,
                              file_patterns: Optional[List[str]] = None,
                              recursive: bool = False,
                              shuffle_order: bool = False,
                              preview_only: bool = False) -> Dict[str, Any]:
        """使用临时重命名的批量重命名文件（避免冲突）
        
        Args:
            target_dir: 目标目录
            rename_pattern: 重命名模式（支持 {name}, {ext}, {index} 等占位符）
            file_patterns: 文件模式列表
            recursive: 是否递归处理
            shuffle_order: 是否打乱文件顺序
            preview_only: 仅预览，不实际重命名
            
        Returns:
            Dict[str, Any]: 重命名结果
        """
        import random
        import uuid
        
        try:
            target_path = validate_path(target_dir, must_exist=True, must_be_dir=True)
            
            self.logger.info(f"开始临时重命名文件: {target_path}")
            self.logger.info(f"重命名模式: {rename_pattern}")
            self.logger.info(f"打乱顺序: {shuffle_order}")
            
            # 获取文件列表
            if file_patterns:
                files_to_rename = []
                for pattern in file_patterns:
                    if recursive:
                        files_to_rename.extend(target_path.rglob(pattern))
                    else:
                        files_to_rename.extend(target_path.glob(pattern))
                files_to_rename = [f for f in set(files_to_rename) if f.is_file()]
            else:
                files_to_rename = get_file_list(target_path, extensions=None, recursive=recursive)
            
            # 按名称排序以确保一致性
            files_to_rename.sort(key=lambda x: x.name)
            
            # 如果需要打乱顺序
            if shuffle_order:
                random.shuffle(files_to_rename)
            
            result = {
                "success": True,
                "target_dir": str(target_path),
                "rename_pattern": rename_pattern,
                "shuffle_order": shuffle_order,
                "preview_only": preview_only,
                "renamed_files": [],
                "failed_files": [],
                "statistics": {
                    "total_files": len(files_to_rename),
                    "renamed_count": 0,
                    "failed_count": 0
                }
            }
            
            if preview_only:
                # 预览模式
                for index, file_path in enumerate(files_to_rename):
                    try:
                        new_name = rename_pattern.format(
                            name=file_path.stem,
                            ext=file_path.suffix,
                            index=index + 1,
                            index0=index
                        )
                        new_path = file_path.parent / new_name
                        
                        result["renamed_files"].append({
                            "success": True,
                            "action": "preview",
                            "old_name": str(file_path),
                            "new_name": str(new_path)
                        })
                    except Exception as e:
                        result["failed_files"].append({
                            "success": False,
                            "action": "failed",
                            "old_name": str(file_path),
                            "error": str(e)
                        })
                        result["statistics"]["failed_count"] += 1
                        result["success"] = False
                
                return result
            
            # 第一阶段：临时重命名（避免冲突）
            temp_mappings = []
            
            for file_path in files_to_rename:
                try:
                    # 生成临时文件名
                    temp_name = f"temp_{uuid.uuid4().hex[:8]}{file_path.suffix}"
                    temp_path = file_path.parent / temp_name
                    
                    # 临时重命名
                    file_path.rename(temp_path)
                    temp_mappings.append((temp_path, file_path.name))
                    
                except Exception as e:
                    result["failed_files"].append({
                        "success": False,
                        "action": "temp_rename_failed",
                        "old_name": str(file_path),
                        "error": str(e)
                    })
                    result["statistics"]["failed_count"] += 1
                    result["success"] = False
            
            # 第二阶段：正式重命名
            for index, (temp_path, original_name) in enumerate(temp_mappings):
                try:
                    # 生成最终文件名
                    new_name = rename_pattern.format(
                        name=Path(original_name).stem,
                        ext=temp_path.suffix,
                        index=index + 1,
                        index0=index
                    )
                    
                    final_path = temp_path.parent / new_name
                    
                    # 确保新文件名唯一
                    if final_path.exists():
                        final_path = get_unique_filename(final_path.parent, final_path.name)
                    
                    # 最终重命名
                    temp_path.rename(final_path)
                    
                    result["renamed_files"].append({
                        "success": True,
                        "action": "renamed",
                        "old_name": original_name,
                        "new_name": str(final_path.name)
                    })
                    result["statistics"]["renamed_count"] += 1
                    
                except Exception as e:
                    # 如果最终重命名失败，尝试恢复原名
                    try:
                        original_path = temp_path.parent / original_name
                        if not original_path.exists():
                            temp_path.rename(original_path)
                    except:
                        pass
                    
                    result["failed_files"].append({
                        "success": False,
                        "action": "final_rename_failed",
                        "old_name": original_name,
                        "error": str(e)
                    })
                    result["statistics"]["failed_count"] += 1
                    result["success"] = False
            
            self.operation_count += result["statistics"]["renamed_count"]
            
            self.logger.info(f"临时重命名完成: {result['statistics']}")
            return result
            
        except Exception as e:
            raise ProcessingError(f"临时重命名失败: {str(e)}")
    
    def rename_images_labels_sync(self, images_dir: str, labels_dir: str,
                                 prefix: str, digits: int = 5,
                                 shuffle_order: bool = False) -> Dict[str, Any]:
        """同步重命名images和labels目录中的对应文件
        
        Args:
            images_dir: 图片目录路径
            labels_dir: 标签目录路径
            prefix: 文件名前缀
            digits: 数字位数
            shuffle_order: 是否打乱顺序
            
        Returns:
            Dict[str, Any]: 重命名结果
        """
        import random
        import uuid
        
        try:
            images_path = validate_path(images_dir, must_exist=True, must_be_dir=True)
            labels_path = validate_path(labels_dir, must_exist=True, must_be_dir=True)
            
            self.logger.info(f"开始同步重命名: {images_path} 和 {labels_path}")
            self.logger.info(f"前缀: {prefix}, 位数: {digits}, 打乱顺序: {shuffle_order}")
            
            # 获取图片文件列表
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(images_path.glob(f"*{ext}"))
                image_files.extend(images_path.glob(f"*{ext.upper()}"))
            
            # 去重处理（避免同一文件被匹配多次）
            image_files = list(set(image_files))
            
            # 记录找到的图片文件
            # self.logger.info(f"找到 {len(image_files)} 个图片文件:")
            # for img_file in image_files:
            #     self.logger.info(f"  图片文件: {img_file.name}")
            
            # 过滤出有对应标签文件的图片
            valid_pairs = []
            for img_file in image_files:
                label_file = labels_path / f"{img_file.stem}.txt"
                if label_file.exists():
                    valid_pairs.append((img_file, label_file))
                    # self.logger.info(f"  有效文件对: {img_file.name} <-> {label_file.name}")
                else:
                    self.logger.warning(f"  图片文件 {img_file.name} 没有对应的标签文件 {img_file.stem}.txt")
            
            if not valid_pairs:
                raise ProcessingError("未找到有效的图片-标签文件对")
            
            # 按图片文件名排序
            valid_pairs.sort(key=lambda x: x[0].name)
            
            # 如果需要打乱顺序
            if shuffle_order:
                random.shuffle(valid_pairs)
            
            result = {
                "success": True,
                "images_dir": str(images_path),
                "labels_dir": str(labels_path),
                "prefix": prefix,
                "digits": digits,
                "shuffle_order": shuffle_order,
                "renamed_pairs": [],
                "failed_pairs": [],
                "statistics": {
                    "total_pairs": len(valid_pairs),
                    "renamed_count": 0,
                    "failed_count": 0
                }
            }
            
            # 第一阶段：临时重命名所有文件
            temp_mappings = []
            
            for img_file, label_file in valid_pairs:
                try:
                    # 检查文件是否真实存在
                    if not img_file.exists():
                        raise FileNotFoundError(f"图片文件不存在: {img_file}")
                    if not label_file.exists():
                        raise FileNotFoundError(f"标签文件不存在: {label_file}")
                    
                    # 生成临时文件名
                    temp_id = uuid.uuid4().hex[:8]
                    temp_img_name = f"temp_img_{temp_id}{img_file.suffix}"
                    temp_label_name = f"temp_label_{temp_id}.txt"
                    
                    temp_img_path = img_file.parent / temp_img_name
                    temp_label_path = label_file.parent / temp_label_name
                    
                    # self.logger.info(f"临时重命名: {img_file.name} -> {temp_img_name}")
                    # self.logger.info(f"临时重命名: {label_file.name} -> {temp_label_name}")
                    
                    # 临时重命名
                    img_file.rename(temp_img_path)
                    label_file.rename(temp_label_path)
                    
                    temp_mappings.append({
                        'temp_img': temp_img_path,
                        'temp_label': temp_label_path,
                        'original_img': img_file.name,
                        'original_label': label_file.name,
                        'img_ext': img_file.suffix
                    })
                    
                except Exception as e:
                    result["failed_pairs"].append({
                        "success": False,
                        "action": "temp_rename_failed",
                        "img_file": str(img_file),
                        "label_file": str(label_file),
                        "error": str(e)
                    })
                    result["statistics"]["failed_count"] += 1
                    result["success"] = False
            
            # 第二阶段：正式重命名
            for index, mapping in enumerate(temp_mappings):
                try:
                    # 生成最终文件名
                    final_name = f"{prefix}_{index+1:0{digits}d}"
                    final_img_name = f"{final_name}{mapping['img_ext']}"
                    final_label_name = f"{final_name}.txt"
                    
                    final_img_path = mapping['temp_img'].parent / final_img_name
                    final_label_path = mapping['temp_label'].parent / final_label_name
                    
                    # 确保文件名唯一
                    if final_img_path.exists():
                        final_img_path = get_unique_filename(final_img_path.parent, final_img_path.name)
                        # 同步更新标签文件名
                        final_label_name = f"{final_img_path.stem}.txt"
                        final_label_path = final_label_path.parent / final_label_name
                    
                    if final_label_path.exists():
                        final_label_path = get_unique_filename(final_label_path.parent, final_label_path.name)
                    
                    # 最终重命名
                    mapping['temp_img'].rename(final_img_path)
                    mapping['temp_label'].rename(final_label_path)
                    
                    result["renamed_pairs"].append({
                        "success": True,
                        "action": "renamed",
                        "old_img": mapping['original_img'],
                        "old_label": mapping['original_label'],
                        "new_img": final_img_path.name,
                        "new_label": final_label_path.name
                    })
                    result["statistics"]["renamed_count"] += 1
                    
                except Exception as e:
                    # 如果最终重命名失败，尝试恢复原名
                    try:
                        original_img_path = mapping['temp_img'].parent / mapping['original_img']
                        original_label_path = mapping['temp_label'].parent / mapping['original_label']
                        
                        if not original_img_path.exists():
                            mapping['temp_img'].rename(original_img_path)
                        if not original_label_path.exists():
                            mapping['temp_label'].rename(original_label_path)
                    except:
                        pass
                    
                    result["failed_pairs"].append({
                        "success": False,
                        "action": "final_rename_failed",
                        "img_file": mapping['original_img'],
                        "label_file": mapping['original_label'],
                        "error": str(e)
                    })
                    result["statistics"]["failed_count"] += 1
                    result["success"] = False
            
            self.operation_count += result["statistics"]["renamed_count"]
            
            # 记录详细的失败信息到日志
            if result["failed_pairs"]:
                self.logger.warning(f"有 {len(result['failed_pairs'])} 个文件对重命名失败:")
                for failed_item in result["failed_pairs"]:
                    self.logger.warning(f"  失败文件: {failed_item.get('img_file', 'N/A')} / {failed_item.get('label_file', 'N/A')}")
                    self.logger.warning(f"  失败原因: {failed_item.get('error', 'N/A')}")
                    self.logger.warning(f"  失败阶段: {failed_item.get('action', 'N/A')}")
            
            self.logger.info(f"同步重命名完成: {result['statistics']}")
            return result
            
        except Exception as e:
            raise ProcessingError(f"同步重命名失败: {str(e)}")
    
    def organize_files_by_extension(self, source_dir: str, target_dir: str,
                                   recursive: bool = False,
                                   create_subdirs: bool = True) -> Dict[str, Any]:
        """按文件扩展名组织文件
        
        Args:
            source_dir: 源目录
            target_dir: 目标目录
            recursive: 是否递归处理
            create_subdirs: 是否为每种扩展名创建子目录
            
        Returns:
            Dict[str, Any]: 组织结果
        """
        try:
            source_path = validate_path(source_dir, must_exist=True, must_be_dir=True)
            target_path = validate_path(target_dir, must_exist=False)
            
            create_directory(target_path)
            
            self.logger.info(f"开始按扩展名组织文件: {source_path} -> {target_path}")
            
            # 获取所有文件
            all_files = get_file_list(source_path, extensions=None, recursive=recursive)
            
            # 按扩展名分组
            files_by_ext = {}
            for file_path in all_files:
                ext = file_path.suffix.lower() or '.no_extension'
                if ext not in files_by_ext:
                    files_by_ext[ext] = []
                files_by_ext[ext].append(file_path)
            
            result = {
                "success": True,
                "source_dir": str(source_path),
                "target_dir": str(target_path),
                "extensions_found": list(files_by_ext.keys()),
                "organized_files": [],
                "failed_files": [],
                "statistics": {
                    "total_files": len(all_files),
                    "organized_count": 0,
                    "failed_count": 0,
                    "extensions_count": len(files_by_ext),
                    "total_size": 0
                }
            }
            
            # 组织文件
            for ext, files in files_by_ext.items():
                if create_subdirs:
                    ext_dir = target_path / ext.lstrip('.')
                    create_directory(ext_dir)
                else:
                    ext_dir = target_path
                
                for file_path in files:
                    try:
                        file_size = file_path.stat().st_size
                        target_file = ext_dir / file_path.name
                        
                        # 确保文件名唯一
                        target_file = get_unique_filename(target_file.parent, target_file.name)
                        
                        # 移动文件
                        move_file_safe(file_path, target_file)
                        
                        result["organized_files"].append({
                            "source_file": str(file_path),
                            "target_file": str(target_file),
                            "extension": ext,
                            "file_size": file_size
                        })
                        
                        result["statistics"]["organized_count"] += 1
                        result["statistics"]["total_size"] += file_size
                        
                    except Exception as e:
                        result["failed_files"].append({
                            "source_file": str(file_path),
                            "error": str(e)
                        })
                        result["statistics"]["failed_count"] += 1
                        result["success"] = False
            
            # 格式化大小信息
            result["statistics"]["total_size_formatted"] = format_file_size(
                result["statistics"]["total_size"]
            )
            
            self.operation_count += result["statistics"]["organized_count"]
            self.total_size_processed += result["statistics"]["total_size"]
            
            self.logger.info(f"文件组织完成: {result['statistics']}")
            return result
            
        except Exception as e:
            raise ProcessingError(f"文件组织失败: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取处理器统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "operation_count": self.operation_count,
            "total_size_processed": self.total_size_processed,
            "total_size_processed_formatted": format_file_size(self.total_size_processed)
        }
    
    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.operation_count = 0
        self.total_size_processed = 0
        self.logger.info("统计信息已重置")