#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_processor.py

YOLO数据集处理器

提供YOLO数据集的验证、清理、转换等功能。
"""

import os
import random
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..config.exceptions import DatasetError, FileProcessingError, ValidationError
from ..core.base import BaseProcessor
from ..core.progress import process_with_progress, progress_context
from ..core.utils import (
    copy_file_safe,
    create_directory,
    delete_file_safe,
    get_file_list,
    move_file_safe,
    validate_path,
)
from .dataset_processor import DatasetProcessor


class YOLOProcessor(DatasetProcessor):
    """YOLO数据集处理器

    提供YOLO数据集的各种处理功能，包括验证、清理、转换等。

    Attributes:
        image_extensions (List[str]): 支持的图像格式
        label_extension (str): 标签文件扩展名
    """

    def __init__(self, **kwargs):
        """初始化YOLO处理器"""
        # 先设置默认属性，确保即使初始化失败也有基本属性
        self.image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        self.label_extension = ".txt"
        self.classes_file = "classes.txt"

        # 调用父类初始化
        super().__init__(**kwargs)

        # 从配置获取支持的格式（覆盖默认值）
        try:
            self.image_extensions = self.get_config(
                "yolo.image_formats", self.image_extensions
            )
            self.label_extension = self.get_config(
                "yolo.label_format", self.label_extension
            )
            self.classes_file = self.get_config("yolo.classes_file", self.classes_file)
        except Exception as e:
            self.logger.warning(f"从配置获取YOLO设置失败，使用默认值: {e}")

    def initialize(self) -> None:
        """初始化处理器"""
        self.logger.info("YOLO处理器初始化完成")
        self.logger.debug(f"支持的图像格式: {self.image_extensions}")
        self.logger.debug(f"标签文件格式: {self.label_extension}")

    def process(self, *args, **kwargs) -> Any:
        """主要处理方法（由子方法实现具体功能）"""
        raise NotImplementedError("请使用具体的处理方法")

    def remove_zero_only_labels(self, dataset_path: str) -> Dict[str, Any]:
        """删除只包含类别0的标签文件及对应图像

        Args:
            dataset_path: 数据集路径

        Returns:
            Dict[str, Any]: 删除结果
        """
        try:
            dataset_dir = validate_path(dataset_path, must_exist=True, must_be_dir=True)
            self.logger.info(f"开始删除只包含类别0的标签: {dataset_dir}")

            # 获取所有标签文件（排除classes.txt）
            all_txt_files = get_file_list(
                dataset_dir, [self.label_extension], recursive=True
            )
            label_files = [
                lbl for lbl in all_txt_files if lbl.name != self.classes_file
            ]

            result = {
                "success": True,
                "dataset_path": str(dataset_dir),
                "deleted_files": {"labels": [], "images": []},
                "statistics": {
                    "total_labels": len(label_files),
                    "deleted_labels": 0,
                    "deleted_images": 0,
                },
            }

            # 检查每个标签文件
            zero_only_files = []
            with progress_context(len(label_files), "检查标签文件") as progress:
                for label_file in label_files:
                    try:
                        if self._is_zero_only_label(label_file):
                            zero_only_files.append(label_file)
                        progress.update_progress(1)
                    except Exception as e:
                        self.logger.warning(f"检查标签文件失败 {label_file}: {str(e)}")

            # 删除只包含类别0的文件
            with progress_context(len(zero_only_files), "删除文件") as progress:
                for label_file in zero_only_files:
                    try:
                        # 删除标签文件
                        delete_file_safe(label_file)
                        result["deleted_files"]["labels"].append(str(label_file))
                        result["statistics"]["deleted_labels"] += 1

                        # 查找并删除对应的图像文件
                        img_file = self._find_corresponding_image(
                            label_file, dataset_dir
                        )
                        if img_file and img_file.exists():
                            delete_file_safe(img_file)
                            result["deleted_files"]["images"].append(str(img_file))
                            result["statistics"]["deleted_images"] += 1

                        progress.update_progress(1)

                    except Exception as e:
                        self.logger.error(f"删除文件失败 {label_file}: {str(e)}")
                        result["success"] = False

            self.logger.info(f"删除完成: {result['statistics']}")
            return result

        except Exception as e:
            raise DatasetError(
                f"删除只包含类别0的标签失败: {str(e)}", dataset_path=dataset_path
            )

    def _validate_label_file(self, label_file: Path) -> bool:
        """验证标签文件格式

        Args:
            label_file: 标签文件路径

        Returns:
            bool: 是否有效

        Raises:
            ValidationError: 验证失败
        """
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # 空文件和只包含空行的文件被认为是有效的
            # 这些情况在分割和检测数据集中都是正常的
            valid_lines = [line.strip() for line in lines if line.strip()]
            if not valid_lines:
                return True  # 空文件被认为是有效的

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue

                parts = line.split()
                if len(parts) != 5:
                    raise ValidationError(
                        f"标签格式错误：第{line_num}行应包含5个值",
                        validation_type="label_format",
                        expected="5 values",
                        actual=f"{len(parts)} values",
                    )

                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # 验证坐标范围
                    if not (
                        0 <= x_center <= 1
                        and 0 <= y_center <= 1
                        and 0 <= width <= 1
                        and 0 <= height <= 1
                    ):
                        raise ValidationError(
                            f"坐标值超出范围：第{line_num}行",
                            validation_type="coordinate_range",
                        )

                    if class_id < 0:
                        raise ValidationError(
                            f"类别ID不能为负数：第{line_num}行",
                            validation_type="class_id",
                        )

                except ValueError as e:
                    raise ValidationError(
                        f"数值格式错误：第{line_num}行 - {str(e)}",
                        validation_type="number_format",
                    )

            return True

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                f"读取标签文件失败: {str(e)}", validation_type="file_read"
            )

    def _is_zero_only_label(self, label_file: Path) -> bool:
        """检查标签文件是否只包含类别0

        Args:
            label_file: 标签文件路径

        Returns:
            bool: 是否只包含类别0
        """
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            has_content = False
            for line in lines:
                line = line.strip()
                if not line:  # 跳过空行
                    continue

                has_content = True
                parts = line.split()
                if len(parts) >= 1:
                    try:
                        class_id = int(parts[0])
                        if class_id != 0:
                            return False
                    except ValueError:
                        return False

            return has_content  # 只有当文件有内容且全是类别0时才返回True

        except Exception:
            return False

    def _find_corresponding_image(
        self, label_file: Path, dataset_dir: Path
    ) -> Optional[Path]:
        """查找标签文件对应的图像文件

        Args:
            label_file: 标签文件路径
            dataset_dir: 数据集目录

        Returns:
            Optional[Path]: 对应的图像文件路径
        """
        label_stem = label_file.stem

        for ext in self.image_extensions:
            img_file = label_file.parent / f"{label_stem}{ext}"
            if img_file.exists():
                return img_file

        return None

    def get_dataset_statistics(self, dataset_path: str) -> Dict[str, Any]:
        """获取数据集统计信息

        Args:
            dataset_path: 数据集路径

        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            original_dataset_path = dataset_path
            dataset_dir = validate_path(dataset_path, must_exist=True, must_be_dir=True)

            # 智能检测数据集根目录，如果传入的是子目录则自动调整
            dataset_dir = self._detect_dataset_root(dataset_dir)

            self.logger.info(f"开始获取YOLO数据集统计信息: {dataset_dir}")

            # 调用父类的验证方法
            validation_result = self._validate_yolo_dataset(
                dataset_dir, check_integrity=True
            )

            # 检查classes文件
            classes_file = dataset_dir / self.classes_file
            classes_file_path = str(classes_file) if classes_file.exists() else None

            stats = validation_result["statistics"].copy()
            stats["dataset_path"] = str(dataset_dir)  # 使用调整后的路径
            stats["original_path"] = original_dataset_path  # 保存原始路径用于比较
            stats["is_valid"] = validation_result.get("success", False)
            stats["has_classes_file"] = classes_file_path is not None

            # 添加类别统计
            if classes_file_path:
                try:
                    with open(classes_file_path, "r", encoding="utf-8") as f:
                        classes = [
                            line.strip() for line in f.readlines() if line.strip()
                        ]
                    stats["num_classes"] = len(classes)
                    stats["class_names"] = classes
                except Exception:
                    stats["num_classes"] = 0
                    stats["class_names"] = []
            else:
                stats["num_classes"] = 0
                stats["class_names"] = []

            # 返回完整的验证结果，包含issues字段
            result = {
                "statistics": stats,
                "valid": validation_result.get("success", False),
                "classes_file": classes_file_path,
                "issues": validation_result.get("issues", []),  # 添加issues字段
            }

            return result

        except Exception as e:
            raise DatasetError(
                f"获取数据集统计信息失败: {str(e)}", dataset_path=dataset_path
            )

    def clean_unmatched_files(
        self, dataset_path: str, dry_run: bool = False
    ) -> Dict[str, Any]:
        """清理数据集中不匹配的文件

        Args:
            dataset_path: 数据集路径
            dry_run: 是否为试运行（不实际删除文件）

        Returns:
            Dict[str, Any]: 清理结果

        Raises:
            DatasetError: 清理失败
        """
        try:
            dataset_dir = validate_path(dataset_path, must_exist=True, must_be_dir=True)
            self.logger.info(f"开始清理不匹配文件: {dataset_dir}")

            # 先验证数据集
            validation_result = self.get_dataset_statistics(dataset_path)

            result = {
                "success": True,
                "dataset_path": str(dataset_dir),
                "dry_run": dry_run,
                "deleted_files": {
                    "orphaned_images": [],
                    "orphaned_labels": [],
                    "invalid_labels": [],
                },
                "statistics": {
                    "total_deleted": 0,
                    "deleted_images": 0,
                    "deleted_labels": 0,
                },
            }

            # 如果数据集已经有效，无需清理
            if validation_result["valid"]:
                self.logger.info("数据集已经有效，无需清理")
                return result

            # 收集需要删除的文件
            files_to_delete = []

            # 调试信息：打印验证结果
            self.logger.info(
                f"验证结果issues数量: {len(validation_result.get('issues', []))}"
            )
            for i, issue in enumerate(validation_result.get("issues", [])):
                self.logger.info(
                    f"Issue {i}: type={issue.get('type')}, count={issue.get('count')}, files_count={len(issue.get('files', []))}"
                )

            # 从验证结果的issues中提取需要删除的文件
            for issue in validation_result.get("issues", []):
                if issue["type"] == "orphaned_images":
                    # 添加孤立的图像文件（没有对应标签的图像）
                    for img_path in issue["files"]:
                        files_to_delete.append((Path(img_path), "orphaned_images"))
                elif issue["type"] == "orphaned_labels":
                    # 添加孤立的标签文件（没有对应图像的标签）
                    for lbl_path in issue["files"]:
                        files_to_delete.append((Path(lbl_path), "orphaned_labels"))
                elif issue["type"] == "invalid_labels":
                    # 添加无效的标签文件
                    for invalid_label in issue["examples"]:
                        files_to_delete.append(
                            (Path(invalid_label["file"]), "invalid_labels")
                        )
                elif issue["type"] == "empty_labels":
                    # 添加空标签文件
                    for empty_label_path in issue["files"]:
                        files_to_delete.append((Path(empty_label_path), "empty_labels"))

            # 删除文件
            with progress_context(len(files_to_delete), "清理不匹配文件") as progress:
                for file_path, file_type in files_to_delete:
                    try:
                        if dry_run:
                            # 试运行模式，只记录不实际删除
                            result["deleted_files"][file_type].append(str(file_path))
                            self.logger.info(f"[试运行] 将删除: {file_path}")
                        else:
                            # 实际删除文件
                            if file_path.exists():
                                delete_file_safe(file_path)
                                result["deleted_files"][file_type].append(
                                    str(file_path)
                                )
                                self.logger.info(f"已删除: {file_path}")

                                # 更新统计
                                if file_type == "orphaned_images":
                                    result["statistics"]["deleted_images"] += 1
                                else:
                                    result["statistics"]["deleted_labels"] += 1

                        progress.update_progress(1)

                    except Exception as e:
                        self.logger.error(f"删除文件失败 {file_path}: {str(e)}")
                        result["success"] = False

            # 更新总删除数
            result["statistics"]["total_deleted"] = (
                result["statistics"]["deleted_images"]
                + result["statistics"]["deleted_labels"]
            )

            if dry_run:
                self.logger.info(f"试运行完成，将删除 {len(files_to_delete)} 个文件")
            else:
                self.logger.info(f"清理完成: {result['statistics']}")

            return result

        except Exception as e:
            raise DatasetError(
                f"清理不匹配文件失败: {str(e)}", dataset_path=dataset_path
            )

    def process_ctds_dataset(
        self, input_path: str, output_name: str = None
    ) -> Dict[str, Any]:
        """处理CTDS标注数据

        基于yolo_dataset_cleaner.py的逻辑，处理CTDS格式的标注数据：
        - 剔除空标签文件或包含非法标注数据的文件
        - 图像和标签文件重命名为统一格式：<项目名>-00001.jpg / .txt
        - 自动复制 obj.names 到 classes.txt
        - 自动生成 images/ 和 labels/ 文件夹
        - 最终重命名整个项目文件夹为 <项目名>-总数

        Args:
            input_path: 输入数据集路径（包含obj.names和obj_train_data文件夹）
            output_name: 输出项目名称，为空时自动从obj.names生成

        Returns:
            Dict[str, Any]: 处理结果

        Raises:
            DatasetError: 处理失败
        """
        try:
            input_dir = validate_path(input_path, must_exist=True, must_be_dir=True)
            self.logger.info(f"开始处理CTDS数据集: {input_dir}")

            # 检查必要的文件和目录
            obj_names_path = input_dir / "obj.names"
            obj_train_data_path = input_dir / "obj_train_data"

            if not obj_names_path.exists():
                raise DatasetError("未找到obj.names文件", dataset_path=str(input_dir))

            if not obj_train_data_path.exists():
                raise DatasetError(
                    "未找到obj_train_data目录", dataset_path=str(input_dir)
                )

            # 获取项目名称
            project_name = self._get_project_name(obj_names_path, output_name)

            # 预检测数据集类型（基于原始数据）
            self.logger.info("正在预检测数据集类型...")
            pre_detection_result = self.detect_dataset_type(str(obj_train_data_path))

            # 在处理前返回检测结果，让调用方进行用户确认
            if pre_detection_result.get("success"):
                # 返回预检测结果，等待用户确认
                return {
                    "success": True,
                    "stage": "pre_detection",
                    "input_path": str(input_dir),
                    "project_name": project_name,
                    "pre_detection_result": pre_detection_result,
                    "obj_names_path": str(obj_names_path),
                    "obj_train_data_path": str(obj_train_data_path),
                }
            else:
                self.logger.warning("预检测失败，将使用默认检测类型进行处理")
                confirmed_type = "detection"  # 默认为检测类型

            # 如果预检测失败，直接进行处理
            return self._execute_ctds_processing(
                input_dir,
                project_name,
                obj_names_path,
                obj_train_data_path,
                confirmed_type,
                pre_detection_result,
            )

        except Exception as e:
            self.logger.error(f"CTDS数据处理失败: {str(e)}")
            raise DatasetError(
                f"CTDS数据处理失败: {str(e)}",
                dataset_path=str(input_dir) if "input_dir" in locals() else input_path,
            )

    def continue_ctds_processing(
        self, pre_result: Dict[str, Any], confirmed_type: str
    ) -> Dict[str, Any]:
        """继续CTDS数据处理（在用户确认数据集类型后）

        Args:
            pre_result: 预检测结果
            confirmed_type: 用户确认的数据集类型

        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            input_dir = Path(pre_result["input_path"])
            project_name = pre_result["project_name"]
            obj_names_path = Path(pre_result["obj_names_path"])
            obj_train_data_path = Path(pre_result["obj_train_data_path"])
            pre_detection_result = pre_result["pre_detection_result"]

            return self._execute_ctds_processing(
                input_dir,
                project_name,
                obj_names_path,
                obj_train_data_path,
                confirmed_type,
                pre_detection_result,
            )

        except Exception as e:
            raise DatasetError(
                f"CTDS数据处理失败: {str(e)}",
                dataset_path=pre_result.get("input_path", ""),
            )

    def _execute_ctds_processing(
        self,
        input_dir: Path,
        project_name: str,
        obj_names_path: Path,
        obj_train_data_path: Path,
        confirmed_type: str,
        pre_detection_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """执行CTDS数据处理的核心逻辑

        Args:
            input_dir: 输入目录
            project_name: 项目名称
            obj_names_path: obj.names文件路径
            obj_train_data_path: obj_train_data目录路径
            confirmed_type: 确认的数据集类型
            pre_detection_result: 预检测结果

        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            # 创建输出目录
            output_dir = input_dir.parent / project_name
            create_directory(output_dir)

            labels_dir = output_dir / "labels"
            images_dir = output_dir / "images"
            create_directory(labels_dir)
            create_directory(images_dir)

            result = {
                "success": True,
                "input_path": str(input_dir),
                "output_path": str(output_dir),
                "project_name": project_name,
                "pre_detection_result": pre_detection_result,
                "confirmed_type": confirmed_type,
                "processed_files": {"images": [], "labels": [], "classes_file": None},
                "invalid_files": [],
                "statistics": {
                    "total_processed": 0,
                    "invalid_removed": 0,
                    "final_count": 0,
                },
            }

            # 获取所有标签文件（排除train.txt）
            all_files = list(obj_train_data_path.iterdir())
            txt_files = [
                f
                for f in all_files
                if f.suffix.lower() == ".txt" and f.name.lower() != "train.txt"
            ]

            self.logger.info(f"找到 {len(txt_files)} 个标签文件")
            self.logger.info(f"使用确认的数据集类型进行验证: {confirmed_type}")

            # 处理标签文件和对应的图像
            count = 1
            invalid_count = 0

            with progress_context(len(txt_files), "处理CTDS数据") as progress:
                for txt_file in txt_files:
                    try:
                        # 检查标签文件是否有效（根据确认的数据集类型）
                        if self._contains_invalid_ctds_data(txt_file, confirmed_type):
                            invalid_count += 1
                            result["invalid_files"].append(str(txt_file))
                            progress.update_progress(1)
                            continue

                        # 生成新的文件名
                        new_txt_name = f"{project_name}-{count:05d}.txt"
                        new_txt_path = labels_dir / new_txt_name

                        # 移动标签文件
                        move_file_safe(txt_file, new_txt_path)
                        result["processed_files"]["labels"].append(str(new_txt_path))

                        # 查找对应的图像文件
                        base_name = txt_file.stem
                        img_file = self._find_corresponding_ctds_image(
                            base_name, obj_train_data_path
                        )

                        if img_file:
                            # 保持原始扩展名
                            new_img_name = (
                                f"{project_name}-{count:05d}{img_file.suffix}"
                            )
                            new_img_path = images_dir / new_img_name

                            # 移动图像文件
                            move_file_safe(img_file, new_img_path)
                            result["processed_files"]["images"].append(
                                str(new_img_path)
                            )
                        else:
                            self.logger.warning(
                                f"未找到标签文件 {txt_file.name} 对应的图像文件"
                            )

                        count += 1
                        progress.update_progress(1)

                    except Exception as e:
                        self.logger.error(f"处理文件失败 {txt_file}: {str(e)}")
                        result["success"] = False

            # 更新统计信息
            total_files = len(txt_files)
            valid_files = count - 1
            result["statistics"]["total_processed"] = total_files
            result["statistics"]["invalid_removed"] = invalid_count
            result["statistics"]["final_count"] = valid_files

            # 输出处理统计信息
            self.logger.info(
                f"CTDS数据处理完成 - 总文件数: {total_files}, 有效文件: {valid_files}, 无效文件: {invalid_count}"
            )

            # 使用OpenCV重新保存图像（如果可用）
            try:
                import cv2

                self._convert_images_with_opencv(images_dir)
            except ImportError:
                self.logger.warning("OpenCV不可用，跳过图像转换")
            except Exception as e:
                self.logger.warning(f"图像转换失败: {str(e)}")

            # 复制obj.names到classes.txt
            classes_file_path = output_dir / "classes.txt"
            try:
                copy_file_safe(obj_names_path, classes_file_path)
                result["processed_files"]["classes_file"] = str(classes_file_path)
                self.logger.info(
                    f"已复制 {obj_names_path.name} 到 {classes_file_path.name}"
                )
            except Exception as e:
                self.logger.error(f"复制classes文件失败: {str(e)}")
                result["success"] = False

            # 重命名项目文件夹，包含文件数
            final_count = count - 1
            final_project_name = f"{project_name}-{final_count:05d}"
            final_output_dir = input_dir.parent / final_project_name

            if output_dir != final_output_dir:
                try:
                    output_dir.rename(final_output_dir)
                    result["output_path"] = str(final_output_dir)
                    result["project_name"] = final_project_name
                    self.logger.info(f"项目文件夹已重命名为: {final_project_name}")
                except Exception as e:
                    self.logger.error(f"重命名项目文件夹失败: {str(e)}")
                    result["success"] = False

            # 更新统计信息
            result["statistics"]["total_processed"] = len(txt_files)
            result["statistics"]["invalid_removed"] = invalid_count
            result["statistics"]["final_count"] = final_count

            # 设置最终的数据集类型检测结果
            result["detected_dataset_type"] = confirmed_type
            result["detection_confidence"] = pre_detection_result.get("confidence", 1.0)
            result["dataset_type_detection"] = pre_detection_result

            return result

        except Exception as e:
            raise DatasetError(
                f"CTDS数据处理失败: {str(e)}", dataset_path=str(input_dir)
            )

    def _get_project_name(self, obj_names_path: Path, manual_name: str = None) -> str:
        """获取项目名称

        Args:
            obj_names_path: obj.names文件路径
            manual_name: 手动指定的名称

        Returns:
            str: 项目名称
        """
        if manual_name and manual_name.strip():
            return manual_name.strip()

        try:
            with open(obj_names_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                if lines:
                    project_name = "-".join(lines)
                    self.logger.info(f"从 obj.names 获取到项目名称: {project_name}")
                    return project_name
        except Exception as e:
            self.logger.warning(f"读取 obj.names 失败: {str(e)}")

        # 默认名称
        return "dataset"

    def _contains_invalid_ctds_data(
        self, label_file: Path, dataset_type: str = "detection"
    ) -> bool:
        """检查CTDS标签文件是否包含无效数据

        Args:
            label_file: 标签文件路径
            dataset_type: 数据集类型 ("detection" 或 "segmentation")

        Returns:
            bool: 是否包含无效数据
        """
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # 空文件和只包含空行的文件被认为是有效的
            # 这些情况在分割和检测数据集中都是正常的

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = line.split()

                    # 根据数据集类型设置不同的验证标准
                    if dataset_type == "segmentation":
                        # 分割数据：至少需要7列（类别 + 至少6个坐标值）
                        if len(data) < 7:
                            self.logger.debug(
                                f"文件 {label_file} 分割数据格式错误，需要至少7列: {line}"
                            )
                            return True

                        # 检查列数是否为奇数（类别 + 偶数个坐标）
                        if len(data) % 2 == 0:
                            self.logger.debug(
                                f"文件 {label_file} 分割数据列数应为奇数: {line}"
                            )
                            return True

                        # 验证所有坐标值
                        coord_values = data[1:]
                    else:
                        # 检测数据：需要5列（类别 + 4个坐标值）
                        if len(data) < 5:
                            self.logger.debug(
                                f"文件 {label_file} 检测数据格式错误，需要至少5列: {line}"
                            )
                            return True

                        # 只验证前4个坐标值
                        coord_values = data[1:5]

                    # 检查类别ID
                    try:
                        class_id = int(data[0])
                        if class_id < 0:
                            self.logger.debug(
                                f"文件 {label_file} 中类别ID不能为负数: {line}"
                            )
                            return True
                    except ValueError:
                        self.logger.debug(f"文件 {label_file} 中类别ID格式错误: {line}")
                        return True

                    # 检查坐标值范围和有效性
                    for i, num_str in enumerate(coord_values):
                        try:
                            num = float(num_str)
                            # 检查是否为负数
                            if num < 0.0:
                                self.logger.debug(
                                    f"文件 {label_file} 中存在负数坐标: {line} (第{i+2}列: {num})"
                                )
                                return True
                            # 检查是否超出范围[0,1]
                            if num > 1.0:
                                self.logger.debug(
                                    f"文件 {label_file} 中坐标超出范围[0,1]: {line} (第{i+2}列: {num})"
                                )
                                return True
                        except ValueError:
                            self.logger.debug(
                                f"文件 {label_file} 中坐标格式错误: {line} (第{i+2}列: {num_str})"
                            )
                            return True

                except ValueError:
                    self.logger.debug(f"文件 {label_file} 中存在无法解析的数据: {line}")
                    return True

            return False

        except Exception as e:
            self.logger.error(f"检查标签文件失败 {label_file}: {str(e)}")
            return True

    def detect_dataset_type(self, dataset_path: str) -> Dict[str, Any]:
        """自动检测数据集类型（检测还是分割）

        随机从labels目录中选10个txt文件，如果只有5列那就是检测，否则是分割。

        Args:
            dataset_path: 数据集路径

        Returns:
            Dict[str, Any]: 检测结果
        """
        try:
            self.logger.info(f"开始检测数据集类型，路径: {dataset_path}")
            dataset_dir = validate_path(dataset_path, must_exist=True, must_be_dir=True)
            self.logger.info(f"验证路径成功: {dataset_dir}")

            # 查找labels目录
            labels_dir = dataset_dir
            self.logger.info(f"查找obj_train_data目录: {labels_dir}")
            if not labels_dir.exists():
                self.logger.error(f"obj_train_data目录不存在: {labels_dir}")
                return {
                    "success": False,
                    "error": "未找到obj_train_data目录",
                    "dataset_type": "unknown",
                }

            # 获取标签文件
            label_files = list(labels_dir.glob("*.txt"))
            self.logger.info(f"找到 {len(label_files)} 个txt文件")
            if not label_files:
                self.logger.error("obj_train_data目录中没有找到txt文件")
                return {
                    "success": False,
                    "error": "obj_train_data目录中没有找到txt文件",
                    "dataset_type": "unknown",
                }

            # 随机选择最多10个文件进行检测
            sample_size = min(100, len(label_files))
            sample_files = random.sample(label_files, sample_size)
            self.logger.info(f"随机选择 {sample_size} 个文件进行分析")

            detection_files = 0
            segmentation_files = 0
            analyzed_files = []

            # 检查每个样本文件
            for i, label_file in enumerate(sample_files):
                try:
                    self.logger.info(f"分析文件 {i+1}/{sample_size}: {label_file.name}")
                    with open(label_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    self.logger.info(f"文件 {label_file.name} 共有 {len(lines)} 行")

                    # 分析文件中的每一行来判断格式
                    detection_lines = 0
                    segmentation_lines = 0
                    valid_lines = 0

                    for line_num, line in enumerate(lines):
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        valid_lines += 1

                        if len(parts) == 5:
                            # 检测格式：class_id x_center y_center width height
                            detection_lines += 1
                            if line_num < 3:  # 只记录前3行的详细信息
                                self.logger.info(
                                    f"文件 {label_file.name} 第 {line_num+1} 行: {len(parts)} 列 - {line[:50]}..."
                                )
                        else:
                            # 其他格式认为是分割
                            segmentation_lines += 1
                            if line_num < 3:  # 只记录前3行的详细信息
                                self.logger.info(
                                    f"文件 {label_file.name} 第 {line_num+1} 行: {len(parts)} 列 - {line[:50]}..."
                                )

                    if valid_lines == 0:
                        self.logger.warning(
                            f"文件 {label_file.name} 没有找到有效数据行"
                        )
                        continue

                    # 根据行数占比判断文件类型（超过50%的行决定文件类型）
                    if detection_lines > segmentation_lines:
                        detection_files += 1
                        file_type = "detection"
                        dominant_columns = 5
                        self.logger.info(
                            f"文件 {label_file.name} 判定为检测格式 (检测行:{detection_lines}, 分割行:{segmentation_lines})"
                        )
                    else:
                        segmentation_files += 1
                        file_type = "segmentation"
                        # 取分割行中最常见的列数
                        dominant_columns = max(
                            [
                                len(line.split())
                                for line in lines
                                if line.strip() and len(line.split()) != 5
                            ],
                            default=0,
                        )
                        self.logger.info(
                            f"文件 {label_file.name} 判定为分割格式 (检测行:{detection_lines}, 分割行:{segmentation_lines})"
                        )

                    analyzed_files.append(
                        {
                            "file": label_file.name,
                            "type": file_type,
                            "columns": dominant_columns,
                            "detection_lines": detection_lines,
                            "segmentation_lines": segmentation_lines,
                            "total_lines": valid_lines,
                        }
                    )

                except Exception as e:
                    self.logger.error(f"分析标签文件失败 {label_file}: {str(e)}")
                    continue

            # 判断数据集类型
            total_analyzed = detection_files + segmentation_files
            self.logger.info(
                f"分析结果: 检测文件={detection_files}, 分割文件={segmentation_files}, 总计={total_analyzed}"
            )

            if total_analyzed == 0:
                dataset_type = "unknown"
                confidence = 0.0
                self.logger.warning("没有成功分析任何文件，数据集类型未知")
            elif detection_files > segmentation_files:
                dataset_type = "detection"
                confidence = detection_files / total_analyzed
                self.logger.info(f"判定为检测数据集，置信度: {confidence:.2f}")
            elif segmentation_files > detection_files:
                dataset_type = "segmentation"
                confidence = segmentation_files / total_analyzed
                self.logger.info(f"判定为分割数据集，置信度: {confidence:.2f}")
            else:
                dataset_type = "mixed"
                confidence = 0.5
                self.logger.info("判定为混合数据集")

            result = {
                "success": True,
                "dataset_type": dataset_type,
                "confidence": confidence,
                "statistics": {
                    "total_files_available": len(label_files),
                    "files_analyzed": total_analyzed,
                    "detection_files": detection_files,
                    "segmentation_files": segmentation_files,
                },
                "sample_files": analyzed_files[:5],  # 只返回前5个样本
            }

            self.logger.info(f"数据集类型检测完成: {result}")
            return result

        except Exception as e:
            self.logger.error(f"数据集类型检测异常: {str(e)}")
            return {"success": False, "error": str(e), "dataset_type": "unknown"}

    def _find_corresponding_ctds_image(
        self, base_name: str, data_dir: Path
    ) -> Optional[Path]:
        """查找对应的图像文件

        Args:
            base_name: 基础文件名（不含扩展名）
            data_dir: 数据目录

        Returns:
            Optional[Path]: 对应的图像文件路径
        """
        # 支持的图像格式
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        for ext in image_extensions:
            img_file = data_dir / f"{base_name}{ext}"
            if img_file.exists():
                return img_file

        return None

    def _convert_images_with_opencv(self, images_dir: Path) -> None:
        """使用OpenCV重新保存图像

        Args:
            images_dir: 图像目录
        """
        try:
            import cv2

            # 只处理jpg文件
            jpg_files = [f for f in images_dir.iterdir() if f.suffix.lower() == ".jpg"]

            if not jpg_files:
                return

            self.logger.info(f"使用OpenCV重新保存 {len(jpg_files)} 个图像文件")

            with progress_context(len(jpg_files), "重新保存图像") as progress:
                for img_file in jpg_files:
                    try:
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            cv2.imwrite(str(img_file), img)
                        else:
                            self.logger.warning(f"无法读取图像文件: {img_file}")
                    except Exception as e:
                        self.logger.warning(f"处理图像文件失败 {img_file}: {str(e)}")

                    progress.update_progress(1)

            self.logger.info("图像重新保存完成")

        except ImportError:
            raise
        except Exception as e:
            self.logger.error(f"OpenCV图像转换失败: {str(e)}")
            raise

    def merge_datasets(
        self,
        dataset_paths: List[str],
        output_path: str,
        output_name: str = None,
        image_prefix: str = "img",
    ) -> Dict[str, Any]:
        """合并多个YOLO数据集

        Args:
            dataset_paths: 数据集路径列表
            output_path: 输出目录路径
            output_name: 输出数据集名称（可选，默认根据classes.txt生成）
            image_prefix: 图片前缀名称（默认为"img"）

        Returns:
            Dict[str, Any]: 合并结果
        """
        try:
            self.logger.info(f"开始合并 {len(dataset_paths)} 个数据集")

            # 验证输入路径
            validated_paths = []
            for path in dataset_paths:
                validated_path = validate_path(path, must_exist=True, must_be_dir=True)
                validated_paths.append(validated_path)

            # 验证所有数据集的classes.txt是否相同
            classes_validation = self._validate_classes_consistency(validated_paths)
            if not classes_validation["consistent"]:
                raise DatasetError(
                    f"数据集classes.txt不一致: {classes_validation['details']}"
                )

            # 获取统一的类别信息
            common_classes = classes_validation["classes"]

            # 生成输出目录名称
            if not output_name:
                output_name = self._generate_output_name(
                    common_classes, validated_paths
                )

            # 创建输出目录
            output_dir = Path(output_path) / output_name
            create_directory(output_dir)

            # 合并数据集
            merge_result = self._merge_dataset_files(
                validated_paths, output_dir, image_prefix, common_classes
            )

            self.logger.info(f"数据集合并完成: {output_dir}")

            return {
                "success": True,
                "output_path": str(output_dir),
                "output_name": output_name,
                "merged_datasets": len(validated_paths),
                "total_images": merge_result["total_images"],
                "total_labels": merge_result["total_labels"],
                "classes": common_classes,
                "statistics": merge_result["statistics"],
            }

        except Exception as e:
            self.logger.error(f"数据集合并失败: {str(e)}")
            raise DatasetError(f"数据集合并失败: {str(e)}")

    def _validate_classes_consistency(
        self, dataset_paths: List[Path]
    ) -> Dict[str, Any]:
        """验证所有数据集的classes.txt是否一致

        Args:
            dataset_paths: 数据集路径列表

        Returns:
            Dict[str, Any]: 验证结果
        """
        classes_info = []

        for dataset_path in dataset_paths:
            classes_file = dataset_path / self.classes_file
            if not classes_file.exists():
                return {
                    "consistent": False,
                    "details": f"数据集 {dataset_path} 缺少 {self.classes_file} 文件",
                    "classes": None,
                }

            try:
                with open(classes_file, "r", encoding="utf-8") as f:
                    classes = [line.strip() for line in f.readlines() if line.strip()]
                classes_info.append({"path": dataset_path, "classes": classes})
            except Exception as e:
                return {
                    "consistent": False,
                    "details": f"读取 {classes_file} 失败: {str(e)}",
                    "classes": None,
                }

        # 检查所有数据集的类别是否相同
        if not classes_info:
            return {
                "consistent": False,
                "details": "没有找到有效的classes.txt文件",
                "classes": None,
            }

        reference_classes = classes_info[0]["classes"]

        for info in classes_info[1:]:
            if info["classes"] != reference_classes:
                return {
                    "consistent": False,
                    "details": f"数据集 {info['path']} 的类别与其他数据集不一致",
                    "classes": None,
                }

        return {
            "consistent": True,
            "details": "所有数据集的类别一致",
            "classes": reference_classes,
        }

    def _generate_output_name(
        self, classes: List[str], dataset_paths: List[Path]
    ) -> str:
        """生成输出目录名称

        Args:
            classes: 类别列表
            dataset_paths: 数据集路径列表

        Returns:
            str: 输出目录名称
        """
        # 计算总图片数量
        total_images = 0
        for dataset_path in dataset_paths:
            image_files = get_file_list(
                dataset_path, self.image_extensions, recursive=True
            )
            total_images += len(image_files)

        # 使用类别名称拼接作为前缀
        classes_prefix = "_".join(classes[:3])  # 最多使用前3个类别名
        if len(classes) > 3:
            classes_prefix += f"_etc{len(classes)}"

        # 生成最终名称
        output_name = f"{classes_prefix}_merged_{total_images}imgs"

        # 清理文件名中的非法字符
        output_name = re.sub(r'[<>:"/\\|?*]', "_", output_name)

        return output_name

    def _merge_dataset_files(
        self,
        dataset_paths: List[Path],
        output_dir: Path,
        image_prefix: str,
        classes: List[str],
    ) -> Dict[str, Any]:
        """合并数据集文件（优化版本）

        Args:
            dataset_paths: 数据集路径列表
            output_dir: 输出目录
            image_prefix: 图片前缀
            classes: 类别列表

        Returns:
            Dict[str, Any]: 合并结果
        """
        # 性能监控开始
        start_time = time.time()
        total_images = 0
        total_labels = 0
        statistics = []
        performance_stats = {
            "start_time": start_time,
            "dataset_times": [],
            "total_size_bytes": 0,
            "avg_file_size": 0,
        }

        # 创建输出子目录
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        create_directory(images_dir)
        create_directory(labels_dir)

        # 复制classes.txt
        classes_file = output_dir / self.classes_file
        with open(classes_file, "w", encoding="utf-8") as f:
            for class_name in classes:
                f.write(f"{class_name}\n")

        current_index = 1

        for i, dataset_path in enumerate(dataset_paths):
            dataset_start_time = time.time()
            self.logger.info(f"处理数据集 {i+1}/{len(dataset_paths)}: {dataset_path}")

            # 获取数据集文件
            image_files = get_file_list(
                dataset_path, self.image_extensions, recursive=True
            )
            label_files = get_file_list(
                dataset_path, [self.label_extension], recursive=True
            )

            # 过滤掉classes.txt
            label_files = [lbl for lbl in label_files if lbl.name != self.classes_file]

            # 计算数据集大小
            dataset_size = sum(f.stat().st_size for f in image_files if f.exists())
            dataset_size += sum(f.stat().st_size for f in label_files if f.exists())
            performance_stats["total_size_bytes"] += dataset_size

            dataset_stats = {
                "dataset_path": str(dataset_path),
                "images_count": len(image_files),
                "labels_count": len(label_files),
                "start_index": current_index,
                "dataset_size_mb": round(dataset_size / (1024 * 1024), 2),
            }

            # 预建立标签映射索引（优化点1）
            mapping_start = time.time()
            self.logger.info(f"建立标签映射索引...")
            label_mapping = self._build_label_mapping(label_files)
            mapping_time = time.time() - mapping_start

            # 使用并行处理（优化点2）
            processing_start = time.time()
            dataset_result = self._merge_dataset_parallel(
                image_files,
                images_dir,
                labels_dir,
                image_prefix,
                current_index,
                label_mapping,
                i + 1,
            )
            processing_time = time.time() - processing_start

            total_images += dataset_result["images_processed"]
            total_labels += dataset_result["labels_processed"]
            current_index += dataset_result["images_processed"]

            dataset_end_time = time.time()
            dataset_total_time = dataset_end_time - dataset_start_time

            # 计算处理速度
            files_per_second = (
                len(image_files) / dataset_total_time if dataset_total_time > 0 else 0
            )
            mb_per_second = (
                (dataset_size / (1024 * 1024)) / dataset_total_time
                if dataset_total_time > 0
                else 0
            )

            dataset_stats.update(
                {
                    "end_index": current_index - 1,
                    "processing_time_seconds": round(dataset_total_time, 2),
                    "mapping_time_seconds": round(mapping_time, 3),
                    "copy_time_seconds": round(processing_time, 2),
                    "files_per_second": round(files_per_second, 1),
                    "mb_per_second": round(mb_per_second, 1),
                    "failed_count": dataset_result.get("failed_count", 0),
                }
            )

            performance_stats["dataset_times"].append(dataset_total_time)
            statistics.append(dataset_stats)

            self.logger.info(
                f"数据集 {i+1} 完成: {len(image_files)} 文件, "
                f"{round(dataset_total_time, 1)}秒, "
                f"{round(files_per_second, 1)} 文件/秒, "
                f"{round(mb_per_second, 1)} MB/秒"
            )

        # 计算总体性能统计
        end_time = time.time()
        total_time = end_time - start_time
        performance_stats.update(
            {
                "end_time": end_time,
                "total_time_seconds": round(total_time, 2),
                "total_time_formatted": self._format_duration(total_time),
                "overall_files_per_second": (
                    round(total_images / total_time, 1) if total_time > 0 else 0
                ),
                "overall_mb_per_second": (
                    round(
                        (performance_stats["total_size_bytes"] / (1024 * 1024))
                        / total_time,
                        1,
                    )
                    if total_time > 0
                    else 0
                ),
                "avg_file_size": (
                    round(
                        performance_stats["total_size_bytes"] / total_images / 1024, 1
                    )
                    if total_images > 0
                    else 0
                ),  # KB
            }
        )

        self.logger.info(f"合并完成! 总计: {total_images} 图像, {total_labels} 标签")
        self.logger.info(f"总耗时: {performance_stats['total_time_formatted']}")
        self.logger.info(
            f"平均速度: {performance_stats['overall_files_per_second']} 文件/秒, "
            f"{performance_stats['overall_mb_per_second']} MB/秒"
        )

        return {
            "total_images": total_images,
            "total_labels": total_labels,
            "statistics": statistics,
            "performance": performance_stats,
        }

    def _build_label_mapping(self, label_files: List[Path]) -> Dict[str, Path]:
        """预建立标签文件映射索引

        Args:
            label_files: 标签文件列表

        Returns:
            Dict[str, Path]: 基础名称到标签文件路径的映射
        """
        label_mapping = {}
        for label_file in label_files:
            if label_file.name != self.classes_file:
                base_name = label_file.stem
                label_mapping[base_name] = label_file
        return label_mapping

    def _format_duration(self, seconds: float) -> str:
        """格式化时间显示

        Args:
            seconds: 秒数

        Returns:
            str: 格式化的时间字符串
        """
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}分{secs:.1f}秒"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}小时{minutes}分{secs:.1f}秒"

    def _merge_dataset_parallel(
        self,
        image_files: List[Path],
        images_dir: Path,
        labels_dir: Path,
        image_prefix: str,
        start_index: int,
        label_mapping: Dict[str, Path],
        dataset_num: int,
    ) -> Dict[str, int]:
        """并行合并数据集文件

        Args:
            image_files: 图像文件列表
            images_dir: 图像输出目录
            labels_dir: 标签输出目录
            image_prefix: 图片前缀
            start_index: 起始索引
            label_mapping: 标签文件映射
            dataset_num: 数据集编号

        Returns:
            Dict[str, int]: 处理结果统计
        """
        images_processed = 0
        labels_processed = 0
        failed_count = 0

        # 线程安全的计数器
        lock = threading.Lock()

        def copy_file_batch(batch_args):
            """批量复制文件对（减少系统调用开销）"""
            batch_tasks = batch_args
            batch_images = 0
            batch_labels = 0
            batch_failed = 0

            # 批量准备文件操作
            copy_operations = []

            for img_file, file_index in batch_tasks:
                try:
                    # 生成新的文件名
                    new_name = f"{image_prefix}_{file_index:05d}{img_file.suffix}"
                    new_img_path = images_dir / new_name

                    # 添加图像复制操作
                    copy_operations.append((img_file, new_img_path, "image"))

                    # 查找对应的标签文件
                    base_name = img_file.stem
                    label_file = label_mapping.get(base_name)

                    if label_file and label_file.exists():
                        new_label_name = f"{image_prefix}_{file_index:05d}.txt"
                        new_label_path = labels_dir / new_label_name
                        copy_operations.append((label_file, new_label_path, "label"))

                except Exception as e:
                    self.logger.error(f"准备文件操作失败 {img_file}: {str(e)}")
                    batch_failed += 1

            # 批量执行文件复制
            for src_file, dst_file, file_type in copy_operations:
                try:
                    # 使用更高效的复制方法
                    if not dst_file.parent.exists():
                        dst_file.parent.mkdir(parents=True, exist_ok=True)

                    # 对于小文件使用shutil.copy2，对于大文件使用shutil.copyfile
                    if src_file.stat().st_size < 1024 * 1024:  # 1MB以下
                        shutil.copy2(src_file, dst_file)
                    else:
                        shutil.copyfile(src_file, dst_file)

                    if file_type == "image":
                        batch_images += 1
                    else:
                        batch_labels += 1

                except Exception as e:
                    self.logger.error(
                        f"复制文件失败 {src_file} -> {dst_file}: {str(e)}"
                    )
                    batch_failed += 1

            # 线程安全更新计数器
            with lock:
                nonlocal images_processed, labels_processed, failed_count
                images_processed += batch_images
                labels_processed += batch_labels
                failed_count += batch_failed

            return len(batch_tasks)

        # 准备任务参数
        tasks = [(img_file, start_index + i) for i, img_file in enumerate(image_files)]

        # 批量处理参数（优化点3：减少线程创建开销）
        batch_size = max(1, len(tasks) // (os.cpu_count() or 4))  # 每个线程处理的文件数
        batch_size = min(batch_size, 100)  # 限制批次大小，避免内存过大

        # 将任务分批
        batches = [tasks[i : i + batch_size] for i in range(0, len(tasks), batch_size)]

        # 确定线程数（基于批次数量）
        max_workers = min(len(batches), (os.cpu_count() or 4) * 2, 16)

        self.logger.info(
            f"使用 {max_workers} 个线程处理 {len(batches)} 个批次，每批次 {batch_size} 个文件"
        )

        # 使用进度条显示处理进度
        with progress_context(
            len(image_files), f"并行合并数据集 {dataset_num}"
        ) as progress:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交批量任务
                future_to_batch = {
                    executor.submit(copy_file_batch, batch): batch for batch in batches
                }

                # 处理完成的批次
                for future in as_completed(future_to_batch):
                    try:
                        processed_count = future.result()
                        progress.update_progress(processed_count)
                    except Exception as e:
                        batch = future_to_batch[future]
                        self.logger.error(
                            f"批次执行异常 (批次大小: {len(batch)}): {str(e)}"
                        )
                        progress.update_progress(len(batch))

        if failed_count > 0:
            self.logger.warning(
                f"数据集 {dataset_num} 有 {failed_count} 个文件复制失败"
            )

        return {
            "images_processed": images_processed,
            "labels_processed": labels_processed,
            "failed_count": failed_count,
        }

    def _find_corresponding_label(
        self, image_file: Path, dataset_path: Path
    ) -> Optional[Path]:
        """查找图像文件对应的标签文件（保留兼容性）

        Args:
            image_file: 图像文件路径
            dataset_path: 数据集根目录

        Returns:
            Optional[Path]: 对应的标签文件路径
        """
        # 获取图像文件的基础名称（不含扩展名）
        base_name = image_file.stem

        # 在同一目录下查找
        label_file = image_file.parent / f"{base_name}.txt"
        if label_file.exists():
            return label_file

        # 在数据集根目录下递归查找
        for label_file in dataset_path.rglob(f"{base_name}.txt"):
            if label_file.name != self.classes_file:
                return label_file

        return None
