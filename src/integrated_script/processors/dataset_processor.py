#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_processor.py

数据集处理器

提供数据集验证、清理、转换等功能，特别针对机器学习数据集。
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict

from ..config.exceptions import DatasetError, ValidationError
from ..core.base import BaseProcessor
from ..core.progress import process_with_progress
from ..core.utils import (
    create_directory,
    get_file_list,
    validate_path,
)


class DatasetProcessor(BaseProcessor):
    """数据集处理器

    提供数据集验证、清理、转换等功能，特别针对机器学习数据集。

    Attributes:
        supported_formats (List[str]): 支持的数据集格式
        image_extensions (List[str]): 支持的图像扩展名
        annotation_extensions (List[str]): 支持的标注文件扩展名
    """

    def __init__(self, **kwargs):
        """初始化数据集处理器"""
        # 先设置属性，再调用父类初始化
        self.supported_formats = ["yolo", "coco", "pascal_voc", "csv"]
        self.image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        self.annotation_extensions = [".txt", ".xml", ".json", ".csv"]

        # 调用父类初始化
        super().__init__(**kwargs)

    def initialize(self) -> None:
        """初始化处理器"""
        self.logger.info("数据集处理器初始化完成")
        self.logger.debug(f"支持的格式: {self.supported_formats}")
        self.logger.debug(f"图像扩展名: {self.image_extensions}")
        self.logger.debug(f"标注扩展名: {self.annotation_extensions}")

    def process(self, *args, **kwargs) -> Any:
        """主要处理方法（由子方法实现具体功能）"""
        raise NotImplementedError("请使用具体的处理方法")

    def analyze_dataset(
        self, dataset_path: str, dataset_format: str = "yolo"
    ) -> Dict[str, Any]:
        """分析数据集

        Args:
            dataset_path: 数据集路径
            dataset_format: 数据集格式

        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            dataset_dir = validate_path(dataset_path, must_exist=True, must_be_dir=True)

            self.logger.info(f"开始分析数据集: {dataset_dir}")

            # 根据格式选择分析方法
            if dataset_format == "yolo":
                return self._analyze_yolo_dataset(dataset_dir)
            else:
                raise ValidationError(f"未实现的数据集格式分析: {dataset_format}")

        except Exception as e:
            raise DatasetError(f"数据集分析失败: {str(e)}")

    def _validate_yolo_dataset(
        self, dataset_dir: Path, check_integrity: bool
    ) -> Dict[str, Any]:
        """验证YOLO格式数据集

        只检查images和labels目录中的文件是否一一匹配，忽略其他文件。
        """
        result = {
            "success": True,
            "dataset_path": str(dataset_dir),
            "format": "yolo",
            "issues": [],
            "statistics": {
                "total_images": 0,
                "total_labels": 0,
                "orphaned_images": 0,
                "orphaned_labels": 0,
                "invalid_labels": 0,
                "empty_labels": 0,
            },
        }

        # 获取图像和标签文件
        # 只检查标准YOLO目录结构（images和labels目录）
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"

        if not images_dir.exists():
            self.logger.warning(f"images目录不存在: {images_dir}")
            return result

        if not labels_dir.exists():
            self.logger.warning(f"labels目录不存在: {labels_dir}")
            return result

        # 只从images和labels目录获取文件，不递归搜索其他目录
        image_files = get_file_list(images_dir, self.image_extensions, recursive=False)
        label_files = get_file_list(labels_dir, [".txt"], recursive=False)

        result["statistics"]["total_images"] = len(image_files)
        result["statistics"]["total_labels"] = len(label_files)

        # 创建文件映射
        image_stems = {f.stem: f for f in image_files}
        label_stems = {f.stem: f for f in label_files}

        # 调试日志：打印文件映射信息
        self.logger.info(f"图像文件stems: {list(image_stems.keys())}")
        self.logger.info(f"标签文件stems: {list(label_stems.keys())}")

        # 检查孤立文件
        orphaned_images = []
        orphaned_labels = []

        for stem, img_file in image_stems.items():
            if stem not in label_stems:
                orphaned_images.append(img_file)
                self.logger.info(f"发现孤立图像: {img_file} (stem: {stem})")

        for stem, label_file in label_stems.items():
            if stem not in image_stems:
                orphaned_labels.append(label_file)
                self.logger.info(f"发现孤立标签: {label_file} (stem: {stem})")

        result["statistics"]["orphaned_images"] = len(orphaned_images)
        result["statistics"]["orphaned_labels"] = len(orphaned_labels)

        # 计算成功配对的数量
        matched_pairs = len(set(image_stems.keys()) & set(label_stems.keys()))
        result["statistics"]["matched_pairs"] = matched_pairs

        # 调试日志：打印孤立文件统计
        self.logger.info(
            f"孤立图像数量: {len(orphaned_images)}, 孤立标签数量: {len(orphaned_labels)}, 成功配对数量: {matched_pairs}"
        )

        if orphaned_images:
            issue = {
                "type": "orphaned_images",
                "count": len(orphaned_images),
                "files": [str(f) for f in orphaned_images],  # 包含所有文件用于清理
                "preview": [
                    str(f) for f in orphaned_images[:10]
                ],  # 只显示前10个用于预览
            }
            result["issues"].append(issue)
            self.logger.info(
                f"添加孤立图像issue到结果: {issue['type']}, count={issue['count']}"
            )

        if orphaned_labels:
            issue = {
                "type": "orphaned_labels",
                "count": len(orphaned_labels),
                "files": [str(f) for f in orphaned_labels],  # 包含所有文件用于清理
                "preview": [
                    str(f) for f in orphaned_labels[:10]
                ],  # 只显示前10个用于预览
            }
            result["issues"].append(issue)
            self.logger.info(
                f"添加孤立标签issue到结果: {issue['type']}, count={issue['count']}"
            )

        # 检查标签文件完整性
        if check_integrity:
            invalid_labels = []
            empty_labels = []

            def validate_label_file(label_file: Path) -> Dict[str, Any]:
                try:
                    with open(label_file, "r", encoding="utf-8") as f:
                        content = f.read().strip()

                    if not content:
                        return {"type": "empty", "file": label_file}

                    lines = content.split("\n")
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) < 5:
                            return {
                                "type": "invalid",
                                "file": label_file,
                                "reason": f"第{line_num}行格式错误: {line}",
                            }

                        try:
                            class_id = int(parts[0])

                            # 检查类别ID
                            if class_id < 0:
                                return {
                                    "type": "invalid",
                                    "file": label_file,
                                    "reason": f"第{line_num}行类别ID不能为负数: {line}",
                                }

                            # 检查坐标值范围和有效性
                            for i, coord_str in enumerate(parts[1:5]):
                                try:
                                    coord = float(coord_str)
                                    # 检查是否为负数
                                    if coord < 0.0:
                                        return {
                                            "type": "invalid",
                                            "file": label_file,
                                            "reason": f"第{line_num}行第{i+2}列坐标不能为负数: {coord} (行内容: {line})",
                                        }
                                    # 检查是否超出范围[0,1]
                                    if coord > 1.0:
                                        return {
                                            "type": "invalid",
                                            "file": label_file,
                                            "reason": f"第{line_num}行第{i+2}列坐标超出范围[0,1]: {coord} (行内容: {line})",
                                        }
                                except ValueError:
                                    return {
                                        "type": "invalid",
                                        "file": label_file,
                                        "reason": f"第{line_num}行第{i+2}列坐标格式错误: {coord_str} (行内容: {line})",
                                    }

                        except ValueError:
                            return {
                                "type": "invalid",
                                "file": label_file,
                                "reason": f"第{line_num}行类别ID格式错误: {line}",
                            }

                    return {"type": "valid", "file": label_file}

                except Exception as e:
                    return {
                        "type": "invalid",
                        "file": label_file,
                        "reason": f"读取文件失败: {str(e)}",
                    }

            # 批量验证标签文件
            validation_results = process_with_progress(
                label_files, validate_label_file, "验证标签文件"
            )

            for val_result in validation_results:
                if val_result:
                    if val_result["type"] == "invalid":
                        invalid_labels.append(val_result)
                    elif val_result["type"] == "empty":
                        empty_labels.append(val_result)

            result["statistics"]["invalid_labels"] = len(invalid_labels)
            result["statistics"]["empty_labels"] = len(empty_labels)

            if invalid_labels:
                result["issues"].append(
                    {
                        "type": "invalid_labels",
                        "count": len(invalid_labels),
                        "examples": invalid_labels,  # 包含所有无效标签用于清理
                        "preview": invalid_labels[:5],  # 只显示前5个例子用于预览
                    }
                )

            if empty_labels:
                result["issues"].append(
                    {
                        "type": "empty_labels",
                        "count": len(empty_labels),
                        "files": [
                            str(item["file"]) for item in empty_labels
                        ],  # 包含所有文件用于清理
                        "preview": [
                            str(item["file"]) for item in empty_labels[:10]
                        ],  # 只显示前10个用于预览
                    }
                )

        # 判断整体验证结果
        self.logger.info(f"验证结束时issues数量: {len(result['issues'])}")
        for i, issue in enumerate(result["issues"]):
            self.logger.info(f"Issue {i}: type={issue['type']}, count={issue['count']}")

        if result["issues"]:
            result["success"] = False
            self.logger.info("由于存在issues，设置success=False")
        else:
            self.logger.info("没有issues，保持success=True")

        self.logger.info(
            f"YOLO数据集验证完成: {result['statistics']}, success={result['success']}"
        )
        return result

    def _detect_dataset_root(self, input_path: Path) -> Path:
        """智能检测数据集根目录

        如果用户输入的是images或labels子目录，自动向上查找数据集根目录

        Args:
            input_path: 用户输入的路径

        Returns:
            Path: 数据集根目录路径
        """
        current_path = input_path

        # 检查当前路径是否为images或labels子目录
        if current_path.name.lower() in ["images", "labels"]:
            parent_path = current_path.parent

            # 检查父目录是否包含images和labels目录（或至少一个）
            images_dir = parent_path / "images"
            labels_dir = parent_path / "labels"
            classes_file = parent_path / "classes.txt"

            # 如果父目录包含标准YOLO结构，使用父目录作为根目录
            if images_dir.exists() or labels_dir.exists() or classes_file.exists():
                self.logger.info(
                    f"检测到YOLO子目录结构，使用父目录作为数据集根目录: {parent_path}"
                )
                return parent_path

        # 如果当前目录不包含images/labels子目录，但包含图像和标签文件
        # 检查是否需要创建标准目录结构
        images_dir = current_path / "images"
        labels_dir = current_path / "labels"

        if not (images_dir.exists() and labels_dir.exists()):
            # 检查当前目录是否直接包含图像和标签文件
            image_files = get_file_list(
                current_path, self.image_extensions, recursive=False
            )
            txt_files = get_file_list(current_path, [".txt"], recursive=False)
            label_files = [f for f in txt_files if f.name != "classes.txt"]

            if image_files and label_files:
                self.logger.info(
                    f"检测到混合目录结构（图像和标签在同一目录），将使用当前目录: {current_path}"
                )

        return current_path

    def _validate_coco_dataset(
        self, dataset_dir: Path, check_integrity: bool
    ) -> Dict[str, Any]:
        """验证COCO格式数据集"""
        # TODO: 实现COCO格式验证
        return {"success": False, "message": "COCO格式验证尚未实现"}

    def _validate_pascal_voc_dataset(
        self, dataset_dir: Path, check_integrity: bool
    ) -> Dict[str, Any]:
        """验证Pascal VOC格式数据集"""
        # TODO: 实现Pascal VOC格式验证
        return {"success": False, "message": "Pascal VOC格式验证尚未实现"}

    def _validate_csv_dataset(
        self, dataset_dir: Path, check_integrity: bool
    ) -> Dict[str, Any]:
        """验证CSV格式数据集"""
        # TODO: 实现CSV格式验证
        return {"success": False, "message": "CSV格式验证尚未实现"}

    def _analyze_yolo_dataset(self, dataset_dir: Path) -> Dict[str, Any]:
        """分析YOLO格式数据集"""
        result = {
            "success": True,
            "dataset_path": str(dataset_dir),
            "format": "yolo",
            "statistics": {
                "total_images": 0,
                "total_labels": 0,
                "total_annotations": 0,
                "class_distribution": {},
                "image_sizes": [],
                "annotations_per_image": [],
            },
        }

        # 获取文件
        image_files = get_file_list(dataset_dir, self.image_extensions, recursive=True)
        label_files = get_file_list(dataset_dir, [".txt"], recursive=True)

        result["statistics"]["total_images"] = len(image_files)
        result["statistics"]["total_labels"] = len(label_files)

        # 分析标签文件
        class_counter = Counter()
        total_annotations = 0

        def analyze_label_file(label_file: Path) -> Dict[str, Any]:
            try:
                with open(label_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                annotations = []
                for line in lines:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            annotations.append(class_id)

                return {
                    "file": label_file,
                    "annotations": annotations,
                    "count": len(annotations),
                }

            except Exception as e:
                return {
                    "file": label_file,
                    "annotations": [],
                    "count": 0,
                    "error": str(e),
                }

        # 批量分析标签文件
        analysis_results = process_with_progress(
            label_files, analyze_label_file, "分析标签文件"
        )

        for analysis_result in analysis_results:
            if analysis_result and "annotations" in analysis_result:
                annotations = analysis_result["annotations"]
                total_annotations += len(annotations)
                result["statistics"]["annotations_per_image"].append(len(annotations))

                for class_id in annotations:
                    class_counter[class_id] += 1

        result["statistics"]["total_annotations"] = total_annotations
        result["statistics"]["class_distribution"] = dict(class_counter)

        # 计算统计信息
        if result["statistics"]["annotations_per_image"]:
            annotations_per_image = result["statistics"]["annotations_per_image"]
            result["statistics"]["avg_annotations_per_image"] = sum(
                annotations_per_image
            ) / len(annotations_per_image)
            result["statistics"]["max_annotations_per_image"] = max(
                annotations_per_image
            )
            result["statistics"]["min_annotations_per_image"] = min(
                annotations_per_image
            )

        # 类别统计
        if class_counter:
            result["statistics"]["num_classes"] = len(class_counter)
            result["statistics"]["most_common_class"] = class_counter.most_common(1)[0]
            result["statistics"]["least_common_class"] = class_counter.most_common()[-1]

        self.logger.info(f"YOLO数据集分析完成: {result['statistics']}")
        return result

    def split_dataset(
        self,
        dataset_path: str,
        output_path: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """分割数据集

        Args:
            dataset_path: 数据集路径
            output_path: 输出路径
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子

        Returns:
            Dict[str, Any]: 分割结果
        """
        import random

        try:
            dataset_dir = validate_path(dataset_path, must_exist=True, must_be_dir=True)
            output_dir = validate_path(output_path, must_exist=False)

            # 检查比例
            if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
                raise ValidationError("训练、验证、测试集比例之和必须等于1")

            create_directory(output_dir)

            self.logger.info(f"开始分割数据集: {dataset_dir} -> {output_dir}")
            self.logger.info(
                f"比例 - 训练: {train_ratio}, 验证: {val_ratio}, 测试: {test_ratio}"
            )

            # 获取图像文件
            image_files = get_file_list(
                dataset_dir, self.image_extensions, recursive=True
            )

            # 设置随机种子
            random.seed(random_seed)
            random.shuffle(image_files)

            # 计算分割点
            total_files = len(image_files)
            train_count = int(total_files * train_ratio)
            val_count = int(total_files * val_ratio)

            # 分割文件列表
            train_files = image_files[:train_count]
            val_files = image_files[train_count : train_count + val_count]
            test_files = image_files[train_count + val_count :]

            # 创建子目录
            splits = {"train": train_files, "val": val_files, "test": test_files}

            result = {
                "success": True,
                "dataset_path": str(dataset_dir),
                "output_path": str(output_dir),
                "splits": {},
                "statistics": {
                    "total_files": total_files,
                    "train_count": len(train_files),
                    "val_count": len(val_files),
                    "test_count": len(test_files),
                },
            }

            # 复制文件到对应目录
            for split_name, files in splits.items():
                if not files:
                    continue

                split_dir = output_dir / split_name
                images_dir = split_dir / "images"
                labels_dir = split_dir / "labels"

                create_directory(images_dir)
                create_directory(labels_dir)

                copied_images = []
                copied_labels = []

                for img_file in files:
                    # 复制图像文件
                    target_img = images_dir / img_file.name
                    try:
                        import shutil

                        shutil.copy2(img_file, target_img)
                        copied_images.append(str(target_img))

                        # 查找对应的标签文件
                        label_file = img_file.with_suffix(".txt")
                        if label_file.exists():
                            target_label = labels_dir / label_file.name
                            shutil.copy2(label_file, target_label)
                            copied_labels.append(str(target_label))

                    except Exception as e:
                        self.logger.warning(f"复制文件失败 {img_file}: {str(e)}")

                result["splits"][split_name] = {
                    "images_count": len(copied_images),
                    "labels_count": len(copied_labels),
                    "images_dir": str(images_dir),
                    "labels_dir": str(labels_dir),
                }

            self.logger.info(f"数据集分割完成: {result['statistics']}")
            return result

        except Exception as e:
            raise DatasetError(f"数据集分割失败: {str(e)}")
