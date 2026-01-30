#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interactive.py

交互式界面

提供交互式用户界面，支持菜单导航和用户输入。
"""

import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config.exceptions import ProcessingError, UserInterruptError, ValidationError
from ..config.settings import ConfigManager
from ..core.logging_config import get_logger, setup_logging
from ..processors import (
    DatasetProcessor,
    FileProcessor,
    ImageProcessor,
    LabelProcessor,
    YOLOProcessor,
)
from .menu import MenuSystem


class InteractiveInterface:
    """交互式界面

    提供交互式用户界面，支持菜单导航和用户输入。

    Attributes:
        config_manager (ConfigManager): 配置管理器
        logger: 日志记录器
        menu_system (MenuSystem): 菜单系统
        processors (Dict): 处理器映射
    """

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """初始化交互式界面

        Args:
            config_manager: 配置管理器，如果为None则创建新实例
        """
        self.config_manager = config_manager or ConfigManager()
        self.logger = get_logger(self.__class__.__name__)
        self.menu_system = MenuSystem()

        # 处理器实例
        self.processors = {
            "yolo": None,
            "image": None,
            "file": None,
            "dataset": None,
            "label": None,
        }

        # 设置菜单
        self._setup_menus()

    def _setup_menus(self) -> None:
        """设置菜单结构"""
        # 主菜单选项
        options = [
            ("YOLO数据集处理", self._yolo_menu),
            ("图像处理", self._image_menu),
            ("文件操作", self._file_menu),
            ("标签处理", self._label_menu),
        ]

        # 在非exe环境下才显示环境检查与配置
        if not self._is_running_as_exe():
            options.append(("环境检查与配置", self._environment_menu))

        options.append(("配置管理", self._config_menu))

        # 主菜单
        main_menu = {"title": "集成脚本工具 - 主菜单", "options": options}

        self.menu_system.set_main_menu(main_menu)

    def _get_processor(self, processor_type: str):
        """获取处理器实例"""
        if self.processors[processor_type] is None:
            try:
                if processor_type == "yolo":
                    self.processors[processor_type] = YOLOProcessor(
                        config=self.config_manager
                    )
                elif processor_type == "image":
                    self.processors[processor_type] = ImageProcessor(
                        config=self.config_manager
                    )
                elif processor_type == "file":
                    self.processors[processor_type] = FileProcessor(
                        config=self.config_manager
                    )
                elif processor_type == "dataset":
                    self.processors[processor_type] = DatasetProcessor(
                        config=self.config_manager
                    )
                elif processor_type == "label":
                    self.processors[processor_type] = LabelProcessor(
                        config=self.config_manager
                    )
            except Exception as e:
                from ..config.exceptions import ProcessingError

                raise ProcessingError(
                    f"处理器初始化失败: {str(e)}", context={"processor": processor_type}
                )

        return self.processors[processor_type]

    def _yolo_menu(self) -> None:
        """YOLO处理菜单"""
        menu = {
            "title": "YOLO数据集处理",
            "options": [
                ("CTDS数据转YOLO格式", self._yolo_process_ctds),
                ("X-label数据转YOLO格式", self._yolo_process_xlabel),
                ("X-label数据转YOLO-分割格式", self._yolo_process_xlabel_segmentation),
                ("YOLO数据转CTDS格式", self._yolo_convert_to_ctds),
                ("目标检测数据集验证", self._yolo_detection_statistics),
                ("目标分割数据集验证", self._yolo_segmentation_statistics),
                ("清理不匹配文件", self._yolo_clean_unmatched),
                ("合并多个数据集(相同类型)", self._yolo_merge_datasets),
                ("合并多个数据集(不同类型)", self._yolo_merge_different_datasets),
                ("返回主菜单", None),
            ],
        }
        self.menu_system.show_menu(menu)
        self._pause()

    def _yolo_process_ctds(self) -> None:
        """CTDS数据转YOLO格式"""
        try:
            print("\n=== CTDS数据转YOLO格式 ===")
            print("此功能将处理CTDS格式的标注数据，包括:")
            print("- 剔除空标签或非法标注数据")
            print("- 重命名图像和标签文件")
            print("- 复制obj.names到classes.txt")
            print("- 生成images/和labels/文件夹")
            print("- 根据处理的文件数量重命名项目文件夹")
            print("- 自动检测数据集类型（检测/分割）")
            print("- 自动调用相应的数据集验证功能")

            dataset_path = self._get_path_input(
                "请输入CTDS数据集路径: ", must_exist=True
            )

            # 获取项目名称
            project_name = input("\n请输入处理后的项目名称（留空自动生成）: ").strip()
            if not project_name:
                project_name = None

            keep_empty_labels = self._get_yes_no_input(
                "\n是否保留空标签文件? (y/N): ", default=False
            )

            processor = self._get_processor("yolo")

            print("\n正在处理CTDS数据集...")
            from pathlib import Path

            dataset_path_obj = Path(dataset_path)

            # 第一阶段：预检测和获取项目名称
            result = processor.process_ctds_dataset(
                str(dataset_path_obj),
                output_name=project_name,
                keep_empty_labels=keep_empty_labels,
            )

            # 检查是否是预检测阶段
            if result.get("stage") == "pre_detection":
                # 显示预检测结果
                pre_detection = result["pre_detection_result"]
                detected_type = pre_detection["dataset_type"]
                confidence = pre_detection["confidence"]

                print(f"\n🔍 数据集类型预检测结果:")
                print(f"  类型: {self._get_dataset_type_display_name(detected_type)}")
                print(f"  置信度: {confidence:.1%}")

                # 显示检测详情
                if pre_detection.get("statistics"):
                    stats = pre_detection["statistics"]
                    print(f"  分析文件数: {stats.get('files_analyzed', 0)}")
                    print(f"  检测格式文件数: {stats.get('detection_files', 0)}")
                    print(f"  分割格式文件数: {stats.get('segmentation_files', 0)}")

                # 获取用户确认的数据集类型
                confirmed_type = self._get_user_confirmed_type(
                    detected_type, confidence
                )
                if not confirmed_type:
                    print("处理已取消")
                    return

                print(
                    f"\n正在处理 {self._get_dataset_type_display_name(confirmed_type)} 数据..."
                )

                # 第二阶段：继续处理
                result = processor.continue_ctds_processing(
                    result, confirmed_type, keep_empty_labels=keep_empty_labels
                )

            # 显示处理结果
            self._display_ctds_result(result)

            # 如果处理成功且检测到数据集类型，询问是否进行验证
            # if result.get("success") and result.get("detected_dataset_type") != "unknown":
            #     self._handle_post_ctds_validation(result)

        except Exception as e:
            print(f"\nCTDS数据转YOLO格式失败: {e}")

        self._pause()

    def _yolo_process_xlabel(self) -> None:
        """X-label数据转YOLO格式"""
        try:
            print("\n=== X-label数据转YOLO格式 ===")
            print("此功能将Labelme/X-label JSON转换为YOLO目标检测格式：")
            print("- 自动扫描类别")
            print("- 支持用户调整类别顺序（class_id）")
            print("- 生成images/labels目录与classes.txt")

            dataset_path = self._get_path_input(
                "请输入X-label数据集路径: ", must_exist=True
            )
            output_path = self._get_input(
                "请输入输出目录（留空自动生成）: ", required=False
            ).strip()
            if not output_path:
                output_path = None

            processor = self._get_processor("yolo")

            classes = processor.detect_xlabel_classes(dataset_path)
            if not classes:
                print("\n❌ 未检测到任何类别")
                self._pause()
                return

            final_classes = self._get_class_order_from_user(list(classes))

            print("\n✅ 最终类别与ID映射：")
            for i, c in enumerate(final_classes):
                print(f"  {i}: {c}")

            print("\n正在转换X-label数据集...")
            result = processor.convert_xlabel_to_yolo(
                dataset_path, output_dir=output_path, class_order=final_classes
            )

            self._display_result(result)

        except Exception as e:
            print(f"\nX-label数据转YOLO失败: {e}")

        self._pause()

    def _yolo_process_xlabel_segmentation(self) -> None:
        """X-label数据转YOLO-分割格式"""
        try:
            print("\n=== X-label数据转YOLO-分割格式 ===")
            print("此功能将Labelme/X-label JSON转换为YOLO分割格式：")
            print("- 自动扫描类别")
            print("- 支持用户调整类别顺序（class_id）")
            print("- 生成images/labels目录与classes.txt")

            dataset_path = self._get_path_input(
                "请输入X-label数据集路径: ", must_exist=True
            )
            output_path = self._get_input(
                "请输入输出目录（留空自动生成）: ", required=False
            ).strip()
            if not output_path:
                output_path = None

            processor = self._get_processor("yolo")

            classes = processor.detect_xlabel_segmentation_classes(dataset_path)
            if not classes:
                print("\n❌ 未检测到任何类别")
                self._pause()
                return

            english_mapping = self._get_english_name_mapping(list(classes))
            final_classes = self._get_class_order_from_user(list(classes))

            print("\n✅ 最终类别与ID映射：")
            for i, c in enumerate(final_classes):
                print(f"  {i}: {c}")

            print("\n正在转换X-label分割数据集...")
            result = processor.convert_xlabel_to_yolo_segmentation(
                dataset_path,
                output_dir=output_path,
                class_order=final_classes,
                english_name_mapping=english_mapping,
            )

            self._display_result(result)

        except Exception as e:
            print(f"\nX-label转YOLO-分割失败: {e}")

        self._pause()

    def _display_ctds_result(self, result: Dict[str, Any]) -> None:
        """显示CTDS处理结果"""
        print("\n" + "=" * 50)
        print("CTDS数据处理结果")
        print("=" * 50)

        if result.get("success"):
            print("✅ 处理成功!")
            print(f"📁 输出路径: {result.get('output_path')}")
            print(f"📝 项目名称: {result.get('project_name')}")

            # 显示处理统计
            stats = result.get("statistics", {})
            print(f"\n📊 处理统计:")
            print(f"  - 总处理文件数: {stats.get('total_processed', 0)}")
            print(f"  - 有效文件数: {stats.get('final_count', 0)}")
            print(f"  - 无效文件数: {stats.get('invalid_removed', 0)}")
            if "missing_images" in stats or "missing_labels" in stats:
                print(f"  - 标签缺图数: {stats.get('missing_images', 0)}")
                print(f"  - 图片缺标数: {stats.get('missing_labels', 0)}")
        else:
            print("❌ 处理失败")
            if "error" in result:
                print(f"错误信息: {result['error']}")

    def _handle_post_ctds_validation(self, result: Dict[str, Any]) -> None:
        """处理CTDS处理后的验证流程"""
        detected_type = result.get("detected_dataset_type")
        confidence = result.get("detection_confidence", 0.0)
        output_path = result.get("output_path")

        # 显示数据集类型检测结果
        print(f"\n🔍 数据集类型检测:")
        if detected_type == "detection":
            print(f"  📋 检测到: 目标检测数据集 (置信度: {confidence:.1%})")
            print(
                f"  💡 说明: 标签文件使用5列格式 (class_id x_center y_center width height)"
            )
        elif detected_type == "segmentation":
            print(f"  🎯 检测到: 目标分割数据集 (置信度: {confidence:.1%})")
            print(f"  💡 说明: 标签文件使用多列格式 (class_id x1 y1 x2 y2 ...)")
        elif detected_type == "mixed":
            print(f"  ⚠️ 检测到: 混合格式数据集 (置信度: {confidence:.1%})")
            print(f"  💡 说明: 数据集包含检测和分割两种格式")
        else:
            print(f"  ❓ 未能确定数据集类型")

        # 显示详细检测信息
        # 优先使用预检测结果，如果没有则使用处理过程中的检测结果
        detection_info = result.get("pre_detection_result") or result.get(
            "dataset_type_detection", {}
        )
        if detection_info.get("success") and detection_info.get("statistics"):
            det_stats = detection_info["statistics"]
            print(f"\n📈 检测详情:")
            print(
                f"  - 分析文件数: {det_stats.get('files_analyzed', det_stats.get('total_files_analyzed', 0))}"
            )
            print(f"  - 总标注行数: {det_stats.get('total_lines', 0)}")
            print(
                f"  - 检测格式文件数: {det_stats.get('detection_files', det_stats.get('detection_lines', 0))}"
            )
            print(
                f"  - 分割格式文件数: {det_stats.get('segmentation_files', det_stats.get('segmentation_lines', 0))}"
            )

        print(f"\n🎯 数据集类型确认")
        print(
            f"检测结果: {self._get_dataset_type_display_name(detected_type)} (置信度: {confidence:.1%})"
        )

        # 让用户确认数据集类型
        if detected_type == "mixed" or confidence < 0.8:
            print(f"\n⚠️ 检测置信度较低或为混合格式，请手动确认数据集类型:")
            print("1. 目标检测数据集")
            print("2. 目标分割数据集")
            print("3. 跳过验证")

            choice = input("\n请选择 (1-3): ").strip()
            if choice == "1":
                confirmed_type = "detection"
            elif choice == "2":
                confirmed_type = "segmentation"
            else:
                print("跳过数据集验证")
                return
        else:
            # 高置信度，询问是否确认
            confirm = (
                input(
                    f"\n确认数据集类型为 {self._get_dataset_type_display_name(detected_type)} 吗？(Y/n): "
                )
                .strip()
                .lower()
            )
            if confirm in ["", "y", "yes"]:
                confirmed_type = detected_type
            else:
                print("\n请手动选择数据集类型:")
                print("1. 目标检测数据集")
                print("2. 目标分割数据集")
                print("3. 跳过验证")

                choice = input("\n请选择 (1-3): ").strip()
                if choice == "1":
                    confirmed_type = "detection"
                elif choice == "2":
                    confirmed_type = "segmentation"
                else:
                    print("跳过数据集验证")
                    return

        # 执行相应的验证
        print(
            f"\n🔍 开始验证 {self._get_dataset_type_display_name(confirmed_type)} 数据集..."
        )

        try:
            processor = self._get_processor("yolo")

            if confirmed_type == "detection":
                # 调用检测数据集验证
                validation_result = processor.get_dataset_statistics(output_path)
                self._display_validation_result(validation_result, "检测")

            elif confirmed_type == "segmentation":
                # 调用分割数据集验证
                validation_result = processor.get_dataset_statistics(output_path)
                self._display_validation_result(validation_result, "分割")

                # 进行分割格式验证
                print("\n正在检查分割标注格式...")
                invalid_files = self._validate_segmentation_format(output_path)

                if invalid_files:
                    print(f"\n⚠️ 发现 {len(invalid_files)} 个不符合分割格式的文件")
                    for file_path, reason in invalid_files[:5]:  # 只显示前5个
                        print(f"  - {file_path.name}: {reason}")
                    if len(invalid_files) > 5:
                        print(f"  ... 还有 {len(invalid_files) - 5} 个文件")
                else:
                    print("✅ 所有标签文件都符合分割格式要求")

            # 显示最终统计
            self._display_final_ctds_summary(result, validation_result, confirmed_type)

        except Exception as e:
            print(f"❌ 验证过程出错: {e}")

    def _get_dataset_type_display_name(self, dataset_type: str) -> str:
        """获取数据集类型的显示名称"""
        type_names = {
            "detection": "目标检测数据集",
            "segmentation": "目标分割数据集",
            "mixed": "混合格式数据集",
            "unknown": "未知类型数据集",
        }
        return type_names.get(dataset_type, "未知类型")

    def _get_user_confirmed_type(self, detected_type: str, confidence: float) -> str:
        """获取用户确认的数据集类型

        Args:
            detected_type: 检测到的数据集类型
            confidence: 检测置信度

        Returns:
            str: 用户确认的数据集类型，如果取消则返回None
        """
        # 如果是混合格式或置信度较低，让用户手动选择
        if detected_type == "mixed" or confidence < 0.8:
            print(f"\n⚠️ 检测置信度较低或为混合格式，请手动确认数据集类型:")
            print("1. 目标检测数据集")
            print("2. 目标分割数据集")
            print("3. 取消处理")

            choice = input("\n请选择 (1-3): ").strip()
            if choice == "1":
                return "detection"
            elif choice == "2":
                return "segmentation"
            else:
                return None
        else:
            # 高置信度，询问是否确认
            confirm = self._get_yes_no_input(
                f"\n确认数据集类型为 {self._get_dataset_type_display_name(detected_type)} 吗？",
                default=True,
            )
            if confirm:
                return detected_type

            print("\n请手动选择数据集类型:")
            print("1. 目标检测数据集")
            print("2. 目标分割数据集")
            print("3. 取消处理")

            choice = input("\n请选择 (1-3): ").strip()
            if choice == "1":
                return "detection"
            elif choice == "2":
                return "segmentation"
            else:
                return None

    def _get_class_order_from_user(self, classes: List[str]) -> List[str]:
        """获取用户确认的类别顺序（class_id）"""
        default = sorted(classes)

        print("\n📌 检测到以下类别（当前顺序 = class_id）：")
        for i, c in enumerate(default):
            print(f"  {i}: {c}")

        print("\n如需修改顺序，请输入新的编号顺序，例如：")
        print("  2 1 0")
        print("直接回车表示使用当前顺序")

        user_input = self._get_input("新的顺序 -> ", required=False).strip()
        if not user_input:
            return default

        try:
            idxs = list(map(int, user_input.split()))
            if len(idxs) != len(default):
                raise ValueError("数量不一致")
            if set(idxs) != set(range(len(default))):
                raise ValueError("编号不合法")
            return [default[i] for i in idxs]
        except Exception as e:
            print(f"❌ 输入非法（{e}），使用默认顺序")
            return default

    def _get_english_name_mapping(self, classes: List[str]) -> Dict[str, str]:
        """获取类别英文名称映射"""
        mapping: Dict[str, str] = {}
        print("\n检测到以下类别，请为每个类别输入对应的英文名称：")
        for class_name in sorted(classes):
            english_name = self._get_input(
                f"请输入 '{class_name}' 的英文名称: ", required=False
            ).strip()
            mapping[class_name] = english_name or class_name
        return mapping

    def _display_validation_result(
        self, result: Dict[str, Any], dataset_type_name: str
    ) -> None:
        """显示验证结果"""
        print(f"\n📋 {dataset_type_name}数据集验证结果:")

        if result.get("valid", False):
            print("✅ 数据集验证通过")
        else:
            print("⚠️ 数据集存在问题")

        stats = result.get("statistics", {})
        print(f"  - 图像文件数: {stats.get('total_images', 0)}")
        print(f"  - 标签文件数: {stats.get('total_labels', 0)}")
        print(f"  - 匹配文件对: {stats.get('matched_pairs', 0)}")

        if stats.get("orphaned_images", 0) > 0:
            print(f"  - 孤立图像: {stats.get('orphaned_images', 0)}")
        if stats.get("orphaned_labels", 0) > 0:
            print(f"  - 孤立标签: {stats.get('orphaned_labels', 0)}")
        if stats.get("invalid_labels", 0) > 0:
            print(f"  - 无效标签: {stats.get('invalid_labels', 0)}")

    def _display_final_ctds_summary(
        self,
        ctds_result: Dict[str, Any],
        validation_result: Dict[str, Any],
        dataset_type: str,
    ) -> None:
        """显示CTDS处理和验证的最终汇总"""
        print("\n" + "=" * 60)
        print("🎉 CTDS数据处理和验证完成汇总")
        print("=" * 60)

        print(f"📁 输出路径: {ctds_result.get('output_path')}")
        print(f"📝 项目名称: {ctds_result.get('project_name')}")
        print(f"🎯 数据集类型: {self._get_dataset_type_display_name(dataset_type)}")

        # CTDS处理统计
        ctds_stats = ctds_result.get("statistics", {})
        print(f"\n📊 处理统计:")
        print(f"  ✅ 成功处理: {ctds_stats.get('final_count', 0)} 个文件对")
        print(f"  ❌ 剔除无效: {ctds_stats.get('invalid_removed', 0)} 个文件")

        # 验证统计
        val_stats = validation_result.get("statistics", {})
        print(f"\n🔍 验证统计:")
        print(f"  📷 图像文件: {val_stats.get('total_images', 0)} 个")
        print(f"  📝 标签文件: {val_stats.get('total_labels', 0)} 个")
        print(f"  🔗 匹配文件对: {val_stats.get('matched_pairs', 0)} 个")

        # 数据集状态
        is_valid = validation_result.get("valid", False)
        print(f"\n🏆 数据集状态: {'✅ 可用于训练' if is_valid else '⚠️ 需要进一步处理'}")

        if not is_valid:
            issues = []
            if val_stats.get("orphaned_images", 0) > 0:
                issues.append(f"孤立图像 {val_stats['orphaned_images']} 个")
            if val_stats.get("orphaned_labels", 0) > 0:
                issues.append(f"孤立标签 {val_stats['orphaned_labels']} 个")
            if val_stats.get("invalid_labels", 0) > 0:
                issues.append(f"无效标签 {val_stats['invalid_labels']} 个")

            if issues:
                print(f"  ⚠️ 发现问题: {', '.join(issues)}")
                print(f"  💡 建议: 使用'清理不匹配文件'功能进行清理")

    def _yolo_detection_statistics(self) -> None:
        """验证YOLO目标检测数据集"""
        try:
            print("\n=== YOLO目标检测数据集验证 ===")
            print("此功能验证数据集中images和labels目录的文件是否一一匹配")
            print("- 检查images目录中的图片是否都有对应的标签文件")
            print("- 检查labels目录中的标签是否都有对应的图片文件")
            print("- 忽略其他目录和文件")

            dataset_path = self._get_path_input("请输入数据集路径: ", must_exist=True)

            processor = self._get_processor("yolo")

            # 路径验证和提示
            path_obj = Path(dataset_path)
            if path_obj.name.lower() in ["images", "labels"]:
                print(f"\n💡 检测到您输入的是 '{path_obj.name}' 子目录")
                print("   系统将自动查找数据集根目录...")

            print("\n正在验证数据集...")
            result = processor.get_dataset_statistics(dataset_path)

            self._display_result(result)

            # 检查数据集是否有效，如果无效则询问是否自动清理
            if "statistics" in result and not result["statistics"].get(
                "is_valid", True
            ):
                stats = result["statistics"]
                has_issues = (
                    stats.get("orphaned_images", 0) > 0
                    or stats.get("orphaned_labels", 0) > 0
                )

                if has_issues:
                    print("\n⚠ 验证发现数据集存在不匹配文件问题")
                    auto_clean = input("是否立即进行自动清理？(Y/n): ").strip().lower()

                    if auto_clean in ["", "y", "yes", "是"]:
                        print("\n开始自动清理...")

                        # 先进行试运行
                        print("\n正在分析需要清理的文件...")
                        clean_result = processor.clean_unmatched_files(
                            dataset_path, dry_run=True
                        )

                        total_files = sum(
                            len(files)
                            for files in clean_result["deleted_files"].values()
                        )

                        if total_files == 0:
                            print("✓ 数据集已经完全匹配，无需清理")
                        else:
                            print(f"\n将删除 {total_files} 个不匹配文件:")

                            if clean_result["deleted_files"]["orphaned_images"]:
                                print(
                                    f"  - 孤立图片: {len(clean_result['deleted_files']['orphaned_images'])} 个"
                                )

                            if clean_result["deleted_files"]["orphaned_labels"]:
                                print(
                                    f"  - 孤立标签: {len(clean_result['deleted_files']['orphaned_labels'])} 个"
                                )

                            if clean_result["deleted_files"]["invalid_labels"]:
                                print(
                                    f"  - 无效标签: {len(clean_result['deleted_files']['invalid_labels'])} 个"
                                )

                            if clean_result["deleted_files"].get("empty_labels"):
                                print(
                                    f"  - 空标签: {len(clean_result['deleted_files']['empty_labels'])} 个"
                                )

                            # 显示具体文件名称（最多10个）
                            self._display_files_to_delete(clean_result["deleted_files"])

                            # 确认删除
                            confirm = (
                                input("\n确认删除这些文件？(Y/n): ").strip().lower()
                            )
                            if confirm in ["", "y", "yes", "是"]:
                                print("\n正在删除文件...")
                                final_result = processor.clean_unmatched_files(
                                    dataset_path, dry_run=False
                                )

                                print("\n=== 清理完成 ===")
                                self._display_clean_result(final_result)

                                # 重新验证数据集
                                print("\n重新验证数据集...")
                                updated_result = processor.get_dataset_statistics(
                                    dataset_path
                                )
                                print("\n=== 清理后的验证结果 ===")
                                self._display_result(updated_result)
                            else:
                                print("\n清理操作已取消")
                    else:
                        print("\n跳过自动清理")

        except Exception as e:
            print(f"\n目标检测数据集验证失败: {e}")

        self._pause()

    def _yolo_segmentation_statistics(self) -> None:
        """验证YOLO目标分割数据集"""
        try:
            print("\n=== YOLO目标分割数据集验证 ===")
            print("此功能验证数据集中images和labels目录的文件是否一一匹配")
            print("- 检查images目录中的图片是否都有对应的标签文件")
            print("- 检查labels目录中的标签是否都有对应的图片文件")
            print("- 验证标签文件是否符合分割格式（至少7列）")
            print("- 忽略其他目录和文件")

            dataset_path = self._get_path_input("请输入数据集路径: ", must_exist=True)

            processor = self._get_processor("yolo")

            # 路径验证和提示
            path_obj = Path(dataset_path)
            if path_obj.name.lower() in ["images", "labels"]:
                print(f"\n💡 检测到您输入的是 '{path_obj.name}' 子目录")
                print("   系统将自动查找数据集根目录...")

            print("\n正在验证分割数据集...")

            # 首先进行常规数据集验证
            result = processor.get_dataset_statistics(dataset_path)
            self._display_result(result)

            # 进行分割数据集特定验证
            print("\n正在检查分割标注格式...")
            invalid_files = self._validate_segmentation_format(dataset_path)

            if invalid_files:
                print(f"\n⚠ 发现 {len(invalid_files)} 个不符合分割格式的标签文件")
                print("分割标签要求每行至少有7列（1个类别 + 至少6个坐标值）")

                # 显示部分无效文件
                print("\n无效文件示例:")
                for i, (file_path, reason) in enumerate(invalid_files[:5]):
                    print(f"  {i+1}. {file_path.name}: {reason}")
                if len(invalid_files) > 5:
                    print(f"  ... 还有 {len(invalid_files) - 5} 个文件")

                # 询问是否移动无效文件
                move_choice = (
                    input("\n是否将无效文件移动到上级目录？(Y/n): ").strip().lower()
                )
                if move_choice in ["", "y", "yes", "是"]:
                    self._move_invalid_segmentation_files(dataset_path, invalid_files)
                else:
                    print("\n跳过文件移动")
            else:
                print("\n✓ 所有标签文件都符合分割格式要求")

            # 检查数据集是否有效，如果无效则询问是否自动清理
            if "statistics" in result and not result["statistics"].get(
                "is_valid", True
            ):
                stats = result["statistics"]
                has_issues = (
                    stats.get("orphaned_images", 0) > 0
                    or stats.get("orphaned_labels", 0) > 0
                )

                if has_issues:
                    print("\n⚠ 验证发现数据集存在不匹配文件问题")
                    auto_clean = input("是否立即进行自动清理？(Y/n): ").strip().lower()

                    if auto_clean in ["", "y", "yes", "是"]:
                        print("\n开始自动清理...")

                        # 先进行试运行
                        print("\n正在分析需要清理的文件...")
                        clean_result = processor.clean_unmatched_files(
                            dataset_path, dry_run=True
                        )

                        total_files = sum(
                            len(files)
                            for files in clean_result["deleted_files"].values()
                        )

                        if total_files == 0:
                            print("✓ 数据集已经完全匹配，无需清理")
                        else:
                            print(f"\n将删除 {total_files} 个不匹配文件:")

                            if clean_result["deleted_files"]["orphaned_images"]:
                                print(
                                    f"  - 孤立图片: {len(clean_result['deleted_files']['orphaned_images'])} 个"
                                )

                            if clean_result["deleted_files"]["orphaned_labels"]:
                                print(
                                    f"  - 孤立标签: {len(clean_result['deleted_files']['orphaned_labels'])} 个"
                                )

                            if clean_result["deleted_files"]["invalid_labels"]:
                                print(
                                    f"  - 无效标签: {len(clean_result['deleted_files']['invalid_labels'])} 个"
                                )

                            if clean_result["deleted_files"].get("empty_labels"):
                                print(
                                    f"  - 空标签: {len(clean_result['deleted_files']['empty_labels'])} 个"
                                )

                            # 显示具体文件名称（最多10个）
                            self._display_files_to_delete(clean_result["deleted_files"])

                            # 确认删除
                            confirm = (
                                input("\n确认删除这些文件？(Y/n): ").strip().lower()
                            )
                            if confirm in ["", "y", "yes", "是"]:
                                print("\n正在删除文件...")
                                final_result = processor.clean_unmatched_files(
                                    dataset_path, dry_run=False
                                )

                                print("\n=== 清理完成 ===")
                                self._display_clean_result(final_result)

                                # 重新验证数据集
                                print("\n重新验证数据集...")
                                updated_result = processor.get_dataset_statistics(
                                    dataset_path
                                )
                                print("\n=== 清理后的验证结果 ===")
                                self._display_result(updated_result)
                            else:
                                print("\n清理操作已取消")
                    else:
                        print("\n跳过自动清理")

        # except KeyboardInterrupt:
        #     print("\n目标分割数据集验证失败: 用户中断操作 (Code: USER_INTERRUPT)")
        except Exception as e:
            print(f"\n目标分割数据集验证失败: {e}")

        self._pause()

    def _validate_segmentation_format(self, dataset_path):
        """验证分割数据集格式

        只检查labels目录中的标签文件格式，确保符合分割数据集要求。
        """
        from pathlib import Path

        dataset_path = Path(dataset_path)

        # 智能检测数据集根目录
        if dataset_path.name.lower() in ["images", "labels"]:
            dataset_path = dataset_path.parent

        labels_dir = dataset_path / "labels"
        if not labels_dir.exists():
            print(f"\n⚠ labels目录不存在: {labels_dir}")
            return []

        invalid_files = []

        for label_file in labels_dir.glob("*.txt"):
            try:
                with open(label_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue

                    parts = line.split()
                    if len(parts) < 7:  # 分割标注至少需要7列（1个类别 + 至少6个坐标值）
                        invalid_files.append(
                            (
                                label_file,
                                f"第{line_num}行只有{len(parts)}列，需要至少7列",
                            )
                        )
                        break  # 一个文件有问题就标记为无效

                    # 检查类别是否为有效整数
                    try:
                        int(parts[0])
                    except ValueError:
                        invalid_files.append(
                            (label_file, f"第{line_num}行类别'{parts[0]}'不是有效整数")
                        )
                        break

                    # 检查坐标是否为有效浮点数
                    try:
                        for coord in parts[1:]:
                            float(coord)
                    except ValueError:
                        invalid_files.append(
                            (label_file, f"第{line_num}行包含无效坐标值")
                        )
                        break

            except Exception as e:
                invalid_files.append((label_file, f"读取文件失败: {e}"))

        return invalid_files

    def _move_invalid_segmentation_files(self, dataset_path, invalid_files):
        """移动无效的分割文件到上级目录"""
        import shutil
        from pathlib import Path

        dataset_path = Path(dataset_path)

        # 智能检测数据集根目录
        if dataset_path.name.lower() in ["images", "labels"]:
            dataset_path = dataset_path.parent

        # 创建无效文件目录
        invalid_dir = dataset_path.parent / "invalid_segmentation_files"
        invalid_images_dir = invalid_dir / "images"
        invalid_labels_dir = invalid_dir / "labels"

        invalid_dir.mkdir(exist_ok=True)
        invalid_images_dir.mkdir(exist_ok=True)
        invalid_labels_dir.mkdir(exist_ok=True)

        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"

        moved_count = 0

        print(f"\n正在移动无效文件到: {invalid_dir}")

        for label_file, reason in invalid_files:
            try:
                # 移动标签文件
                target_label = invalid_labels_dir / label_file.name
                shutil.move(str(label_file), str(target_label))

                # 查找对应的图片文件并移动
                label_stem = label_file.stem
                image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

                for ext in image_extensions:
                    image_file = images_dir / f"{label_stem}{ext}"
                    if image_file.exists():
                        target_image = invalid_images_dir / image_file.name
                        shutil.move(str(image_file), str(target_image))
                        break

                moved_count += 1
                print(f"  移动: {label_file.name} ({reason})")

            except Exception as e:
                print(f"  ❌ 移动失败 {label_file.name}: {e}")

        # 复制classes.txt文件
        classes_file = dataset_path / "classes.txt"
        if classes_file.exists():
            try:
                target_classes = invalid_dir / "classes.txt"
                shutil.copy2(str(classes_file), str(target_classes))
                print(f"  ✓ 已复制 classes.txt")
            except Exception as e:
                print(f"  ❌ 复制classes.txt失败: {e}")

        print(f"\n✓ 已移动 {moved_count} 个无效文件对到: {invalid_dir}")
        print(f"  - 无效标签: {moved_count} 个")
        print(f"  - 对应图片: {moved_count} 个")
        if classes_file.exists():
            print(f"  - 类别文件: 1 个")

    def _yolo_clean_unmatched(self) -> None:
        """清理YOLO数据集中不匹配的文件"""
        try:
            print("\n=== 清理不匹配文件 ===")
            print("此功能将删除images和labels目录中没有配对的文件")
            print("- 删除images目录中没有对应标签的图片")
            print("- 删除labels目录中没有对应图片的标签")
            print("- 删除格式无效的标签文件")
            print("- 只处理images和labels目录，忽略其他文件")

            dataset_path = self._get_path_input("请输入数据集路径: ", must_exist=True)

            # 询问是否先进行试运行
            dry_run_choice = (
                input("\n是否先进行试运行（查看将要删除的文件但不实际删除）？(y/N): ")
                .strip()
                .lower()
            )
            dry_run = dry_run_choice in ["y", "yes", "是"]

            processor = self._get_processor("yolo")

            if dry_run:
                print("\n正在进行试运行...")
                result = processor.clean_unmatched_files(dataset_path, dry_run=True)

                print("\n=== 试运行结果 ===")
                total_files = sum(
                    len(files) for files in result["deleted_files"].values()
                )

                if total_files == 0:
                    print("✓ 数据集已经完全匹配，无需清理")
                else:
                    print(f"将删除 {total_files} 个文件:")

                    if result["deleted_files"]["orphaned_images"]:
                        print(
                            f"  - 孤立图片: {len(result['deleted_files']['orphaned_images'])} 个"
                        )
                        for img in result["deleted_files"]["orphaned_images"][
                            :5
                        ]:  # 只显示前5个
                            print(f"    {img}")
                        if len(result["deleted_files"]["orphaned_images"]) > 5:
                            print(
                                f"    ... 还有 {len(result['deleted_files']['orphaned_images']) - 5} 个"
                            )

                    if result["deleted_files"]["orphaned_labels"]:
                        print(
                            f"  - 孤立标签: {len(result['deleted_files']['orphaned_labels'])} 个"
                        )
                        for lbl in result["deleted_files"]["orphaned_labels"][
                            :5
                        ]:  # 只显示前5个
                            print(f"    {lbl}")
                        if len(result["deleted_files"]["orphaned_labels"]) > 5:
                            print(
                                f"    ... 还有 {len(result['deleted_files']['orphaned_labels']) - 5} 个"
                            )

                    if result["deleted_files"]["invalid_labels"]:
                        print(
                            f"  - 无效标签: {len(result['deleted_files']['invalid_labels'])} 个"
                        )
                        for lbl in result["deleted_files"]["invalid_labels"][
                            :5
                        ]:  # 只显示前5个
                            print(f"    {lbl}")
                        if len(result["deleted_files"]["invalid_labels"]) > 5:
                            print(
                                f"    ... 还有 {len(result['deleted_files']['invalid_labels']) - 5} 个"
                            )

                    if result["deleted_files"].get("empty_labels"):
                        print(
                            f"  - 空标签: {len(result['deleted_files']['empty_labels'])} 个"
                        )
                        for lbl in result["deleted_files"]["empty_labels"][
                            :5
                        ]:  # 只显示前5个
                            print(f"    {lbl}")
                        if len(result["deleted_files"]["empty_labels"]) > 5:
                            print(
                                f"    ... 还有 {len(result['deleted_files']['empty_labels']) - 5} 个"
                            )

                    # 询问是否继续实际删除
                    confirm = input("\n确认要删除这些文件吗？(y/N): ").strip().lower()
                    if confirm in ["y", "yes", "是"]:
                        print("\n正在删除文件...")
                        result = processor.clean_unmatched_files(
                            dataset_path, dry_run=False
                        )
                        self._display_clean_result(result)
                    else:
                        print("\n操作已取消")
            else:
                # 直接删除，但先确认
                confirm = (
                    input("\n确认要直接删除不匹配的文件吗？(y/N): ").strip().lower()
                )
                if confirm in ["y", "yes", "是"]:
                    print("\n正在清理文件...")
                    result = processor.clean_unmatched_files(
                        dataset_path, dry_run=False
                    )
                    self._display_clean_result(result)
                else:
                    print("\n操作已取消")

        except Exception as e:
            print(f"\n清理失败: {e}")

        self._pause()

    def _display_clean_result(self, result: dict) -> None:
        """显示清理结果"""
        print("\n=== 清理完成 ===")

        if result["statistics"]["total_deleted"] == 0:
            print("✓ 数据集已经完全匹配，无文件被删除")
        else:
            print(f"✓ 成功删除 {result['statistics']['total_deleted']} 个文件")
            print(f"  - 删除图片: {result['statistics']['deleted_images']} 个")
            print(f"  - 删除标签: {result['statistics']['deleted_labels']} 个")

        if not result["success"]:
            print("⚠ 部分文件删除失败，请检查日志")

    def _display_files_to_delete(self, deleted_files: dict) -> None:
        """显示待删除的文件列表，最多显示10个文件"""
        print("\n待删除的文件:")

        # 收集所有文件
        image_files = []
        label_files = []

        # 收集孤立图片
        if deleted_files.get("orphaned_images"):
            image_files.extend(deleted_files["orphaned_images"])

        # 收集孤立标签、无效标签、空标签
        for key in ["orphaned_labels", "invalid_labels", "empty_labels"]:
            if deleted_files.get(key):
                label_files.extend(deleted_files[key])

        # 显示逻辑：图片和标签各最多5个，如果某一种不够则用另一种补齐
        max_display = 10
        max_per_type = 5

        # 取前5个图片和前5个标签
        display_images = image_files[:max_per_type]
        display_labels = label_files[:max_per_type]

        # 如果图片不够5个，用标签补齐
        if len(display_images) < max_per_type and len(label_files) > max_per_type:
            remaining_slots = max_per_type - len(display_images)
            additional_labels = label_files[
                max_per_type : max_per_type + remaining_slots
            ]
            display_labels.extend(additional_labels)

        # 如果标签不够5个，用图片补齐
        if len(display_labels) < max_per_type and len(image_files) > max_per_type:
            remaining_slots = max_per_type - len(display_labels)
            additional_images = image_files[
                max_per_type : max_per_type + remaining_slots
            ]
            display_images.extend(additional_images)

        # 显示图片文件
        if display_images:
            print(f"  图片文件 ({len(display_images)} 个):")
            for img in display_images:
                print(f"    {img}")

        # 显示标签文件
        if display_labels:
            print(f"  标签文件 ({len(display_labels)} 个):")
            for lbl in display_labels:
                print(f"    {lbl}")

        # 显示总数统计
        total_files = len(image_files) + len(label_files)
        displayed_files = len(display_images) + len(display_labels)
        if total_files > displayed_files:
            print(f"  ... 还有 {total_files - displayed_files} 个文件未显示")

    def _yolo_merge_datasets(self) -> None:
        """合并多个YOLO数据集"""
        try:
            print("\n=== 合并YOLO数据集 ===")
            print("此功能将合并多个YOLO格式数据集，包括:")
            print("- 验证所有数据集的classes.txt一致性")
            print("- 自动生成输出目录名称")
            print("- 统一图片前缀并格式化为5位数字")
            print("- 合并所有图片和标签文件")
            print("- 提供详细的合并统计信息")

            # 收集数据集路径
            dataset_paths = []
            print("\n请输入要合并的数据集路径（至少2个）:")

            while True:
                prompt = f"数据集 {len(dataset_paths) + 1} 路径（回车结束输入）: "
                path = input(prompt).strip()

                if not path:
                    if len(dataset_paths) < 2:
                        print("⚠ 至少需要输入2个数据集路径")
                        continue
                    else:
                        break

                # 验证路径
                if not Path(path).exists():
                    print(f"⚠ 路径不存在: {path}")
                    continue

                if not Path(path).is_dir():
                    print(f"⚠ 路径不是目录: {path}")
                    continue

                dataset_paths.append(path)
                print(f"✓ 已添加数据集: {path}")

            print(f"\n共收集到 {len(dataset_paths)} 个数据集")

            # 获取可选参数
            print("\n=== 可选设置 ===")

            # 输出路径
            output_path = input("输出路径（留空使用当前目录）: ").strip()
            if not output_path:
                output_path = "."
            else:
                # 验证输出路径
                output_path_obj = Path(output_path)
                if not output_path_obj.exists():
                    create_parent = (
                        input(f"路径 {output_path} 不存在，是否创建？(y/N): ")
                        .strip()
                        .lower()
                    )
                    if create_parent in ["y", "yes", "是"]:
                        try:
                            output_path_obj.mkdir(parents=True, exist_ok=True)
                            print(f"✓ 已创建输出路径: {output_path}")
                        except Exception as e:
                            print(f"❌ 创建路径失败: {e}")
                            self._pause()
                            return
                    else:
                        print("操作已取消")
                        self._pause()
                        return
                elif not output_path_obj.is_dir():
                    print(f"❌ 指定的路径不是目录: {output_path}")
                    self._pause()
                    return

            # 输出目录名称
            output_dir = input("输出目录名称（留空自动生成）: ").strip()
            if not output_dir:
                output_dir = None

            # 图片前缀
            image_prefix = input("图片前缀（留空使用默认）: ").strip()
            if not image_prefix:
                image_prefix = None

            processor = self._get_processor("yolo")

            # 先验证classes.txt一致性
            print("\n正在验证数据集兼容性...")
            path_objects = [Path(path) for path in dataset_paths]
            validation_result = processor._validate_classes_consistency(path_objects)

            if not validation_result["consistent"]:
                print(f"❌ 数据集验证失败: {validation_result['details']}")
                print("\n请确保所有数据集具有相同的classes.txt文件内容")
                self._pause()
                return

            print("✓ 数据集兼容性验证通过")
            print(f"类别列表: {', '.join(validation_result['classes'])}")

            # 生成输出目录名称预览
            if not output_dir:
                suggested_name = processor._generate_output_name(
                    classes=validation_result["classes"], dataset_paths=path_objects
                )
                print(f"建议输出目录名: {suggested_name}")

            # 确认合并
            print("\n=== 合并确认 ===")
            print(f"数据集数量: {len(dataset_paths)}")
            for i, path in enumerate(dataset_paths, 1):
                print(f"  {i}. {path}")

            print(f"输出路径: {output_path}")

            if output_dir:
                print(f"输出目录名称: {output_dir}")
            else:
                print("输出目录名称: 自动生成")

            if image_prefix:
                print(f"图片前缀: {image_prefix}")
            else:
                print("图片前缀: 自动生成")

            confirm = input("\n确认开始合并？(y/N): ").strip().lower()
            if confirm not in ["y", "yes", "是"]:
                print("\n操作已取消")
                self._pause()
                return

            # 执行合并
            print("\n正在合并数据集...")
            result = processor.merge_datasets(
                dataset_paths=path_objects,
                output_path=output_path,
                output_name=output_dir,
                image_prefix=image_prefix,
            )

            # 显示结果
            if result["success"]:
                print("\n✅ 数据集合并成功！")
                print(f"输出目录: {result['output_path']}")
                print(f"\n合并统计:")
                print(f"  - 总图片数: {result['total_images']}")
                print(f"  - 总标签数: {result['total_labels']}")
                print(f"  - 类别数: {len(result['classes'])}")
                print(f"  - 合并数据集数: {result['merged_datasets']}")

                if "statistics" in result:
                    stats = result["statistics"]
                    if "source_stats" in stats:
                        print(f"\n各数据集统计:")
                        for source, source_stats in stats["source_stats"].items():
                            print(
                                f"  {Path(source).name}: {source_stats['images']} 图片, {source_stats['labels']} 标签"
                            )

                print(f"\n✓ 合并后的数据集已保存到: {result['output_path']}")
            else:
                print(f"\n❌ 数据集合并失败: {result.get('error', '未知错误')}")

        except KeyboardInterrupt:
            print("\n合并数据集失败: 用户中断操作 (Code: USER_INTERRUPT)")
        except Exception as e:
            print(f"\n合并数据集失败: {e}")

        self._pause()

    def _yolo_merge_different_datasets(self) -> None:
        """合并多个不同类型的YOLO数据集"""
        try:
            print("\n=== 合并不同类型YOLO数据集 ===")
            print("此功能将合并多个不同类型的YOLO格式数据集，包括:")
            print("- 自动处理不同数据集的类别差异")
            print("- 生成统一的类别映射")
            print("- 支持用户自定义数据集处理顺序")
            print("- 自动重命名类别ID避免冲突")
            print("- 统一图片前缀并格式化为5位数字")
            print("- 提供详细的合并统计信息")

            # 收集数据集路径
            dataset_paths = []
            print("\n请输入要合并的数据集路径（至少2个）:")

            while True:
                prompt = f"数据集 {len(dataset_paths) + 1} 路径（回车结束输入）: "
                path = input(prompt).strip()

                if not path:
                    if len(dataset_paths) < 2:
                        print("⚠ 至少需要输入2个数据集路径")
                        continue
                    else:
                        break

                # 验证路径
                if not Path(path).exists():
                    print(f"⚠ 路径不存在: {path}")
                    continue

                if not Path(path).is_dir():
                    print(f"⚠ 路径不是目录: {path}")
                    continue

                dataset_paths.append(path)
                print(f"✓ 已添加数据集: {path}")

            print(f"\n共收集到 {len(dataset_paths)} 个数据集")

            # 显示数据集类别信息
            processor = self._get_processor("yolo")
            path_objects = [Path(path) for path in dataset_paths]
            
            print("\n=== 数据集类别信息 ===")
            all_classes_info = processor._collect_all_classes_info(path_objects)
            
            for i, info in enumerate(all_classes_info):
                print(f"数据集 {i+1}: {info['dataset_path'].name}")
                print(f"  类别数: {len(info['classes'])}")
                print(f"  类别: {', '.join(info['classes'][:5])}")
                if len(info['classes']) > 5:
                    print(f"  ... 等共 {len(info['classes'])} 个类别")
                print()

            # 询问是否调整数据集顺序
            adjust_order = input("是否需要调整数据集处理顺序？(y/N): ").strip().lower()
            dataset_order = None
            
            if adjust_order in ["y", "yes", "是"]:
                print("\n当前数据集顺序:")
                for i, path in enumerate(dataset_paths):
                    print(f"  {i}: {Path(path).name}")
                
                print("\n请输入新的处理顺序（用空格分隔的数字，如: 1 0 2）:")
                order_input = input("新顺序: ").strip()
                
                try:
                    dataset_order = [int(x) for x in order_input.split()]
                    if len(dataset_order) != len(dataset_paths):
                        print("⚠ 顺序数量与数据集数量不匹配，将使用默认顺序")
                        dataset_order = None
                    elif set(dataset_order) != set(range(len(dataset_paths))):
                        print("⚠ 顺序包含无效索引，将使用默认顺序")
                        dataset_order = None
                    else:
                        print("✓ 已设置自定义处理顺序")
                        reordered_paths = [dataset_paths[i] for i in dataset_order]
                        print("新的处理顺序:")
                        for i, path in enumerate(reordered_paths):
                            print(f"  {i+1}. {Path(path).name}")
                except ValueError:
                    print("⚠ 输入格式错误，将使用默认顺序")
                    dataset_order = None

            # 获取可选参数
            print("\n=== 可选设置 ===")

            # 输出路径
            output_path = input("输出路径（留空使用当前目录）: ").strip()
            if not output_path:
                output_path = "."
            else:
                # 验证输出路径
                output_path_obj = Path(output_path)
                if not output_path_obj.exists():
                    create_parent = (
                        input(f"路径 {output_path} 不存在，是否创建？(y/N): ")
                        .strip()
                        .lower()
                    )
                    if create_parent in ["y", "yes", "是"]:
                        try:
                            output_path_obj.mkdir(parents=True, exist_ok=True)
                            print(f"✓ 已创建输出路径: {output_path}")
                        except Exception as e:
                            print(f"❌ 创建路径失败: {e}")
                            self._pause()
                            return
                    else:
                        print("操作已取消")
                        self._pause()
                        return
                elif not output_path_obj.is_dir():
                    print(f"❌ 指定的路径不是目录: {output_path}")
                    self._pause()
                    return

            # 输出目录名称
            output_dir = input("输出目录名称（留空自动生成）: ").strip()
            if not output_dir:
                output_dir = None

            # 图片前缀
            image_prefix = input("图片前缀（留空使用默认）: ").strip()
            if not image_prefix:
                image_prefix = None

            # 预览统一类别映射
            print("\n正在分析类别映射...")
            unified_classes, class_mappings = processor._create_unified_class_mapping(all_classes_info)
            
            print(f"\n=== 统一类别映射预览 ===")
            print(f"合并后总类别数: {len(unified_classes)}")
            print(f"统一类别列表: {', '.join(unified_classes[:10])}")
            if len(unified_classes) > 10:
                print(f"... 等共 {len(unified_classes)} 个类别")

            print("\n各数据集类别映射:")
            for i, (info, mapping) in enumerate(zip(all_classes_info, class_mappings)):
                print(f"数据集 {i+1} ({info['dataset_path'].name}):")
                for old_id, new_id in mapping.items():
                    old_class = info['classes'][old_id]
                    new_class = unified_classes[new_id]
                    if old_id != new_id:
                        print(f"  {old_id}({old_class}) -> {new_id}({new_class})")
                    else:
                        print(f"  {old_id}({old_class}) -> 保持不变")

            # 生成输出目录名称预览
            if not output_dir:
                suggested_name = processor._generate_different_output_name(
                    unified_classes=unified_classes, dataset_paths=path_objects
                )
                print(f"\n建议输出目录名: {suggested_name}")

            # 确认合并
            print("\n=== 合并确认 ===")
            print(f"数据集数量: {len(dataset_paths)}")
            for i, path in enumerate(dataset_paths, 1):
                print(f"  {i}. {path}")

            print(f"输出路径: {output_path}")

            if output_dir:
                print(f"输出目录名称: {output_dir}")
            else:
                print("输出目录名称: 自动生成")

            if image_prefix:
                print(f"图片前缀: {image_prefix}")
            else:
                print("图片前缀: 使用默认(img)")

            if dataset_order:
                print(f"处理顺序: 自定义")
            else:
                print("处理顺序: 默认")

            confirm = input("\n确认开始合并？(y/N): ").strip().lower()
            if confirm not in ["y", "yes", "是"]:
                print("\n操作已取消")
                self._pause()
                return

            # 执行合并
            print("\n正在合并不同类型数据集...")
            result = processor.merge_different_type_datasets(
                dataset_paths=dataset_paths,
                output_path=output_path,
                output_name=output_dir,
                image_prefix=image_prefix,
                dataset_order=dataset_order,
            )

            # 显示结果
            if result["success"]:
                print("\n✅ 不同类型数据集合并成功！")
                print(f"输出目录: {result['output_path']}")
                print(f"\n合并统计:")
                print(f"  - 总图片数: {result['total_images']}")
                print(f"  - 总标签数: {result['total_labels']}")
                print(f"  - 统一类别数: {len(result['unified_classes'])}")
                print(f"  - 合并数据集数: {result['merged_datasets']}")

                if "statistics" in result:
                    stats = result["statistics"]
                    print(f"\n各数据集处理统计:")
                    for i, stat in enumerate(stats):
                        dataset_name = Path(stat['dataset_path']).name
                        print(f"  {i+1}. {dataset_name}:")
                        print(f"     图片: {stat['images_processed']}/{stat['images_count']}")
                        print(f"     标签: {stat['labels_processed']}/{stat['labels_count']}")
                        print(f"     索引范围: {stat['start_index']}-{stat['end_index']}")
                        print(f"     处理时间: {stat['processing_time']}秒")

                print(f"\n类别映射信息:")
                print(f"  - 原始类别总数: {sum(len(info['classes']) for info in all_classes_info)}")
                print(f"  - 统一后类别数: {len(result['unified_classes'])}")
                print(f"  - 统一类别列表: {', '.join(result['unified_classes'][:5])}")
                if len(result['unified_classes']) > 5:
                    print(f"    ... 等共 {len(result['unified_classes'])} 个类别")

                print(f"\n✓ 合并后的数据集已保存到: {result['output_path']}")
            else:
                print(f"\n❌ 不同类型数据集合并失败: {result.get('error', '未知错误')}")

        except KeyboardInterrupt:
            print("\n合并数据集失败: 用户中断操作 (Code: USER_INTERRUPT)")
        except Exception as e:
            print(f"\n合并数据集失败: {e}")

        self._pause()

    def _image_menu(self) -> None:
        """图像处理菜单"""
        menu = {
            "title": "图像处理",
            "options": [
                ("格式转换", self._image_convert),
                ("尺寸调整", self._image_resize),
                ("图像压缩", self._image_compress),
                (
                    "修复 OpenCV 读取错误的图像",
                    self._image_repair_corrupted_images,
                ),
                ("获取图像信息", self._image_info),
                ("返回主菜单", None),
            ],
        }

        self.menu_system.show_menu(menu)

    def _image_convert(self) -> None:
        """图像格式转换"""
        try:
            print("\n=== 图像格式转换 ===")

            input_path = self._get_path_input(
                "请输入输入路径 (文件或目录): ", must_exist=True
            )

            # 生成默认输出路径
            input_path_obj = Path(input_path)
            if input_path_obj.is_file():
                # 单个文件：生成 文件名_converted.目标格式
                stem = input_path_obj.stem
                default_output = str(input_path_obj.parent / f"{stem}_converted")
            else:
                # 目录：生成 目录名_converted
                default_output = str(
                    input_path_obj.parent / f"{input_path_obj.name}_converted"
                )

            output_path = self._get_input(f"输出路径 (默认: {default_output}): ")
            if not output_path.strip():
                output_path = default_output

            print("\n支持的格式: jpg, jpeg, png, bmp, tiff, webp")
            target_format = self._get_input("目标格式: ", required=True)

            quality = 95
            if target_format.lower() in ["jpg", "jpeg"]:
                quality = self._get_int_input(
                    "JPEG质量 (1-100, 默认95): ", default=95, min_val=1, max_val=100
                )

            recursive = False
            if Path(input_path).is_dir():
                recursive = self._get_yes_no_input("是否递归处理子目录?", default=True)

            # 多进程分批处理设置（已移除线程处理选项）
            import os

            cpu_count = os.cpu_count() or 4

            print("\n=== 多进程分批处理设置 ===")
            print(f"检测到 {cpu_count} 个CPU核心")

            batch_count = self._get_int_input(
                "批次数量 (默认100): ", default=100, min_val=1, max_val=1000
            )

            max_processes = self._get_int_input(
                f"最大进程数 (推荐{cpu_count}): ",
                default=cpu_count,
                min_val=1,
                max_val=cpu_count * 2,  # 允许超过CPU核心数，适应不同工作负载
            )

            processor = self._get_processor("image")

            print("\n正在转换图像格式...")
            result = processor.convert_format(
                input_path,
                target_format,
                output_path=output_path if output_path else None,
                quality=quality,
                recursive=recursive,
            )

            self._display_result(result)

        except UserInterruptError:
            print(f"\n转换失败: 用户中断操作 (Code: USER_INTERRUPT)")
            print("\n按回车键继续...")
            input()
        except Exception as e:
            print(f"\n转换失败: {e}")

        self._pause()

    def _image_resize(self) -> None:
        """图像尺寸调整"""
        try:
            print("\n=== 图像尺寸调整 ===")

            input_path = self._get_path_input(
                "请输入输入路径 (文件或目录): ", must_exist=True
            )

            # 生成默认输出路径
            input_path_obj = Path(input_path)
            if input_path_obj.is_dir():
                default_output = str(
                    input_path_obj.parent / f"{input_path_obj.name}_sized"
                )
            else:
                # 对于单文件，在同目录下生成 文件名_sized.扩展名
                stem = input_path_obj.stem
                suffix = input_path_obj.suffix
                default_output = str(input_path_obj.parent / f"{stem}_sized{suffix}")

            output_path = self._get_input(f"输出路径 (默认: {default_output}): ")
            if not output_path:
                output_path = default_output

            print("\n尺寸格式: WxH (如 800x600) 或单个数字 (如 800)")
            size_str = self._get_input("目标尺寸: ", required=True)
            size = self._parse_size(size_str)

            keep_aspect = self._get_yes_no_input("是否保持宽高比?", default=True)

            recursive = False
            if Path(input_path).is_dir():
                recursive = self._get_yes_no_input("是否递归处理子目录?", default=True)

            # 多进程分批处理选项
            import os

            cpu_count = os.cpu_count() or 4

            print("\n=== 多进程分批处理设置 ===")
            print(f"检测到 {cpu_count} 个CPU核心")

            batch_count = self._get_int_input(
                "批次数量 (默认100): ", default=100, min_val=1, max_val=1000
            )

            max_processes = self._get_int_input(
                f"最大进程数 (推荐{cpu_count}): ",
                default=cpu_count,
                min_val=1,
                max_val=cpu_count * 2,  # 允许超过CPU核心数，适应不同工作负载
            )

            processor = self._get_processor("image")

            print("\n正在调整图像尺寸...")
            result = processor.resize_images(
                input_path,
                output_path,
                target_size=size,
                maintain_aspect_ratio=keep_aspect,
                recursive=recursive,
            )

            self._display_result(result)

        except UserInterruptError:
            print(f"\n调整失败: 用户中断操作 (Code: USER_INTERRUPT)")
            print("\n按回车键继续...")
            input()
        except Exception as e:
            print(f"\n调整失败: {e}")

        self._pause()

    def _image_info(self) -> None:
        """获取图像信息"""
        try:
            print("\n=== 获取图像信息 ===")

            image_path = self._get_path_input(
                "请输入图像路径 (文件或目录): ", must_exist=True
            )

            recursive = False
            if Path(image_path).is_dir():
                recursive = self._get_yes_no_input("是否递归处理子目录?", default=True)

            processor = self._get_processor("image")

            print("\n正在获取图像信息...")
            result = processor.get_image_info(image_path, recursive=recursive)

            # 增强显示效果
            self._display_enhanced_image_info(result)

        except UserInterruptError:
            print(f"\n获取信息失败: 用户中断操作 (Code: USER_INTERRUPT)")
            print("\n按回车键继续...")
            input()
        except Exception as e:
            print(f"\n获取图像信息失败: {e}")

        self._pause()

    def _image_repair_corrupted_images(self) -> None:
        """修复因 OpenCV 加载失败的图像"""
        try:
            print("\n=== 修复 OpenCV 读取错误的图像 ===")

            directory = self._get_path_input(
                "请输入要检查的图像目录: ", must_exist=True, must_be_dir=True
            )

            recursive = self._get_yes_no_input("是否递归子目录?", default=True)
            extensions = None

            processor = self._get_processor("image")

            print("\n正在尝试用 OpenCV 读取图像，若读取失败将重新保存...")
            result = processor.repair_images_with_opencv(
                directory,
                extensions=extensions,
                recursive=recursive,
                include_hidden=False,
            )

            total = result.get("total_files", 0)
            loaded = result.get("loaded_without_issue", 0)
            repaired = result.get("repaired_count", 0)
            failed = result.get("failed_count", 0)

            print(
                f"\n处理完成: 共检查 {total} 张，OpenCV 成功加载 {loaded} 张，重新保存 {repaired} 张，失败 {failed} 张"
            )

            if failed:
                print("修复失败的文件:")
                for failure in result.get("failed_files", []):
                    print(f"  - {failure.get('file')}: {failure.get('error')}")

        except UserInterruptError:
            print(
                "\n修复操作被用户中断 (Code: USER_INTERRUPT)，按回车键继续..."
            )
            input()
        except Exception as e:
            print(f"\n修复失败: {e}")
        finally:
            self._pause()

    def _display_enhanced_image_info(self, result: Dict[str, Any]) -> None:
        """增强的图像信息显示"""
        if not result.get("success", False):
            self._display_result(result)
            return

        # 单文件处理
        if "file_path" in result:
            self._display_single_image_info(result)
        # 目录处理
        elif "input_dir" in result:
            self._display_directory_image_info(result)
        else:
            self._display_result(result)

    def _display_single_image_info(self, info: Dict[str, Any]) -> None:
        """显示单个图像的详细信息"""
        print("\n" + "=" * 50)
        print("✓ 图像信息获取成功")
        print("=" * 50)

        print(f"文件路径: {info.get('file_path', 'N/A')}")
        print(
            f"文件大小: {info.get('file_size_formatted', 'N/A')} ({info.get('file_size', 0)} 字节)"
        )
        print(f"图像格式: {info.get('format', 'N/A').upper()}")

        width = info.get("width", 0)
        height = info.get("height", 0)
        if width > 0 and height > 0:
            print(f"分辨率: {width} x {height}")
            print(f"宽高比: {info.get('aspect_ratio', 0):.3f}")
            print(f"总像素数: {info.get('total_pixels', 0):,}")

            # 清晰度分析
            quality_level = self._analyze_image_quality(width, height)
            print(f"清晰度级别: {quality_level}")

        if "mode" in info:
            print(f"颜色模式: {info['mode']}")
        if "has_transparency" in info:
            transparency = "是" if info["has_transparency"] else "否"
            print(f"包含透明度: {transparency}")

        print("=" * 50)

    def _display_directory_image_info(self, result: Dict[str, Any]) -> None:
        """显示目录图像信息统计"""
        print("\n" + "=" * 50)
        print("✓ 目录图像信息统计")
        print("=" * 50)

        stats = result.get("statistics", {})
        print(f"输入目录: {result.get('input_dir', 'N/A')}")
        print(f"递归处理: {'是' if result.get('recursive', False) else '否'}")
        print(f"总文件数: {stats.get('total_files', 0)}")
        print(f"处理成功: {stats.get('processed_count', 0)}")
        print(f"处理失败: {stats.get('failed_count', 0)}")
        print(f"总文件大小: {stats.get('total_size_formatted', 'N/A')}")
        print(f"总像素数: {stats.get('total_pixels', 0):,}")

        if stats.get("processed_count", 0) > 0:
            avg_size = stats.get("average_file_size", 0)
            print(f"平均文件大小: {self._format_file_size(avg_size)}")

        # 分辨率统计和清晰度分析
        image_info_list = result.get("image_info_list", [])
        if image_info_list:
            self._display_resolution_statistics(image_info_list)

        print("=" * 50)

    def _display_resolution_statistics(
        self, image_info_list: List[Dict[str, Any]]
    ) -> None:
        """显示分辨率统计和清晰度分析"""
        resolution_stats = {}
        quality_stats = {}

        # 统计分辨率和清晰度
        for info in image_info_list:
            if not info.get("success", False):
                continue

            width = info.get("width", 0)
            height = info.get("height", 0)

            if width > 0 and height > 0:
                resolution = f"{width}x{height}"
                resolution_stats[resolution] = resolution_stats.get(resolution, 0) + 1

                quality_level = self._analyze_image_quality(width, height)
                quality_stats[quality_level] = quality_stats.get(quality_level, 0) + 1

        total_images = len(
            [info for info in image_info_list if info.get("success", False)]
        )

        if total_images == 0:
            return

        print("\n📊 分辨率统计:")
        print("-" * 30)

        # 显示前10个最常见的分辨率
        sorted_resolutions = sorted(
            resolution_stats.items(), key=lambda x: x[1], reverse=True
        )
        for i, (resolution, count) in enumerate(sorted_resolutions[:10]):
            percentage = (count / total_images) * 100
            print(f"{resolution:>15}: {count:>4} 张 ({percentage:>5.1f}%)")

        if len(sorted_resolutions) > 10:
            other_count = sum(count for _, count in sorted_resolutions[10:])
            other_percentage = (other_count / total_images) * 100
            print(f"{'其他':>15}: {other_count:>4} 张 ({other_percentage:>5.1f}%)")

        print("\n🎯 清晰度分析:")
        print("-" * 30)

        # 按清晰度级别排序显示
        quality_order = ["4K", "2K", "Full HD", "HD", "SD", "低清"]
        for quality in quality_order:
            if quality in quality_stats:
                count = quality_stats[quality]
                percentage = (count / total_images) * 100
                print(f"{quality:>10}: {count:>4} 张 ({percentage:>5.1f}%)")

    def _analyze_image_quality(self, width: int, height: int) -> str:
        """分析图像清晰度级别"""
        try:
            config = self.config_manager.get_config()
            quality_config = config.get("image_processing", {}).get(
                "quality_analysis", {}
            )

            # 获取自定义清晰度级别
            custom_levels = quality_config.get("custom_levels", [])

            # 如果没有自定义级别，使用默认判断
            if not custom_levels:
                if width >= 3840 and height >= 2160:
                    return "超清 4K"
                elif width >= 2560 and height >= 1440:
                    return "超清 2K"
                elif width >= 1920 and height >= 1080:
                    return "全高清"
                elif width >= 1280 and height >= 720:
                    return "高清"
                elif width >= 720 and height >= 480:
                    return "标清"
                else:
                    return "低清"

            # 按阈值从高到低排序
            sorted_levels = sorted(
                custom_levels,
                key=lambda x: x["threshold"][0] * x["threshold"][1],
                reverse=True,
            )

            for level in sorted_levels:
                threshold = level["threshold"]
                if width >= threshold[0] and height >= threshold[1]:
                    return level["name"]

            # 如果没有匹配的自定义级别，使用默认判断
            return "低清"

        except Exception as e:
            # 配置读取失败时使用硬编码阈值
            if width >= 3840 and height >= 2160:
                return "超清 4K"
            elif width >= 2560 and height >= 1440:
                return "超清 2K"
            elif width >= 1920 and height >= 1080:
                return "全高清"
            elif width >= 1280 and height >= 720:
                return "高清"
            elif width >= 720 and height >= 480:
                return "标清"
            else:
                return "低清"

    def _format_file_size(self, size_bytes: float) -> str:
        """格式化文件大小"""
        if size_bytes == 0:
            return "0 B"

        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math

        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"

    def _image_compress(self) -> None:
        """图像压缩"""
        try:
            print("\n=== 图像压缩 ===")

            input_path = self._get_path_input(
                "请输入输入路径 (文件或目录): ", must_exist=True
            )

            # 生成默认输出路径
            input_path_obj = Path(input_path)
            if input_path_obj.is_file():
                # 单个文件：生成 文件名_compressed.扩展名
                stem = input_path_obj.stem
                suffix = input_path_obj.suffix
                default_output = str(
                    input_path_obj.parent / f"{stem}_compressed{suffix}"
                )
            else:
                # 目录：生成 目录名_compressed
                default_output = str(
                    input_path_obj.parent / f"{input_path_obj.name}_compressed"
                )

            output_path = self._get_input(f"输出路径 (默认: {default_output}): ")
            if not output_path.strip():
                output_path = default_output

            quality = self._get_int_input(
                "压缩质量 (1-100, 默认85): ", default=85, min_val=1, max_val=100
            )

            print("\n目标格式选项:")
            print("1. 保持原格式")
            print("2. 转换为 JPG (推荐，压缩效果最好)")
            print("3. 转换为 PNG")
            print("4. 转换为 WebP")

            format_choice = self._get_int_input(
                "请选择目标格式 (1-4): ", min_val=1, max_val=4
            )

            target_format = None
            if format_choice == 2:
                target_format = "jpg"
            elif format_choice == 3:
                target_format = "png"
            elif format_choice == 4:
                target_format = "webp"

            # 询问是否限制最大尺寸
            limit_size = self._get_yes_no_input("是否限制图像最大尺寸?", default=False)
            max_size = None
            if limit_size:
                print("\n常用尺寸选项:")
                print("1. 1920x1080 (Full HD)")
                print("2. 1280x720 (HD)")
                print("3. 800x600")
                print("4. 自定义")

                size_choice = self._get_int_input(
                    "请选择尺寸 (1-4): ", min_val=1, max_val=4
                )

                if size_choice == 1:
                    max_size = (1920, 1080)
                elif size_choice == 2:
                    max_size = (1280, 720)
                elif size_choice == 3:
                    max_size = (800, 600)
                elif size_choice == 4:
                    size_str = self._get_input(
                        "请输入最大尺寸 (格式: WxH，如 1024x768): ", required=True
                    )
                    max_size = self._parse_size(size_str)

            recursive = False
            if Path(input_path).is_dir():
                recursive = self._get_yes_no_input("是否递归处理子目录?", default=True)

            # 多进程分批处理设置
            import os

            cpu_count = os.cpu_count() or 4

            # 统计图片数量
            processor = self._get_processor("image")
            print("\n正在统计图片数量...")

            if Path(input_path).is_file():
                total_images = 1
            else:
                # 统计目录中的图片文件数量
                image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
                total_images = 0

                if recursive:
                    for file_path in Path(input_path).rglob("*"):
                        if file_path.suffix.lower() in image_extensions:
                            total_images += 1
                else:
                    for file_path in Path(input_path).iterdir():
                        if (
                            file_path.is_file()
                            and file_path.suffix.lower() in image_extensions
                        ):
                            total_images += 1

            print(f"发现图片文件: {total_images} 张")

            # 固定每批次1000张图片，根据总数计算批次数
            batch_size = 1000
            batch_count = max(
                1, (total_images + batch_size - 1) // batch_size
            )  # 向上取整

            print(f"每批次处理: {batch_size} 张图片")
            print(f"总批次数: {batch_count} 个批次")

            # 最大进程数设置
            default_processes = min(cpu_count, batch_count)  # 进程数不超过批次数
            max_processes = self._get_int_input(
                f"最大进程数 (默认{default_processes}): ",
                default=default_processes,
                min_val=1,
                max_val=min(cpu_count * 2, batch_count),
            )

            print("\n正在压缩图像...")
            print(f"输入路径: {input_path}")
            if Path(input_path).is_dir():
                print(f"输出目录: {output_path or '输入目录/compressed'}")
            else:
                print(f"输出文件: {output_path or '自动生成'}")
            print(f"压缩质量: {quality}")
            if target_format:
                print(f"目标格式: {target_format.upper()}")
            if max_size:
                print(f"最大尺寸: {max_size[0]}x{max_size[1]}")
            if Path(input_path).is_dir():
                print(f"递归处理: {'是' if recursive else '否'}")
            print(f"处理模式: 多进程分批处理")
            print(f"每批次大小: {batch_size} 张图片")
            print(f"总批次数: {batch_count} 个批次")
            print(f"最大进程数: {max_processes}")
            print()

            # 使用多进程分批处理
            result = processor.compress_images_multiprocess_batch(
                input_dir=input_path,
                output_dir=output_path if output_path else None,
                quality=quality,
                target_format=target_format,
                recursive=recursive,
                max_size=max_size,
                batch_count=batch_count,
                max_processes=max_processes,
            )

            self._display_result(result)

            # 显示压缩统计信息
            if result.get("success") and "statistics" in result:
                stats = result["statistics"]
                print("\n=== 压缩统计 ===")
                print(f"总文件数: {stats['total_files']}")
                print(f"成功压缩: {stats['compressed_count']}")
                print(f"失败文件: {stats['failed_count']}")
                print(f"原始总大小: {stats['total_input_size_formatted']}")
                print(f"压缩后大小: {stats['total_output_size_formatted']}")
                print(f"节省空间: {stats['space_saved_formatted']}")
                print(f"压缩比: {stats['overall_compression_ratio']:.2f}")
                print(f"空间节省率: {stats['overall_space_saved_percentage']:.1f}%")

        except UserInterruptError:
            print(f"\n压缩失败: 用户中断操作 (Code: USER_INTERRUPT)")
            print("\n按回车键继续...")
            input()
        except Exception as e:
            print(f"\n压缩失败: {e}")
            # 创建一个失败的结果对象，避免后续代码出错
            result = {
                "success": False,
                "message": str(e),
                "statistics": {
                    "total_files": 0,
                    "compressed_count": 0,
                    "failed_count": 0,
                },
            }
            self._display_result(result)

        self._pause()

    def _file_menu(self) -> None:
        """文件操作菜单"""
        menu = {
            "title": "文件操作",
            "options": [
                ("单目录重命名", self._file_rename_single_dir),
                ("数据集重命名", self._file_rename_images_labels),
                ("数据集重命名（传统模式）", self._file_rename_images_labels_legacy),
                ("按扩展名组织文件", self._file_organize),
                ("递归删除JSON文件", self._file_delete_json_recursive),
                ("批量复制文件", self._file_copy),
                ("批量移动文件", self._file_move),
                ("按数量移动图片", self._file_move_images_by_count),
                ("返回主菜单", None),
            ],
        }

        self.menu_system.show_menu(menu)

    def _file_organize(self) -> None:
        """按扩展名组织文件"""
        try:
            print("\n=== 按扩展名组织文件 ===")

            source_dir = self._get_path_input(
                "请输入源目录: ", must_exist=True, must_be_dir=True
            )
            output_dir = self._get_input("输出目录 (默认为源目录): ")
            copy_files = self._get_yes_no_input("是否复制文件而不是移动? (y/n): ")

            processor = self._get_processor("file")

            print("\n正在组织文件...")
            result = processor.organize_by_extension(
                source_dir,
                output_dir=output_dir if output_dir else None,
                copy_files=copy_files,
            )

            self._display_result(result)

        except UserInterruptError:
            print(f"\n组织失败: 用户中断操作 (Code: USER_INTERRUPT)")
            print("\n按回车键继续...")
            input()
        except Exception as e:
            print(f"\n组织失败: {e}")

        self._pause()

    def _file_copy(self) -> None:
        """批量复制文件"""
        try:
            print("\n=== 批量复制文件 ===")

            source_path = self._get_path_input("请输入源路径: ", must_exist=True)
            dest_path = self._get_path_input("请输入目标路径: ", must_exist=False)

            recursive = False
            if Path(source_path).is_dir():
                recursive = self._get_yes_no_input("是否递归复制? (y/n): ")

            processor = self._get_processor("file")

            print("\n正在复制文件...")
            result = processor.copy_files(source_path, dest_path, recursive=recursive)

            self._display_result(result)

        except UserInterruptError:
            print(f"\n复制失败: 用户中断操作 (Code: USER_INTERRUPT)")
            print("\n按回车键继续...")
            input()
        except Exception as e:
            print(f"\n复制失败: {e}")

        self._pause()

    def _file_move(self) -> None:
        """批量移动文件"""
        try:
            print("\n=== 批量移动文件 ===")

            source_path = self._get_path_input("请输入源路径: ", must_exist=True)
            dest_path = self._get_path_input("请输入目标路径: ", must_exist=False)

            recursive = False
            if Path(source_path).is_dir():
                recursive = self._get_yes_no_input("是否递归移动? (y/n): ")

            processor = self._get_processor("file")

            print("\n正在移动文件...")
            result = processor.move_files(source_path, dest_path, recursive=recursive)

            self._display_result(result)

        except UserInterruptError:
            print(f"\n移动失败: 用户中断操作 (Code: USER_INTERRUPT)")
            print("\n按回车键继续...")
            input()
        except Exception as e:
            print(f"\n移动失败: {e}")

        self._pause()

    def _file_move_images_by_count(self) -> None:
        """按数量移动图片"""
        try:
            print("\n=== 按数量移动图片 ===")
            print("规则: 先处理源目录下图片，再按子目录名称顺序处理子目录内图片")
            print("提示: 输入 9999 表示移动全部图片")

            source_path = self._get_path_input(
                "请输入源目录: ", must_exist=True, must_be_dir=True
            )
            dest_path = self._get_path_input("请输入目标目录: ", must_exist=False)

            count_str = self._get_input("请输入要移动的图片数量: ", required=True)
            try:
                count = int(count_str)
            except ValueError:
                print("数量必须为整数")
                self._pause()
                return

            overwrite = self._get_yes_no_input("目标存在同名文件时覆盖? (y/n): ")

            processor = self._get_processor("file")

            print("\n正在移动图片...")
            result = processor.move_images_by_count(
                source_path, dest_path, count=count, overwrite=overwrite
            )

            self._display_result(result)

        except UserInterruptError:
            print(f"\n移动失败: 用户中断操作 (Code: USER_INTERRUPT)")
            print("\n按回车键继续...")
            input()
        except Exception as e:
            print(f"\n移动失败: {e}")

        self._pause()

    def _file_rename_images_labels_legacy(self) -> None:
        """Images/Labels同步重命名（传统模式，不补零）"""
        try:
            print("\n=== Images/Labels同步重命名（传统模式） ===")
            print("此功能会同时重命名images和labels子目录中的对应文件")
            print("文件名将直接使用序号（如: 1.jpg, 2.jpg），不补零")

            source_dir = self._get_path_input(
                "请输入包含images和labels子目录的根目录: ",
                must_exist=True,
                must_be_dir=True,
            )

            # 检查images和labels目录是否存在
            source_path = Path(source_dir)
            images_dir = source_path / "images"
            labels_dir = source_path / "labels"

            if not images_dir.exists():
                print(f"错误: 未找到images目录: {images_dir}")
                self._pause()
                return

            if not labels_dir.exists():
                print(f"错误: 未找到labels目录: {labels_dir}")
                self._pause()
                return

            print(f"找到images目录: {images_dir}")
            print(f"找到labels目录: {labels_dir}")

            prefix = self._get_input(
                "请输入文件名前缀（空格表示无前缀）: ",
                required=True,
                allow_space_empty=True,
            )

            shuffle_order = self._get_yes_no_input(
                "是否打乱文件顺序? (默认: 否) (y/n): ", default=False
            )

            if prefix:
                print(f"\n重命名前缀: {prefix}")
            else:
                print("\n重命名前缀: （无前缀）")
            print("重命名模式: 1, 2, 3...（不补零）")
            print(f"打乱顺序: {'是' if shuffle_order else '否'}")

            if not self._get_yes_no_input("\n确认开始同步重命名? (y/n): "):
                print("操作已取消")
                return

            processor = self._get_processor("file")

            print("\n正在同步重命名images和labels文件...")
            result = processor.rename_images_labels_sync(
                str(images_dir), str(labels_dir), prefix, 0, shuffle_order
            )

            self._display_result(result)

        except UserInterruptError:
            print(f"\n重命名失败: 用户中断操作 (Code: USER_INTERRUPT)")
            print("\n按回车键继续...")
            input()
        except Exception as e:
            print(f"\n重命名失败: {e}")

        self._pause()

    def _file_rename_single_dir(self) -> None:
        """单目录重命名文件"""
        try:
            print("\n=== 单目录重命名文件 ===")

            source_dir = self._get_path_input(
                "请输入源目录: ", must_exist=True, must_be_dir=True
            )

            # 获取文件名前缀
            prefix = self._get_input(
                "请输入文件名前缀（空格表示无前缀）: ",
                required=True,
                allow_space_empty=True,
            )

            # 获取数字位数，默认为5位
            digits_input = self._get_input("请输入数字位数 (默认: 5): ")
            try:
                digits = int(digits_input) if digits_input else 5
                if digits < 1 or digits > 10:
                    print("位数必须在1-10之间，使用默认值5")
                    digits = 5
            except ValueError:
                print("无效的位数输入，使用默认值5")
                digits = 5

            # 是否打乱顺序
            shuffle_order = self._get_yes_no_input(
                "是否打乱文件顺序? (默认: 否) (y/n): ", default=False
            )

            # 检测目录中的文件后缀
            source_path = Path(source_dir)
            file_extensions = set()
            for file_path in source_path.iterdir():
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext:  # 只添加有后缀的文件
                        file_extensions.add(ext)

            # 获取文件后缀
            if file_extensions:
                extensions_list = sorted(list(file_extensions))
                print(f"\n检测到的文件后缀: {', '.join(extensions_list)}")
                default_ext = (
                    extensions_list[0]
                    if len(extensions_list) == 1
                    else extensions_list[0]
                )
                suffix_input = self._get_input(
                    f"请输入文件后缀 (默认: {default_ext}): "
                )
                suffix = suffix_input if suffix_input else default_ext
            else:
                print("\n未检测到文件后缀")
                suffix = self._get_input("请输入文件后缀 (如: .jpg): ", required=True)

            # 确保后缀以点开头
            if not suffix.startswith("."):
                suffix = "." + suffix

            # 构建重命名模式
            if prefix:
                pattern = f"{prefix}_{{index:0{digits}d}}{suffix}"
            else:
                pattern = f"{{index:0{digits}d}}{suffix}"

            print(f"\n重命名模式: {pattern}")
            # 显示示例时使用正确的格式
            if prefix:
                example_pattern = f"{prefix}_{{:0{digits}d}}{suffix}"
            else:
                example_pattern = f"{{:0{digits}d}}{suffix}"
            print(
                f"示例: {example_pattern.format(1)}, {example_pattern.format(2)}, {example_pattern.format(3)}..."
            )
            print(f"打乱顺序: {'是' if shuffle_order else '否'}")

            # 确认操作
            if not self._get_yes_no_input("\n确认使用此重命名模式? (y/n): "):
                print("操作已取消")
                return

            processor = self._get_processor("file")

            print("\n正在重命名文件...")
            result = processor.rename_files_with_temp(
                source_dir, pattern, shuffle_order=shuffle_order
            )

            self._display_result(result)

        except UserInterruptError:
            print(f"\n重命名失败: 用户中断操作 (Code: USER_INTERRUPT)")
            print("\n按回车键继续...")
            input()
        except Exception as e:
            print(f"\n重命名失败: {e}")

        self._pause()

    def _file_rename_images_labels(self) -> None:
        """Images和Labels目录同步重命名"""
        try:
            print("\n=== Images/Labels同步重命名 ===")
            print("此功能会同时重命名images和labels子目录中的对应文件")

            source_dir = self._get_path_input(
                "请输入包含images和labels子目录的根目录: ",
                must_exist=True,
                must_be_dir=True,
            )

            # 检查images和labels目录是否存在
            source_path = Path(source_dir)
            images_dir = source_path / "images"
            labels_dir = source_path / "labels"

            if not images_dir.exists():
                print(f"错误: 未找到images目录: {images_dir}")
                self._pause()
                return

            if not labels_dir.exists():
                print(f"错误: 未找到labels目录: {labels_dir}")
                self._pause()
                return

            print(f"找到images目录: {images_dir}")
            print(f"找到labels目录: {labels_dir}")

            # 获取文件名前缀
            prefix = self._get_input(
                "请输入文件名前缀（空格表示无前缀）: ",
                required=True,
                allow_space_empty=True,
            )

            # 获取数字位数，默认为5位
            digits_input = self._get_input("请输入数字位数 (默认: 5): ")
            try:
                digits = int(digits_input) if digits_input else 5
                if digits < 1 or digits > 10:
                    print("位数必须在1-10之间，使用默认值5")
                    digits = 5
            except ValueError:
                print("无效的位数输入，使用默认值5")
                digits = 5

            # 是否打乱顺序
            shuffle_order = self._get_yes_no_input(
                "是否打乱文件顺序? (默认: 否) (y/n): ", default=False
            )

            print(f"\n重命名前缀: {prefix}")
            print(f"数字位数: {digits}")
            print(f"打乱顺序: {'是' if shuffle_order else '否'}")

            # 确认操作
            if not self._get_yes_no_input("\n确认开始同步重命名? (y/n): "):
                print("操作已取消")
                return

            processor = self._get_processor("file")

            print("\n正在同步重命名images和labels文件...")
            result = processor.rename_images_labels_sync(
                str(images_dir), str(labels_dir), prefix, digits, shuffle_order
            )

            self._display_result(result)

        except UserInterruptError:
            print(f"\n重命名失败: 用户中断操作 (Code: USER_INTERRUPT)")
            print("\n按回车键继续...")
            input()
        except Exception as e:
            print(f"\n重命名失败: {e}")

        self._pause()

    def _file_delete_json_recursive(self) -> None:
        """递归删除目录中的所有JSON文件"""
        try:
            print("\n=== 递归删除JSON文件 ===")

            target_dir = self._get_path_input(
                "请输入目标目录: ", must_exist=True, must_be_dir=True
            )

            # 先扫描目录，统计JSON文件数量
            json_files = []
            target_path = Path(target_dir)

            print("\n正在扫描目录...")
            for json_file in target_path.rglob("*.json"):
                if json_file.is_file():
                    json_files.append(json_file)

            if not json_files:
                print("\n未找到任何JSON文件")
                self._pause()
                return

            print(f"\n找到 {len(json_files)} 个JSON文件:")

            # 显示前10个文件作为预览
            for i, json_file in enumerate(json_files[:10]):
                print(f"  {i+1}. {json_file.relative_to(target_path)}")

            if len(json_files) > 10:
                print(f"  ... 还有 {len(json_files) - 10} 个文件")

            # 确认删除
            if not self._get_yes_no_input(
                f"\n警告: 此操作将永久删除 {len(json_files)} 个JSON文件，是否继续? (y/n): "
            ):
                print("操作已取消")
                return

            # 执行删除
            deleted_count = 0
            failed_files = []

            print("\n正在删除JSON文件...")
            for json_file in json_files:
                try:
                    json_file.unlink()
                    deleted_count += 1
                    if deleted_count % 10 == 0 or deleted_count == len(json_files):
                        print(f"已删除 {deleted_count}/{len(json_files)} 个文件")
                except Exception as e:
                    failed_files.append((json_file, str(e)))

            # 显示结果
            print(f"\n删除完成!")
            print(f"成功删除: {deleted_count} 个文件")

            if failed_files:
                print(f"删除失败: {len(failed_files)} 个文件")
                print("\n失败的文件:")
                for failed_file, error in failed_files[:5]:  # 只显示前5个失败的文件
                    print(f"  {failed_file.relative_to(target_path)}: {error}")
                if len(failed_files) > 5:
                    print(f"  ... 还有 {len(failed_files) - 5} 个失败的文件")

        except UserInterruptError:
            print(f"\n删除失败: 用户中断操作 (Code: USER_INTERRUPT)")
            print("\n按回车键继续...")
            input()
        except Exception as e:
            print(f"\n删除JSON文件失败: {e}")

        self._pause()

    def _dataset_menu(self) -> None:
        """数据集处理菜单"""
        menu = {
            "title": "数据集处理",
            "options": [
                ("分割数据集", self._dataset_split),
                ("验证数据集", self._dataset_validate),
                ("验证分割数据集", self._dataset_validate_segmentation),
                ("分析数据集", self._dataset_analyze),
                ("清理数据集", self._dataset_clean),
                ("返回主菜单", None),
            ],
        }

        self.menu_system.show_menu(menu)

    def _dataset_split(self) -> None:
        """分割数据集"""
        try:
            print("\n=== 分割数据集 ===")

            dataset_path = self._get_path_input(
                "请输入数据集路径: ", must_exist=True, must_be_dir=True
            )
            output_dir = self._get_input("输出目录 (可选): ")

            print("\n支持的格式: yolo, coco, pascal_voc")
            dataset_format = self._get_input(
                "数据集格式 (默认: yolo): ", default="yolo"
            )

            train_ratio = self._get_float_input(
                "训练集比例 (默认: 0.8): ", default=0.8, min_val=0.1, max_val=0.9
            )
            val_ratio = self._get_float_input(
                "验证集比例 (默认: 0.1): ", default=0.1, min_val=0.0, max_val=0.5
            )
            test_ratio = self._get_float_input(
                "测试集比例 (默认: 0.1): ", default=0.1, min_val=0.0, max_val=0.5
            )

            # 检查比例总和
            total_ratio = train_ratio + val_ratio + test_ratio
            if abs(total_ratio - 1.0) > 0.001:
                print(f"\n警告: 比例总和为 {total_ratio:.3f}，将自动调整")

            processor = self._get_processor("dataset")

            print("\n正在分割数据集...")
            result = processor.split_dataset(
                dataset_path,
                output_path=output_dir if output_dir else None,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                dataset_format=dataset_format,
            )

            self._display_result(result)

        except UserInterruptError:
            pass
        except Exception as e:
            print(f"\n数据集分割失败: {e}")

        self._pause()

    def _dataset_validate(self) -> None:
        """验证数据集"""
        try:
            print("\n=== 验证数据集 ===")

            dataset_path = self._get_path_input(
                "请输入数据集路径: ", must_exist=True, must_be_dir=True
            )

            print("\n支持的格式: yolo, coco, pascal_voc")
            dataset_format = self._get_input(
                "数据集格式 (默认: yolo): ", default="yolo"
            )

            processor = self._get_processor("dataset")

            print("\n正在验证数据集...")
            result = processor.get_dataset_statistics(dataset_path)

            self._display_result(result)

        except UserInterruptError:
            print(f"\n删除失败: 用户中断操作 (Code: USER_INTERRUPT)")
            print("\n按回车键继续...")
            input()
        except Exception as e:
            print(f"\n恢复失败: {e}")

        self._pause()

    def _dataset_analyze(self) -> None:
        """分析数据集"""
        try:
            print("\n=== 分析数据集 ===")

            dataset_path = self._get_path_input(
                "请输入数据集路径: ", must_exist=True, must_be_dir=True
            )

            print("\n支持的格式: yolo, coco, pascal_voc")
            dataset_format = self._get_input(
                "数据集格式 (默认: yolo): ", default="yolo"
            )

            output_file = self._get_input("分析结果输出文件 (可选): ")

            processor = self._get_processor("dataset")

            print("\n正在分析数据集...")
            result = processor.analyze_dataset(
                dataset_path,
                dataset_format=dataset_format,
                output_file=output_file if output_file else None,
            )

            self._display_result(result)

        except UserInterruptError:
            print(f"\n转换失败: 用户中断操作 (Code: USER_INTERRUPT)")
            print("\n按回车键继续...")
            input()
        except Exception as e:
            print(f"\n分析失败: {e}")

        self._pause()

    def _dataset_validate_segmentation(self) -> None:
        """验证分割数据集"""
        try:
            print("\n=== 验证分割数据集 ===")
            print("分割数据集要求每个标签文件至少有7列（类别 + 至少4个分割点坐标）")

            dataset_path = self._get_path_input(
                "请输入分割数据集路径: ", must_exist=True, must_be_dir=True
            )

            # 询问是否移动无效文件
            move_invalid = self._get_yes_no_input(
                "是否移动无效文件到上级目录? (y/n, 默认: y): ", default=True
            )

            custom_dir_name = None
            if move_invalid:
                custom_dir_name = self._get_input("自定义无效文件目录名称 (可选): ")
                if not custom_dir_name:
                    custom_dir_name = None

            # 导入分割验证器
            try:
                import os
                import sys

                # 添加项目根目录到路径
                project_root = os.path.dirname(
                    os.path.dirname(os.path.dirname(__file__))
                )
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)

                from test_segmentation_dataset_validation import (
                    SegmentationDatasetValidator,
                )

                validator = SegmentationDatasetValidator()

                print("\n正在验证分割数据集...")
                result = validator.validate_and_clean_dataset(
                    dataset_path=dataset_path,
                    move_invalid=move_invalid,
                    custom_invalid_dir_name=custom_dir_name,
                )

                self._display_result(result)

            except ImportError as e:
                print(f"\n导入分割验证器失败: {e}")
                print("请确保 test_segmentation_dataset_validation.py 文件存在")

        except UserInterruptError:
            print(f"\n分析失败: 用户中断操作 (Code: USER_INTERRUPT)")
            print("\n按回车键继续...")
            input()
        except Exception as e:
            print(f"\n分割数据集验证失败: {e}")

        self._pause()

    def _label_menu(self) -> None:
        """标签处理菜单"""
        menu = {
            "title": "标签处理",
            "options": [
                ("创建空标签文件", self._label_create_empty),
                ("翻转标签坐标", self._label_flip),
                ("过滤标签类别", self._label_filter),
                ("删除空标签", self._label_remove_empty),
                ("删除只包含指定类别标签", self._label_remove_class),
                ("返回主菜单", None),
            ],
        }

        self.menu_system.show_menu(menu)

    def _label_create_empty(self) -> None:
        """创建空标签文件"""
        try:
            print("\n=== 创建空标签文件 ===")

            images_dir = self._get_path_input(
                "请输入图像目录: ", must_exist=True, must_be_dir=True
            )

            # 计算默认标签目录（与图像目录同级的labels目录）
            images_path = Path(images_dir)
            default_labels_dir = images_path.parent / "labels"

            labels_dir = self._get_input(f"标签目录 (默认为 {default_labels_dir}): ")
            overwrite = self._get_yes_no_input(
                "是否覆盖已存在的标签文件?", default=False
            )

            processor = self._get_processor("label")

            print("\n正在创建空标签文件...")
            result = processor.create_empty_labels(
                images_dir,
                labels_dir=labels_dir if labels_dir else str(default_labels_dir),
                overwrite=overwrite,
            )

            self._display_result(result)

        except UserInterruptError:
            print(f"\n创建失败: 用户中断操作 (Code: USER_INTERRUPT)")
            print("\n按回车键继续...")
            input()
        except Exception as e:
            print(f"\n创建空标签文件失败: {e}")

        self._pause()

    def _label_flip(self) -> None:
        """翻转标签坐标"""
        try:
            print("\n=== 翻转标签坐标 ===")

            labels_dir = self._get_path_input(
                "请输入标签目录: ", must_exist=True, must_be_dir=True
            )

            print("\n翻转类型: horizontal, vertical, both")
            flip_type = self._get_input(
                "翻转类型 (默认: horizontal): ", default="horizontal"
            )

            backup = self._get_yes_no_input("是否备份原文件?", default=True)

            processor = self._get_processor("label")

            print("\n正在翻转标签坐标...")
            result = processor.flip_labels(
                labels_dir, flip_type=flip_type, backup=backup
            )

            self._display_result(result)

        except UserInterruptError:
            print(f"\n翻转失败: 用户中断操作 (Code: USER_INTERRUPT)")
            print("\n按回车键继续...")
            input()
        except Exception as e:
            print(f"\n翻转标签坐标失败: {e}")

        self._pause()

    def _label_filter(self) -> None:
        """过滤标签类别"""
        try:
            print("\n=== 过滤标签类别 ===")

            labels_dir = self._get_path_input(
                "请输入标签目录: ", must_exist=True, must_be_dir=True
            )

            classes_str = self._get_input(
                "目标类别 (逗号分隔，如: 0,1,2): ", required=True
            )
            classes = [int(c.strip()) for c in classes_str.split(",")]

            print("\n操作类型: keep (保留), remove (移除)")
            action = self._get_input("操作类型 (默认: keep): ", default="keep")

            backup = self._get_yes_no_input("是否备份原文件?", default=True)

            processor = self._get_processor("label")

            print("\n正在过滤标签类别...")
            result = processor.filter_labels_by_class(
                labels_dir, target_classes=classes, action=action, backup=backup
            )

            self._display_result(result)

        except UserInterruptError:
            print(f"\n过滤失败: 用户中断操作 (Code: USER_INTERRUPT)")
            print("\n按回车键继续...")
            input()
        except Exception as e:
            print(f"\n过滤标签类别失败: {e}")

        self._pause()

    def _label_remove_empty(self) -> None:
        """删除空标签"""
        try:
            print("\n=== 删除空标签及对应图像 ===")

            dataset_dir = self._get_path_input(
                "请输入数据集目录: ", must_exist=True, must_be_dir=True
            )
            images_dir = self._get_input(
                "图像子目录名 (默认: images): ", default="images"
            )
            labels_dir = self._get_input(
                "标签子目录名 (默认: labels): ", default="labels"
            )

            # 确认操作
            if not self._get_yes_no_input(
                "\n警告: 此操作将永久删除文件，是否继续?", default=False
            ):
                print("操作已取消")
                return

            processor = self._get_processor("label")

            print("\n正在删除空标签及对应图像...")
            result = processor.remove_empty_labels_and_images(
                dataset_dir, images_subdir=images_dir, labels_subdir=labels_dir
            )

            self._display_result(result)

        except UserInterruptError:
            print(f"\n删除失败: 用户中断操作 (Code: USER_INTERRUPT)")
            print("\n按回车键继续...")
            input()
        except Exception as e:
            print(f"\n删除空标签失败: {e}")

        self._pause()

    def _label_remove_class(self) -> None:
        """删除指定类别标签"""
        try:
            print("\n=== 删除只包含指定类别的标签及图像 ===")

            dataset_dir = self._get_path_input(
                "请输入数据集目录: ", must_exist=True, must_be_dir=True
            )

            # 读取classes.txt文件
            classes_file = Path(dataset_dir) / "classes.txt"
            if not classes_file.exists():
                print(f"\n❌ 未找到classes.txt文件: {classes_file}")
                print("请确保数据集目录包含classes.txt文件")
                self._pause()
                return

            try:
                with open(classes_file, "r", encoding="utf-8") as f:
                    classes = [line.strip() for line in f.readlines() if line.strip()]

                if not classes:
                    print("\n❌ classes.txt文件为空")
                    self._pause()
                    return

                # 展示类别列表
                print("\n=== 数据集类别列表 ===")
                for i, class_name in enumerate(classes):
                    print(f"  {i}: {class_name}")

                # 让用户选择要删除的类别
                target_class = self._get_int_input(
                    f"\n请选择要删除的类别编号 (0-{len(classes)-1}): ",
                    min_val=0,
                    max_val=len(classes) - 1,
                    required=True,
                )

                class_name = classes[target_class]
                print(f"\n选择的类别: {target_class} - {class_name}")

            except Exception as e:
                print(f"\n❌ 读取classes.txt文件失败: {e}")
                self._pause()
                return

            images_dir = self._get_input(
                "图像子目录名 (默认: images): ", default="images"
            )
            labels_dir = self._get_input(
                "标签子目录名 (默认: labels): ", default="labels"
            )

            # 确认操作
            if not self._get_yes_no_input(
                f"\n警告: 此操作将永久删除只包含类别{target_class}({class_name})的文件，是否继续?",
                default=False,
            ):
                print("操作已取消")
                return

            processor = self._get_processor("label")

            print(f"\n正在删除只包含类别{target_class}({class_name})的标签及图像...")
            result = processor.remove_labels_with_only_class(
                dataset_dir,
                target_class=target_class,
                images_subdir=images_dir,
                labels_subdir=labels_dir,
            )

            self._display_result(result)

        except UserInterruptError:
            print(f"\n删除失败: 用户中断操作 (Code: USER_INTERRUPT)")
            print("\n按回车键继续...")
            input()
        except Exception as e:
            print(f"\n删除指定类别标签失败: {e}")

        self._pause()

    def _auto_fix_all_environment(self) -> None:
        """一键检查并修复所有环境问题"""
        try:
            print("\n=== 一键检查并修复所有环境 ===")
            print("正在执行全面环境检查和自动修复...\n")

            # 1. 检查系统环境
            print("📋 步骤 1/5: 检查系统环境...")
            try:
                self._check_system_environment()
                print("✅ 系统环境检查完成")
            except Exception as e:
                print(f"⚠️ 系统环境检查出现问题: {e}")

            # 2. 检查并自动安装Python依赖
            print("\n📦 步骤 2/5: 检查Python依赖...")
            try:
                # 先检查依赖
                missing_deps = []
                if os.path.exists("requirements.txt"):
                    with open("requirements.txt", "r", encoding="utf-8") as f:
                        requirements = f.readlines()

                    # 包名到导入名的映射
                    package_import_map = {
                        "Pillow": "PIL",
                        "opencv-python": "cv2",
                        "opencv-python-headless": "cv2",
                        "PyYAML": "yaml",
                        "pyyaml": "yaml",
                        "scikit-learn": "sklearn",
                        "beautifulsoup4": "bs4",
                        "python-dateutil": "dateutil",
                    }

                    for req in requirements:
                        req = req.strip()
                        if req and not req.startswith("#"):
                            package_name = (
                                req.split("==")[0]
                                .split(">=")[0]
                                .split("<=")[0]
                                .split(">")[0]
                                .split("<")[0]
                                .strip()
                            )
                            import_name = package_import_map.get(
                                package_name, package_name.replace("-", "_").lower()
                            )

                            try:
                                __import__(import_name)
                            except ImportError:
                                missing_deps.append(package_name)

                    if missing_deps:
                        print(f"发现缺失依赖: {', '.join(missing_deps)}")
                        print("正在自动安装缺失依赖...")
                        self._auto_install_dependencies()
                    else:
                        print("✅ 所有Python依赖已满足")
                else:
                    print("⚠️ 未找到requirements.txt文件")
            except Exception as e:
                print(f"⚠️ Python依赖检查出现问题: {e}")

            # 3. 检查并创建配置文件
            print("\n⚙️ 步骤 3/5: 检查配置文件...")
            try:
                config_issues = []

                # 检查主配置文件
                if not os.path.exists("config.json"):
                    config_issues.append("config.json")

                # 检查默认配置文件
                default_config_path = os.path.join("config", "default_config.yaml")
                if not os.path.exists(default_config_path):
                    config_issues.append("default_config.yaml")

                if config_issues:
                    print(f"发现配置文件问题: {', '.join(config_issues)}")
                    print("正在创建缺失的配置文件...")
                    self._check_config_files()
                else:
                    print("✅ 配置文件检查完成")
            except Exception as e:
                print(f"⚠️ 配置文件检查出现问题: {e}")

            # 4. 初始化工作目录
            print("\n📁 步骤 4/5: 检查工作目录...")
            try:
                required_dirs = ["logs", "temp", "config"]
                missing_dirs = []

                for dir_name in required_dirs:
                    if not os.path.exists(dir_name):
                        missing_dirs.append(dir_name)

                if missing_dirs:
                    print(f"发现缺失目录: {', '.join(missing_dirs)}")
                    print("正在创建缺失目录...")
                    self._initialize_workspace()
                else:
                    print("✅ 工作目录检查完成")
            except Exception as e:
                print(f"⚠️ 工作目录检查出现问题: {e}")

            # 5. 最终验证
            print("\n🔍 步骤 5/5: 最终环境验证...")
            try:
                self._comprehensive_environment_check()
                print("✅ 最终验证完成")
            except Exception as e:
                print(f"⚠️ 最终验证出现问题: {e}")

            print("\n🎉 一键环境检查和修复完成!")
            print("\n=== 修复总结 ===")
            print("✅ 系统环境: 已检查")
            print("✅ Python依赖: 已检查并自动安装缺失项")
            print("✅ 配置文件: 已检查并创建缺失项")
            print("✅ 工作目录: 已检查并创建缺失目录")
            print("✅ 最终验证: 已完成")
            print("\n现在您的环境应该已经完全配置好了!")

        except KeyboardInterrupt:
            print("\n❌ 用户中断操作")
            raise KeyboardInterrupt()
        except Exception as e:
            print(f"\n❌ 一键修复过程中出现错误: {e}")
            print("建议手动执行各个检查步骤以获取详细信息")

        self._pause()

    def _is_running_as_exe(self) -> bool:
        """检测是否以exe方式运行

        Returns:
            bool: 如果是exe运行返回True，否则返回False
        """
        return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")

    def _silent_environment_check(self) -> None:
        """静默环境检查（用于exe启动时）"""
        print("正在进行环境检查...")

        try:
            # 1. 静默检查并创建配置文件
            self._silent_check_config_files()

            # 2. 静默初始化工作目录
            self._silent_initialize_workspace()

            # 3. 检查核心模块（静默）
            try:
                from ..config.settings import ConfigManager
                from ..processors import FileProcessor, ImageProcessor, YOLOProcessor
            except ImportError:
                pass  # 静默忽略导入错误

            print("环境检查完成")

        except Exception:
            pass  # 静默忽略所有错误

    def _silent_check_config_files(self) -> None:
        """静默检查配置文件"""
        try:
            config_files = [
                "config.json",
                "config/default_config.yaml",
                "src/config.json",
            ]

            for config_file in config_files:
                config_path = Path(config_file)
                if config_path.exists():
                    print(f"✓ {config_file} 存在")
                    try:
                        if config_file.endswith(".json"):
                            import json

                            with open(config_path, "r", encoding="utf-8") as f:
                                json.load(f)
                            print(f"  - JSON格式有效")
                        elif config_file.endswith(".yaml") or config_file.endswith(
                            ".yml"
                        ):
                            try:
                                import yaml

                                with open(config_path, "r", encoding="utf-8") as f:
                                    yaml.safe_load(f)
                                print(f"  - YAML格式有效")
                            except ImportError:
                                pass  # 静默忽略yaml库缺失
                    except Exception:
                        pass  # 静默忽略格式错误
                else:
                    print(f"❌ {config_file} 不存在")

            # 检查ConfigManager是否能正常加载
            try:
                config_manager = ConfigManager()
                print("✓ ConfigManager初始化成功")
            except Exception:
                pass  # 静默忽略初始化错误

        except Exception:
            pass  # 静默忽略所有错误

    def _silent_initialize_workspace(self) -> None:
        """静默初始化工作目录"""
        try:
            # 创建必要的目录
            directories = ["logs", "temp", "config"]

            for directory in directories:
                dir_path = Path(directory)
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    print(f"✓ 创建目录: {directory}")
                else:
                    print(f"✓ 目录已存在: {directory}")

            # 检查并创建默认配置文件
            default_config_path = Path("config/default_config.yaml")
            if not default_config_path.exists():
                default_config_content = """# 默认配置文件
logging:
  level: INFO
  file: logs/integrated_script.log

processing:
  batch_size: 100
  max_workers: 4

image:
  quality: 95
  format: JPEG
"""
                with open(default_config_path, "w", encoding="utf-8") as f:
                    f.write(default_config_content)
                print(f"✓ 创建默认配置文件: {default_config_path}")
            else:
                print(f"✓ 默认配置文件已存在: {default_config_path}")

        except Exception:
            pass  # 静默忽略所有错误

    def _environment_menu(self) -> None:
        """环境检查与配置菜单"""
        menu = {
            "title": "环境检查与配置",
            "options": [
                ("一键检查并修复所有环境", self._auto_fix_all_environment),
                ("仅检查Python依赖", self._check_python_dependencies),
                ("仅安装缺失依赖", self._auto_install_dependencies),
                ("仅初始化工作目录", self._initialize_workspace),
                ("返回主菜单", None),
            ],
        }

        self.menu_system.show_menu(menu)

    def _config_menu(self) -> None:
        """配置管理菜单"""
        menu = {
            "title": "配置管理",
            "options": [
                ("查看当前配置", self._config_view),
                ("修改配置", self._config_modify),
                ("加载配置文件", self._config_load),
                ("保存配置文件", self._config_save),
                ("重置为默认配置", self._config_reset),
                ("返回主菜单", None),
            ],
        }

        self.menu_system.show_menu(menu)

    def _config_view(self) -> None:
        """查看当前配置"""
        try:
            print("\n=== 当前配置 ===")
            config = self.config_manager.get_all()

            # 定义中文字段映射
            section_names = {
                "version": "版本",
                "debug": "调试模式",
                "log_level": "日志级别",
                "paths": "路径配置",
                "processing": "处理配置",
                "ui": "界面配置",
                "yolo": "YOLO配置",
                "image_processing": "图像处理配置",
                "_metadata": "元数据",
            }

            field_names = {
                # paths 字段
                "input_dir": "输入目录",
                "output_dir": "输出目录",
                "temp_dir": "临时目录",
                "log_dir": "日志目录",
                # processing 字段
                "batch_size": "批处理大小",
                "max_workers": "最大工作线程",
                "timeout": "超时时间(秒)",
                "retry_count": "重试次数",
                # ui 字段
                "language": "语言",
                "theme": "主题",
                "show_progress": "显示进度",
                # yolo 字段
                "image_formats": "图像格式",
                "label_format": "标签格式",
                "classes_file": "类别文件",
                "validate_on_load": "加载时验证",
                # image_processing 字段
                "default_output_format": "默认输出格式",
                "jpeg_quality": "JPEG质量",
                "png_compression": "PNG压缩级别",
                "webp_quality": "WebP质量",
                "quality_analysis": "清晰度分析设置",
                "resize": "尺寸调整设置",
                "auto_orient": "自动旋转",
                "strip_metadata": "移除元数据",
                "parallel_processing": "并行处理",
                "chunk_size": "分块大小",
                # metadata 字段
                "last_updated": "最后更新时间",
                "version": "版本",
            }

            for section, values in config.items():
                # 显示中文节名称
                chinese_section = section_names.get(section, section)
                print(f"\n[{chinese_section}]")

                if isinstance(values, dict):
                    for key, value in values.items():
                        # 显示中文字段名称
                        chinese_key = field_names.get(key, key)

                        # 特殊处理复杂的嵌套配置
                        if key == "quality_analysis" and isinstance(value, dict):
                            print(f"  {chinese_key}:")
                            if "custom_levels" in value:
                                print(f"    清晰度级别:")
                                for level in value["custom_levels"]:
                                    name = level.get("name", "未知")
                                    threshold = level.get("threshold", [0, 0])
                                    print(
                                        f"      - {name}: {threshold[0]}x{threshold[1]}"
                                    )
                        elif key == "resize" and isinstance(value, dict):
                            print(f"  {chinese_key}:")
                            for resize_key, resize_value in value.items():
                                resize_chinese = {
                                    "maintain_aspect_ratio": "保持宽高比",
                                    "interpolation": "插值方法",
                                    "default_size": "默认尺寸",
                                }.get(resize_key, resize_key)
                                print(f"    {resize_chinese}: {resize_value}")
                        else:
                            print(f"  {chinese_key}: {value}")
                else:
                    print(f"  {values}")

        except Exception as e:
            print(f"\n查看配置失败: {e}")

        self._pause()

    def _config_load(self) -> None:
        """加载配置文件"""
        try:
            print("\n=== 加载配置文件 ===")

            config_file = self._get_path_input("请输入配置文件路径: ", must_exist=True)

            # 创建新的ConfigManager实例来加载指定文件
            temp_config = ConfigManager(config_file=config_file, auto_save=False)

            # 将加载的配置更新到当前配置管理器
            loaded_config = temp_config.get_all()
            self.config_manager.update(loaded_config)

            print(f"\n配置文件已加载: {config_file}")

        except UserInterruptError:
            print(f"\n加载配置失败: 用户中断操作 (Code: USER_INTERRUPT)")
            print("\n按回车键继续...")
            input()
        except Exception as e:
            print(f"\n加载配置失败: {e}")

        self._pause()

    def _config_save(self) -> None:
        """保存配置文件"""
        try:
            print("\n=== 保存配置文件 ===")

            config_file = self._get_input("请输入配置文件路径: ", required=True)

            # 创建新的ConfigManager实例来保存到指定文件
            temp_config = ConfigManager(config_file=config_file, auto_save=False)

            # 将当前配置更新到临时配置管理器并保存
            current_config = self.config_manager.get_all()
            temp_config.update(current_config)
            temp_config.save()

            print(f"\n配置已保存到: {config_file}")

        except UserInterruptError:
            print(f"\n保存配置失败: 用户中断操作 (Code: USER_INTERRUPT)")
            print("\n按回车键继续...")
            input()
        except Exception as e:
            print(f"\n保存配置失败: {e}")

        self._pause()

    def _config_reset(self) -> None:
        """重置为默认配置"""
        try:
            print("\n=== 重置为默认配置 ===")

            if self._get_yes_no_input("确认重置为默认配置? (y/n): "):
                self.config_manager.reset()
                print("\n配置已重置为默认值")
            else:
                print("\n操作已取消")

        except Exception as e:
            print(f"\n重置配置失败: {e}")

        self._pause()

    def _config_log_level(self) -> None:
        """设置日志级别"""
        try:
            print("\n=== 设置日志级别 ===")

            # 显示当前日志级别
            current_level = self.config_manager.get("log_level", "INFO")
            print(f"\n当前日志级别: {current_level}")

            # 显示级别选择菜单
            print("\n============================================================")
            print("                        选择日志级别")
            print("============================================================")
            print(" 0. DEBUG   - 详细调试信息")
            print(" 1. INFO    - 一般信息 (推荐)")
            print(" 2. WARNING - 警告信息")
            print(" 3. ERROR   - 错误信息")
            print("============================================================")

            choice = self._get_input("请选择日志级别 (0-3, 默认: 1): ", default="1")

            # 映射选择到日志级别
            level_map = {"0": "DEBUG", "1": "INFO", "2": "WARNING", "3": "ERROR"}

            if choice not in level_map:
                print(f"\n❌ 无效的选择: {choice}")
                print("请选择 0-3 之间的数字")
                return

            log_level = level_map[choice]

            # 更新配置中的日志级别设置
            self.config_manager.set("log_level", log_level)

            # 立即应用到日志系统
            from ..core.logging_config import set_log_level

            set_log_level(log_level)

            print(f"\n✅ 日志级别已设置为: {log_level}")
            print("\n测试日志输出:")

            # 测试不同级别的日志输出
            test_logger = self.logger
            print(f"当前日志记录器级别: {test_logger.level}")
            print(f"根日志记录器级别: {test_logger.parent.level}")

            if log_level == "DEBUG":
                test_logger.debug("🔍 这是DEBUG级别的日志")
                print("DEBUG日志已发送")
            test_logger.info("ℹ️ 这是INFO级别的日志")
            test_logger.warning("⚠️ 这是WARNING级别的日志")
            test_logger.error("❌ 这是ERROR级别的日志")

        except Exception as e:
            print(f"\n设置日志级别失败: {e}")

        self._pause()

    def _config_modify(self) -> None:
        """配置修改菜单"""
        menu = {
            "title": "配置修改",
            "options": [
                ("日志级别设置", self._config_log_level),
                ("路径配置", self._config_modify_paths),
                ("处理配置", self._config_modify_processing),
                ("图像处理配置", self._config_modify_image),
                ("YOLO配置", self._config_modify_yolo),
                ("界面配置", self._config_modify_ui),
                ("返回主菜单", self._return_to_main_menu),
            ],
        }

        self.menu_system.show_menu(menu)

    def _config_modify_paths(self) -> None:
        """修改路径配置"""
        try:
            print("\n=== 路径配置修改 ===")

            # 显示当前路径配置
            paths = self.config_manager.get("paths", {})
            print("\n当前路径配置:")
            print(f"  输入目录: {paths.get('input_dir', '')}")
            print(f"  输出目录: {paths.get('output_dir', '')}")
            print(f"  临时目录: {paths.get('temp_dir', 'temp')}")
            print(f"  日志目录: {paths.get('log_dir', 'logs')}")

            print("\n请输入新的路径配置 (留空保持不变):")

            # 获取新的路径配置
            input_dir = self._get_input(f"输入目录 [{paths.get('input_dir', '')}]: ")
            if input_dir:
                self.config_manager.set("paths.input_dir", input_dir)

            output_dir = self._get_input(f"输出目录 [{paths.get('output_dir', '')}]: ")
            if output_dir:
                self.config_manager.set("paths.output_dir", output_dir)

            temp_dir = self._get_input(f"临时目录 [{paths.get('temp_dir', 'temp')}]: ")
            if temp_dir:
                self.config_manager.set("paths.temp_dir", temp_dir)

            log_dir = self._get_input(f"日志目录 [{paths.get('log_dir', 'logs')}]: ")
            if log_dir:
                self.config_manager.set("paths.log_dir", log_dir)

            print("\n✅ 路径配置已更新")

        except Exception as e:
            print(f"\n修改路径配置失败: {e}")

        self._pause()

    def _config_modify_processing(self) -> None:
        """修改处理配置"""
        try:
            print("\n=== 处理配置修改 ===")

            # 显示当前处理配置
            processing = self.config_manager.get("processing", {})
            print("\n当前处理配置:")
            print(f"  批处理大小: {processing.get('batch_size', 100)}")
            print(f"  最大工作线程: {processing.get('max_workers', 4)}")
            print(f"  超时时间(秒): {processing.get('timeout', 300)}")
            print(f"  重试次数: {processing.get('retry_count', 3)}")

            print("\n请输入新的处理配置 (留空保持不变):")

            # 获取新的处理配置
            batch_size = self._get_int_input(
                f"批处理大小 [{processing.get('batch_size', 100)}]: ",
                min_val=1,
                max_val=1000,
            )
            if batch_size is not None:
                self.config_manager.set("processing.batch_size", batch_size)

            import os

            cpu_count = os.cpu_count() or 4
            max_workers = self._get_int_input(
                f"最大工作线程 [{processing.get('max_workers', 4)}]: ",
                min_val=1,
                max_val=cpu_count,  # 最大值设置为机器的CPU核心数
            )
            if max_workers is not None:
                self.config_manager.set("processing.max_workers", max_workers)

            timeout = self._get_int_input(
                f"超时时间(秒) [{processing.get('timeout', 300)}]: ",
                min_val=30,
                max_val=3600,
            )
            if timeout is not None:
                self.config_manager.set("processing.timeout", timeout)

            retry_count = self._get_int_input(
                f"重试次数 [{processing.get('retry_count', 3)}]: ",
                min_val=0,
                max_val=10,
            )
            if retry_count is not None:
                self.config_manager.set("processing.retry_count", retry_count)

            print("\n✅ 处理配置已更新")

        except Exception as e:
            print(f"\n修改处理配置失败: {e}")

        self._pause()

    def _config_modify_image(self) -> None:
        """修改图像处理配置"""
        try:
            print("\n=== 图像处理配置修改 ===")

            # 显示当前图像处理配置
            image_config = self.config_manager.get("image_processing", {})
            print("\n当前图像处理配置:")
            print(f"  默认输出格式: {image_config.get('default_output_format', 'jpg')}")
            print(f"  JPEG质量: {image_config.get('jpeg_quality', 95)}")
            print(f"  PNG压缩级别: {image_config.get('png_compression', 6)}")
            print(f"  WebP质量: {image_config.get('webp_quality', 90)}")
            print(f"  自动旋转: {image_config.get('auto_orient', True)}")
            print(f"  移除元数据: {image_config.get('strip_metadata', False)}")
            print(f"  并行处理: {image_config.get('parallel_processing', True)}")
            print(f"  分块大小: {image_config.get('chunk_size', 50)}")

            print("\n请输入新的图像处理配置 (留空保持不变):")

            # 获取新的图像处理配置
            output_format = self._get_input(
                f"默认输出格式 (jpg/png/webp) [{image_config.get('default_output_format', 'jpg')}]: "
            )
            if output_format and output_format.lower() in [
                "jpg",
                "jpeg",
                "png",
                "webp",
            ]:
                self.config_manager.set(
                    "image_processing.default_output_format", output_format.lower()
                )

            jpeg_quality = self._get_int_input(
                f"JPEG质量 (1-100) [{image_config.get('jpeg_quality', 95)}]: ",
                min_val=1,
                max_val=100,
            )
            if jpeg_quality is not None:
                self.config_manager.set("image_processing.jpeg_quality", jpeg_quality)

            png_compression = self._get_int_input(
                f"PNG压缩级别 (0-9) [{image_config.get('png_compression', 6)}]: ",
                min_val=0,
                max_val=9,
            )
            if png_compression is not None:
                self.config_manager.set(
                    "image_processing.png_compression", png_compression
                )

            webp_quality = self._get_int_input(
                f"WebP质量 (1-100) [{image_config.get('webp_quality', 90)}]: ",
                min_val=1,
                max_val=100,
            )
            if webp_quality is not None:
                self.config_manager.set("image_processing.webp_quality", webp_quality)

            auto_orient = self._get_yes_no_input(
                f"自动旋转 [{image_config.get('auto_orient', True)}]: "
            )
            if auto_orient is not None:
                self.config_manager.set("image_processing.auto_orient", auto_orient)

            strip_metadata = self._get_yes_no_input(
                f"移除元数据 [{image_config.get('strip_metadata', False)}]: "
            )
            if strip_metadata is not None:
                self.config_manager.set(
                    "image_processing.strip_metadata", strip_metadata
                )

            parallel_processing = self._get_yes_no_input(
                f"并行处理 [{image_config.get('parallel_processing', True)}]: "
            )
            if parallel_processing is not None:
                self.config_manager.set(
                    "image_processing.parallel_processing", parallel_processing
                )

            chunk_size = self._get_int_input(
                f"分块大小 [{image_config.get('chunk_size', 50)}]: ",
                min_val=1,
                max_val=1000,
            )
            if chunk_size is not None:
                self.config_manager.set("image_processing.chunk_size", chunk_size)

            print("\n✅ 图像处理配置已更新")

        except Exception as e:
            print(f"\n修改图像处理配置失败: {e}")

        self._pause()

    def _config_modify_yolo(self) -> None:
        """修改YOLO配置"""
        try:
            print("\n=== YOLO配置修改 ===")

            # 显示当前YOLO配置
            yolo_config = self.config_manager.get("yolo", {})
            print("\n当前YOLO配置:")
            print(f"  图像格式: {yolo_config.get('image_formats', [])}")
            print(f"  标签格式: {yolo_config.get('label_format', '.txt')}")
            print(f"  类别文件: {yolo_config.get('classes_file', 'classes.txt')}")
            print(f"  加载时验证: {yolo_config.get('validate_on_load', True)}")

            print("\n请输入新的YOLO配置 (留空保持不变):")

            # 获取新的YOLO配置
            label_format = self._get_input(
                f"标签格式 [{yolo_config.get('label_format', '.txt')}]: "
            )
            if label_format:
                if not label_format.startswith("."):
                    label_format = "." + label_format
                self.config_manager.set("yolo.label_format", label_format)

            classes_file = self._get_input(
                f"类别文件 [{yolo_config.get('classes_file', 'classes.txt')}]: "
            )
            if classes_file:
                self.config_manager.set("yolo.classes_file", classes_file)

            validate_on_load = self._get_yes_no_input(
                f"加载时验证 [{yolo_config.get('validate_on_load', True)}]: "
            )
            if validate_on_load is not None:
                self.config_manager.set("yolo.validate_on_load", validate_on_load)

            print("\n✅ YOLO配置已更新")

        except Exception as e:
            print(f"\n修改YOLO配置失败: {e}")

        self._pause()

    def _config_modify_ui(self) -> None:
        """修改界面配置"""
        try:
            print("\n=== 界面配置修改 ===")

            # 显示当前界面配置
            ui_config = self.config_manager.get("ui", {})
            print("\n当前界面配置:")
            print(f"  语言: {ui_config.get('language', 'zh_CN')}")
            print(f"  主题: {ui_config.get('theme', 'default')}")
            print(f"  显示进度: {ui_config.get('show_progress', True)}")

            print("\n请输入新的界面配置 (留空保持不变):")

            # 获取新的界面配置
            language = self._get_input(
                f"语言 (zh_CN/en_US) [{ui_config.get('language', 'zh_CN')}]: "
            )
            if language and language in ["zh_CN", "en_US"]:
                self.config_manager.set("ui.language", language)

            theme = self._get_input(
                f"主题 (default/dark/light) [{ui_config.get('theme', 'default')}]: "
            )
            if theme and theme in ["default", "dark", "light"]:
                self.config_manager.set("ui.theme", theme)

            show_progress = self._get_yes_no_input(
                f"显示进度 [{ui_config.get('show_progress', True)}]: "
            )
            if show_progress is not None:
                self.config_manager.set("ui.show_progress", show_progress)

            print("\n✅ 界面配置已更新")

        except Exception as e:
            print(f"\n修改界面配置失败: {e}")

        self._pause()

    def _get_int_input(
        self, prompt: str, min_val: int = None, max_val: int = None
    ) -> int:
        """获取整数输入"""
        while True:
            try:
                user_input = self._get_input(prompt).strip()
                if not user_input:
                    return None

                value = int(user_input)

                if min_val is not None and value < min_val:
                    print(f"值必须大于等于 {min_val}")
                    continue

                if max_val is not None and value > max_val:
                    print(f"值必须小于等于 {max_val}")
                    continue

                return value

            except ValueError:
                print("请输入有效的数字")
            except KeyboardInterrupt:
                return None

    def _get_yes_no_input(self, prompt: str) -> bool:
        """获取是否输入"""
        while True:
            try:
                user_input = self._get_input(prompt).strip().lower()
                if not user_input:
                    return None

                if user_input in ["y", "yes", "是", "true", "1"]:
                    return True
                elif user_input in ["n", "no", "否", "false", "0"]:
                    return False
                else:
                    print("请输入 y/yes/是/true/1 或 n/no/否/false/0")

            except KeyboardInterrupt:
                return None

    def _return_to_main_menu(self) -> None:
        """返回主菜单"""
        # 清空菜单栈，直接返回主菜单
        self.menu_system.menu_stack.clear()
        self.menu_system.current_menu = self.menu_system.main_menu

    def _exit_program(self) -> None:
        """退出程序"""
        print("\n感谢使用集成脚本工具！")
        sys.exit(0)

    # 输入辅助方法
    def _get_input(
        self,
        prompt: str,
        default: str = None,
        required: bool = False,
        allow_space_empty: bool = False,
    ) -> str:
        """获取用户输入"""
        while True:
            try:
                if default:
                    raw_input = input(f"{prompt}[{default}] ")
                    if raw_input.strip() == "":
                        if allow_space_empty and raw_input != "":
                            return ""
                        return default
                    user_input = raw_input.strip()
                else:
                    raw_input = input(prompt)
                    if allow_space_empty and raw_input != "" and raw_input.strip() == "":
                        return ""
                    user_input = raw_input.strip()

                if required and not user_input:
                    print("此项为必填项，请重新输入")
                    continue

                return user_input

            except KeyboardInterrupt:
                raise UserInterruptError("用户中断操作")
            except EOFError:
                raise UserInterruptError("输入结束")

    def _get_yes_no_input(self, prompt: str, default: bool = None) -> bool:
        """获取是/否输入"""
        # 如果有默认值，在提示中显示
        if default is not None:
            default_text = "y" if default else "n"
            display_prompt = f"{prompt} (默认: {default_text}): "
        else:
            display_prompt = prompt

        while True:
            try:
                response = self._get_input(display_prompt).strip().lower()

                # 如果输入为空且有默认值，使用默认值
                if not response and default is not None:
                    return default

                if response in ["y", "yes", "是", "1", "true"]:
                    return True
                elif response in ["n", "no", "否", "0", "false"]:
                    return False
                else:
                    print("请输入 y 或 n")

            except KeyboardInterrupt:
                raise UserInterruptError("用户中断操作")
            except EOFError:
                raise UserInterruptError("输入结束")

    def _get_path_input(
        self, prompt: str, must_exist: bool = False, must_be_dir: bool = False
    ) -> str:
        """获取路径输入"""
        while True:
            try:
                path_str = self._get_input(prompt, required=True)

                # 处理Windows路径格式
                # 移除可能的引号
                path_str = path_str.strip("\"'")

                # 处理反斜杠转义问题 - 将双反斜杠转换为单反斜杠
                if "\\\\" in path_str:
                    path_str = path_str.replace("\\\\", "\\")

                # 规范化路径分隔符
                path_str = path_str.replace("/", os.sep)

                # 展开用户目录和环境变量
                path_str = os.path.expanduser(path_str)
                path_str = os.path.expandvars(path_str)

                path = Path(path_str)

                # Windows路径特殊处理：检查是否为Windows绝对路径格式
                is_windows_absolute = (
                    len(path_str) >= 3
                    and path_str[1:3] == ":\\"
                    and path_str[0].isalpha()
                ) or (
                    len(path_str) >= 3
                    and path_str[1:3] == ":/"
                    and path_str[0].isalpha()
                )

                # 如果不是绝对路径且不是Windows绝对路径格式，则解析为相对路径
                if not path.is_absolute() and not is_windows_absolute:
                    path = path.resolve()
                elif is_windows_absolute and not path.is_absolute():
                    # 强制创建Windows绝对路径
                    path = Path(path_str)

                if must_exist and not path.exists():
                    print(f"路径不存在: {path}")
                    print(f"提示: 请确保路径格式正确")
                    print(f"Windows路径示例: C:\\Users\\username\\folder")
                    print(f"或者使用正斜杠: C:/Users/username/folder")
                    continue

                if must_be_dir and path.exists() and not path.is_dir():
                    print(f"路径不是目录: {path}")
                    continue

                return str(path)

            except UserInterruptError:
                raise
            except Exception as e:
                print(f"无效路径: {e}")
                print(f"提示: 请检查路径格式")
                print(f"Windows路径示例: C:\\Users\\username\\folder")
                print(f"或者使用正斜杠: C:/Users/username/folder")

    def _get_int_input(
        self,
        prompt: str,
        default: int = None,
        required: bool = False,
        min_val: int = None,
        max_val: int = None,
    ) -> int:
        """获取整数输入"""
        while True:
            try:
                input_str = self._get_input(
                    prompt, str(default) if default is not None else None, required
                )

                if not input_str and default is not None:
                    return default

                value = int(input_str)

                if min_val is not None and value < min_val:
                    print(f"值不能小于 {min_val}")
                    continue

                if max_val is not None and value > max_val:
                    print(f"值不能大于 {max_val}")
                    continue

                return value

            except ValueError:
                print("请输入有效的整数")

    def _get_float_input(
        self,
        prompt: str,
        default: float = None,
        required: bool = False,
        min_val: float = None,
        max_val: float = None,
    ) -> float:
        """获取浮点数输入"""
        while True:
            try:
                input_str = self._get_input(
                    prompt, str(default) if default is not None else None, required
                )

                if not input_str and default is not None:
                    return default

                value = float(input_str)

                if min_val is not None and value < min_val:
                    print(f"值不能小于 {min_val}")
                    continue

                if max_val is not None and value > max_val:
                    print(f"值不能大于 {max_val}")
                    continue

                return value

            except ValueError:
                print("请输入有效的数字")

    def _parse_size(self, size_str: str) -> tuple:
        """解析尺寸字符串"""
        if "x" in size_str.lower():
            parts = size_str.lower().split("x")
            if len(parts) == 2:
                return (int(parts[0]), int(parts[1]))
        else:
            size = int(size_str)
            return (size, size)

        raise ValueError(f"无效的尺寸格式: {size_str}")

    def _display_result(self, result: Dict[str, Any]) -> None:
        """显示结果"""
        # 字段名中英文映射
        field_translations = {
            "total_images": "总图像数",
            "total_labels": "总标签数",
            "matched_pairs": "匹配对数",
            "orphaned_images": "孤立图像",
            "orphaned_labels": "孤立标签",
            "invalid_labels": "无效标签",
            "empty_labels": "空标签",
            "dataset_path": "数据集路径",
            "is_valid": "数据集有效性",
            "has_classes_file": "包含类别文件",
            "num_classes": "类别数量",
            "class_names": "类别名称",
            "total_processed": "总处理文件数",
            "invalid_removed": "无效文件数",
            "final_count": "有效文件数",
            "input_path": "输入路径",
            "output_path": "输出路径",
            "project_name": "项目名称",
            "valid": "数据集有效",
            "classes_file": "类别文件路径",
            # 图像处理相关字段
            "total_files": "总文件数",
            "converted_count": "转换成功数",
            "failed_count": "重命名失败数",
            "target_class_only_labels": "仅包含目标类别的标签数",
            "removed_images": "删除的图像数",
            "removed_labels": "删除的标签数",
            "dataset_dir": "数据集目录",
            "images_dir": "图像目录",
            "labels_dir": "标签目录",
            "target_class": "目标类别",
            "total_input_size": "输入总大小",
            "total_output_size": "输出总大小",
            "total_input_size_formatted": "输入总大小",
            "total_output_size_formatted": "输出总大小",
            "overall_compression_ratio": "总体压缩比",
            "input_dir": "输入目录",
            "output_dir": "输出目录",
            "target_format": "目标格式",
            "quality": "图像质量",
            "resized_count": "调整成功数",
            "target_size": "目标尺寸",
            "maintain_aspect_ratio": "保持宽高比",
            "copied_count": "复制成功数",
            "moved_count": "移动成功数",
            "deleted_count": "删除成功数",
            "renamed_count": "重命名成功数",
            # 重命名功能相关字段
            "total_pairs": "总文件对数",
            "rename_pattern": "重命名模式",
            "shuffle_order": "打乱顺序",
            "preview_only": "仅预览",
            "target_dir": "目标目录",
            "prefix": "文件前缀",
            "digits": "数字位数",
            # 图像信息相关字段
            "file_path": "文件路径",
            "file_size": "文件大小(字节)",
            "file_size_formatted": "文件大小",
            "format": "图像格式",
            "width": "宽度",
            "height": "高度",
            "aspect_ratio": "宽高比",
            "total_pixels": "总像素数",
            "mode": "颜色模式",
            "has_transparency": "包含透明度",
        }

        print("\n" + "=" * 50)

        # 检查是否为统计信息结果
        is_statistics_result = (
            "statistics" in result
            and isinstance(result["statistics"], dict)
            and "dataset_path" in result["statistics"]
            and "is_valid" in result["statistics"]
        )

        if is_statistics_result:
            # 这是统计信息结果
            if result["statistics"].get("is_valid", False):
                print("✓ 数据集验证通过")
            else:
                print("⚠ 数据集存在问题")
        elif result.get("success", False):
            print("✓ 操作成功完成")
        else:
            # 对于统计信息结果，不显示操作失败
            if not is_statistics_result:
                print("✗ 操作失败")
                if "message" in result:
                    print(f"错误信息: {result['message']}")

        # 显示统计信息
        if "statistics" in result:
            print("\n统计信息:")
            stats = result["statistics"]
            for key, value in stats.items():
                chinese_key = field_translations.get(key, key)
                # 特殊处理数据集路径，如果路径被调整则显示提示
                if key == "dataset_path":
                    print(f"  {chinese_key}: {value}")
                    # 检查是否路径被调整（通过比较original_path和dataset_path）
                    original_path = stats.get("original_path")
                    if original_path and str(original_path) != str(value):
                        print(f"    💡 已自动调整为数据集根目录")
                elif key != "original_path":  # 不显示original_path字段
                    print(f"  {chinese_key}: {value}")

        # 显示其他重要信息
        for key, value in result.items():
            if key not in ["success", "statistics", "message"] and not key.endswith(
                "_list"
            ):
                if isinstance(value, (str, int, float, bool)):
                    chinese_key = field_translations.get(key, key)
                    # 对布尔值进行中文化
                    if isinstance(value, bool):
                        value_text = "是" if value else "否"
                    else:
                        value_text = value
                    print(f"{chinese_key}: {value_text}")

        # 显示失败文件详情
        if "failed_pairs" in result and result["failed_pairs"]:
            print("\n失败文件详情:")
            for i, failed_item in enumerate(result["failed_pairs"], 1):
                print(f"  {i}. 图像文件: {failed_item.get('img_file', 'N/A')}")
                print(f"     标签文件: {failed_item.get('label_file', 'N/A')}")
                print(f"     失败原因: {failed_item.get('error', 'N/A')}")
                print(f"     失败阶段: {failed_item.get('action', 'N/A')}")
                print()

        print("=" * 50)

    def _check_system_environment(self) -> None:
        """检查系统环境"""
        print("\n" + "=" * 50)
        print("系统环境检查")
        print("=" * 50)

        try:
            import platform
            import sys

            print(f"操作系统: {platform.system()} {platform.release()}")
            print(f"Python版本: {sys.version}")
            print(f"Python路径: {sys.executable}")
            print(f"当前工作目录: {os.getcwd()}")

            # 检查虚拟环境
            if hasattr(sys, "real_prefix") or (
                hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
            ):
                print("✓ 当前运行在虚拟环境中")
            else:
                print("⚠ 当前未运行在虚拟环境中，建议使用虚拟环境")

            print("\n系统环境检查完成")

        except Exception as e:
            print(f"系统环境检查失败: {e}")

        self._pause()

    def _check_python_dependencies(self) -> None:
        """检查Python依赖"""
        print("\n" + "=" * 50)
        print("Python依赖检查")
        print("=" * 50)

        # 读取requirements.txt
        requirements_file = Path("requirements.txt")
        if not requirements_file.exists():
            print("❌ requirements.txt文件不存在")
            self._pause()
            return

        try:
            with open(requirements_file, "r", encoding="utf-8") as f:
                requirements = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]

            missing_packages = []
            installed_packages = []

            # 包名到导入名的映射
            package_import_map = {
                "Pillow": "PIL",
                "opencv-python": "cv2",
                "opencv-python-headless": "cv2",
                "PyYAML": "yaml",
                "pyyaml": "yaml",
                "scikit-learn": "sklearn",
                "beautifulsoup4": "bs4",
                "python-dateutil": "dateutil",
            }

            for requirement in requirements:
                package_name = (
                    requirement.split("==")[0]
                    .split(">=")[0]
                    .split("<=")[0]
                    .split(">")[0]
                    .split("<")[0]
                    .strip()
                )

                # 获取实际的导入名
                import_name = package_import_map.get(
                    package_name, package_name.replace("-", "_").lower()
                )

                try:
                    __import__(import_name)
                    installed_packages.append(package_name)
                    print(f"✓ {package_name}")
                except ImportError:
                    missing_packages.append(requirement)
                    print(f"❌ {package_name}")

            print(f"\n已安装: {len(installed_packages)}个包")
            print(f"缺失: {len(missing_packages)}个包")

            if missing_packages:
                print("\n缺失的包:")
                for pkg in missing_packages:
                    print(f"  - {pkg}")
            else:
                print("\n✓ 所有依赖包都已安装")

        except Exception as e:
            print(f"依赖检查失败: {e}")

        self._pause()

    def _yolo_convert_to_ctds(self) -> None:
        """将YOLO数据集重新封装为CTDS"""
        try:
            print("\n=== YOLO数据转CTDS格式 ===")
            print("将现有 YOLO 数据集复制到 CTDS 结构（obj.names + obj_train_data）")
            dataset_path = self._get_path_input(
                "请输入YOLO数据集路径: ", must_exist=True
            )
            output_path = input(
                "\n请输入CTDS输出目录（留空使用默认）："
            ).strip()
            output_path = output_path or None

            processor = self._get_processor("yolo")
            print("\n正在转换数据集...")
            result = processor.convert_yolo_to_ctds_dataset(
                dataset_path, output_path=output_path
            )

            print("\n转换结果：")
            if result.get("success"):
                print(f"✅ 输出路径: {result.get('output_path')}")
                stats = result.get("statistics", {})
                print(f"  - 标签数: {stats.get('total_labels', 0)}")
                print(f"  - 复制标签: {stats.get('labels_copied', 0)}")
                print(f"  - 复制图像: {stats.get('images_copied', 0)}")
                missing = stats.get("missing_images", 0)
                print(f"  - 缺失图像: {missing}")
                if missing:
                    missing_list = result.get("missing_images", [])
                    print(
                        f"  - 未找到图像的标签: {', '.join(missing_list[:5])}"
                        + (" ..." if len(missing_list) > 5 else "")
                    )
            else:
                print("❌ 转换失败")
                if result.get("error"):
                    print(f"错误信息: {result['error']}")

        except Exception as e:
            print(f"\nYOLO数据转CTDS失败: {e}")

        self._pause()

    def _auto_install_dependencies(self) -> None:
        """自动安装缺失依赖"""
        print("\n" + "=" * 50)
        print("自动安装缺失依赖")
        print("=" * 50)

        requirements_file = Path("requirements.txt")
        if not requirements_file.exists():
            print("❌ requirements.txt文件不存在")
            self._pause()
            return

        try:
            import subprocess
            import sys

            print("正在检查并安装缺失的依赖...")

            # 使用pip安装requirements.txt中的所有依赖
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print("✓ 依赖安装成功")
                print("\n安装输出:")
                print(result.stdout)
            else:
                print("❌ 依赖安装失败")
                print("\n错误信息:")
                print(result.stderr)

        except Exception as e:
            print(f"自动安装依赖失败: {e}")

        self._pause()

    def _check_config_files(self) -> None:
        """检查配置文件"""
        print("\n" + "=" * 50)
        print("配置文件检查")
        print("=" * 50)

        config_files = ["config.json", "config/default_config.yaml", "src/config.json"]

        for config_file in config_files:
            config_path = Path(config_file)
            if config_path.exists():
                print(f"✓ {config_file} 存在")
                try:
                    if config_file.endswith(".json"):
                        import json

                        with open(config_path, "r", encoding="utf-8") as f:
                            json.load(f)
                        print(f"  - JSON格式有效")
                    elif config_file.endswith(".yaml") or config_file.endswith(".yml"):
                        try:
                            import yaml

                            with open(config_path, "r", encoding="utf-8") as f:
                                yaml.safe_load(f)
                            print(f"  - YAML格式有效")
                        except ImportError:
                            print(f"  - 无法验证YAML格式（缺少yaml库）")
                except Exception as e:
                    print(f"  - ❌ 配置文件格式错误: {e}")
            else:
                print(f"❌ {config_file} 不存在")

        # 检查ConfigManager是否能正常加载
        try:
            config_manager = ConfigManager()
            print("\n✓ ConfigManager初始化成功")
        except Exception as e:
            print(f"\n❌ ConfigManager初始化失败: {e}")

        self._pause()

    def _initialize_workspace(self) -> None:
        """初始化工作目录"""
        print("\n" + "=" * 50)
        print("初始化工作目录")
        print("=" * 50)

        try:
            # 创建必要的目录
            directories = ["logs", "temp", "config"]

            for directory in directories:
                dir_path = Path(directory)
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    print(f"✓ 创建目录: {directory}")
                else:
                    print(f"✓ 目录已存在: {directory}")

            # 检查并创建默认配置文件
            default_config_path = Path("config/default_config.yaml")
            if not default_config_path.exists():
                default_config_content = """# 默认配置文件
logging:
  level: INFO
  file: logs/integrated_script.log

processing:
  batch_size: 100
  max_workers: 4

image:
  quality: 95
  format: JPEG
"""
                with open(default_config_path, "w", encoding="utf-8") as f:
                    f.write(default_config_content)
                print(f"✓ 创建默认配置文件: {default_config_path}")
            else:
                print(f"✓ 默认配置文件已存在: {default_config_path}")

            print("\n工作目录初始化完成")

        except Exception as e:
            print(f"工作目录初始化失败: {e}")

        self._pause()

    def _comprehensive_environment_check(self) -> None:
        """完整环境检查"""
        print("\n" + "=" * 50)
        print("完整环境检查")
        print("=" * 50)

        checks = [
            ("系统环境", self._check_system_info),
            ("Python依赖", self._check_dependencies_info),
            ("配置文件", self._check_config_info),
            ("工作目录", self._check_workspace_info),
            ("核心模块", self._check_core_modules),
        ]

        results = []

        for check_name, check_func in checks:
            print(f"\n检查 {check_name}...")
            try:
                result = check_func()
                if result:
                    print(f"✓ {check_name} 检查通过")
                    results.append(True)
                else:
                    print(f"❌ {check_name} 检查失败")
                    results.append(False)
            except Exception as e:
                print(f"❌ {check_name} 检查出错: {e}")
                results.append(False)

        # 显示总结
        print("\n" + "=" * 50)
        print("环境检查总结")
        print("=" * 50)

        passed = sum(results)
        total = len(results)

        for i, (check_name, _) in enumerate(checks):
            status = "✓" if results[i] else "❌"
            print(f"{status} {check_name}")

        print(f"\n通过: {passed}/{total}")

        if passed == total:
            print("\n🎉 所有环境检查都通过了！")
        else:
            print(f"\n⚠ 有 {total - passed} 项检查未通过，建议修复后再使用")

        self._pause()

    def _check_system_info(self) -> bool:
        """检查系统信息"""
        try:
            import platform
            import sys

            # 基本检查
            if sys.version_info < (3, 8):
                return False

            return True
        except Exception:
            return False

    def _check_dependencies_info(self) -> bool:
        """检查依赖信息"""
        try:
            requirements_file = Path("requirements.txt")
            if not requirements_file.exists():
                return False

            with open(requirements_file, "r", encoding="utf-8") as f:
                requirements = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]

            # 包名到导入名的映射
            package_import_map = {
                "Pillow": "PIL",
                "opencv-python": "cv2",
                "opencv-python-headless": "cv2",
                "PyYAML": "yaml",
                "pyyaml": "yaml",
                "scikit-learn": "sklearn",
                "beautifulsoup4": "bs4",
                "python-dateutil": "dateutil",
            }

            for requirement in requirements:
                package_name = (
                    requirement.split("==")[0]
                    .split(">=")[0]
                    .split("<=")[0]
                    .split(">")[0]
                    .split("<")[0]
                    .strip()
                )

                # 获取实际的导入名
                import_name = package_import_map.get(
                    package_name, package_name.replace("-", "_").lower()
                )

                try:
                    __import__(import_name)
                except ImportError:
                    return False

            return True
        except Exception:
            return False

    def _check_config_info(self) -> bool:
        """检查配置信息"""
        try:
            config_manager = ConfigManager()
            return True
        except Exception:
            return False

    def _check_workspace_info(self) -> bool:
        """检查工作空间信息"""
        try:
            required_dirs = ["logs", "temp"]
            for directory in required_dirs:
                if not Path(directory).exists():
                    return False
            return True
        except Exception:
            return False

    def _check_core_modules(self) -> bool:
        """检查核心模块"""
        try:
            from ..config.settings import ConfigManager
            from ..processors import FileProcessor, ImageProcessor, YOLOProcessor

            return True
        except Exception:
            return False

    def _pause(self) -> None:
        """暂停等待用户按键"""
        try:
            input("\n按回车键继续...")
        except KeyboardInterrupt:
            pass
        except EOFError:
            pass

    def run(self) -> None:
        """运行交互式界面"""
        try:
            # 设置日志
            setup_logging(log_level="INFO")

            print("\n" + "=" * 60)
            print("欢迎使用集成脚本工具 - 交互式界面")
            print("版本: 1.0.0")
            print("=" * 60)

            # 如果是exe环境，自动进行静默环境检查
            if self._is_running_as_exe():
                print("\n🔧 正在进行环境检查...")
                try:
                    self._silent_environment_check()
                    print("✅ 环境检查完成")
                except Exception as e:
                    print(f"⚠️ 环境检查出现问题: {e}")
                print()

            # 显示主菜单
            self.menu_system.run()

        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
        except Exception as e:
            print(f"\n\n程序运行出错: {e}")
        finally:
            print("\n感谢使用集成脚本工具！")


def main():
    """主入口函数"""
    interface = InteractiveInterface()
    interface.run()


if __name__ == "__main__":
    main()
