#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cli.py

命令行界面

提供命令行参数解析和处理功能。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config.exceptions import ConfigurationError, ProcessingError
from ..config.settings import ConfigManager
from ..core.logging_config import get_logger, setup_logging
from ..processors import (
    DatasetProcessor,
    FileProcessor,
    ImageProcessor,
    LabelProcessor,
    YOLOProcessor,
)


class CLIInterface:
    """命令行界面

    提供命令行参数解析和处理功能。

    Attributes:
        parser (argparse.ArgumentParser): 参数解析器
        config_manager (ConfigManager): 配置管理器
        logger: 日志记录器
    """

    def __init__(self):
        """初始化命令行界面"""
        self.parser = self._create_parser()
        self.config_manager = ConfigManager()
        self.logger = None  # 将在setup_logging后初始化

        # 处理器映射
        self.processors = {
            "yolo": YOLOProcessor,
            "image": ImageProcessor,
            "file": FileProcessor,
            "dataset": DatasetProcessor,
            "label": LabelProcessor,
        }

    def _create_parser(self) -> argparse.ArgumentParser:
        """创建参数解析器"""
        parser = argparse.ArgumentParser(
            prog="integrated_script",
            description="集成脚本工具 - 提供多种数据处理功能",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例用法:
  %(prog)s yolo validate /path/to/dataset
  %(prog)s image convert /path/to/images --format jpg
  %(prog)s file organize /path/to/files --by-extension
  %(prog)s dataset split /path/to/dataset --train-ratio 0.8
  %(prog)s label create-empty /path/to/images

更多信息请访问: https://github.com/your-repo/integrated_script
            """,
        )

        # 全局参数
        parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

        parser.add_argument("--config", type=str, help="配置文件路径")

        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            default="INFO",
            help="日志级别 (默认: INFO)",
        )

        parser.add_argument("--log-file", type=str, help="日志文件路径")

        parser.add_argument(
            "--quiet", action="store_true", help="静默模式，只输出错误信息"
        )

        parser.add_argument(
            "--verbose", action="store_true", help="详细模式，输出调试信息"
        )

        return parser

    def setup_parsers(self, subparsers):
        """设置子命令解析器

        Args:
            subparsers: 子命令解析器组
        """
        # YOLO处理器命令
        self._add_yolo_commands(subparsers)

        # 图像处理器命令
        self._add_image_commands(subparsers)

        # 文件处理器命令
        self._add_file_commands(subparsers)

        # 标签处理器命令
        self._add_label_commands(subparsers)

    def _add_yolo_commands(self, subparsers):
        """添加YOLO处理器命令"""
        yolo_parser = subparsers.add_parser(
            "yolo", help="YOLO数据集处理", description="YOLO数据集验证、清理和转换功能"
        )

        yolo_subparsers = yolo_parser.add_subparsers(
            dest="yolo_action", help="YOLO操作"
        )

        # 验证数据集
        validate_parser = yolo_subparsers.add_parser("validate", help="验证YOLO数据集")
        validate_parser.add_argument("dataset_path", help="数据集路径")
        validate_parser.add_argument(
            "--images-dir", default="images", help="图像目录名 (默认: images)"
        )
        validate_parser.add_argument(
            "--labels-dir", default="labels", help="标签目录名 (默认: labels)"
        )

        # 删除只包含类别0的数据
        remove_class0_parser = yolo_subparsers.add_parser(
            "remove-class0", help="删除只包含类别0的标签和图像"
        )
        remove_class0_parser.add_argument("dataset_path", help="数据集路径")
        remove_class0_parser.add_argument(
            "--images-dir", default="images", help="图像目录名 (默认: images)"
        )
        remove_class0_parser.add_argument(
            "--labels-dir", default="labels", help="标签目录名 (默认: labels)"
        )

        # CTDS数据处理
        ctds_parser = yolo_subparsers.add_parser(
            "process-ctds", help="CTDS数据转YOLO格式"
        )
        ctds_parser.add_argument("dataset_path", help="CTDS数据集路径")
        ctds_parser.add_argument(
            "--project-name", help="处理后的项目名称（为空时自动生成）"
        )
        ctds_parser.add_argument(
            "--keep-empty-labels",
            action="store_true",
            help="保留空标签文件（默认不保留）",
        )

        # X-label数据处理
        xlabel_parser = yolo_subparsers.add_parser(
            "process-xlabel", help="X-label数据转YOLO格式"
        )
        xlabel_parser.add_argument("dataset_path", help="X-label数据集路径")
        xlabel_parser.add_argument(
            "--output-path", help="输出目录（为空时自动生成）"
        )
        xlabel_parser.add_argument(
            "--class-order",
            help="类别顺序，使用逗号分隔，例如: dam,bridge,spillway",
        )

        # X-label分割数据处理
        xlabel_seg_parser = yolo_subparsers.add_parser(
            "process-xlabel-seg", help="X-label数据转YOLO-分割格式"
        )
        xlabel_seg_parser.add_argument("dataset_path", help="X-label数据集路径")
        xlabel_seg_parser.add_argument(
            "--output-path", help="输出目录（为空时自动生成）"
        )
        xlabel_seg_parser.add_argument(
            "--class-order",
            help="类别顺序，使用逗号分隔，例如: dam,bridge,spillway",
        )
        xlabel_seg_parser.add_argument(
            "--class-map",
            help="类别英文映射JSON文件（键为中文/原始名称，值为英文名称）",
        )

    def _add_image_commands(self, subparsers):
        """添加图像处理器命令"""
        image_parser = subparsers.add_parser(
            "image", help="图像处理", description="图像格式转换、尺寸调整等功能"
        )

        image_subparsers = image_parser.add_subparsers(
            dest="image_action", help="图像操作"
        )

        # 格式转换
        convert_parser = image_subparsers.add_parser("convert", help="转换图像格式")
        convert_parser.add_argument("input_path", help="输入路径（文件或目录）")
        convert_parser.add_argument("--output-path", help="输出路径")
        convert_parser.add_argument(
            "--format",
            choices=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
            required=True,
            help="目标格式",
        )
        convert_parser.add_argument(
            "--quality", type=int, default=95, help="JPEG质量 (1-100, 默认: 95)"
        )
        convert_parser.add_argument(
            "--recursive", action="store_true", help="递归处理子目录"
        )

        # 尺寸调整
        resize_parser = image_subparsers.add_parser("resize", help="调整图像尺寸")
        resize_parser.add_argument("input_path", help="输入路径（文件或目录）")
        resize_parser.add_argument("--output-path", help="输出路径")
        resize_parser.add_argument(
            "--size", required=True, help="目标尺寸 (WxH 或 单个数字)"
        )
        resize_parser.add_argument(
            "--keep-aspect", action="store_true", help="保持宽高比"
        )
        resize_parser.add_argument(
            "--recursive", action="store_true", help="递归处理子目录"
        )

        # 图像压缩
        compress_parser = image_subparsers.add_parser("compress", help="压缩图像")
        compress_parser.add_argument("input_dir", help="输入目录")
        compress_parser.add_argument(
            "--output-dir", help="输出目录（默认为输入目录下的compressed子目录）"
        )
        compress_parser.add_argument(
            "--quality", type=int, default=85, help="压缩质量 (1-100, 默认: 85)"
        )
        compress_parser.add_argument(
            "--format",
            choices=["jpg", "jpeg", "png", "webp"],
            help="目标格式（默认保持原格式）",
        )
        compress_parser.add_argument(
            "--max-size", help="最大尺寸限制 (WxH，如 1920x1080)"
        )
        compress_parser.add_argument(
            "--recursive", action="store_true", help="递归处理子目录"
        )
        compress_parser.add_argument(
            "--no-concurrent", action="store_true", help="禁用并发处理（默认启用）"
        )
        compress_parser.add_argument(
            "--max-workers",
            type=int,
            help="最大线程数（默认为CPU核心数的20%，最大为CPU核心数）",
        )

    def _add_file_commands(self, subparsers):
        """添加文件处理器命令"""
        file_parser = subparsers.add_parser(
            "file", help="文件操作", description="文件复制、移动、删除、组织等功能"
        )

        file_subparsers = file_parser.add_subparsers(
            dest="file_action", help="文件操作"
        )

        # 按扩展名组织
        organize_parser = file_subparsers.add_parser(
            "organize", help="按扩展名组织文件"
        )
        organize_parser.add_argument("source_dir", help="源目录")
        organize_parser.add_argument("--output-dir", help="输出目录（默认为源目录）")
        organize_parser.add_argument(
            "--by-extension", action="store_true", help="按扩展名分组"
        )
        organize_parser.add_argument(
            "--copy", action="store_true", help="复制文件而不是移动"
        )

        # 批量复制
        copy_parser = file_subparsers.add_parser("copy", help="批量复制文件")
        copy_parser.add_argument("source_path", help="源路径")
        copy_parser.add_argument("dest_path", help="目标路径")
        copy_parser.add_argument("--recursive", action="store_true", help="递归复制")

        # 批量移动
        move_parser = file_subparsers.add_parser("move", help="批量移动文件")
        move_parser.add_argument("source_path", help="源路径")
        move_parser.add_argument("dest_path", help="目标路径")
        move_parser.add_argument("--recursive", action="store_true", help="递归移动")

    def _add_dataset_commands(self, subparsers):
        """添加数据集处理器命令"""
        dataset_parser = subparsers.add_parser(
            "dataset", help="数据集处理", description="数据集验证、分析、分割等功能"
        )

        dataset_subparsers = dataset_parser.add_subparsers(
            dest="dataset_action", help="数据集操作"
        )

        # 分割数据集
        split_parser = dataset_subparsers.add_parser("split", help="分割数据集")
        split_parser.add_argument("dataset_path", help="数据集路径")
        split_parser.add_argument("--output-dir", help="输出目录")
        split_parser.add_argument(
            "--train-ratio", type=float, default=0.8, help="训练集比例 (默认: 0.8)"
        )
        split_parser.add_argument(
            "--val-ratio", type=float, default=0.1, help="验证集比例 (默认: 0.1)"
        )
        split_parser.add_argument(
            "--test-ratio", type=float, default=0.1, help="测试集比例 (默认: 0.1)"
        )
        split_parser.add_argument(
            "--format",
            choices=["yolo", "coco", "pascal_voc"],
            default="yolo",
            help="数据集格式 (默认: yolo)",
        )

        # 验证数据集
        validate_parser = dataset_subparsers.add_parser("validate", help="验证数据集")
        validate_parser.add_argument("dataset_path", help="数据集路径")
        validate_parser.add_argument(
            "--format",
            choices=["yolo", "coco", "pascal_voc"],
            default="yolo",
            help="数据集格式 (默认: yolo)",
        )
        validate_parser.add_argument(
            "--segmentation",
            action="store_true",
            help="验证分割数据集格式（要求至少7列）",
        )
        validate_parser.add_argument(
            "--move-invalid",
            action="store_true",
            default=True,
            help="移动无效文件到上级目录（默认: True）",
        )
        validate_parser.add_argument(
            "--no-move", action="store_true", help="只验证不移动文件"
        )
        validate_parser.add_argument("--output-dir", help="自定义无效文件目录名称")

        # 分析数据集
        analyze_parser = dataset_subparsers.add_parser("analyze", help="分析数据集")
        analyze_parser.add_argument("dataset_path", help="数据集路径")
        analyze_parser.add_argument(
            "--format",
            choices=["yolo", "coco", "pascal_voc"],
            default="yolo",
            help="数据集格式 (默认: yolo)",
        )
        analyze_parser.add_argument("--output-file", help="分析结果输出文件")

    def _add_label_commands(self, subparsers):
        """添加标签处理器命令"""
        label_parser = subparsers.add_parser(
            "label", help="标签处理", description="标签文件创建、修改、转换等功能"
        )

        label_subparsers = label_parser.add_subparsers(
            dest="label_action", help="标签操作"
        )

        # 创建空标签
        create_empty_parser = label_subparsers.add_parser(
            "create-empty", help="为图像创建空标签文件"
        )
        create_empty_parser.add_argument("images_dir", help="图像目录")
        create_empty_parser.add_argument(
            "--labels-dir", help="标签目录（默认为图像目录）"
        )
        create_empty_parser.add_argument(
            "--overwrite", action="store_true", help="覆盖已存在的标签文件"
        )

        # 翻转标签
        flip_parser = label_subparsers.add_parser("flip", help="翻转标签坐标")
        flip_parser.add_argument("labels_dir", help="标签目录")
        flip_parser.add_argument(
            "--type",
            choices=["horizontal", "vertical", "both"],
            default="horizontal",
            help="翻转类型 (默认: horizontal)",
        )
        flip_parser.add_argument("--backup", action="store_true", help="备份原文件")

        # 过滤标签
        filter_parser = label_subparsers.add_parser("filter", help="根据类别过滤标签")
        filter_parser.add_argument("labels_dir", help="标签目录")
        filter_parser.add_argument(
            "--classes", required=True, help="目标类别（逗号分隔）"
        )
        filter_parser.add_argument(
            "--action",
            choices=["keep", "remove"],
            default="keep",
            help="操作类型 (默认: keep)",
        )
        filter_parser.add_argument("--backup", action="store_true", help="备份原文件")

        # 删除空标签
        remove_empty_parser = label_subparsers.add_parser(
            "remove-empty", help="删除空标签文件及对应图像"
        )
        remove_empty_parser.add_argument("dataset_dir", help="数据集目录")
        remove_empty_parser.add_argument(
            "--images-dir", default="images", help="图像子目录名 (默认: images)"
        )
        remove_empty_parser.add_argument(
            "--labels-dir", default="labels", help="标签子目录名 (默认: labels)"
        )

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """解析命令行参数"""
        return self.parser.parse_args(args)

    def setup_logging_from_args(self, args: argparse.Namespace) -> None:
        """根据参数设置日志"""
        log_level = (
            "ERROR" if args.quiet else ("DEBUG" if args.verbose else args.log_level)
        )

        setup_logging(
            log_level=log_level, enable_error_file=True
        )

        self.logger = get_logger(self.__class__.__name__)

    def load_config_from_args(self, args: argparse.Namespace) -> None:
        """根据参数加载配置"""
        if args.config:
            try:
                # 创建新的ConfigManager实例来加载指定配置文件
                self.config_manager = ConfigManager(config_file=args.config)
                self.logger.info(f"已加载配置文件: {args.config}")
            except Exception as e:
                self.logger.error(f"加载配置文件失败: {e}")
                raise ConfigurationError(f"配置文件加载失败: {e}")

    def execute_command(self, args: argparse.Namespace) -> int:
        """执行命令

        Returns:
            int: 退出码 (0表示成功)
        """
        try:
            if not args.command:
                self.parser.print_help()
                return 1

            # 根据命令执行相应操作
            if args.command == "yolo":
                return self._execute_yolo_command(args)
            elif args.command == "image":
                return self._execute_image_command(args)
            elif args.command == "file":
                return self._execute_file_command(args)

            elif args.command == "label":
                return self._execute_label_command(args)
            else:
                self.logger.error(f"未知命令: {args.command}")
                return 1

        except KeyboardInterrupt:
            self.logger.info("操作被用户中断")
            return 130
        except ProcessingError as e:
            self.logger.error(f"处理错误: {e}")
            return 1
        except Exception as e:
            self.logger.error(f"未知错误: {e}")
            return 1

    def _execute_yolo_command(self, args: argparse.Namespace) -> int:
        """执行YOLO命令"""
        if not args.yolo_action:
            self.logger.error("请指定YOLO操作")
            return 1

        processor = YOLOProcessor()

        try:
            if args.yolo_action == "validate":
                result = processor.get_dataset_statistics(args.dataset_path)
                self._print_result(result)

            elif args.yolo_action == "remove-class0":
                result = processor.remove_zero_only_labels(args.dataset_path)
                self._print_result(result)

            elif args.yolo_action == "process-ctds":
                result = processor.process_ctds_dataset(
                    args.dataset_path,
                    output_name=getattr(args, "project_name", None),
                    keep_empty_labels=getattr(args, "keep_empty_labels", False),
                )
                self._print_result(result)

            elif args.yolo_action == "process-xlabel":
                class_order = None
                if getattr(args, "class_order", None):
                    class_order = [
                        name.strip()
                        for name in args.class_order.split(",")
                        if name.strip()
                    ]
                result = processor.convert_xlabel_to_yolo(
                    args.dataset_path,
                    output_dir=getattr(args, "output_path", None),
                    class_order=class_order,
                )
                self._print_result(result)

            elif args.yolo_action == "process-xlabel-seg":
                class_order = None
                if getattr(args, "class_order", None):
                    class_order = [
                        name.strip()
                        for name in args.class_order.split(",")
                        if name.strip()
                    ]
                english_name_mapping = None
                if getattr(args, "class_map", None):
                    with open(args.class_map, "r", encoding="utf-8") as f:
                        english_name_mapping = json.load(f)
                    if not isinstance(english_name_mapping, dict):
                        raise ValueError("class-map必须为JSON对象")
                result = processor.convert_xlabel_to_yolo_segmentation(
                    args.dataset_path,
                    output_dir=getattr(args, "output_path", None),
                    class_order=class_order,
                    english_name_mapping=english_name_mapping,
                )
                self._print_result(result)

            else:
                self.logger.error(f"未知YOLO操作: {args.yolo_action}")
                return 1

            return 0 if result.get("success", False) else 1

        except Exception as e:
            self.logger.error(f"YOLO操作失败: {e}")
            return 1

    def _execute_image_command(self, args: argparse.Namespace) -> int:
        """执行图像命令"""
        if not args.image_action:
            self.logger.error("请指定图像操作")
            return 1

        processor = ImageProcessor()

        try:
            if args.image_action == "convert":
                result = processor.convert_format(
                    args.input_path,
                    args.format,
                    output_path=args.output_path,
                    quality=args.quality,
                    recursive=args.recursive,
                )
                self._print_result(result)

            elif args.image_action == "resize":
                # 解析尺寸参数
                size = self._parse_size(args.size)
                result = processor.resize_images(
                    args.input_path,
                    args.output_path if args.output_path else args.input_path,
                    target_size=size,
                    maintain_aspect_ratio=args.keep_aspect,
                    recursive=args.recursive,
                )
                self._print_result(result)

            elif args.image_action == "compress":
                # 解析最大尺寸参数
                max_size = None
                if args.max_size:
                    max_size = self._parse_size(args.max_size)

                compress_kwargs = {
                    "input_dir": args.input_dir,
                    "output_dir": args.output_dir,
                    "quality": args.quality,
                    "target_format": args.format,
                    "recursive": args.recursive,
                }
                if max_size is not None:
                    compress_kwargs["max_size"] = max_size
                
                result = processor.compress_images(**compress_kwargs)
                self._print_result(result)

                # 显示压缩统计信息
                if result.get("success") and "statistics" in result:
                    stats = result["statistics"]
                    print(f"\n压缩统计:")
                    print(f"  总文件数: {stats['total_files']}")
                    print(f"  成功压缩: {stats['compressed_count']}")
                    print(f"  失败文件: {stats['failed_count']}")
                    print(f"  原始总大小: {stats['total_input_size_formatted']}")
                    print(f"  压缩后大小: {stats['total_output_size_formatted']}")
                    print(f"  节省空间: {stats['space_saved_formatted']}")
                    print(f"  压缩比: {stats['overall_compression_ratio']:.2f}")
                    print(
                        f"  空间节省率: {stats['overall_space_saved_percentage']:.1f}%"
                    )

            else:
                self.logger.error(f"未知图像操作: {args.image_action}")
                return 1

            return 0 if result.get("success", False) else 1

        except Exception as e:
            self.logger.error(f"图像操作失败: {e}")
            return 1

    def _execute_file_command(self, args: argparse.Namespace) -> int:
        """执行文件命令"""
        if not args.file_action:
            self.logger.error("请指定文件操作")
            return 1

        processor = FileProcessor()

        try:
            if args.file_action == "organize":
                result = processor.organize_files_by_extension(
                    args.source_dir, target_dir=args.output_dir if args.output_dir else args.source_dir, create_subdirs=args.by_extension
                )
                self._print_result(result)

            elif args.file_action == "copy":
                result = processor.copy_files(
                    args.source_path, args.dest_path, recursive=args.recursive
                )
                self._print_result(result)

            elif args.file_action == "move":
                result = processor.move_files(
                    args.source_path, args.dest_path, recursive=args.recursive
                )
                self._print_result(result)

            else:
                self.logger.error(f"未知文件操作: {args.file_action}")
                return 1

            return 0 if result.get("success", False) else 1

        except Exception as e:
            self.logger.error(f"文件操作失败: {e}")
            return 1

    def _execute_dataset_command(self, args: argparse.Namespace) -> int:
        """执行数据集命令"""
        if not args.dataset_action:
            self.logger.error("请指定数据集操作")
            return 1

        processor = DatasetProcessor()

        try:
            if args.dataset_action == "split":
                result = processor.split_dataset(
                    args.dataset_path,
                    output_path=args.output_dir or f"{args.dataset_path}_split",
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                )
                self._print_result(result)

            elif args.dataset_action == "validate":
                if args.segmentation:
                    # 使用分割数据集验证
                    from test_segmentation_dataset_validation import (
                        SegmentationDatasetValidator,
                    )

                    validator = SegmentationDatasetValidator()

                    # 处理移动文件参数
                    move_invalid = args.move_invalid and not args.no_move

                    result = validator.validate_and_clean_dataset(
                        dataset_path=args.dataset_path,
                        move_invalid=move_invalid,
                        custom_invalid_dir_name=args.output_dir,
                    )
                else:
                    # 使用标准数据集验证
                    result = processor.analyze_dataset(args.dataset_path)
                self._print_result(result)

            elif args.dataset_action == "analyze":
                result = processor.analyze_dataset(
                    args.dataset_path,
                    dataset_format=args.format,
                    output_file=args.output_file,
                )
                self._print_result(result)

            else:
                self.logger.error(f"未知数据集操作: {args.dataset_action}")
                return 1

            return 0 if result.get("success", False) else 1

        except Exception as e:
            self.logger.error(f"数据集操作失败: {e}")
            return 1

    def _execute_label_command(self, args: argparse.Namespace) -> int:
        """执行标签命令"""
        if not args.label_action:
            self.logger.error("请指定标签操作")
            return 1

        processor = LabelProcessor()

        try:
            if args.label_action == "create-empty":
                result = processor.create_empty_labels(
                    args.images_dir,
                    labels_dir=args.labels_dir,
                    overwrite=args.overwrite,
                )
                self._print_result(result)

            elif args.label_action == "flip":
                result = processor.flip_labels(
                    args.labels_dir, flip_type=args.type, backup=args.backup
                )
                self._print_result(result)

            elif args.label_action == "filter":
                # 解析类别参数
                classes = [int(c.strip()) for c in args.classes.split(",")]
                result = processor.filter_labels_by_class(
                    args.labels_dir,
                    target_classes=classes,
                    action=args.action,
                    backup=args.backup,
                )
                self._print_result(result)

            elif args.label_action == "remove-empty":
                result = processor.remove_empty_labels_and_images(
                    args.dataset_dir,
                    images_subdir=args.images_dir,
                    labels_subdir=args.labels_dir,
                )
                self._print_result(result)

            else:
                self.logger.error(f"未知标签操作: {args.label_action}")
                return 1

            return 0 if result.get("success", False) else 1

        except Exception as e:
            self.logger.error(f"标签操作失败: {e}")
            return 1

    def _parse_size(self, size_str: str) -> Tuple[int, int]:
        """解析尺寸字符串"""
        if "x" in size_str.lower():
            parts = size_str.lower().split("x")
            if len(parts) == 2:
                return (int(parts[0]), int(parts[1]))
        else:
            size = int(size_str)
            return (size, size)

        raise ValueError(f"无效的尺寸格式: {size_str}")

    def _print_result(self, result: Dict[str, Any]) -> None:
        """打印结果"""
        if result.get("success", False):
            self.logger.info("操作成功完成")

            # 打印统计信息
            if "statistics" in result:
                stats = result["statistics"]
                self.logger.info("统计信息:")
                for key, value in stats.items():
                    self.logger.info(f"  {key}: {value}")
        else:
            self.logger.error("操作失败")
            if "message" in result:
                self.logger.error(f"错误信息: {result['message']}")

    def run(self, args: Optional[List[str]] = None) -> int:
        """运行命令行界面

        Args:
            args: 命令行参数列表（用于测试）

        Returns:
            int: 退出码
        """
        try:
            # 解析参数
            parsed_args = self.parse_args(args)

            # 设置日志
            self.setup_logging_from_args(parsed_args)

            # 加载配置
            self.load_config_from_args(parsed_args)

            # 执行命令
            return self.execute_command(parsed_args)

        except SystemExit as e:
            # SystemExit.code 可能是 int、str 或 None，需要确保返回 int
            if e.code is None:
                return 0
            elif isinstance(e.code, int):
                return e.code
            else:
                # 如果是字符串或其他类型，尝试转换为int，失败则返回1
                try:
                    return int(e.code)
                except (ValueError, TypeError):
                    return 1
        except Exception as e:
            print(f"致命错误: {e}", file=sys.stderr)
            return 1


def main():
    """主入口函数"""
    cli = CLIInterface()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()
