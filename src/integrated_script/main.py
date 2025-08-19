#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

主程序入口

提供命令行和交互式两种运行模式。
"""

import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

from .config import ConfigManager
from .core.logging_config import setup_logging, get_logger
from .ui.cli import CLIInterface
from .ui.interactive import InteractiveInterface


def setup_argument_parser() -> argparse.ArgumentParser:
    """设置命令行参数解析器
    
    Returns:
        argparse.ArgumentParser: 配置好的参数解析器
    """
    parser = argparse.ArgumentParser(
        prog='integrated_script',
        description='集成脚本工具 - 提供YOLO数据集处理、图像处理、文件管理等功能',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s --interactive                    # 启动交互式界面
  %(prog)s yolo validate /path/to/dataset   # 验证YOLO数据集
  %(prog)s image convert /path/to/images    # 转换图像格式
  %(prog)s file organize /path/to/files     # 组织文件
  %(prog)s dataset split /path/to/dataset   # 分割数据集
  %(prog)s label create /path/to/images     # 创建标签文件

更多信息请访问: https://github.com/your-repo/integrated_script
        """
    )
    
    # 全局选项
    parser.add_argument(
        '--version', 
        action='version', 
        version='%(prog)s 1.3.1'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        help='配置文件路径 (JSON或YAML格式)'
    )
    
    parser.add_argument(
        '--log-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='日志级别 (默认: INFO)'
    )
    
    parser.add_argument(
        '--log-file', 
        type=str, 
        help='日志文件路径'
    )
    
    parser.add_argument(
        '--interactive', 
        action='store_true',
        help='启动交互式界面'
    )
    
    parser.add_argument(
        '--no-color', 
        action='store_true',
        help='禁用彩色输出'
    )
    
    parser.add_argument(
        '--quiet', 
        action='store_true',
        help='静默模式，减少输出'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='详细模式，增加输出'
    )
    
    parser.add_argument(
        '--build', 
        action='store_true',
        help='调用根目录的build_exe.py进行打包'
    )
    
    # 创建子命令解析器
    subparsers = parser.add_subparsers(
        dest='command',
        title='可用命令',
        description='选择要执行的操作类型',
        help='使用 "%(prog)s <command> --help" 查看具体命令的帮助信息'
    )
    
    # 添加CLI接口的子命令
    cli = CLIInterface()
    cli.setup_parsers(subparsers)
    
    return parser


def setup_logging_from_args(args: argparse.Namespace) -> None:
    """根据命令行参数设置日志
    
    Args:
        args: 命令行参数
    """
    # 确定日志级别
    log_level = args.log_level
    if args.quiet:
        log_level = 'WARNING'
    elif args.verbose:
        log_level = 'DEBUG'
    
    # 设置日志
    log_dir = "logs"
    if args.log_file:
        # 如果指定了日志文件，使用其目录作为日志目录
        log_dir = str(Path(args.log_file).parent)
    
    setup_logging(
        log_dir=log_dir,
        log_level=log_level,
        enable_error_file=True
    )


def load_config_from_args(args: argparse.Namespace) -> ConfigManager:
    """根据命令行参数加载配置
    
    Args:
        args: 命令行参数
        
    Returns:
        ConfigManager: 配置管理器实例
    """
    config_manager = ConfigManager()
    
    if args.config:
        try:
            config_manager.load_config(args.config)
            logger = get_logger(__name__)
            logger.info(f"已加载配置文件: {args.config}")
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"加载配置文件失败: {e}")
            sys.exit(1)
    
    return config_manager


def run_interactive_mode(config_manager: ConfigManager) -> int:
    """运行交互式模式
    
    Args:
        config_manager: 配置管理器
        
    Returns:
        int: 退出代码
    """
    try:
        interface = InteractiveInterface(config_manager)
        interface.run()
        return 0
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        return 130
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"交互式模式运行失败: {e}")
        return 1


def run_cli_mode(args: argparse.Namespace, config_manager: ConfigManager) -> int:
    """运行命令行模式
    
    Args:
        args: 命令行参数
        config_manager: 配置管理器
        
    Returns:
        int: 退出代码
    """
    try:
        cli = CLIInterface(config_manager)
        return cli.run(args)
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        return 130
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"命令行模式运行失败: {e}")
        return 1


def run_build_mode() -> int:
    """运行打包模式
    
    Returns:
        int: 退出代码
    """
    logger = get_logger(__name__)
    
    # 查找项目根目录的build_exe.py
    current_dir = Path(__file__).parent
    # 向上查找到项目根目录
    project_root = current_dir
    while project_root.parent != project_root:
        build_script = project_root / "build_exe.py"
        if build_script.exists():
            break
        project_root = project_root.parent
    else:
        # 如果没找到，尝试相对于当前文件的几个可能位置
        possible_paths = [
            current_dir / "../../../build_exe.py",  # 从src/integrated_script向上3级
            current_dir / "../../build_exe.py",     # 从src/integrated_script向上2级
            current_dir / "../build_exe.py",        # 从src/integrated_script向上1级
        ]
        
        build_script = None
        for path in possible_paths:
            if path.exists():
                build_script = path.resolve()
                project_root = build_script.parent
                break
        
        if not build_script:
            logger.error("找不到build_exe.py文件")
            print("错误: 找不到build_exe.py文件")
            return 1
    
    logger.info(f"找到打包脚本: {build_script}")
    print(f"正在调用打包脚本: {build_script}")
    
    try:
        # 调用build_exe.py
        result = subprocess.run(
            [sys.executable, str(build_script)],
            cwd=str(project_root),
            capture_output=False  # 让输出直接显示给用户
        )
        
        if result.returncode == 0:
            logger.info("打包完成")
            print("\n✅ 打包完成!")
        else:
            logger.error(f"打包失败，退出代码: {result.returncode}")
            print(f"\n❌ 打包失败，退出代码: {result.returncode}")
        
        return result.returncode
        
    except Exception as e:
        logger.error(f"调用打包脚本失败: {e}")
        print(f"错误: 调用打包脚本失败: {e}")
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """主函数
    
    Args:
        argv: 命令行参数列表，如果为None则使用sys.argv
        
    Returns:
        int: 退出代码
    """
    # 解析命令行参数
    parser = setup_argument_parser()
    args = parser.parse_args(argv)
    
    # 设置日志
    setup_logging_from_args(args)
    logger = get_logger(__name__)
    
    try:
        # 检查是否为打包模式
        if args.build:
            logger.info("启动打包模式")
            return run_build_mode()
        
        # 加载配置
        config_manager = load_config_from_args(args)
        
        # 根据参数选择运行模式
        if args.interactive or not args.command:
            # 交互式模式
            logger.info("启动交互式模式")
            return run_interactive_mode(config_manager)
        else:
            # 命令行模式
            logger.info(f"执行命令: {args.command}")
            return run_cli_mode(args, config_manager)
            
    except Exception as e:
        logger.error(f"程序运行失败: {e}")
        return 1


def console_entry_point() -> None:
    """控制台入口点
    
    用于setup.py中的console_scripts配置。
    """
    sys.exit(main())


if __name__ == '__main__':
    sys.exit(main())