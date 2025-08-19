#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_exe.py

PyInstaller打包脚本

用于将integrated_script项目打包为Windows可执行文件
"""

import os
import sys
import shutil
from pathlib import Path

def build_exe():
    """构建可执行文件"""
    
    # 项目根目录
    project_root = Path(__file__).parent
    src_dir = project_root / "src"
    main_script = project_root / "main.py"
    
    # 确保主脚本存在
    if not main_script.exists():
        print(f"错误: 找不到主脚本 {main_script}")
        return False
    
    # 清理之前的构建
    build_dir = project_root / "build"
    dist_dir = project_root / "dist"
    
    # 只清理build目录，dist目录让PyInstaller自动处理
    if build_dir.exists():
        try:
            shutil.rmtree(build_dir)
        except PermissionError:
            print("警告: 无法删除build目录，继续执行...")
    
    # PyInstaller命令
    cmd_parts = [
        "pyinstaller",
        "--onefile",  # 打包为单个exe文件
        "--console",  # 显示控制台窗口
        "--name=integrated_script",  # 可执行文件名称
        f"--add-data={src_dir};src",  # 添加源代码目录
        f"--add-data={project_root / 'requirements.txt'};.",  # 添加requirements.txt
        f"--add-data={project_root / 'config'};config",  # 添加配置目录
        "--hidden-import=integrated_script",
        "--hidden-import=integrated_script.config",
        "--hidden-import=integrated_script.core",
        "--hidden-import=integrated_script.processors",
        "--hidden-import=integrated_script.ui",
        "--hidden-import=PIL",
        "--hidden-import=cv2",
        "--hidden-import=yaml",
        "--hidden-import=tqdm",
        "--hidden-import=logging.handlers",
        "--hidden-import=logging.config",
        "--hidden-import=argparse",
        "--hidden-import=pathlib",
        "--hidden-import=json",
        "--hidden-import=datetime",
        "--hidden-import=typing",
        "--collect-all=integrated_script",
        str(main_script)
    ]
    
    # 执行PyInstaller命令
    cmd = " ".join(cmd_parts)
    print(f"执行命令: {cmd}")
    
    result = os.system(cmd)
    
    if result == 0:
        exe_path = dist_dir / "integrated_script.exe"
        if exe_path.exists():
            print(f"\n✅ 打包成功!")
            print(f"可执行文件位置: {exe_path}")
            print(f"文件大小: {exe_path.stat().st_size / 1024 / 1024:.1f} MB")
            return True
        else:
            print("❌ 打包失败: 找不到生成的exe文件")
            return False
    else:
        print("❌ 打包失败: PyInstaller执行出错")
        return False

if __name__ == "__main__":
    print("开始打包integrated_script项目...")
    success = build_exe()
    if success:
        print("\n🎉 打包完成!")
    else:
        print("\n💥 打包失败!")
        sys.exit(1)