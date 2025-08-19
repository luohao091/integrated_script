#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

项目主入口脚本

这是项目的主要入口点，可以直接运行或通过模块方式调用。
"""

import sys
import os

# 添加src目录到Python路径
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 初始化Windows兼容性（在导入其他模块之前）
if sys.platform == 'win32':
    try:
        from integrated_script.core.windows_compat import initialize_windows_compatibility
        compat_result = initialize_windows_compatibility()
        # 可选：显示兼容性初始化结果（仅在调试模式下）
        if '--debug-compat' in sys.argv:
            print(f"Windows兼容性初始化: {compat_result}")
    except Exception as e:
        # 如果兼容性初始化失败，继续运行但给出警告
        print(f"警告: Windows兼容性初始化失败: {e}")

from integrated_script.main import main
  
if __name__ == '__main__':
    sys.exit(main())