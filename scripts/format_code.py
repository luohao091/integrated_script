#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
format_code.py

代码格式化和质量检查脚本

自动运行 black、isort、flake8 和 mypy 进行代码格式化和质量检查
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


class CodeFormatter:
    """代码格式化器"""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.src_dirs = [
            "src/integrated_script",
            "scripts",
        ]

    def run_command(self, command: List[str]) -> Tuple[bool, str]:
        """运行命令并返回结果"""
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False,
            )
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)

    def format_with_black(self) -> bool:
        """使用 Black 格式化代码"""
        print("🔧 运行 Black 代码格式化...")
        command = ["black"] + self.src_dirs
        success, output = self.run_command(command)
        if success:
            print("✅ Black 格式化完成")
        else:
            print(f"❌ Black 格式化失败: {output}")
        return success

    def sort_imports_with_isort(self) -> bool:
        """使用 isort 整理导入"""
        print("🔧 运行 isort 导入整理...")
        command = ["isort"] + self.src_dirs
        success, output = self.run_command(command)
        if success:
            print("✅ isort 导入整理完成")
        else:
            print(f"❌ isort 导入整理失败: {output}")
        return success

    def check_with_flake8(self) -> bool:
        """使用 Flake8 检查代码质量"""
        print("🔧 运行 Flake8 代码质量检查...")
        command = ["flake8"] + self.src_dirs
        success, output = self.run_command(command)
        if success:
            print("✅ Flake8 检查通过")
        else:
            print(f"⚠️ Flake8 发现问题:\n{output}")
        return success

    def check_with_mypy(self) -> bool:
        """使用 MyPy 进行类型检查"""
        print("🔧 运行 MyPy 类型检查...")
        command = ["mypy", "src/integrated_script"]
        success, output = self.run_command(command)
        if success:
            print("✅ MyPy 类型检查通过")
        else:
            print(f"⚠️ MyPy 发现类型问题:\n{output}")
        return success

    def format_all(self, check_only: bool = False) -> bool:
        """运行所有格式化和检查"""
        print("\n" + "=" * 60)
        print("🚀 开始代码格式化和质量检查")
        print("=" * 60)

        results = []

        if not check_only:
            # 格式化步骤
            results.append(("Black 格式化", self.format_with_black()))
            results.append(("isort 导入整理", self.sort_imports_with_isort()))

        # 检查步骤
        results.append(("Flake8 质量检查", self.check_with_flake8()))
        results.append(("MyPy 类型检查", self.check_with_mypy()))

        # 显示结果
        print("\n" + "=" * 60)
        print("📊 检查结果汇总")
        print("=" * 60)

        all_passed = True
        for name, success in results:
            status = "✅ 通过" if success else "❌ 失败"
            print(f"{name}: {status}")
            if not success:
                all_passed = False

        if all_passed:
            print("\n🎉 所有检查都通过了！")
        else:
            print("\n⚠️ 部分检查未通过，请查看上面的详细信息")

        return all_passed


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="代码格式化和质量检查工具")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="仅进行检查，不进行格式化",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="仅进行格式化，不进行质量检查",
    )

    args = parser.parse_args()

    formatter = CodeFormatter()

    if args.format_only:
        # 仅格式化
        success1 = formatter.format_with_black()
        success2 = formatter.sort_imports_with_isort()
        sys.exit(0 if success1 and success2 else 1)
    else:
        # 完整流程
        success = formatter.format_all(check_only=args.check_only)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
