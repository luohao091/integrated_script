#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
release.py

一键发布脚本

简化发布流程，包括版本管理、构建、测试和发布
"""

import os
import sys
import subprocess
import time
import requests
import json
from pathlib import Path
from typing import Optional, Dict, Any

# 添加脚本目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from version_manager import VersionManager


class ReleaseManager:
    """发布管理器"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.vm = VersionManager(self.project_root)
        
    def check_git_status(self) -> bool:
        """检查 Git 状态"""
        try:
            # 检查是否有未提交的更改
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.stdout.strip():
                print("❌ 检测到未提交的更改:")
                print(result.stdout)
                print("请先提交所有更改后再发布")
                return False
                
            print("✅ Git 状态检查通过")
            return True
            
        except subprocess.CalledProcessError:
            print("❌ Git 状态检查失败")
            return False
        except FileNotFoundError:
            print("❌ 找不到 Git 命令")
            return False
    
    def run_tests(self) -> bool:
        """运行测试"""
        print("🧪 运行测试...")
        
        # 检查是否有测试文件
        test_dirs = ["tests", "test"]
        has_tests = any((self.project_root / d).exists() for d in test_dirs)
        
        if not has_tests:
            print("⚠️  未找到测试目录，跳过测试")
            return True
        
        try:
            # 尝试运行 pytest
            result = subprocess.run(
                ["python", "-m", "pytest", "-v"],
                cwd=self.project_root,
                check=True
            )
            
            print("✅ 所有测试通过")
            return True
            
        except subprocess.CalledProcessError:
            print("❌ 测试失败")
            return False
        except FileNotFoundError:
            print("⚠️  pytest 未安装，跳过测试")
            return True
    
    def build_executable(self) -> bool:
        """构建可执行文件"""
        print("🔨 构建可执行文件...")
        
        build_script = self.project_root / "build_exe.py"
        if not build_script.exists():
            print(f"❌ 找不到构建脚本: {build_script}")
            return False
        
        try:
            result = subprocess.run(
                ["python", "build_exe.py"],
                cwd=self.project_root,
                check=True
            )
            
            # 检查生成的文件
            exe_path = self.project_root / "dist" / "integrated_script.exe"
            if exe_path.exists():
                size_mb = exe_path.stat().st_size / 1024 / 1024
                print(f"✅ 构建成功! 文件大小: {size_mb:.1f} MB")
                return True
            else:
                print("❌ 构建失败: 找不到生成的可执行文件")
                return False
                
        except subprocess.CalledProcessError:
            print("❌ 构建失败")
            return False
    
    def test_executable(self) -> bool:
        """测试可执行文件"""
        print("🧪 测试可执行文件...")
        
        exe_path = self.project_root / "dist" / "integrated_script.exe"
        if not exe_path.exists():
            print("❌ 找不到可执行文件")
            return False
        
        try:
            # 测试 --version 参数
            result = subprocess.run(
                [str(exe_path), "--version"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print(f"✅ 可执行文件测试通过: {result.stdout.strip()}")
                return True
            else:
                print(f"❌ 可执行文件测试失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ 可执行文件测试超时")
            return False
        except Exception as e:
            print(f"❌ 可执行文件测试失败: {e}")
            return False
    
    def push_to_github(self, version: str) -> bool:
        """推送到 GitHub"""
        print("📤 推送到 GitHub...")
        
        try:
            # 获取当前分支名
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                check=True
            )
            current_branch = result.stdout.strip()
            
            # 推送当前分支
            subprocess.run(["git", "push", "origin", current_branch], check=True)
            print(f"✅ 已推送分支 {current_branch}")
            
            # 推送标签
            subprocess.run(["git", "push", "origin", f"v{version}"], check=True)
            print(f"✅ 已推送标签 v{version}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 推送失败: {e}")
            return False
    
    def get_github_workflow_status(self, version: str) -> Dict[str, Any]:
        """获取 GitHub Actions 工作流状态"""
        try:
            # GitHub API URL
            api_url = "https://api.github.com/repos/luohao091/integrated_script/actions/runs"
            
            # 发送请求
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # 首先查找与当前版本完全匹配的工作流
            target_branch = f"v{version}"
            for run in data.get('workflow_runs', []):
                if run.get('head_branch') == target_branch and run.get('event') == 'push':
                    return {
                        'status': run.get('status'),
                        'conclusion': run.get('conclusion'),
                        'html_url': run.get('html_url'),
                        'created_at': run.get('created_at'),
                        'updated_at': run.get('updated_at'),
                        'name': run.get('name'),
                        'head_branch': run.get('head_branch')
                    }
            
            # 如果没找到精确匹配，返回最新的工作流（可能是刚触发的）
            if data.get('workflow_runs'):
                latest_run = data['workflow_runs'][0]
                return {
                    'status': latest_run.get('status'),
                    'conclusion': latest_run.get('conclusion'),
                    'html_url': latest_run.get('html_url'),
                    'created_at': latest_run.get('created_at'),
                    'updated_at': latest_run.get('updated_at'),
                    'name': latest_run.get('name'),
                    'head_branch': latest_run.get('head_branch'),
                    'is_latest': True
                }
            
            return {'status': 'not_found'}
            
        except Exception as e:
            print(f"⚠️  无法获取 GitHub Actions 状态: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def wait_for_github_actions(self, version: str, timeout: int = 600) -> bool:
        """等待 GitHub Actions 完成"""
        print("⏳ 等待 GitHub Actions 构建完成...")
        print(f"   可以在以下链接查看进度:")
        print(f"   https://github.com/luohao091/integrated_script/actions")
        
        start_time = time.time()
        check_interval = 30  # 每30秒检查一次
        
        while time.time() - start_time < timeout:
            # 获取工作流状态
            status_info = self.get_github_workflow_status(version)
            
            if status_info.get('status') == 'error':
                print("⚠️  API 检查失败，切换到简单等待模式")
                break
            elif status_info.get('status') == 'not_found':
                print("   🔍 等待工作流启动...")
            elif status_info.get('status') == 'queued':
                branch_info = f" (分支: {status_info.get('head_branch', 'unknown')})" if status_info.get('head_branch') else ""
                print(f"   ⏳ 工作流已排队等待执行{branch_info}")
            elif status_info.get('status') == 'in_progress':
                branch_info = f" (分支: {status_info.get('head_branch', 'unknown')})" if status_info.get('head_branch') else ""
                print(f"   🔄 工作流正在执行中{branch_info}")
                if status_info.get('is_latest') and status_info.get('head_branch') != f"v{version}":
                    print(f"   ⚠️  注意: 正在监控最新工作流，可能不是当前版本 v{version} 的工作流")
            elif status_info.get('status') == 'completed':
                conclusion = status_info.get('conclusion')
                branch_info = f" (分支: {status_info.get('head_branch', 'unknown')})" if status_info.get('head_branch') else ""
                
                if conclusion == 'success':
                    print(f"   ✅ GitHub Actions 构建成功!{branch_info}")
                    if status_info.get('html_url'):
                        print(f"   🔗 查看详情: {status_info['html_url']}")
                    
                    # 检查是否是正确的版本
                    if status_info.get('is_latest') and status_info.get('head_branch') != f"v{version}":
                        print(f"   ⚠️  注意: 完成的工作流分支 ({status_info.get('head_branch')}) 与当前版本 (v{version}) 不匹配")
                        print(f"   💡 建议检查 GitHub Actions 页面确认正确的工作流状态")
                    
                    return True
                elif conclusion == 'failure':
                    print(f"   ❌ GitHub Actions 构建失败!{branch_info}")
                    if status_info.get('html_url'):
                        print(f"   🔗 查看详情: {status_info['html_url']}")
                    return False
                else:
                    print(f"   ⚠️  工作流完成，状态: {conclusion}{branch_info}")
                    return False
            
            # 等待下次检查
            elapsed = int(time.time() - start_time)
            print(f"   等待中... ({elapsed}s/{timeout}s)")
            time.sleep(check_interval)
        
        print("⏰ 等待超时，请手动检查 GitHub Actions 状态")
        return True
    
    def release(self, version_type: str = "patch", 
                skip_tests: bool = False,
                skip_build: bool = False,
                auto_push: bool = False,
                message: Optional[str] = None) -> bool:
        """执行完整的发布流程
        
        Args:
            version_type: 版本类型 (major, minor, patch)
            skip_tests: 跳过测试
            skip_build: 跳过构建
            auto_push: 自动推送到 GitHub
            message: 发布消息
            
        Returns:
            发布是否成功
        """
        print("🚀 开始自动化发布流程...")
        print(f"📦 版本类型: {version_type}")
        
        # 1. 检查 Git 状态
        if not self.check_git_status():
            return False
        
        # 2. 运行测试
        if not skip_tests and not self.run_tests():
            print("❌ 发布失败: 测试未通过")
            return False
        
        # 3. 构建可执行文件
        if not skip_build and not self.build_executable():
            print("❌ 发布失败: 构建失败")
            return False
        
        # 4. 测试可执行文件
        if not skip_build and not self.test_executable():
            print("❌ 发布失败: 可执行文件测试失败")
            return False
        
        # 5. 更新版本并创建标签
        new_version = self.vm.release(version_type, message)
        
        # 6. 推送到 GitHub (可选)
        if auto_push:
            if not self.push_to_github(new_version):
                print("❌ 发布失败: 推送失败")
                return False
            
            # 7. 等待 GitHub Actions
            self.wait_for_github_actions(new_version)
        else:
            # 获取当前分支名用于显示
            try:
                result = subprocess.run(
                    ["git", "branch", "--show-current"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                current_branch = result.stdout.strip()
            except:
                current_branch = "main"  # 默认分支名
                
            print("\n📋 手动推送命令:")
            print(f"   git push origin {current_branch}")
            print(f"   git push origin v{new_version}")
        
        print(f"\n🎉 发布流程完成! 版本: {new_version}")
        
        if auto_push:
            print("\n🔗 相关链接:")
            print(f"   GitHub Release: https://github.com/your-username/integrated_script/releases/tag/v{new_version}")
            print(f"   GitHub Actions: https://github.com/your-username/integrated_script/actions")
        
        return True


def interactive_release():
    """交互式发布"""
    rm = ReleaseManager()
    
    print("\n" + "=" * 50)
    print("     🚀 集成脚本工具 - 交互式发布")
    print("=" * 50)
    
    # 显示当前版本
    try:
        current_version = rm.vm.get_current_version()
        print(f"\n📦 当前版本: {current_version}")
    except Exception as e:
        print(f"❌ 无法获取当前版本: {e}")
        return False
    
    # 选择版本类型
    print("\n请选择发布类型:")
    print("  1. patch  - 补丁版本 (修复bug)")
    print("  2. minor  - 次要版本 (新功能)")
    print("  3. major  - 主要版本 (重大更新)")
    
    while True:
        choice = input("\n请输入选择 (1-3, 默认为 1): ").strip()
        if choice == "" or choice == "1":
            version_type = "patch"
            break
        elif choice == "2":
            version_type = "minor"
            break
        elif choice == "3":
            version_type = "major"
            break
        else:
            print("❌ 无效选择，请输入 1-3")
    
    # 发布选项
    print("\n📋 发布选项:")
    
    skip_tests = input("跳过测试? (y/N): ").strip().lower() in ['y', 'yes']
    skip_build = input("跳过构建? (y/N): ").strip().lower() in ['y', 'yes']
    auto_push_input = input("自动推送到 GitHub? (Y/n): ").strip().lower()
    auto_push = auto_push_input not in ['n', 'no']  # 默认为 True
    
    message = input("发布消息 (可选): ").strip()
    if not message:
        message = None
    
    # 确认发布
    print(f"\n🎯 发布配置:")
    print(f"  版本类型: {version_type}")
    print(f"  跳过测试: {'是' if skip_tests else '否'}")
    print(f"  跳过构建: {'是' if skip_build else '否'}")
    print(f"  自动推送: {'是' if auto_push else '否'}")
    if message:
        print(f"  发布消息: {message}")
    
    confirm = input("\n确认开始发布? (Y/n): ").strip().lower()
    if confirm in ['n', 'no']:
        print("❌ 发布已取消")
        return False
    
    # 执行发布
    print("\n🚀 开始发布流程...")
    try:
        success = rm.release(
            version_type=version_type,
            skip_tests=skip_tests,
            skip_build=skip_build,
            auto_push=auto_push,
            message=message
        )
        return success
    except KeyboardInterrupt:
        print("\n❌ 发布被用户中断")
        return False
    except Exception as e:
        print(f"❌ 发布失败: {e}")
        return False


def main():
    """主函数"""
    import argparse
    
    # 如果没有命令行参数，启动交互模式
    if len(sys.argv) == 1:
        success = interactive_release()
        sys.exit(0 if success else 1)
    
    # 命令行模式
    parser = argparse.ArgumentParser(
        description="一键发布工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python release.py                         # 交互式发布
  python release.py patch                   # 发布补丁版本
  python release.py minor --auto-push      # 发布次要版本并自动推送
  python release.py major --skip-tests     # 发布主要版本，跳过测试
  python release.py patch --skip-build     # 只更新版本，不构建
        """
    )
    
    parser.add_argument(
        "version_type",
        choices=["major", "minor", "patch"],
        default="patch",
        nargs="?",
        help="版本类型 (默认: patch)"
    )
    
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="跳过测试"
    )
    
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="跳过构建"
    )
    
    parser.add_argument(
        "--auto-push",
        action="store_true",
        help="自动推送到 GitHub"
    )
    
    parser.add_argument(
        "-m", "--message",
        help="发布消息"
    )
    
    args = parser.parse_args()
    
    rm = ReleaseManager()
    
    try:
        success = rm.release(
            version_type=args.version_type,
            skip_tests=args.skip_tests,
            skip_build=args.skip_build,
            auto_push=args.auto_push,
            message=args.message
        )
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n❌ 发布被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 发布失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()