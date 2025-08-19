#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
menu.py

菜单系统

提供菜单显示和导航功能。
"""

import os
import sys
from typing import Dict, Any, List, Optional, Callable, Tuple

from ..config.exceptions import UserInterruptError


class MenuSystem:
    """菜单系统
    
    提供菜单显示和导航功能。
    
    Attributes:
        main_menu (Dict): 主菜单配置
        current_menu (Dict): 当前菜单
        menu_stack (List): 菜单栈，用于返回上级菜单
    """
    
    def __init__(self):
        """初始化菜单系统"""
        self.main_menu = None
        self.current_menu = None
        self.menu_stack = []
    
    def set_main_menu(self, menu: Dict[str, Any]) -> None:
        """设置主菜单
        
        Args:
            menu: 菜单配置字典
                {
                    'title': '菜单标题',
                    'options': [
                        ('选项1', callback1),
                        ('选项2', callback2),
                        ...
                    ]
                }
        """
        self.main_menu = menu
        self.current_menu = menu
    
    def show_menu(self, menu: Dict[str, Any] = None) -> None:
        """显示菜单
        
        Args:
            menu: 要显示的菜单，如果为None则显示当前菜单
        """
        if menu:
            self.menu_stack.append(self.current_menu)
            self.current_menu = menu
        
        while True:
            try:
                self._display_current_menu()
                choice = self._get_user_choice()
                
                if choice == 0:  # 返回上级菜单或退出
                    if self.menu_stack:
                        self.current_menu = self.menu_stack.pop()
                        continue
                    else:
                        # 在主菜单选择0时直接退出程序
                        import sys
                        sys.exit(0)
                
                # 执行选择的操作
                options = self.current_menu.get('options', [])
                if 1 <= choice <= len(options):
                    option_name, callback = options[choice - 1]
                    
                    if callback is None:  # 返回上级菜单
                        if self.menu_stack:
                            self.current_menu = self.menu_stack.pop()
                        else:
                            break
                    elif callable(callback):
                        try:
                            callback()
                        except UserInterruptError:
                            # 用户中断时直接返回，不显示额外信息
                            pass
                        except Exception as e:
                            print(f"\n操作执行失败: {e}")
                            self._pause()
                    else:
                        print(f"\n无效的回调函数: {callback}")
                        self._pause()
                else:
                    print(f"\n无效选择: {choice}")
                    self._pause()
                    
            except UserInterruptError:
                # 用户中断时直接返回上级菜单，不显示额外信息
                if self.menu_stack:
                    self.current_menu = self.menu_stack.pop()
                else:
                    # 在主菜单时直接退出程序
                    import sys
                    sys.exit(0)
            except Exception as e:
                print(f"\n\n菜单系统错误: {e}")
                self._pause()
    
    def _display_current_menu(self) -> None:
        """显示当前菜单"""
        self._clear_screen()
        
        # 显示标题
        title = self.current_menu.get('title', '菜单')
        print("\n" + "="*60)
        print(f"{title:^60}")
        print("="*60)
        
        # 显示选项
        options = self.current_menu.get('options', [])
        for i, (option_name, _) in enumerate(options, 1):
            print(f"{i:2d}. {option_name}")
        
        # 显示返回选项
        if self.menu_stack:
            print(f" 0. 返回上级菜单")
        else:
            print(f" 0. 退出程序")
        
        print("="*60)
    
    def _get_user_choice(self) -> int:
        """获取用户选择
        
        Returns:
            int: 用户选择的选项编号
        """
        while True:
            try:
                choice_str = input("\n请选择操作 (输入数字): ").strip()
                
                if not choice_str:
                    print("请输入有效的数字")
                    continue
                
                choice = int(choice_str)
                
                options_count = len(self.current_menu.get('options', []))
                if 0 <= choice <= options_count:
                    return choice
                else:
                    print(f"请输入 0 到 {options_count} 之间的数字")
                    
            except ValueError:
                print("请输入有效的数字")
            except KeyboardInterrupt:
                raise UserInterruptError("用户中断操作")
            except EOFError:
                raise UserInterruptError("输入结束")
    
    def _clear_screen(self) -> None:
        """清屏"""
        try:
            if os.name == 'nt':  # Windows
                os.system('cls')
            else:  # Unix/Linux/MacOS
                os.system('clear')
        except Exception:
            # 如果清屏失败，打印空行
            print('\n' * 50)
    
    def _pause(self) -> None:
        """暂停等待用户按键"""
        try:
            input("\n按回车键继续...")
        except KeyboardInterrupt:
            pass
        except EOFError:
            pass
    
    def run(self) -> None:
        """运行菜单系统"""
        if not self.main_menu:
            raise ValueError("未设置主菜单")
        
        try:
            self.show_menu()
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
        except Exception as e:
            print(f"\n\n菜单系统运行错误: {e}")
    
    def add_menu_option(self, option_name: str, callback: Callable, 
                       menu: Dict[str, Any] = None) -> None:
        """添加菜单选项
        
        Args:
            option_name: 选项名称
            callback: 回调函数
            menu: 要添加到的菜单，默认为当前菜单
        """
        target_menu = menu if menu else self.current_menu
        
        if 'options' not in target_menu:
            target_menu['options'] = []
        
        target_menu['options'].append((option_name, callback))
    
    def remove_menu_option(self, option_name: str, 
                          menu: Dict[str, Any] = None) -> bool:
        """移除菜单选项
        
        Args:
            option_name: 要移除的选项名称
            menu: 要从中移除的菜单，默认为当前菜单
            
        Returns:
            bool: 是否成功移除
        """
        target_menu = menu if menu else self.current_menu
        
        if 'options' not in target_menu:
            return False
        
        options = target_menu['options']
        for i, (name, _) in enumerate(options):
            if name == option_name:
                options.pop(i)
                return True
        
        return False
    
    def create_submenu(self, title: str, options: List[Tuple[str, Callable]]) -> Dict[str, Any]:
        """创建子菜单
        
        Args:
            title: 菜单标题
            options: 菜单选项列表
            
        Returns:
            Dict[str, Any]: 菜单配置字典
        """
        return {
            'title': title,
            'options': options
        }
    
    def show_message(self, message: str, title: str = "信息", 
                    wait_for_input: bool = True) -> None:
        """显示消息
        
        Args:
            message: 要显示的消息
            title: 消息标题
            wait_for_input: 是否等待用户输入
        """
        self._clear_screen()
        
        print("\n" + "="*60)
        print(f"{title:^60}")
        print("="*60)
        print(f"\n{message}")
        print("\n" + "="*60)
        
        if wait_for_input:
            self._pause()
    
    def show_confirmation(self, message: str, title: str = "确认") -> bool:
        """显示确认对话框
        
        Args:
            message: 确认消息
            title: 对话框标题
            
        Returns:
            bool: 用户是否确认
        """
        self._clear_screen()
        
        print("\n" + "="*60)
        print(f"{title:^60}")
        print("="*60)
        print(f"\n{message}")
        print("\n" + "="*60)
        
        while True:
            try:
                response = input("\n确认操作? (y/n): ").strip().lower()
                if response in ['y', 'yes', '是', '1', 'true']:
                    return True
                elif response in ['n', 'no', '否', '0', 'false']:
                    return False
                else:
                    print("请输入 y 或 n")
                    
            except KeyboardInterrupt:
                return False
            except EOFError:
                return False
    
    def show_input_dialog(self, prompt: str, title: str = "输入", 
                         default: str = None, required: bool = False) -> Optional[str]:
        """显示输入对话框
        
        Args:
            prompt: 输入提示
            title: 对话框标题
            default: 默认值
            required: 是否必填
            
        Returns:
            Optional[str]: 用户输入的值，如果取消则返回None
        """
        self._clear_screen()
        
        print("\n" + "="*60)
        print(f"{title:^60}")
        print("="*60)
        
        while True:
            try:
                if default:
                    user_input = input(f"\n{prompt} [{default}]: ").strip()
                    if not user_input:
                        return default
                else:
                    user_input = input(f"\n{prompt}: ").strip()
                
                if required and not user_input:
                    print("此项为必填项，请重新输入")
                    continue
                
                return user_input if user_input else None
                
            except KeyboardInterrupt:
                return None
            except EOFError:
                return None
    
    def show_list_selection(self, items: List[str], title: str = "选择", 
                           allow_multiple: bool = False) -> Optional[List[int]]:
        """显示列表选择对话框
        
        Args:
            items: 选项列表
            title: 对话框标题
            allow_multiple: 是否允许多选
            
        Returns:
            Optional[List[int]]: 选择的索引列表，如果取消则返回None
        """
        self._clear_screen()
        
        print("\n" + "="*60)
        print(f"{title:^60}")
        print("="*60)
        
        # 显示选项
        for i, item in enumerate(items, 1):
            print(f"{i:2d}. {item}")
        
        print(f" 0. 取消")
        print("="*60)
        
        if allow_multiple:
            print("\n多选模式: 输入数字，用逗号分隔 (如: 1,3,5)")
        
        while True:
            try:
                if allow_multiple:
                    choice_str = input("\n请选择 (多个选项用逗号分隔): ").strip()
                else:
                    choice_str = input("\n请选择: ").strip()
                
                if not choice_str:
                    print("请输入有效的选择")
                    continue
                
                if choice_str == '0':
                    return None
                
                if allow_multiple:
                    try:
                        choices = [int(x.strip()) for x in choice_str.split(',')]
                        
                        # 验证选择
                        valid_choices = []
                        for choice in choices:
                            if 1 <= choice <= len(items):
                                valid_choices.append(choice - 1)  # 转换为0基索引
                            else:
                                print(f"无效选择: {choice}")
                                break
                        else:
                            return valid_choices
                        
                    except ValueError:
                        print("请输入有效的数字")
                else:
                    try:
                        choice = int(choice_str)
                        if 1 <= choice <= len(items):
                            return [choice - 1]  # 转换为0基索引
                        else:
                            print(f"请输入 1 到 {len(items)} 之间的数字")
                    except ValueError:
                        print("请输入有效的数字")
                        
            except KeyboardInterrupt:
                return None
            except EOFError:
                return None
    
    def show_progress_menu(self, title: str, progress_callback: Callable) -> None:
        """显示进度菜单
        
        Args:
            title: 进度标题
            progress_callback: 进度回调函数，应该接受一个更新函数作为参数
        """
        self._clear_screen()
        
        print("\n" + "="*60)
        print(f"{title:^60}")
        print("="*60)
        
        def update_progress(current: int, total: int, message: str = ""):
            """更新进度显示"""
            if total > 0:
                percentage = (current / total) * 100
                bar_length = 40
                filled_length = int(bar_length * current // total)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                
                print(f"\r进度: |{bar}| {percentage:.1f}% ({current}/{total}) {message}", end='', flush=True)
            else:
                print(f"\r处理中... {message}", end='', flush=True)
        
        try:
            progress_callback(update_progress)
            print("\n\n操作完成！")
        except Exception as e:
            print(f"\n\n操作失败: {e}")
        
        self._pause()
    
    def get_current_menu_path(self) -> List[str]:
        """获取当前菜单路径
        
        Returns:
            List[str]: 菜单路径列表
        """
        path = []
        
        # 添加主菜单
        if self.main_menu:
            path.append(self.main_menu.get('title', '主菜单'))
        
        # 添加菜单栈中的菜单
        for menu in self.menu_stack:
            path.append(menu.get('title', '菜单'))
        
        # 添加当前菜单（如果不是主菜单）
        if self.current_menu and self.current_menu != self.main_menu:
            path.append(self.current_menu.get('title', '当前菜单'))
        
        return path
    
    def show_breadcrumb(self) -> None:
        """显示面包屑导航"""
        path = self.get_current_menu_path()
        if len(path) > 1:
            breadcrumb = ' > '.join(path)
            print(f"\n位置: {breadcrumb}")
    
    def reset_to_main_menu(self) -> None:
        """重置到主菜单"""
        self.current_menu = self.main_menu
        self.menu_stack.clear()


def create_simple_menu(title: str, options: List[Tuple[str, Callable]]) -> Dict[str, Any]:
    """创建简单菜单的辅助函数
    
    Args:
        title: 菜单标题
        options: 选项列表，每个选项为 (名称, 回调函数) 的元组
        
    Returns:
        Dict[str, Any]: 菜单配置字典
    """
    return {
        'title': title,
        'options': options
    }


def create_menu_option(name: str, callback: Callable) -> Tuple[str, Callable]:
    """创建菜单选项的辅助函数
    
    Args:
        name: 选项名称
        callback: 回调函数
        
    Returns:
        Tuple[str, Callable]: 菜单选项元组
    """
    return (name, callback)