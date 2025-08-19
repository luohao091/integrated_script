#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ui模块

用户界面相关功能

提供命令行界面、交互式界面等用户交互功能。
"""

from .cli import CLIInterface
from .interactive import InteractiveInterface
from .menu import MenuSystem

__all__ = ["CLIInterface", "InteractiveInterface", "MenuSystem"]

__version__ = "1.0.0"
__author__ = "Integrated Script Team"
