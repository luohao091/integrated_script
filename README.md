# 集成脚本工具 (Integrated Script)

一个以交互式菜单为主的多功能数据处理工具，面向计算机视觉数据集、图像与标签批量处理场景。

## 项目特点

- 交互式菜单驱动，开箱即用
- 覆盖 YOLO 数据集全流程（验证、清理、合并、转换）
- 图像/文件/标签批量处理能力完整
- 配置可加载/保存/重置，支持自定义
- Windows 兼容性适配

## 功能清单（以当前程序为准）

### YOLO 数据集处理

- CTDS 数据集转 YOLO（自动检测数据集类型并验证）
- YOLO 数据集转 CTDS（生成 `obj.names` + `obj_train_data` 结构）
- YOLO 数据集转 X-label（自动识别检测/分割）
- X-label/Labelme JSON 转 YOLO（自动识别检测/分割）
- 目标检测数据集验证（自动定位根目录）
- 目标分割数据集验证（含分割格式检查与无效标签处理）
- 清理不匹配文件（支持试运行）
- 合并多个数据集（相同类型 / 不同类型）

### 图像处理

- 图像格式转换
- 图像尺寸调整
- 图像压缩（支持批量与限尺寸）
- 修复 OpenCV 读取失败的图像
- 获取图像信息与统计

### 文件操作

- 数据集重命名（images/labels 同步）
- 数据集重命名（传统模式）
- 按扩展名组织文件
- 递归删除 JSON 文件
- 批量复制 / 批量移动

### 标签处理

- 创建空标签文件
- 翻转标签坐标
- 过滤标签类别
- 删除空标签及对应图像
- 删除仅包含指定类别的标签及对应图像

### 环境检查与配置（非 EXE 环境显示）

- 一键检查并修复环境
- 检查 Python 依赖
- 安装缺失依赖
- 初始化工作目录

### 配置管理

- 查看 / 修改配置
- 加载 / 保存配置文件
- 重置为默认配置

## 安装与运行

### 环境要求

- Python 3.7+
- Windows / Linux / macOS

### 安装依赖

```bash
pip install -r requirements.txt
```

或开发模式：

```bash
pip install -e .
```

### 启动方式

```bash
# 方式一：直接运行入口脚本
python main.py

# 方式二：安装后使用命令
integrated-script
```

### 常用命令行参数

```bash
integrated-script --config path/to/config.yaml
integrated-script --log-level DEBUG
integrated-script --build
```

说明：当前程序以交互式菜单为主，命令行参数仅用于配置、日志与打包辅助。

## 构建可执行文件

PyInstaller 需安装在当前 Python 环境中：

```bash
pip install pyinstaller
python build_exe.py
```

生成产物位于 `dist/` 目录。

## 配置文件

默认配置文件路径：`config/default_config.yaml`。

可通过交互式菜单的“配置管理”进行查看、修改、保存或重置。

## 项目结构

```
integrated_script/
├── src/
│   └── integrated_script/
│       ├── config/          # 配置管理
│       ├── core/            # 核心功能
│       ├── processors/      # 处理器模块
│       └── ui/              # 交互式界面
├── config/                  # 配置文件
├── main.py                  # 主入口
├── build_exe.py             # 打包脚本
└── requirements.txt         # 依赖列表
```

## 主要依赖

- `opencv-python-headless` (>=4.5.0)
- `Pillow` (>=9.0.0)
- `PyYAML` (>=6.0)
- `tqdm` (>=4.64.0)

## 开发说明

```bash
# 运行测试
pytest

# 代码格式化
black .

# 风格检查
flake8 .

# 类型检查
mypy src/
```

## 自动化发布

项目支持脚本化发布流程：

```bash
python scripts/release.py
```

支持 `patch/minor/major` 版本更新，详情见 `scripts/release.py` 注释。

## 许可证

MIT License，详见 `LICENSE`。
