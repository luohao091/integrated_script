# 集成脚本工具 (Integrated Script)

一个功能强大的模块化数据处理整合脚本项目，专为计算机视觉和数据处理任务而设计。

## 🚀 项目特色

**多功能数据处理平台** - 集成了YOLO数据集处理、图像处理、文件管理等多种功能于一体，提供统一的操作界面和工作流程。

## 📋 主要功能模块

- **YOLO数据集处理** - 支持YOLO格式数据集的验证、转换和管理
- **图像处理** - 提供图像格式转换、批量处理等功能
- **数据集管理** - 数据集分割、组织和预处理
- **标签处理** - 标签文件的创建、编辑和验证
- **文件管理** - 智能文件组织和批量操作

## 🎯 使用场景

- **计算机视觉项目** - YOLO模型训练数据准备
- **数据科学工作流** - 大规模图像数据预处理
- **文件批量处理** - 自动化文件组织和管理
- **标注工作辅助** - 标签文件生成和验证

## 🛠️ 安装说明

### 环境要求

- Python 3.7+
- Windows/Linux/macOS

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/your-username/integrated_script.git
cd integrated_script

# 安装依赖
pip install -r requirements.txt

# 或使用开发模式安装
pip install -e .
```

### 开发环境安装

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 安装文档依赖
pip install -e ".[docs]"
```

## 📖 使用方法

### 交互式模式

```bash
# 启动交互式界面
python main.py --interactive
```

### 命令行模式

```bash
# YOLO数据集验证
python main.py yolo validate /path/to/dataset

# 图像格式转换
python main.py image convert /path/to/images

# 文件组织
python main.py file organize /path/to/files

# 数据集分割
python main.py dataset split /path/to/dataset

# 创建标签文件
python main.py label create /path/to/images
```

### 可执行文件使用

从 [Releases](https://github.com/luohao091/integrated_script/releases) 页面下载最新的 `integrated_script.exe` 文件，直接运行：

```cmd
# Windows 命令行
integrated_script.exe --help
integrated_script.exe yolo validate /path/to/dataset
integrated_script.exe image convert /path/to/images
```

### 配置文件

项目支持YAML配置文件自定义，默认配置文件位于 `config/default_config.yaml`。

## 💡 技术特点

- **模块化设计** - 清晰的架构，易于扩展和维护
- **双模式操作** - 支持命令行和交互式界面
- **跨平台兼容** - 特别优化了Windows系统兼容性
- **进度可视化** - 内置进度条和日志系统
- **配置灵活** - 支持YAML配置文件自定义

## 📁 项目结构

```
integrated_script/
├── src/
│   └── integrated_script/
│       ├── config/          # 配置管理
│       ├── core/            # 核心功能
│       ├── processors/      # 处理器模块
│       └── ui/              # 用户界面
├── config/                  # 配置文件
├── main.py                  # 主入口
├── pyproject.toml          # 项目配置
└── requirements.txt        # 依赖列表
```

## 🔧 核心依赖

- **OpenCV** (>=4.5.0) - 图像处理核心引擎
- **Pillow** (>=9.0.0) - 图像格式支持
- **PyYAML** (>=6.0) - 配置文件解析
- **tqdm** (>=4.64.0) - 进度显示

## 🤝 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📝 开发说明

### 环境要求

- Python 3.8+
- Windows 10/11 (主要支持平台)
- 推荐使用虚拟环境

### 开发安装

```bash
# 克隆仓库
git clone https://github.com/luohao091/integrated_script
cd integrated-script

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows

# 安装开发依赖
pip install -e .
pip install -r requirements-dev.txt
```

### 自动化发布

项目支持完全自动化的发布流程：

#### 快速发布（推荐）

```cmd
# Windows 一键发布
release.bat
```

#### 手动发布

```bash
# 发布补丁版本 (1.0.0 → 1.0.1)
python scripts/release.py patch

# 发布次要版本 (1.0.0 → 1.1.0)
python scripts/release.py minor --message "添加新功能"

# 发布主要版本 (1.0.0 → 2.0.0)
python scripts/release.py major --auto-push

# 跳过测试和构建（快速发布）
python scripts/release.py patch --skip-tests --skip-build
```

#### 版本管理

```bash
# 查看当前版本
python scripts/version_manager.py current

# 手动更新版本
python scripts/version_manager.py update 1.2.3
```

#### 发布流程说明

1. **本地发布**：运行发布脚本，自动更新版本、运行测试、构建EXE
2. **推送代码**：自动或手动推送代码和标签到GitHub
3. **自动构建**：GitHub Actions自动构建并发布到Releases
4. **下载使用**：用户可从Releases页面下载最新版本

### 运行测试

```bash
# 运行所有测试
pytest

# 运行测试并生成覆盖率报告
pytest --cov=integrated_script
```

### 代码格式化

```bash
# 格式化代码
black .

# 检查代码风格
flake8 .

# 类型检查
mypy src/
```

### 代码规范

- 使用 Black 进行代码格式化
- 使用 isort 进行导入排序
- 遵循 PEP 8 编码规范
- 编写单元测试

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 👥 作者

- **luohao091** - *初始工作* - [luohao.622@gmail.com](mailto:luohao.622@gmail.com)

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者和用户。

---

**适合数据科学家、计算机视觉工程师和需要处理大量图像数据的开发者使用。**