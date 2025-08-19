# 代码质量自动化配置

本项目已配置完整的代码质量自动化工具链，包括代码格式化、导入排序、代码质量检查和类型检查。

## 🛠️ 已安装的工具

- **Black**: 代码格式化工具
- **isort**: 导入语句排序工具
- **Flake8**: 代码质量检查工具
- **MyPy**: 静态类型检查工具
- **pre-commit**: Git 提交前检查工具

## 📁 配置文件

- `pyproject.toml`: 包含所有工具的配置
- `.pre-commit-config.yaml`: pre-commit hooks 配置
- `.github/workflows/code-quality.yml`: GitHub Actions 自动化流程
- `scripts/format_code.py`: 跨平台代码格式化脚本


## 🚀 使用方法

### 1. 基本命令

```bash
# 格式化代码
make format
# 或者
python scripts/format_code.py --format-only

# 检查代码质量（不修改文件）
make format-check
# 或者
python scripts/format_code.py

# 运行所有检查
make format-all
```

### 2. Pre-commit 检查

```bash
# 运行 pre-commit 检查所有文件
pre-commit run --all-files

# 运行 pre-commit 检查暂存文件
pre-commit run
```

### 3. 单独运行工具

```bash
# Black 格式化
black src/integrated_script/ scripts/

# isort 导入排序
isort src/integrated_script/ scripts/

# Flake8 质量检查
flake8 src/integrated_script/ scripts/

# MyPy 类型检查
mypy src/integrated_script/
```

## 🔧 工具配置说明

### Black 配置
- 行长度: 88 字符
- 目标 Python 版本: 3.9+
- 跳过字符串标准化: false

### isort 配置
- 配置文件: Black 兼容模式
- 行长度: 88 字符
- 多行输出模式: 3 (垂直悬挂缩进)
- 强制单行导入: true

### Flake8 配置
- 最大行长度: 88 字符
- 排除目录: `__pycache__`, `.git`, `build`, `dist`
- 忽略错误: E203, W503 (与 Black 兼容)

### MyPy 配置
- Python 版本: 3.9
- 忽略缺失导入: true
- 显示错误代码: true
- 较宽松的类型检查设置

## 🔄 自动化流程

### GitHub Actions
项目配置了 GitHub Actions 工作流，在以下情况下自动运行：
- 推送到任何分支
- 创建 Pull Request

工作流包括：
1. **代码质量检查**: 运行 Black、isort、Flake8、MyPy
2. **自动格式化**: 推送到 main 分支时自动格式化并提交

### Pre-commit Hooks
项目已配置 `.pre-commit-config.yaml`，可以直接使用：

```bash
# 安装 Git hooks
pre-commit install

# 运行所有检查
pre-commit run --all-files
```

## 📊 检查项目

Pre-commit hooks 包含以下检查：

1. **行尾空白字符检查**: 移除行尾多余空格
2. **文件末尾换行符检查**: 确保文件以换行符结尾
3. **YAML/JSON/TOML 语法检查**: 验证配置文件语法
4. **合并冲突检查**: 检查是否有未解决的合并冲突
5. **大文件检查**: 防止提交过大的文件
6. **Black 格式检查**: 验证代码格式
7. **isort 检查**: 验证导入排序
8. **Flake8 质量检查**: 代码质量和风格检查
9. **MyPy 类型检查**: 静态类型检查

## 🎯 最佳实践

1. **提交前检查**: 每次提交前运行 `pre-commit run --all-files`
2. **自动修复**: 使用 `make format` 自动修复格式问题
3. **持续集成**: 依赖 GitHub Actions 进行自动化检查
4. **代码审查**: 关注 Flake8 报告的代码质量问题

## ⚠️ 注意事项

1. **Flake8 警告**: 当前代码库存在一些 Flake8 警告，主要是：
   - 未使用的导入 (F401)
   - 行长度超限 (E501)
   - f-string 缺少占位符 (F541)
   - 变量重定义 (F811)

2. **MyPy 类型检查**: 已配置为较宽松模式，当前存在一些类型错误需要逐步修复

3. **跨平台兼容**: 所有脚本都支持 Windows 和 Unix 系统

## 🔍 故障排除

如果遇到问题，请检查：

1. **Python 版本**: 确保使用 Python 3.9+
2. **依赖安装**: 运行 `pip install -r requirements-dev.txt`
3. **路径问题**: 确保在项目根目录运行命令
4. **网络问题**: 如果 pre-commit 初始化失败，可以使用 `make format` 等命令代替

---

通过这套自动化配置，项目代码质量将得到有效保障，开发效率也会显著提升。
