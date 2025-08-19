# 🚀 快速开始指南

## 自动化发布流程

### 1. 首次设置

#### 配置 GitHub 仓库

```bash
# 在 GitHub 上创建新仓库
# 然后在本地项目中添加远程仓库
git remote add origin https://github.com/your-username/integrated-script.git
git branch -M master
git push -u origin master
```

#### 配置 GitHub Token（用于自动发布）

1. 访问 GitHub Settings → Developer settings → Personal access tokens
2. 创建新的 token，权限选择：`repo`, `workflow`
3. 在仓库设置中添加 Secret：`GITHUB_TOKEN`

### 2. 一键发布（推荐）

```cmd
# Windows 用户直接双击或运行
release.bat
```

这个脚本会：
- 显示当前版本
- 让你选择发布类型（patch/minor/major）
- 可选跳过测试/构建
- 可选自动推送到 GitHub
- 执行完整的发布流程

### 3. 手动发布

```bash
# 发布补丁版本（推荐用于 bug 修复）
python scripts/release.py patch

# 发布次要版本（推荐用于新功能）
python scripts/release.py minor --message "添加图像批量处理功能"

# 发布主要版本（推荐用于重大更改）
python scripts/release.py major --auto-push
```

### 4. 发布流程说明

```
本地开发 → 运行发布脚本 → 自动化流程
    ↓              ↓           ↓
代码修改      版本更新     GitHub Actions
    ↓              ↓           ↓
功能测试      本地构建     自动构建 EXE
    ↓              ↓           ↓
提交代码      推送标签     发布到 Releases
```

## 常用命令

### 版本管理

```bash
# 查看当前版本
python scripts/version_manager.py current

# 手动设置版本
python scripts/version_manager.py update 1.2.3

# 递增版本
python scripts/version_manager.py increment patch  # 1.0.0 → 1.0.1
python scripts/version_manager.py increment minor  # 1.0.0 → 1.1.0
python scripts/version_manager.py increment major  # 1.0.0 → 2.0.0
```

### 构建可执行文件

```bash
# 构建 EXE 文件
python build_exe.py

# 构建后的文件位置
# dist/integrated_script.exe
```

### Git 操作

```bash
# 查看状态
git status

# 提交更改
git add .
git commit -m "feat: 添加新功能"

# 推送到 GitHub
git push origin master

# 推送标签（触发自动发布）
git push origin v1.0.1
```

## 发布选项说明

### 版本类型

- **patch** (1.0.0 → 1.0.1): 用于 bug 修复
- **minor** (1.0.0 → 1.1.0): 用于新功能添加
- **major** (1.0.0 → 2.0.0): 用于重大更改或不兼容更新

### 发布参数

- `--skip-tests`: 跳过测试（快速发布）
- `--skip-build`: 跳过本地构建（依赖 GitHub Actions）
- `--auto-push`: 自动推送到 GitHub
- `--message "描述"`: 添加发布说明

## 故障排除

### 常见问题

1. **Python 找不到**
   ```bash
   # 确保 Python 在 PATH 中
   python --version
   ```

2. **Git 推送失败**
   ```bash
   # 检查远程仓库配置
   git remote -v
   
   # 检查认证
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

3. **构建失败**
   ```bash
   # 检查依赖
   pip install -r requirements.txt
   
   # 手动测试构建
   python build_exe.py
   ```

4. **GitHub Actions 失败**
   - 检查 GitHub 仓库的 Actions 页面
   - 确保 GITHUB_TOKEN 权限正确
   - 检查 workflow 文件语法

### 获取帮助

```bash
# 查看发布脚本帮助
python scripts/release.py --help

# 查看版本管理帮助
python scripts/version_manager.py --help

# 查看主程序帮助
python main.py --help
```

## 最佳实践

1. **发布前检查**
   - 运行测试确保代码质量
   - 更新文档和 CHANGELOG
   - 检查版本号是否合理

2. **发布说明**
   - 使用有意义的发布消息
   - 遵循语义化版本规范
   - 记录重要更改

3. **自动化优先**
   - 优先使用 `release.bat` 一键发布
   - 让 GitHub Actions 处理构建
   - 定期检查自动化流程

---

🎉 现在你可以开始使用自动化发布流程了！有问题请查看 README.md 或提交 Issue。