# OpenClaw 开发环境搭建 - GitHub & Reddit 资料

**来源**: Grok-mcp Web Search - GitHub & Reddit
**查询时间**: 2026-02-23
**查询关键词**: "openclaw development setup from source 2025 2026"
**用途**: 基础编译流程、开发模式配置、故障排查实战

---

## 搜索结果概览

本文档收集了 GitHub 和 Reddit 上关于 OpenClaw 从源码搭建开发环境的最新资料和社区经验。

---

## 1. OpenClaw 官方仓库开发从源码搭建

**标题**: OpenClaw 官方仓库开发从源码搭建
**链接**: https://github.com/openclaw/openclaw
**来源**: GitHub 官方仓库

### 内容摘要

2026 版 OpenClaw 源码开发设置：git clone 仓库，pnpm install 安装依赖，pnpm ui:build 构建 UI，pnpm build 编译项目，pnpm gateway:watch 进入自动重载开发模式。Node.js >=22 必需。

### 完整开发流程

#### 1. 系统要求

```bash
# Node.js 版本
node --version  # 应该 >= 22.12.0

# pnpm 版本
pnpm --version  # 应该 = 10.23.0
```

#### 2. 克隆仓库

```bash
# HTTPS
git clone https://github.com/openclaw/openclaw.git

# SSH
git clone git@github.com:openclaw/openclaw.git

cd openclaw
```

#### 3. 安装依赖

```bash
# 安装所有依赖
pnpm install

# 如果遇到构建审批问题
pnpm approve-builds
```

#### 4. 构建项目

```bash
# 构建 UI（React + Vite）
pnpm ui:build

# 编译 TypeScript
pnpm build
```

#### 5. 开发模式

```bash
# 方式 1: 使用 gateway:watch（推荐）
pnpm gateway:watch

# 方式 2: 使用 dev 脚本
pnpm dev

# 方式 3: 使用 gateway:dev
pnpm gateway:dev
```

#### 6. 全局链接（可选）

```bash
# 全局链接 openclaw 命令
pnpm link --global

# 验证
openclaw --version
```

### 开发工作流

```bash
# 1. 修改代码
vim src/index.ts

# 2. 自动重新编译（如果运行了 gateway:watch）
# 或手动编译
pnpm build

# 3. 运行测试
pnpm test

# 4. 代码质量检查
pnpm check
```

---

## 2. OpenClaw Ansible 开发模式从源码安装

**标题**: OpenClaw Ansible 开发模式从源码安装
**链接**: https://github.com/openclaw/openclaw-ansible
**来源**: GitHub 自动化工具

### 内容摘要

Ansible 自动化开发安装：clone ansible 仓库，ansible-playbook -e openclaw_install_mode=development，从 GitHub 源码克隆构建，提供 openclaw-rebuild 等开发别名，适用于 Debian/Ubuntu。

### 自动化安装流程

#### 1. 安装 Ansible

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ansible

# macOS
brew install ansible
```

#### 2. 克隆 Ansible 仓库

```bash
git clone https://github.com/openclaw/openclaw-ansible.git
cd openclaw-ansible
```

#### 3. 配置 inventory

```ini
# inventory.ini
[openclaw]
localhost ansible_connection=local
```

#### 4. 运行 playbook（开发模式）

```bash
ansible-playbook \
  -i inventory.ini \
  -e openclaw_install_mode=development \
  playbook.yml
```

### 开发别名

安装后会自动创建以下别名：

```bash
# 重新构建 OpenClaw
openclaw-rebuild

# 更新源码
openclaw-update

# 查看日志
openclaw-logs

# 重启服务
openclaw-restart
```

### 配置文件位置

```bash
# 源码目录
~/openclaw

# 配置文件
~/.config/openclaw/

# 日志文件
~/.local/share/openclaw/logs/
```

---

## 3. Harbor OpenClaw Docker 从源码构建指南

**标题**: Harbor OpenClaw Docker 从源码构建指南
**链接**: https://github.com/av/harbor/wiki/2.3.70-Satellite-OpenClaw
**来源**: Harbor 项目 Wiki

### 内容摘要

Docker 推荐从源码构建：harbor build openclaw，设置 OPENCLAW_GIT_REF 为主分支，配置模型和 Control UI，支持本地 Gateway 开发，2026 版持久化设置。

### Docker 构建流程

#### 1. 安装 Harbor

```bash
# 克隆 Harbor
git clone https://github.com/av/harbor.git
cd harbor

# 安装依赖
pnpm install
```

#### 2. 配置 OpenClaw

```yaml
# harbor.yml
satellites:
  openclaw:
    git_ref: main  # 或特定分支/标签
    build_from_source: true
    models:
      - provider: openai
        model: gpt-4
    control_ui:
      enabled: true
      port: 3000
```

#### 3. 构建 OpenClaw

```bash
# 从源码构建
harbor build openclaw

# 查看构建日志
harbor logs openclaw
```

#### 4. 运行 OpenClaw

```bash
# 启动
harbor start openclaw

# 查看状态
harbor status openclaw

# 停止
harbor stop openclaw
```

### Docker Compose 配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  openclaw:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ./src:/app/src
      - ./config:/app/config
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=development
      - OPENCLAW_GIT_REF=main
    command: pnpm gateway:watch
```

### 本地开发配置

```dockerfile
# Dockerfile.dev
FROM node:22-alpine

RUN corepack enable pnpm

WORKDIR /app

# 复制依赖文件
COPY package.json pnpm-lock.yaml ./

# 安装依赖
RUN pnpm install --frozen-lockfile

# 复制源码
COPY . .

# 构建
RUN pnpm ui:build && pnpm build

# 开发模式
CMD ["pnpm", "gateway:watch"]
```

---

## 4. OpenClaw Termux Android 从源码构建

**标题**: OpenClaw Termux Android 从源码构建
**链接**: https://github.com/mithun50/openclaw-termux
**来源**: 社区移动端适配

### 内容摘要

Android Flutter 应用源码构建：git clone 仓库，cd flutter_app，flutter build apk --release，实现一键 AI Gateway 设置，适用于移动开发环境。

### Android 构建流程

#### 1. 安装 Termux

```bash
# 从 F-Droid 或 GitHub 安装 Termux
# https://github.com/termux/termux-app
```

#### 2. 安装依赖

```bash
# 更新包
pkg update && pkg upgrade

# 安装 Node.js
pkg install nodejs

# 安装 pnpm
corepack enable pnpm

# 安装 Git
pkg install git
```

#### 3. 克隆并构建

```bash
# 克隆仓库
git clone https://github.com/mithun50/openclaw-termux.git
cd openclaw-termux

# 安装依赖
pnpm install

# 构建
pnpm build

# 运行
pnpm start
```

### Flutter 应用构建

```bash
# 进入 Flutter 应用目录
cd flutter_app

# 获取依赖
flutter pub get

# 构建 APK
flutter build apk --release

# 安装到设备
flutter install
```

---

## 开发环境最佳实践

### 1. 版本管理

```bash
# 使用 asdf 管理 Node.js 版本
asdf install nodejs 22.12.0
asdf global nodejs 22.12.0

# 或使用 nvm
nvm install 22.12.0
nvm use 22.12.0
```

### 2. 依赖管理

```bash
# 使用 pnpm（推荐）
corepack enable pnpm
pnpm install

# 清理缓存
pnpm store prune

# 更新依赖
pnpm update
```

### 3. 开发工具

```bash
# VSCode 扩展
code --install-extension dbaeumer.vscode-eslint
code --install-extension esbenp.prettier-vscode
code --install-extension bradlc.vscode-tailwindcss

# 配置 Git hooks
pnpm husky install
```

### 4. 调试配置

```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "node",
      "request": "launch",
      "name": "Debug OpenClaw",
      "runtimeExecutable": "pnpm",
      "runtimeArgs": ["dev"],
      "console": "integratedTerminal",
      "internalConsoleOptions": "neverOpen"
    }
  ]
}
```

---

## 常见问题排查

### Q: pnpm install 失败

**可能原因**：
1. Node.js 版本不符合要求
2. 网络问题
3. 权限问题

**解决方案**：
```bash
# 检查 Node.js 版本
node --version

# 清理缓存
pnpm store prune

# 使用镜像
pnpm config set registry https://registry.npmmirror.com

# 重新安装
rm -rf node_modules pnpm-lock.yaml
pnpm install
```

### Q: pnpm build 失败

**可能原因**：
1. TypeScript 类型错误
2. 依赖缺失
3. 配置错误

**解决方案**：
```bash
# 类型检查
pnpm check

# 重新安装依赖
pnpm install

# 查看详细错误
pnpm build --verbose
```

### Q: gateway:watch 无法启动

**可能原因**：
1. 端口被占用
2. UI 未构建
3. 环境变量缺失

**解决方案**：
```bash
# 检查端口
lsof -i :3000

# 构建 UI
pnpm ui:build

# 检查环境变量
cat .env

# 重新启动
pnpm gateway:watch
```

---

## 生产环境部署

### 1. 构建生产版本

```bash
# 设置环境变量
export NODE_ENV=production

# 构建
pnpm ui:build
pnpm build

# 验证
node dist/index.js --version
```

### 2. 使用 PM2 管理

```bash
# 安装 PM2
pnpm add -g pm2

# 启动
pm2 start dist/index.js --name openclaw

# 查看状态
pm2 status

# 查看日志
pm2 logs openclaw

# 重启
pm2 restart openclaw
```

### 3. 使用 Docker 部署

```bash
# 构建镜像
docker build -t openclaw:latest .

# 运行容器
docker run -d \
  --name openclaw \
  -p 3000:3000 \
  -v $(pwd)/config:/app/config \
  openclaw:latest

# 查看日志
docker logs -f openclaw
```

---

**参考资料**:
- OpenClaw GitHub: https://github.com/openclaw/openclaw
- OpenClaw 文档: https://docs.openclaw.ai/
- Harbor 项目: https://github.com/av/harbor
