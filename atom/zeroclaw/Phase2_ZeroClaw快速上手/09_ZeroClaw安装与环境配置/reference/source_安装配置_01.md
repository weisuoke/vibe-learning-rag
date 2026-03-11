---
type: source_code_analysis
source: sourcecode/zeroclaw
analyzed_files:
  - Cargo.toml
  - scripts/bootstrap.sh
  - zeroclaw_install.sh
  - Dockerfile
  - .github/workflows/ci.yml
  - .github/workflows/release.yml
  - src/config/mod.rs
  - src/main.rs
  - rust-toolchain.toml
analyzed_at: 2026-03-11
knowledge_point: 09_ZeroClaw安装与环境配置
---

# 源码分析：ZeroClaw 安装与环境配置

## 分析的文件

- `Cargo.toml` - 项目元数据、依赖管理、Feature Flags、构建优化配置
- `scripts/bootstrap.sh` - 一键安装引导脚本
- `zeroclaw_install.sh` - 本地安装脚本
- `Dockerfile` - 多阶段 Docker 构建
- `.github/workflows/` - CI/CD 工作流
- `src/config/mod.rs` - 配置系统实现
- `src/main.rs` - CLI 入口

## 关键发现

### 1. 安装方式（5种）

#### A. 一键远程安装
```bash
curl -LsSf https://raw.githubusercontent.com/zeroclaw-labs/zeroclaw/master/scripts/bootstrap.sh | bash
```

#### B. 本地引导安装
```bash
./zeroclaw_install.sh --guided           # 交互式
./zeroclaw_install.sh --prefer-prebuilt  # 优先预编译
./zeroclaw_install.sh --force-source-build  # 强制源码编译
./zeroclaw_install.sh --install-system-deps --install-rust  # 全自动
```

**关键参数：**
- `--guided` - 交互式引导
- `--install-system-deps` - 自动安装系统依赖
- `--install-rust` - 自动安装 Rust
- `--prefer-prebuilt` - 优先预编译二进制
- `--prebuilt-only` - 仅使用预编译（不回退到源码）
- `--force-source-build` - 强制源码编译
- `--docker` - Docker 模式
- `--onboard` - 安装后自动 onboard
- `--api-key <key>` - 非交互式 API key
- `--provider <id>` - 指定 Provider（默认 openrouter）
- `--model <id>` - 指定模型

#### C. Cargo 安装
```bash
cargo install --path . --force --locked
```

#### D. Docker 安装
```bash
docker build -t zeroclaw-bootstrap:local .
docker run -it zeroclaw-bootstrap:local
```

#### E. 预编译二进制
支持平台：
- x86_64-unknown-linux-gnu
- aarch64-unknown-linux-gnu
- armv7-unknown-linux-gnueabihf
- x86_64-apple-darwin
- aarch64-apple-darwin
- x86_64-pc-windows-msvc

下载地址：`https://github.com/zeroclaw-labs/zeroclaw/releases/latest/download/zeroclaw-{target}.tar.gz`

### 2. 系统要求

#### Rust 版本
- MSRV: 1.87
- 推荐: Stable 最新版

#### 构建资源要求
| 资源 | 最低要求 | 推荐 |
|------|---------|------|
| RAM + swap | 2 GB | 4 GB+ |
| 磁盘空间 | 6 GB | 10 GB+ |

#### 平台依赖

**Linux (Debian/Ubuntu):** build-essential, pkg-config, git, curl, openssl-dev, perl, ca-certificates

**Linux (Fedora/RHEL):** gcc, gcc-c++, make, pkgconf-pkg-config, git, curl, openssl-devel, perl

**Linux (Arch):** gcc, make, pkgconf, git, curl, openssl, perl, ca-certificates

**Linux (Alpine):** bash, build-base, pkgconf, git, curl, openssl-dev, perl, ca-certificates

**macOS:** Xcode Command Line Tools (`xcode-select --install`)

**Windows:** Visual Studio Build Tools 2022 (MSVC + Windows SDK, "Desktop development with C++")

### 3. Feature Flags

**默认：** 无（最小构建）

| Feature | 说明 | 新增依赖 |
|---------|------|---------|
| `hardware` | USB + 串口支持 | nusb, tokio-serial |
| `channel-matrix` | Matrix 协议 | matrix-sdk |
| `channel-lark` | 飞书通道 | prost |
| `memory-postgres` | PostgreSQL 后端 | postgres |
| `observability-otel` | OpenTelemetry | opentelemetry |
| `peripheral-rpi` | 树莓派 GPIO | rppal |
| `browser-native` | 浏览器自动化 | fantoccini |
| `sandbox-landlock` | Landlock 沙箱 | - |
| `sandbox-bubblewrap` | Bubblewrap 沙箱 | - |
| `probe` | STM32 探测 | probe-rs (~50 deps) |
| `rag-pdf` | PDF 文档 RAG | pdf-extract |
| `whatsapp-web` | WhatsApp 客户端 | wa-rs |

### 4. 环境变量（完整列表）

#### 核心运行时
| 变量 | 用途 | 默认值 |
|------|------|-------|
| ZEROCLAW_API_KEY | Provider API Key | 无 |
| API_KEY | 通用回退 API Key | 无 |
| ZEROCLAW_PROVIDER | 默认 Provider | openrouter |
| ZEROCLAW_MODEL | 默认模型 | 无 |
| ZEROCLAW_TEMPERATURE | 温度 (0.0-2.0) | 0.7 |
| ZEROCLAW_WORKSPACE | 工作区目录 | ~/.zeroclaw |
| ZEROCLAW_CONFIG_DIR | 配置目录 | ~/.zeroclaw |

#### Gateway 配置
| 变量 | 用途 | 默认值 |
|------|------|-------|
| ZEROCLAW_GATEWAY_PORT | 网关端口 | 42617 |
| ZEROCLAW_GATEWAY_HOST | 绑定地址 | [::] |
| ZEROCLAW_ALLOW_PUBLIC_BIND | 允许公网绑定 | false |

#### 存储与数据库
| 变量 | 用途 | 默认值 |
|------|------|-------|
| ZEROCLAW_STORAGE_PROVIDER | 存储后端 | sqlite |
| ZEROCLAW_STORAGE_DB_URL | 数据库连接 URL | 无 |

#### 代理配置
| 变量 | 用途 | 默认值 |
|------|------|-------|
| ZEROCLAW_PROXY_ENABLED | 启用代理 | false |
| ZEROCLAW_HTTP_PROXY | HTTP 代理 | 无 |
| ZEROCLAW_HTTPS_PROXY | HTTPS 代理 | 无 |
| ZEROCLAW_ALL_PROXY | SOCKS 代理 | 无 |
| ZEROCLAW_NO_PROXY | 排除域名 | 无 |

#### Provider API Keys
```
OPENROUTER_API_KEY=sk-or-v1-...
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
GROQ_API_KEY=...
MISTRAL_API_KEY=...
DEEPSEEK_API_KEY=...
OLLAMA_API_KEY=...
```

### 5. 配置文件系统

#### 配置解析优先级
1. `ZEROCLAW_CONFIG_DIR` 环境变量
2. `ZEROCLAW_WORKSPACE` 环境变量
3. 活跃工作区标记 (`active_workspace.toml`)
4. 默认: `~/.zeroclaw/config.toml`

#### config.toml 结构
```toml
workspace_dir = "/path/to/workspace"
api_key = "sk-..."
api_url = "http://custom-endpoint"
default_provider = "openrouter"
default_model = "anthropic/claude-sonnet-4-6"
default_temperature = 0.7

[gateway]
port = 42617
host = "[::]"
allow_public_bind = true

[autonomy]
[security]
[runtime]
[reliability]
[scheduler]
[agent]
[skills]
[memory]
[channels_config]
[observability]
```

### 6. 构建优化配置

```toml
[profile.release]
opt-level = "z"        # 体积优化
lto = "fat"            # 最大链接时优化
codegen-units = 1      # 低内存设备友好
strip = true           # 去除调试符号
panic = "abort"        # 减小二进制体积

[profile.release-fast]
inherits = "release"
codegen-units = 8      # 并行编译（需 16GB+ RAM）
```

### 7. Docker 多阶段构建

- **Stage 1 (builder):** Rust 1.93-slim, 编译 release 二进制
- **Stage 2 (dev):** Debian trixie-slim, 开发运行时
- **Stage 3 (release):** Distroless cc-debian13, 生产运行时

### 8. 性能指标

| 指标 | 数值 |
|------|------|
| 二进制大小 | ~8.8 MB |
| 运行内存 | ~3.9-4.1 MB |
| 启动时间 | <10ms (0.8GHz) |
| 编译时间 | ~30-40分钟（视硬件） |

## 代码片段

### bootstrap.sh 预检逻辑
```bash
# RAM check
ZEROCLAW_BOOTSTRAP_MIN_RAM_MB=${ZEROCLAW_BOOTSTRAP_MIN_RAM_MB:-2048}
# Disk check
ZEROCLAW_BOOTSTRAP_MIN_DISK_MB=${ZEROCLAW_BOOTSTRAP_MIN_DISK_MB:-6144}
```

### 配置加载路径
```rust
// src/config/mod.rs - 配置路径解析
fn resolve_config_dir() -> PathBuf {
    if let Ok(dir) = env::var("ZEROCLAW_CONFIG_DIR") { return dir.into(); }
    if let Ok(dir) = env::var("ZEROCLAW_WORKSPACE") { return dir.into(); }
    // ... 最终 fallback 到 ~/.zeroclaw
}
```
