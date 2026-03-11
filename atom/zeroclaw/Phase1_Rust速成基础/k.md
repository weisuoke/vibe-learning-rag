# Phase 1: Rust 速成基础 - 知识点列表

> 目标：掌握 20% 的 Rust 核心知识，能读懂 ZeroClaw 源码
> 学习时长：第 1-2 周
> 前置要求：TypeScript/前端开发经验

---

## 知识点列表

### 01. 所有权与借用
- 所有权三规则、移动语义、借用（&T / &mut T）
- 前端类比：JS 值传递 vs 引用传递
- ZeroClaw 场景：理解函数签名中的 `&self`、`&mut self`、`self`

### 02. Struct 与 Enum
- struct 定义与 impl、enum 与模式匹配、derive 宏
- 前端类比：TypeScript interface / type / union type
- ZeroClaw 场景：Agent、Message、Config、Role 等核心数据结构

### 03. Trait 与泛型
- trait 定义与实现、泛型约束（where）、关联类型
- 前端类比：TypeScript interface + generics
- ZeroClaw 场景：Provider/Tool/Memory/Channel 四大 Trait 定义

### 04. 错误处理 Result/Option
- Result<T, E>、Option<T>、? 运算符、unwrap 与 expect
- 前端类比：try/catch + optional chaining (?.)
- ZeroClaw 场景：每个函数都返回 `Result<T, anyhow::Error>`

### 05. 动态分发与 Trait Object
- Box<dyn Trait>、Arc<dyn Trait>、对象安全、静态 vs 动态分发
- 前端类比：依赖注入容器、多态接口
- ZeroClaw 场景：`Box<dyn Provider>`、`Arc<dyn Memory>`、`Vec<Box<dyn Tool>>`

### 06. async/await 与 Tokio
- async fn、.await、Future trait、tokio::spawn、Send + 'static
- 前端类比：JS Promise + Event Loop + async/await
- ZeroClaw 场景：整个运行时基于 Tokio，所有 Trait 方法都是 async

### 07. Cargo 与模块系统
- Cargo.toml、依赖管理、workspace、mod/use/pub、feature flags
- 前端类比：npm/package.json + ES modules
- ZeroClaw 场景：理解项目结构、模块组织、条件编译

### 08. 常用库速查（serde/anyhow/clap）
- serde 序列化、anyhow 错误处理、clap CLI 解析、tracing 日志
- 前端类比：axios/yargs/zod/winston
- ZeroClaw 场景：配置解析、错误处理、CLI 命令、日志输出
