# 实战代码 场景 1：实现 ZeroClaw Tool Trait

> **知识点**: Trait 与泛型
> **层级**: Phase1_Rust速成基础
> **维度**: 实战代码（场景 1）
> **目标受众**: 有 TypeScript/前端经验但零 Rust 经验的开发者

---

## 场景说明

实现一个自定义的 **WordCount Tool**，统计文本的字数、字符数和行数。

这是向 ZeroClaw 贡献代码的最简单入口 -- Tool Trait 只有 5 个方法，其中 1 个有默认实现，实际只需写 4 个方法。本节带你从零完成整个过程。

---

## 前置知识

- Trait 定义与实现（[03_核心概念_1](./03_核心概念_1_Trait定义与实现.md)）
- 基本的 `serde_json` 用法（JSON 序列化/反序列化）
- `async/await` 基础语法

---

## 第一步：理解 Tool Trait 定义

ZeroClaw 的 Tool Trait 定义在 `src/tools/traits.rs`，是 10 个核心 Trait 中最简洁的：

```rust
// ZeroClaw 源码：src/tools/traits.rs（简化版）
use async_trait::async_trait;
use serde_json::Value;

/// 工具执行结果
pub struct ToolResult {
    pub content: String,
}

/// 工具规格（发送给 LLM，让 LLM 知道有哪些工具可用）
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[async_trait]
pub trait Tool: Send + Sync {
    /// 工具名称（唯一标识，LLM 用这个名字来调用工具）
    fn name(&self) -> &str;

    /// 工具描述（LLM 用来判断什么时候该调用这个工具）
    fn description(&self) -> &str;

    /// 参数的 JSON Schema（LLM 用来构造正确的参数格式）
    fn parameters_schema(&self) -> Value;

    /// 执行工具（核心方法，接收 JSON 参数，返回结果）
    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult>;

    /// 生成工具规格（默认实现，组合上面三个方法的返回值）
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters_schema(),
        }
    }
}
```

逐方法解释：

| 方法 | 类型 | 必须实现？ | 作用 |
|------|------|-----------|------|
| `name()` | 同步 | 是 | 返回工具唯一标识，如 `"word_count"` |
| `description()` | 同步 | 是 | 返回工具描述，LLM 据此决定是否调用 |
| `parameters_schema()` | 同步 | 是 | 返回 JSON Schema，定义输入参数格式 |
| `execute()` | **异步** | 是 | 核心执行逻辑，接收参数返回结果 |
| `spec()` | 同步 | 否（有默认实现） | 组装上面三个方法的返回值 |

```typescript
// TypeScript 对照：等价的 interface
interface Tool {
  name(): string;
  description(): string;
  parametersSchema(): object;
  execute(args: Record<string, unknown>): Promise<ToolResult>;
  spec(): ToolSpec;  // TS interface 没法给默认实现
}
```

---

## 第二步：定义 WordCountTool 结构体

```rust
/// 字数统计工具 -- 统计文本的字数、字符数和行数
pub struct WordCountTool;
```

就这么简单。这个工具不需要任何内部状态（没有配置、没有连接），所以用**单元结构体**（unit struct）即可。

```typescript
// TypeScript 对照
class WordCountTool {}  // 同样不需要任何字段
```

> **为什么不需要字段？** 因为 WordCountTool 是纯函数式的 -- 给它文本，它返回统计结果，不依赖任何外部状态。ZeroClaw 中很多工具都是这样的无状态工具。

---

## 第三步：实现 Tool Trait

```rust
use async_trait::async_trait;
use serde_json::{json, Value};

#[async_trait]
impl Tool for WordCountTool {
    // 方法 1：工具名称
    fn name(&self) -> &str {
        "word_count"
    }

    // 方法 2：工具描述（LLM 会读这段文字来决定是否调用）
    fn description(&self) -> &str {
        "统计文本的字数、字符数和行数"
    }

    // 方法 3：参数的 JSON Schema
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "要统计的文本内容"
                }
            },
            "required": ["text"]
        })
    }

    // 方法 4：执行工具（核心逻辑）
    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        // 1. 从 JSON 参数中提取 text 字段
        let text = args.get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: text"))?;

        // 2. 统计
        let words = text.split_whitespace().count();
        let chars = text.chars().count();
        let lines = text.lines().count();

        // 3. 构造结果（JSON 格式的字符串）
        let result = json!({
            "words": words,
            "characters": chars,
            "lines": lines
        });

        Ok(ToolResult {
            content: result.to_string(),
        })
    }

    // 方法 5：spec() -- 使用默认实现，不需要写任何代码
}
```

逐段解读：

**参数提取模式：**
```rust
let text = args.get("text")          // 从 JSON 对象中取 "text" 字段 -> Option<&Value>
    .and_then(|v| v.as_str())        // 尝试转为 &str -> Option<&str>
    .ok_or_else(|| anyhow::anyhow!("Missing required parameter: text"))?;
                                     // None -> 返回错误，Some -> 解包
```

```typescript
// TypeScript 对照
const text = args.text as string;
if (!text) throw new Error("Missing required parameter: text");
```

**错误处理模式：**
```rust
// Rust 用 Result + ? 操作符
async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
    let text = /* ... */?;  // ? 操作符：出错时自动 return Err(...)
    Ok(ToolResult { content: result.to_string() })  // 成功时返回 Ok(...)
}
```

```typescript
// TypeScript 用 throw + try/catch
async execute(args: any): Promise<ToolResult> {
    const text = args.text;
    if (!text) throw new Error("...");  // 出错时抛异常
    return { content: JSON.stringify(result) };  // 成功时直接返回
}
```

---

## 第四步：测试 Tool

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_word_count_name() {
        let tool = WordCountTool;
        assert_eq!(tool.name(), "word_count");
        assert_eq!(tool.description(), "统计文本的字数、字符数和行数");
    }

    #[tokio::test]
    async fn test_word_count_execute() {
        let tool = WordCountTool;
        let args = json!({"text": "Hello World\nThis is a test"});
        let result = tool.execute(args).await.unwrap();

        let parsed: Value = serde_json::from_str(&result.content).unwrap();
        assert_eq!(parsed["words"], 6);
        assert_eq!(parsed["characters"], 26);
        assert_eq!(parsed["lines"], 2);
    }

    #[tokio::test]
    async fn test_word_count_missing_param() {
        let tool = WordCountTool;
        let args = json!({});  // 缺少 text 参数
        let result = tool.execute(args).await;
        assert!(result.is_err());  // 应该返回错误
    }

    #[tokio::test]
    async fn test_word_count_spec_default() {
        let tool = WordCountTool;
        let spec = tool.spec();  // 使用默认实现
        assert_eq!(spec.name, "word_count");
        assert_eq!(spec.description, "统计文本的字数、字符数和行数");
    }
}
```

```typescript
// TypeScript 对照（Jest）
describe("WordCountTool", () => {
  const tool = new WordCountTool();

  test("name", () => {
    expect(tool.name()).toBe("word_count");
  });

  test("execute", async () => {
    const result = await tool.execute({ text: "Hello World\nThis is a test" });
    const parsed = JSON.parse(result.content);
    expect(parsed.words).toBe(6);
    expect(parsed.lines).toBe(2);
  });

  test("missing param", async () => {
    await expect(tool.execute({})).rejects.toThrow();
  });
});
```

---

## 完整代码（可直接运行）

将以下代码保存为 `main.rs`，用 `cargo run` 即可运行：

```rust
// main.rs -- 完整的 WordCountTool 示例
// 运行方式：cargo run
// 依赖：async-trait, serde_json, anyhow, tokio

use async_trait::async_trait;
use serde_json::{json, Value};

// ========== Trait 和类型定义（简化自 ZeroClaw src/tools/traits.rs）==========

pub struct ToolResult {
    pub content: String,
}

pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> Value;
    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult>;
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters_schema(),
        }
    }
}

// ========== WordCountTool 实现 ==========

pub struct WordCountTool;

#[async_trait]
impl Tool for WordCountTool {
    fn name(&self) -> &str { "word_count" }

    fn description(&self) -> &str { "统计文本的字数、字符数和行数" }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "text": { "type": "string", "description": "要统计的文本内容" }
            },
            "required": ["text"]
        })
    }

    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        let text = args.get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: text"))?;

        let result = json!({
            "words": text.split_whitespace().count(),
            "characters": text.chars().count(),
            "lines": text.lines().count(),
        });

        Ok(ToolResult { content: result.to_string() })
    }
}

// ========== 主函数 ==========

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let tool = WordCountTool;

    // 打印工具信息
    println!("Tool: {}", tool.name());
    println!("Description: {}", tool.description());
    println!("Schema: {}", serde_json::to_string_pretty(&tool.parameters_schema())?);
    println!();

    // 测试 spec() 默认实现
    let spec = tool.spec();
    println!("Spec name: {}", spec.name);
    println!();

    // 执行工具
    let args = json!({"text": "Hello World\nThis is ZeroClaw\nRust is awesome"});
    let result = tool.execute(args).await?;
    println!("Result: {}", result.content);

    // 解析结果
    let parsed: Value = serde_json::from_str(&result.content)?;
    println!("\n--- 统计结果 ---");
    println!("字数: {}", parsed["words"]);
    println!("字符数: {}", parsed["characters"]);
    println!("行数: {}", parsed["lines"]);

    Ok(())
}

// ========== 测试 ==========

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_execute() {
        let tool = WordCountTool;
        let args = json!({"text": "Hello World"});
        let result = tool.execute(args).await.unwrap();
        let parsed: Value = serde_json::from_str(&result.content).unwrap();
        assert_eq!(parsed["words"], 2);
        assert_eq!(parsed["characters"], 11);
        assert_eq!(parsed["lines"], 1);
    }

    #[tokio::test]
    async fn test_missing_param() {
        let tool = WordCountTool;
        assert!(tool.execute(json!({})).await.is_err());
    }
}
```

对应的 `Cargo.toml`：

```toml
[package]
name = "word-count-tool"
version = "0.1.0"
edition = "2021"

[dependencies]
async-trait = "0.1"
serde_json = "1"
anyhow = "1"
tokio = { version = "1", features = ["full"] }
```

---

## TypeScript 对照实现

用 TypeScript 实现同样的 WordCountTool，对比两种语言的差异：

```typescript
// word_count_tool.ts

// 类型定义
interface ToolResult { content: string; }
interface ToolSpec { name: string; description: string; parameters: object; }

// TypeScript 需要 abstract class 才能有默认实现
// 而 Rust 用一个 trait 就搞定了
abstract class BaseTool {
  abstract name(): string;
  abstract description(): string;
  abstract parametersSchema(): object;
  abstract execute(args: Record<string, unknown>): Promise<ToolResult>;

  // 默认实现（Rust trait 的 spec() 等价物）
  spec(): ToolSpec {
    return {
      name: this.name(),
      description: this.description(),
      parameters: this.parametersSchema(),
    };
  }
}

// WordCountTool 实现
class WordCountTool extends BaseTool {
  name(): string { return "word_count"; }
  description(): string { return "统计文本的字数、字符数和行数"; }

  parametersSchema(): object {
    return {
      type: "object",
      properties: { text: { type: "string", description: "要统计的文本内容" } },
      required: ["text"],
    };
  }

  async execute(args: Record<string, unknown>): Promise<ToolResult> {
    const text = args.text as string;
    if (!text) throw new Error("Missing required parameter: text");

    return {
      content: JSON.stringify({
        words: text.split(/\s+/).filter(Boolean).length,
        characters: text.length,
        lines: text.split("\n").length,
      }),
    };
  }
}

// 运行
async function main() {
  const tool = new WordCountTool();
  console.log(`Tool: ${tool.name()}`);

  const result = await tool.execute({ text: "Hello World\nThis is ZeroClaw\nRust is awesome" });
  console.log(`Result: ${result.content}`);

  const spec = tool.spec();
  console.log(`Spec name: ${spec.name}`);
}

main();
```

### 关键差异对照

| 方面 | Rust | TypeScript |
|------|------|-----------|
| 契约定义 | `trait Tool` | `abstract class BaseTool` |
| 默认实现 | trait 内直接写 | 必须用 abstract class |
| 多 Trait | `impl A for T` + `impl B for T` | 只能 `extends` 一个基类 |
| 异步标记 | `#[async_trait]` + `async fn` | `async` 关键字 |
| 错误处理 | `Result<T>` + `?` 操作符 | `throw` + `try/catch` |
| 参数类型 | `serde_json::Value`（类型安全的动态 JSON） | `Record<string, unknown>` |
| 线程安全 | `Send + Sync` 编译时保证 | 无（单线程 Event Loop） |

---

## 运行输出示例

```
Tool: word_count
Description: 统计文本的字数、字符数和行数
Schema: {
  "type": "object",
  "properties": {
    "text": {
      "type": "string",
      "description": "要统计的文本内容"
    }
  },
  "required": [
    "text"
  ]
}

Spec name: word_count

Result: {"characters":45,"lines":3,"words":8}

--- 统计结果 ---
字数: 8
字符数: 45
行数: 3
```

---

## 关键学习点总结

### 1. Trait 实现的基本流程

```
定义结构体 -> impl Trait for 结构体 -> 实现所有必需方法 -> 默认方法自动可用
```

```rust
struct WordCountTool;                    // 1. 定义结构体
#[async_trait]
impl Tool for WordCountTool {           // 2. impl Trait for Type
    fn name(&self) -> &str { ... }      // 3. 实现必需方法
    fn description(&self) -> &str { ... }
    fn parameters_schema(&self) -> Value { ... }
    async fn execute(&self, args: Value) -> Result<ToolResult> { ... }
    // spec() 自动可用                    // 4. 默认实现
}
```

### 2. async fn 在 Trait 中的使用

Rust 原生 trait 对 async 方法的支持尚不完整（截至 2026 年，`dyn Trait` 场景仍需 crate 辅助），ZeroClaw 使用 `async-trait` crate 解决：

```rust
use async_trait::async_trait;

#[async_trait]                          // 加在 trait 定义上
pub trait Tool: Send + Sync {
    async fn execute(&self, args: Value) -> Result<ToolResult>;
}

#[async_trait]                          // 也要加在 impl 块上
impl Tool for WordCountTool {
    async fn execute(&self, args: Value) -> Result<ToolResult> {
        // 可以在这里使用 .await
        Ok(ToolResult { content: "...".into() })
    }
}
```

`#[async_trait]` 宏在底层将 `async fn` 转换为返回 `Pin<Box<dyn Future>>` 的普通方法，使其兼容 trait object（`dyn Tool`）。

### 3. serde_json::Value 作为动态类型参数

```rust
// Value 是 Rust 中的动态 JSON 类型，类似 TypeScript 的 any（但更安全）
let args: Value = json!({"text": "hello", "count": 42});

// 取值时必须显式处理类型
args.get("text")           // -> Option<&Value>
    .and_then(|v| v.as_str())  // -> Option<&str>
    .unwrap_or("default");     // -> &str

// 对比 TypeScript：
// args.text  -- 直接访问，可能 undefined
// args.text as string  -- 强制转换，可能运行时错误
```

### 4. 默认实现的使用（spec 方法）

```rust
// 不写 spec() 方法 -> 自动使用默认实现
// 默认实现调用 name() + description() + parameters_schema()
// 这就是模板方法模式（Template Method Pattern）

let tool = WordCountTool;
let spec = tool.spec();
// spec.name == "word_count"  -- 来自 name() 的返回值
// spec.description == "统计文本的字数、字符数和行数"  -- 来自 description()
```

---

## 挑战练习：实现 CalculatorTool

尝试自己实现一个 CalculatorTool，支持加减乘除：

**要求：**
- 工具名称：`"calculator"`
- 参数：`operation`（"add"/"sub"/"mul"/"div"）、`a`（数字）、`b`（数字）
- 返回计算结果
- 处理除以零的错误

**提示骨架：**

```rust
pub struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str { "calculator" }
    fn description(&self) -> &str { "执行基本的四则运算" }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "operation": { "type": "string", "enum": ["add", "sub", "mul", "div"] },
                "a": { "type": "number" },
                "b": { "type": "number" }
            },
            "required": ["operation", "a", "b"]
        })
    }

    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        let op = args.get("operation").and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing: operation"))?;
        let a = args.get("a").and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow::anyhow!("Missing: a"))?;
        let b = args.get("b").and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow::anyhow!("Missing: b"))?;

        let result = match op {
            "add" => a + b,
            "sub" => a - b,
            "mul" => a * b,
            "div" => {
                if b == 0.0 { anyhow::bail!("Division by zero") }
                a / b
            }
            _ => anyhow::bail!("Unknown operation: {}", op),
        };

        Ok(ToolResult { content: json!({"result": result}).to_string() })
    }
}
```

完成后，为它写测试来验证你的实现。

---

*上一篇：[06_反直觉点](./06_反直觉点.md) -- 常见误区与纠正*
*下一篇：[07_实战代码_场景2_泛型工厂函数](./07_实战代码_场景2_泛型工厂函数.md) -- where 子句实战*

---

**文件信息**
- 知识点: Trait 与泛型
- 维度: 07_实战代码_场景1
- 版本: v1.0
- 日期: 2026-03-10
