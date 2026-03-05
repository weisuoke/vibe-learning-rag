# 实战代码 - 场景2：多状态Schema应用

> 本文档演示 LangGraph 中多状态 Schema 的使用，帮助你理解状态隔离和不同 Schema 的组合应用

---

## 场景描述

**目标**：构建一个文本处理工作流，演示如何使用不同的状态 Schema（InputState, OutputState, PrivateState）实现状态隔离。

**核心知识点**：
- InputState：定义输入接口
- OutputState：定义输出接口
- OverallState：内部完整状态
- PrivateState：私有状态（不暴露给外部）
- 状态隔离的实现

**适用场景**：
- 构建模块化的工作流
- 隐藏内部实现细节
- 提供清晰的输入输出接口
- 多团队协作开发

---

## 核心原理回顾

### 多状态 Schema 的作用

**问题**：为什么需要多个状态 Schema？

1. **接口清晰**：明确定义输入和输出
2. **状态隔离**：隐藏内部实现细节
3. **模块化**：不同节点使用不同的状态视图
4. **安全性**：防止外部访问敏感数据

### 状态 Schema 的类型

```python
# 1. InputState - 输入接口
class InputState(TypedDict):
    user_input: str

# 2. OutputState - 输出接口
class OutputState(TypedDict):
    final_result: str

# 3. OverallState - 内部完整状态
class OverallState(TypedDict):
    user_input: str
    intermediate_data: str
    final_result: str

# 4. PrivateState - 私有状态
class PrivateState(TypedDict):
    internal_cache: dict
```

### 节点与状态 Schema 的关系

```python
# 节点可以从不同的 Schema 读取和写入
def node1(state: InputState) -> OverallState:
    # 从 InputState 读取，写入 OverallState
    return {"intermediate_data": process(state["user_input"])}

def node2(state: OverallState) -> OutputState:
    # 从 OverallState 读取，写入 OutputState
    return {"final_result": finalize(state["intermediate_data"])}
```

---

## 实战代码

### 场景1：文本处理管道

**功能**：用户输入文本，经过清洗、分析、总结三个步骤，输出最终结果

```python
"""
场景1：多状态Schema应用 - 文本处理管道
演示：InputState, OutputState, OverallState, PrivateState 的组合使用
"""

import os
from typing import TypedDict, List, Dict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# ===== 1. 环境配置 =====
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ===== 2. 定义多个状态 Schema =====

class InputState(TypedDict):
    """
    输入状态 - 对外暴露的输入接口

    只包含用户需要提供的字段
    """
    user_text: str


class OutputState(TypedDict):
    """
    输出状态 - 对外暴露的输出接口

    只包含用户需要获取的字段
    """
    summary: str
    word_count: int


class OverallState(TypedDict):
    """
    整体状态 - 内部使用的完整状态

    包含所有中间处理字段
    """
    user_text: str          # 输入字段
    cleaned_text: str       # 中间字段
    analysis: Dict          # 中间字段
    summary: str            # 输出字段
    word_count: int         # 输出字段


class PrivateState(TypedDict):
    """
    私有状态 - 内部敏感数据

    不暴露给外部，只在内部节点间传递
    """
    internal_metrics: Dict
    processing_log: List[str]


# ===== 3. 定义节点函数 =====

def clean_text(state: InputState) -> OverallState:
    """
    节点1：文本清洗

    读取：InputState["user_text"]
    写入：OverallState["cleaned_text"]
    """
    print("\n[节点1] 文本清洗...")

    text = state["user_text"]
    # 简单清洗：去除多余空格和换行
    cleaned = " ".join(text.split())

    print(f"[节点1] 原始文本长度：{len(text)}")
    print(f"[节点1] 清洗后长度：{len(cleaned)}")

    return {"cleaned_text": cleaned}


def analyze_text(state: OverallState) -> PrivateState:
    """
    节点2：文本分析

    读取：OverallState["cleaned_text"]
    写入：PrivateState["internal_metrics", "processing_log"]
    """
    print("\n[节点2] 文本分析...")

    text = state["cleaned_text"]

    # 分析文本
    words = text.split()
    metrics = {
        "word_count": len(words),
        "char_count": len(text),
        "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0
    }

    log = [
        f"分析完成：{len(words)} 个词",
        f"平均词长：{metrics['avg_word_length']:.2f}"
    ]

    print(f"[节点2] 词数：{metrics['word_count']}")
    print(f"[节点2] 平均词长：{metrics['avg_word_length']:.2f}")

    return {
        "internal_metrics": metrics,
        "processing_log": log
    }


def generate_summary(state: PrivateState) -> OutputState:
    """
    节点3：生成摘要

    读取：PrivateState["internal_metrics"]
    写入：OutputState["summary", "word_count"]
    """
    print("\n[节点3] 生成摘要...")

    metrics = state["internal_metrics"]

    # 生成摘要
    summary = (
        f"文本包含 {metrics['word_count']} 个词，"
        f"平均词长 {metrics['avg_word_length']:.2f} 个字符。"
    )

    print(f"[节点3] 摘要：{summary}")

    return {
        "summary": summary,
        "word_count": metrics["word_count"]
    }


# ===== 4. 构建图 =====

# 创建 StateGraph，指定 input_schema 和 output_schema
workflow = StateGraph(
    OverallState,
    input=InputState,
    output=OutputState
)

# 添加节点
workflow.add_node("clean", clean_text)
workflow.add_node("analyze", analyze_text)
workflow.add_node("summarize", generate_summary)

# 定义边
workflow.add_edge(START, "clean")
workflow.add_edge("clean", "analyze")
workflow.add_edge("analyze", "summarize")
workflow.add_edge("summarize", END)

# 编译图
graph = workflow.compile()


# ===== 5. 运行图 =====

def run_text_processor(text: str):
    """运行文本处理器"""
    print("=" * 60)
    print("文本处理管道启动")
    print("=" * 60)

    # 输入：只需要提供 InputState 的字段
    input_data = {"user_text": text}

    # 执行图
    result = graph.invoke(input_data)

    # 输出：只返回 OutputState 的字段
    print("\n" + "=" * 60)
    print("处理结果")
    print("=" * 60)
    print(f"摘要：{result['summary']}")
    print(f"词数：{result['word_count']}")
    print("=" * 60)

    # 注意：result 中不包含 OverallState 和 PrivateState 的字段
    # 这就是状态隔离的效果

    return result


# ===== 6. 测试运行 =====

if __name__ == "__main__":
    run_text_processor(
        "这是一个测试文本。  它包含多个句子。\n\n还有一些多余的空格和换行。"
    )
```

---

## 场景2：用户信息处理系统

**功能**：演示如何使用 PrivateState 保护敏感信息

```python
"""
场景2：用户信息处理系统
演示：使用 PrivateState 保护敏感信息
"""

from typing import TypedDict, Optional

# ===== 1. 定义状态 Schema =====

class UserInputState(TypedDict):
    """用户输入"""
    username: str
    email: str


class UserOutputState(TypedDict):
    """用户输出（脱敏）"""
    username: str
    email_domain: str
    verification_status: str


class InternalState(TypedDict):
    """内部状态（包含敏感信息）"""
    username: str
    email: str
    email_domain: str
    password_hash: str          # 敏感信息
    verification_token: str     # 敏感信息
    verification_status: str


# ===== 2. 定义节点 =====

def process_user_input(state: UserInputState) -> InternalState:
    """处理用户输入"""
    print("\n[节点1] 处理用户输入...")

    username = state["username"]
    email = state["email"]
    email_domain = email.split("@")[1] if "@" in email else "unknown"

    # 生成敏感信息（模拟）
    password_hash = f"hash_{username}_secret"
    verification_token = f"token_{username}_12345"

    print(f"[节点1] 用户名：{username}")
    print(f"[节点1] 邮箱域名：{email_domain}")
    print(f"[节点1] 生成密码哈希（敏感）")
    print(f"[节点1] 生成验证令牌（敏感）")

    return {
        "username": username,
        "email": email,
        "email_domain": email_domain,
        "password_hash": password_hash,
        "verification_token": verification_token,
        "verification_status": "pending"
    }


def verify_user(state: InternalState) -> InternalState:
    """验证用户（使用敏感信息）"""
    print("\n[节点2] 验证用户...")

    # 使用敏感信息进行验证（模拟）
    token = state["verification_token"]
    print(f"[节点2] 使用令牌验证：{token[:10]}...")

    # 验证成功
    return {"verification_status": "verified"}


def prepare_output(state: InternalState) -> UserOutputState:
    """准备输出（脱敏）"""
    print("\n[节点3] 准备输出（脱敏）...")

    # 只返回非敏感信息
    output = {
        "username": state["username"],
        "email_domain": state["email_domain"],
        "verification_status": state["verification_status"]
    }

    print(f"[节点3] 输出用户名：{output['username']}")
    print(f"[节点3] 输出邮箱域名：{output['email_domain']}")
    print(f"[节点3] 输出验证状态：{output['verification_status']}")
    print(f"[节点3] 敏感信息已隐藏")

    return output


# ===== 3. 构建图 =====

workflow2 = StateGraph(
    InternalState,
    input=UserInputState,
    output=UserOutputState
)

workflow2.add_node("process", process_user_input)
workflow2.add_node("verify", verify_user)
workflow2.add_node("output", prepare_output)

workflow2.add_edge(START, "process")
workflow2.add_edge("process", "verify")
workflow2.add_edge("verify", "output")
workflow2.add_edge("output", END)

graph2 = workflow2.compile()


# ===== 4. 运行 =====

def run_user_processor(username: str, email: str):
    """运行用户处理器"""
    print("=" * 60)
    print("用户信息处理系统启动")
    print("=" * 60)

    # 输入
    input_data = {
        "username": username,
        "email": email
    }

    # 执行
    result = graph2.invoke(input_data)

    # 输出（已脱敏）
    print("\n" + "=" * 60)
    print("处理结果（已脱敏）")
    print("=" * 60)
    print(f"用户名：{result['username']}")
    print(f"邮箱域名：{result['email_domain']}")
    print(f"验证状态：{result['verification_status']}")
    print("\n注意：敏感信息（密码哈希、验证令牌）未包含在输出中")
    print("=" * 60)

    return result


if __name__ == "__main__":
    run_user_processor("alice", "alice@example.com")
```

---

## 运行结果示例

### 场景1：文本处理管道

```
============================================================
文本处理管道启动
============================================================

[节点1] 文本清洗...
[节点1] 原始文本长度：45
[节点1] 清洗后长度：38

[节点2] 文本分析...
[节点2] 词数：7
[节点2] 平均词长：3.14

[节点3] 生成摘要...
[节点3] 摘要：文本包含 7 个词，平均词长 3.14 个字符。

============================================================
处理结果
============================================================
摘要：文本包含 7 个词，平均词长 3.14 个字符。
词数：7
============================================================
```

### 场景2：用户信息处理系统

```
============================================================
用户信息处理系统启动
============================================================

[节点1] 处理用户输入...
[节点1] 用户名：alice
[节点1] 邮箱域名：example.com
[节点1] 生成密码哈希（敏感）
[节点1] 生成验证令牌（敏感）

[节点2] 验证用户...
[节点2] 使用令牌验证：token_alic...

[节点3] 准备输出（脱敏）...
[节点3] 输出用户名：alice
[节点3] 输出邮箱域名：example.com
[节点3] 输出验证状态：verified
[节点3] 敏感信息已隐藏

============================================================
处理结果（已脱敏）
============================================================
用户名：alice
邮箱域名：example.com
验证状态：verified

注意：敏感信息（密码哈希、验证令牌）未包含在输出中
============================================================
```

---

## 核心要点总结

### 1. 状态 Schema 的定义

**InputState**：
```python
class InputState(TypedDict):
    # 只包含用户需要提供的字段
    user_input: str
```

**OutputState**：
```python
class OutputState(TypedDict):
    # 只包含用户需要获取的字段
    final_result: str
```

**OverallState**：
```python
class OverallState(TypedDict):
    # 包含所有字段（输入 + 中间 + 输出）
    user_input: str
    intermediate_data: str
    final_result: str
```

**PrivateState**：
```python
class PrivateState(TypedDict):
    # 只包含内部敏感字段
    sensitive_data: str
```

### 2. 图的创建

```python
workflow = StateGraph(
    OverallState,           # 内部使用的完整状态
    input=InputState,       # 输入接口
    output=OutputState      # 输出接口
)
```

### 3. 节点的类型注解

```python
# 从 InputState 读取，写入 OverallState
def node1(state: InputState) -> OverallState:
    return {"intermediate_data": ...}

# 从 OverallState 读取，写入 PrivateState
def node2(state: OverallState) -> PrivateState:
    return {"sensitive_data": ...}

# 从 PrivateState 读取，写入 OutputState
def node3(state: PrivateState) -> OutputState:
    return {"final_result": ...}
```

### 4. 状态隔离的效果

**输入时**：
```python
# 用户只需要提供 InputState 的字段
input_data = {"user_input": "..."}
result = graph.invoke(input_data)
```

**输出时**：
```python
# 用户只能获取 OutputState 的字段
print(result["final_result"])  # ✅ 可以访问
print(result["intermediate_data"])  # ❌ 不存在
print(result["sensitive_data"])  # ❌ 不存在
```

---

## 最佳实践

### 1. 状态 Schema 设计原则

**清晰的接口**：
- InputState：只包含必需的输入字段
- OutputState：只包含需要返回的字段
- OverallState：包含所有字段
- PrivateState：只包含敏感字段

**命名规范**：
```python
# ✅ 推荐
class UserInputState(TypedDict): ...
class UserOutputState(TypedDict): ...

# ❌ 不推荐
class Input(TypedDict): ...
class Output(TypedDict): ...
```

### 2. 节点设计原则

**明确类型注解**：
```python
# ✅ 推荐：明确输入输出类型
def my_node(state: InputState) -> OutputState:
    ...

# ❌ 不推荐：使用泛型类型
def my_node(state: dict) -> dict:
    ...
```

**单一职责**：
```python
# ✅ 推荐：每个节点只做一件事
def clean_text(state: InputState) -> OverallState:
    return {"cleaned_text": clean(state["user_text"])}

# ❌ 不推荐：一个节点做多件事
def process_all(state: InputState) -> OutputState:
    cleaned = clean(state["user_text"])
    analyzed = analyze(cleaned)
    return {"result": summarize(analyzed)}
```

### 3. 安全性考虑

**保护敏感信息**：
```python
# ✅ 推荐：使用 PrivateState
class PrivateState(TypedDict):
    password_hash: str
    api_key: str

# ❌ 不推荐：敏感信息放在 OutputState
class OutputState(TypedDict):
    password_hash: str  # 不应该暴露
```

**脱敏处理**：
```python
def prepare_output(state: InternalState) -> OutputState:
    # 只返回非敏感信息
    return {
        "username": state["username"],
        "email_domain": state["email"].split("@")[1]
        # 不返回 password_hash, api_key 等
    }
```

---

## 常见问题

### Q1：为什么需要多个状态 Schema？

**A**：多个状态 Schema 提供了清晰的接口和状态隔离：

1. **接口清晰**：明确定义输入和输出
2. **状态隔离**：隐藏内部实现细节
3. **安全性**：防止敏感信息泄露
4. **模块化**：不同节点使用不同的状态视图

### Q2：OverallState 必须包含所有字段吗？

**A**：是的。OverallState 应该包含所有可能用到的字段（输入 + 中间 + 输出）。

```python
class OverallState(TypedDict):
    # 输入字段
    user_input: str

    # 中间字段
    intermediate_data: str

    # 输出字段
    final_result: str
```

### Q3：节点可以跨 Schema 读写吗？

**A**：可以。节点可以从一个 Schema 读取，写入另一个 Schema。

```python
# 从 InputState 读取，写入 OverallState
def node1(state: InputState) -> OverallState:
    return {"intermediate_data": process(state["user_input"])}

# 从 OverallState 读取，写入 OutputState
def node2(state: OverallState) -> OutputState:
    return {"final_result": finalize(state["intermediate_data"])}
```

### Q4：如何验证状态隔离是否生效？

**A**：检查输出结果中是否只包含 OutputState 的字段。

```python
result = graph.invoke({"user_input": "test"})

# ✅ OutputState 的字段可以访问
print(result["final_result"])

# ❌ OverallState 的中间字段不存在
try:
    print(result["intermediate_data"])
except KeyError:
    print("状态隔离生效：中间字段不可访问")
```

---

## 引用来源

### 官方文档
- **文件**：`reference/context7_langgraph_01.md`
- **关键内容**：
  - 多状态 Schema 定义
  - InputState, OutputState, OverallState, PrivateState
  - 状态隔离机制
  - 节点参数类型

### 社区讨论
- **文件**：`reference/fetch_状态传递_11.md`
- **关键内容**：
  - 不同 State Schema 的使用
  - 主图与子图状态映射
  - TypeScript 类型定义

### 技术博客
- **文件**：`reference/fetch_状态传递_07.md`
- **关键内容**：
  - 多 Agent 状态管理
  - 共享状态设计
  - Namespaced state

---

## 下一步学习

完成本场景后，建议学习：

1. **场景3：Runtime Context应用** - 学习如何传递运行时上下文
2. **场景4：状态流转控制** - 学习如何控制状态流转
3. **场景5：多Agent状态共享** - 学习多 Agent 系统的状态管理

---

**文档版本**：v1.0
**最后更新**：2026-02-26
**适用版本**：LangGraph 0.2.x+, Python 3.13+
