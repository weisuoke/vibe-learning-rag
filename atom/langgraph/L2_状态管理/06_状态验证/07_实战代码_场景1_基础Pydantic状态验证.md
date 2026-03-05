# 实战代码 - 场景1：基础 Pydantic 状态验证

## 场景说明

本场景演示如何在 LangGraph 中使用 Pydantic `BaseModel` 定义带验证的状态。你将看到：

1. 用 `Field` 约束定义状态 schema
2. LangGraph 图的正常调用流程
3. Pydantic 自动类型转换的行为
4. 验证失败时的错误捕获与诊断

这是状态验证的最基础用法——不需要自定义验证器，仅靠 `Field` 约束就能拦截大量非法输入。

[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]
[来源: reference/context7_pydantic_01.md | Pydantic 官方文档]

---

## 完整代码

```python
"""
LangGraph 状态验证实战 - 场景1：基础 Pydantic 状态验证
演示：使用 Pydantic BaseModel 定义带验证的状态

运行环境：Python 3.13+, langgraph, pydantic
安装依赖：uv add langgraph pydantic
"""

from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. 定义带验证的状态
# ============================================================

class ChatState(BaseModel):
    """聊天状态 - 带基础 Field 约束验证

    每个字段都通过 Field() 设置了约束条件：
    - user_query: 不能为空，最长 1000 字符
    - response: 默认空字符串，AI 回复内容
    - turn_count: 非负整数，记录对话轮次
    - model_name: 必须匹配指定的模型名称模式
    """
    user_query: str = Field(
        min_length=1,
        max_length=1000,
        description="用户查询，不能为空"
    )
    response: str = Field(
        default="",
        description="AI 回复内容"
    )
    turn_count: int = Field(
        default=0,
        ge=0,
        description="对话轮次，不能为负数"
    )
    model_name: str = Field(
        default="gpt-4",
        pattern=r"^(gpt-4|gpt-3\.5|claude).*$",
        description="模型名称，必须是 gpt-4/gpt-3.5/claude 开头"
    )


# ============================================================
# 2. 定义节点函数
# ============================================================

def process_query(state: ChatState):
    """处理用户查询的节点

    接收经过 Pydantic 验证的 ChatState 实例，
    返回普通字典，LangGraph 会自动合并到状态中。
    """
    print(f"  处理查询: {state.user_query}")
    print(f"  使用模型: {state.model_name}")
    print(f"  当前轮次: {state.turn_count}")
    return {
        "response": f"这是对 '{state.user_query}' 的回复（by {state.model_name}）",
        "turn_count": state.turn_count + 1,
    }


# ============================================================
# 3. 构建 LangGraph 图
# ============================================================

builder = StateGraph(ChatState)
builder.add_node("process", process_query)
builder.add_edge(START, "process")
builder.add_edge("process", END)
graph = builder.compile()


# ============================================================
# 4. 正常调用 —— 验证通过
# ============================================================

print("=" * 60)
print("=== 测试 1：正常调用 ===")
print("=" * 60)

result = graph.invoke({"user_query": "什么是 RAG？"})
print(f"  回复: {result['response']}")
print(f"  轮次: {result['turn_count']}")
print(f"  模型: {result['model_name']}")


# ============================================================
# 5. 自动类型转换 —— Pydantic 的隐藏能力
# ============================================================

print()
print("=" * 60)
print("=== 测试 2：自动类型转换 ===")
print("=" * 60)

# turn_count 传入字符串 "5"，Pydantic 会自动转为 int 5
result = graph.invoke({"user_query": "测试类型转换", "turn_count": "5"})
print(f"  turn_count 值: {result['turn_count']}")
print(f"  turn_count 类型: {type(result['turn_count'])}")

# model_name 使用 claude 前缀
result = graph.invoke({
    "user_query": "测试 Claude 模型",
    "model_name": "claude-3-opus"
})
print(f"  模型名: {result['model_name']}")


# ============================================================
# 6. 验证失败演示 —— 各种约束的拦截效果
# ============================================================

print()
print("=" * 60)
print("=== 测试 3：验证失败场景 ===")
print("=" * 60)

# PLACEHOLDER_SECTION_6

test_cases = [
    # 测试 3a：空查询 → min_length=1 拦截
    {
        "name": "空查询（min_length=1 拦截）",
        "input": {"user_query": ""},
    },
    # 测试 3b：负数轮次 → ge=0 拦截
    {
        "name": "负数轮次（ge=0 拦截）",
        "input": {"user_query": "测试", "turn_count": -1},
    },
    # 测试 3c：无效模型名 → pattern 正则拦截
    {
        "name": "无效模型名（pattern 正则拦截）",
        "input": {"user_query": "测试", "model_name": "invalid-model"},
    },
    # 测试 3d：查询超长 → max_length=1000 拦截
    {
        "name": "超长查询（max_length=1000 拦截）",
        "input": {"user_query": "x" * 1001},
    },
    # 测试 3e：缺少必填字段
    {
        "name": "缺少必填字段 user_query",
        "input": {},
    },
]

for i, case in enumerate(test_cases):
    print(f"\n--- 测试 3{chr(97+i)}: {case['name']} ---")
    try:
        graph.invoke(case["input"])
        print("  ⚠️ 意外通过，未触发验证")
    except ValidationError as e:
        print(f"  ✅ ValidationError 捕获成功")
        for err in e.errors():
            field = " → ".join(str(loc) for loc in err["loc"])
            print(f"     字段: {field}")
            print(f"     错误: {err['msg']}")
            print(f"     类型: {err['type']}")
    except Exception as e:
        print(f"  ❌ 其他异常: {type(e).__name__}: {e}")

# PLACEHOLDER_SECTION_7


# ============================================================
# 7. 补充：直接用 Pydantic 模型测试验证（不经过 LangGraph）
# ============================================================

print()
print("=" * 60)
print("=== 测试 4：直接 Pydantic 验证（脱离 LangGraph） ===")
print("=" * 60)

# 这种方式方便你在单元测试中验证 schema 定义是否正确
print("\n--- 正常创建 ---")
state = ChatState(user_query="直接创建状态对象")
print(f"  query: {state.user_query}")
print(f"  model: {state.model_name}")
print(f"  turn:  {state.turn_count}")

print("\n--- 自动转换 ---")
state = ChatState(user_query="类型转换测试", turn_count="10")
print(f"  turn_count: {state.turn_count} (类型: {type(state.turn_count).__name__})")

print("\n--- 验证失败 ---")
try:
    ChatState(user_query="", model_name="bad-model", turn_count=-5)
except ValidationError as e:
    print(f"  共 {e.error_count()} 个错误:")
    for err in e.errors():
        field = " → ".join(str(loc) for loc in err["loc"])
        print(f"    [{field}] {err['msg']}")
```

---

## 预期输出

```text
============================================================
=== 测试 1：正常调用 ===
============================================================
  处理查询: 什么是 RAG？
  使用模型: gpt-4
  当前轮次: 0
  回复: 这是对 '什么是 RAG？' 的回复（by gpt-4）
  轮次: 1
  模型: gpt-4

============================================================
=== 测试 2：自动类型转换 ===
============================================================
  处理查询: 测试类型转换
  使用模型: gpt-4
  当前轮次: 5
  turn_count 值: 6
  turn_count 类型: <class 'int'>
  处理查询: 测试 Claude 模型
  使用模型: claude-3-opus
  当前轮次: 0
  模型名: claude-3-opus

============================================================
=== 测试 3：验证失败场景 ===
============================================================

--- 测试 3a: 空查询（min_length=1 拦截） ---
  ✅ ValidationError 捕获成功
     字段: user_query
     错误: String should have at least 1 character
     类型: string_too_short

--- 测试 3b: 负数轮次（ge=0 拦截） ---
  ✅ ValidationError 捕获成功
     字段: turn_count
     错误: Input should be greater than or equal to 0
     类型: greater_than_equal

--- 测试 3c: 无效模型名（pattern 正则拦截） ---
  ✅ ValidationError 捕获成功
     字段: model_name
     错误: String should match pattern '^(gpt-4|gpt-3\.5|claude).*$'
     类型: string_pattern_mismatch

--- 测试 3d: 超长查询（max_length=1000 拦截） ---
  ✅ ValidationError 捕获成功
     字段: user_query
     错误: String should have at most 1000 characters
     类型: string_too_long

--- 测试 3e: 缺少必填字段 user_query ---
  ✅ ValidationError 捕获成功
     字段: user_query
     错误: Field required
     类型: missing

PLACEHOLDER_OUTPUT_REST
```

---

## 关键知识点解析

### 1. Field 约束 vs 自定义验证器

本场景只使用了 `Field()` 内置约束，没有写任何 `@field_validator`。这是有意为之的——**能用声明式约束解决的问题，就不要写命令式代码**。

| 需求 | 用 Field 约束 | 用 field_validator |
|------|--------------|-------------------|
| 字符串长度限制 | `Field(min_length=1)` | 不需要 |
| 数值范围限制 | `Field(ge=0, le=100)` | 不需要 |
| 正则匹配 | `Field(pattern=r"...")` | 不需要 |
| 清洗输入（去空格） | 做不到 | 需要 |
| 跨字段联合验证 | 做不到 | 需要 model_validator |
| 复杂业务逻辑 | 做不到 | 需要 |

**原则**：先用 `Field`，不够再用 `field_validator`，跨字段用 `model_validator`。

### 2. 验证触发时机

在 LangGraph 中，Pydantic 验证在以下时机触发：

```
graph.invoke(input_dict)
    ↓
LangGraph 调用 ChatState(**input_dict)   ← 这里触发验证
    ↓
验证通过 → 创建 ChatState 实例 → 传入节点
验证失败 → 抛出 ValidationError → 不会进入任何节点
```

这意味着：**非法输入永远不会到达你的节点函数**。节点内部可以放心使用 `state.user_query`，不需要再做空值检查。

### 3. 自动类型转换的边界

Pydantic 默认开启「宽松模式」（lax mode），会尝试合理的类型转换：

| 输入 | 目标类型 | 结果 |
|------|---------|------|
| `"5"` | `int` | `5`（字符串转整数） |
| `"0.95"` | `float` | `0.95`（字符串转浮点） |
| `5` | `str` | `"5"`（整数转字符串） |
| `"true"` | `bool` | `True` |
| `"不是数字"` | `int` | ValidationError |

这在 RAG 系统中很实用——前端传来的 JSON 数据经常把数字序列化为字符串，Pydantic 会自动处理。

### 4. 错误信息的结构

`ValidationError` 的 `.errors()` 方法返回结构化的错误列表，每个错误包含：

```python
{
    "type": "string_too_short",       # 错误类型（机器可读）
    "loc": ("user_query",),           # 字段路径（支持嵌套）
    "msg": "String should have ...",  # 人类可读的错误消息
    "input": "",                      # 导致错误的输入值
    "ctx": {"min_length": 1},         # 约束上下文
}
```

你可以用这些信息构建用户友好的错误提示，或者在日志中记录详细的验证失败原因。

---

## 在 RAG 系统中的应用

这种基础验证模式在 RAG 系统中的典型应用：

```python
class RAGState(BaseModel):
    """RAG 管道状态 - 用 Field 约束保护关键参数"""

    # 用户输入：防止空查询和超长输入
    query: str = Field(min_length=1, max_length=2000)

    # 检索参数：限制合理范围，避免资源浪费
    top_k: int = Field(default=5, ge=1, le=50)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # 生成参数：约束 LLM 行为
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=4096)

    # 模型选择：正则限制合法模型名
    model_name: str = Field(
        default="gpt-4",
        pattern=r"^(gpt-4|gpt-3\.5|claude).*$"
    )

    # 流程状态
    retrieved_docs: list = Field(default_factory=list)
    response: str = Field(default="")
```

**好处**：
- 前端传入 `top_k=1000` 会被拦截，避免向量数据库查询过载
- `temperature=5.0` 会被拦截，避免 LLM 生成乱码
- 空查询会被拦截，避免浪费 API 调用
- 所有约束集中在状态定义中，一目了然

---

## 学习检查清单

- [ ] 理解 `Field()` 的常用约束：`min_length`、`max_length`、`ge`、`le`、`pattern`
- [ ] 理解 Pydantic 自动类型转换的行为和边界
- [ ] 能捕获 `ValidationError` 并提取结构化错误信息
- [ ] 理解验证在 LangGraph 中的触发时机（`invoke` 时，节点执行前）
- [ ] 知道何时用 `Field` 约束，何时需要升级到 `field_validator`

---

## 下一步

掌握了基础 `Field` 约束后，进入 **场景2：自定义验证器实战**，学习：
- `@field_validator`：字段级自定义验证（清洗、转换、复杂校验）
- `@model_validator`：跨字段联合验证（字段间的依赖关系）
- 两者的组合使用模式
