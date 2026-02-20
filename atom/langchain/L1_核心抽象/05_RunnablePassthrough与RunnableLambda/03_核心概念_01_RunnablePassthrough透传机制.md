# 03_核心概念_01_RunnablePassthrough透传机制

> **学习目标**：深入理解 RunnablePassthrough 的透传机制、assign() 方法和数据保留模式

---

## 概述

RunnablePassthrough 是 LangChain LCEL 中最简单但最重要的 Runnable 之一，它实现了**恒等函数**（Identity Function）的概念，同时通过 `assign()` 方法提供了强大的数据扩展能力。

**核心价值**：
- 在链式处理中保留原始数据
- 通过 assign() 添加新字段而不丢失原有信息
- 与其他 Runnable 组合构建复杂数据流

---

## 透传机制的本质

### 恒等函数（Identity Function）

**数学定义**：
```
f(x) = x
```

**RunnablePassthrough 实现**：
```python
class RunnablePassthrough(Runnable):
    def invoke(self, input: Input) -> Input:
        """直接返回输入，不做任何转换"""
        return input
```

**特性**：
- 输入类型 = 输出类型
- 不做任何数据转换
- 保持数据完整性

### 为什么需要透传？

**问题场景**：
```python
# 没有透传机制
retriever = ...  # 输出：List[Document]
llm = ...        # 输入：str 或 dict

# 直接连接
chain = retriever | llm

# 问题：原始查询丢失了！
result = chain.invoke("什么是 LangChain?")
# retriever 输出：[doc1, doc2, doc3]
# llm 输入：[doc1, doc2, doc3]（没有查询！）
```

**解决方案**：
```python
from langchain_core.runnables import RunnablePassthrough

# 使用透传保留查询
chain = (
    RunnablePassthrough.assign(context=retriever)
    | prompt
    | llm
)

result = chain.invoke({"query": "什么是 LangChain?"})
# 数据流：
# 1. {"query": "什么是 LangChain?"}
# 2. {"query": "什么是 LangChain?", "context": [doc1, doc2, doc3]}
# 3. prompt 可以同时访问 query 和 context
```

---

## assign() 方法详解

### 方法签名

```python
@classmethod
def assign(cls, **kwargs: Union[Runnable, Callable, Any]) -> Runnable:
    """
    创建一个新的 Runnable，保留输入并添加新字段

    参数:
        **kwargs: 键值对，值可以是：
            - Runnable: 调用 invoke() 方法
            - Callable: 调用函数
            - Any: 直接赋值

    返回:
        RunnableAssign: 新的 Runnable 对象
    """
    return RunnableAssign(kwargs)
```

### 为什么是类方法？

**设计决策**：
```python
# 类方法：不需要实例化
chain = RunnablePassthrough.assign(key=func)  # ✅ 简洁

# 如果是实例方法
chain = RunnablePassthrough().assign(key=func)  # ❌ 冗余
```

**优势**：
- 使用更简洁
- 符合函数式编程范式
- 不需要管理实例状态

### assign() 的三种用法

#### 1. 使用 Lambda 函数

```python
from langchain_core.runnables import RunnablePassthrough

# 从输入中计算新字段
chain = RunnablePassthrough.assign(
    upper=lambda x: x["text"].upper(),
    length=lambda x: len(x["text"])
)

result = chain.invoke({"text": "hello"})
# 输出: {"text": "hello", "upper": "HELLO", "length": 5}
```

**特点**：
- Lambda 接收完整的输入字典
- 可以访问输入的任意字段
- 返回值赋给新字段

#### 2. 使用 Runnable

```python
from langchain_core.runnables import RunnablePassthrough

# 使用检索器作为 Runnable
chain = RunnablePassthrough.assign(
    context=retriever  # retriever 是 Runnable
)

result = chain.invoke({"query": "什么是 LangChain?"})
# 输出: {"query": "什么是 LangChain?", "context": [doc1, doc2, doc3]}
```

**特点**：
- Runnable 自动调用 invoke() 方法
- 输入是完整的字典
- 输出赋给新字段

#### 3. 使用常量

```python
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime

# 直接赋值常量
chain = RunnablePassthrough.assign(
    timestamp=datetime.now().isoformat(),
    version="1.0"
)

result = chain.invoke({"query": "hello"})
# 输出: {"query": "hello", "timestamp": "2026-02-18T...", "version": "1.0"}
```

**特点**：
- 常量在创建链时计算
- 所有调用使用相同的值
- 适用于静态配置

---

## 数据保留模式

### 模式 1：完全保留

**场景**：保留所有输入字段，只添加新字段

```python
chain = RunnablePassthrough.assign(
    new_field=lambda x: compute(x)
)

input_data = {
    "query": "hello",
    "user": "alice",
    "session_id": "123"
}

result = chain.invoke(input_data)
# 输出: {
#     "query": "hello",
#     "user": "alice",
#     "session_id": "123",
#     "new_field": "computed_value"
# }
```

**特点**：
- 所有原始字段保留
- 新字段添加
- 数据完整性保证

---

### 模式 2：字段覆盖

**场景**：新字段与原字段同名，会被覆盖

```python
chain = RunnablePassthrough.assign(
    query=lambda x: x["query"].upper()  # 覆盖原 query
)

input_data = {"query": "hello", "user": "alice"}

result = chain.invoke(input_data)
# 输出: {"query": "HELLO", "user": "alice"}
# 注意：原 query 被覆盖了
```

**注意事项**：
- ⚠️ 同名字段会被覆盖
- ⚠️ 原始值丢失
- ✅ 建议使用不同的键名

**最佳实践**：
```python
# ✅ 使用不同的键名
chain = RunnablePassthrough.assign(
    original_query=lambda x: x["query"],
    processed_query=lambda x: x["query"].upper()
)
```

---

### 模式 3：多字段扩展

**场景**：同时添加多个字段

```python
from datetime import datetime

def get_metadata(x):
    return {
        "timestamp": datetime.now().isoformat(),
        "user": x.get("user", "anonymous")
    }

def get_stats(x):
    query = x["query"]
    return {
        "length": len(query),
        "word_count": len(query.split())
    }

# 同时添加多个字段
chain = RunnablePassthrough.assign(
    context=retriever,
    metadata=get_metadata,
    stats=get_stats
)

result = chain.invoke({"query": "什么是 LangChain?", "user": "alice"})
# 输出: {
#     "query": "什么是 LangChain?",
#     "user": "alice",
#     "context": [doc1, doc2, doc3],
#     "metadata": {"timestamp": "...", "user": "alice"},
#     "stats": {"length": 15, "word_count": 3}
# }
```

**执行顺序**：
- assign 中的字段是**串行执行**的
- 按照 kwargs 的顺序依次计算
- 后面的字段可以访问前面添加的字段

**示例**：
```python
chain = RunnablePassthrough.assign(
    field1=lambda x: "value1",
    field2=lambda x: x["field1"] + "_extended"  # 可以访问 field1
)

result = chain.invoke({"query": "hello"})
# 输出: {"query": "hello", "field1": "value1", "field2": "value1_extended"}
```

---

## 与 RunnableParallel 的集成

### 并行处理模式

**场景**：需要并行执行多个操作

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 并行处理
parallel = RunnableParallel({
    "original": RunnablePassthrough(),  # 保留原始输入
    "context": retriever,                # 检索上下文
    "metadata": RunnableLambda(get_metadata)  # 获取元数据
})

result = parallel.invoke({"query": "什么是 LangChain?"})
# 输出: {
#     "original": {"query": "什么是 LangChain?"},
#     "context": [doc1, doc2, doc3],
#     "metadata": {"timestamp": "...", "user": "..."}
# }
```

**对比 assign()**：
```python
# assign：串行执行
chain1 = RunnablePassthrough.assign(
    context=retriever,
    metadata=get_metadata
)
# 执行顺序：retriever → get_metadata

# RunnableParallel：并行执行
chain2 = RunnableParallel({
    "original": RunnablePassthrough(),
    "context": retriever,
    "metadata": get_metadata
})
# 执行顺序：retriever 和 get_metadata 同时执行
```

**性能对比**：
```
场景：retriever 耗时 1 秒，get_metadata 耗时 1 秒

assign（串行）：
- 总耗时：2 秒

RunnableParallel（并行）：
- 总耗时：1 秒

性能提升：2 倍
```

---

## 实现原理

### RunnableAssign 的实现

```python
class RunnableAssign(Runnable[dict, dict]):
    """
    RunnablePassthrough.assign() 返回的实际类型
    """

    def __init__(self, mappers: dict):
        """
        参数:
            mappers: 字段名到 mapper 的映射
        """
        self.mappers = mappers

    def invoke(self, input: dict) -> dict:
        """
        执行数据扩展

        参数:
            input: 输入字典

        返回:
            扩展后的字典
        """
        # 1. 浅拷贝输入（避免修改原对象）
        output = input.copy()

        # 2. 遍历所有 mapper
        for key, mapper in self.mappers.items():
            if isinstance(mapper, Runnable):
                # 如果是 Runnable，调用 invoke
                output[key] = mapper.invoke(input)
            elif callable(mapper):
                # 如果是函数，直接调用
                output[key] = mapper(input)
            else:
                # 否则直接赋值
                output[key] = mapper

        return output

    async def ainvoke(self, input: dict) -> dict:
        """
        异步版本
        """
        output = input.copy()

        for key, mapper in self.mappers.items():
            if isinstance(mapper, Runnable):
                output[key] = await mapper.ainvoke(input)
            elif asyncio.iscoroutinefunction(mapper):
                output[key] = await mapper(input)
            elif callable(mapper):
                output[key] = mapper(input)
            else:
                output[key] = mapper

        return output
```

### 性能考虑

#### 1. 浅拷贝 vs 深拷贝

```python
# 浅拷贝（当前实现）
output = input.copy()

# 优点：
# - 快速（O(n)，n 是键的数量）
# - 值是引用，不占额外空间
# - 适用于大多数场景

# 缺点：
# - 嵌套对象是引用，修改会影响原对象
```

**示例**：
```python
input_data = {
    "query": "hello",
    "metadata": {"user": "alice"}  # 嵌套对象
}

chain = RunnablePassthrough.assign(context="retrieved")
result = chain.invoke(input_data)

# 修改嵌套对象
result["metadata"]["user"] = "bob"

# 原始输入也被修改了！
print(input_data["metadata"]["user"])  # 输出: "bob"
```

**解决方案**：
```python
import copy

# 如果需要深拷贝
def deep_copy_assign(**kwargs):
    def _invoke(input_dict):
        output = copy.deepcopy(input_dict)  # 深拷贝
        for key, mapper in kwargs.items():
            if callable(mapper):
                output[key] = mapper(input_dict)
            else:
                output[key] = mapper
        return output
    return RunnableLambda(_invoke)
```

#### 2. 大数据场景

```python
# 大数据场景
large_data = {
    "query": "hello",
    "embeddings": np.array([...])  # 1GB 的向量
}

# assign 只是浅拷贝，不复制 embeddings 数组
result = RunnablePassthrough.assign(context="retrieved").invoke(large_data)

# 内存占用：~1GB（不是 2GB）
# embeddings 数组是引用，不占额外空间
```

---

## 类型系统

### 类型签名

```python
from typing import TypeVar, Dict

Input = TypeVar("Input", bound=Dict)

class RunnablePassthrough(Runnable[Input, Input]):
    """
    类型签名：Runnable[Input, Input]
    含义：输入类型 = 输出类型
    """
    pass

class RunnableAssign(Runnable[Dict, Dict]):
    """
    类型签名：Runnable[Dict, Dict]
    含义：输入和输出都是字典
    """
    pass
```

### 类型推断

```python
# 类型系统自动推断
chain = RunnablePassthrough.assign(
    context=retriever  # retriever: Runnable[str, List[Document]]
)

# 类型推断：
# - 输入类型：Dict[str, Any]
# - 输出类型：Dict[str, Any]
# - context 字段类型：List[Document]
```

### 类型检查

```python
from typing import TypedDict

class InputType(TypedDict):
    query: str
    user: str

class OutputType(TypedDict):
    query: str
    user: str
    context: list

# 类型注解
chain: Runnable[InputType, OutputType] = RunnablePassthrough.assign(
    context=retriever
)

# 类型检查器会验证：
# - 输入必须有 query 和 user 字段
# - 输出会有 query、user 和 context 字段
```

---

## 常见模式

### 模式 1：RAG 数据保留

```python
# 标准 RAG 模式
rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: retriever.invoke(x["query"])
    )
    | prompt
    | llm
)
```

### 模式 2：多级数据增强

```python
# 逐步添加字段
chain = (
    RunnablePassthrough.assign(
        rewritten_query=rewrite_func
    )
    | RunnablePassthrough.assign(
        context=lambda x: retriever.invoke(x["rewritten_query"])
    )
    | RunnablePassthrough.assign(
        metadata=get_metadata
    )
    | llm
)
```

### 模式 3：条件字段添加

```python
def conditional_add(x):
    if x.get("use_cache"):
        return get_cached_context(x["query"])
    else:
        return retriever.invoke(x["query"])

chain = RunnablePassthrough.assign(
    context=conditional_add
)
```

### 模式 4：字段转换

```python
# 转换现有字段
chain = RunnablePassthrough.assign(
    query=lambda x: x["query"].lower().strip(),
    user=lambda x: x.get("user", "anonymous")
)
```

---

## 调试技巧

### 1. 打印中间结果

```python
chain = (
    RunnableLambda(lambda x: print(f"输入: {x}") or x)
    | RunnablePassthrough.assign(context=retriever)
    | RunnableLambda(lambda x: print(f"assign 后: {x}") or x)
    | llm
)
```

### 2. 使用 LangSmith 追踪

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"

# 自动记录所有中间步骤
chain.invoke(input_data)
```

### 3. 单元测试

```python
def test_assign_preserves_input():
    """测试 assign 是否保留原始输入"""
    chain = RunnablePassthrough.assign(new_field="value")
    result = chain.invoke({"original": "data", "user": "alice"})

    assert "original" in result
    assert "user" in result
    assert "new_field" in result
    assert result["new_field"] == "value"

def test_assign_overwrites_same_key():
    """测试 assign 是否覆盖同名字段"""
    chain = RunnablePassthrough.assign(query="new")
    result = chain.invoke({"query": "old", "user": "alice"})

    assert result["query"] == "new"  # 被覆盖
    assert result["user"] == "alice"  # 保留
```

---

## 2025-2026 最新特性

### 1. 类型推断增强（LangChain v0.3+）

```python
# 自动类型推断
chain = RunnablePassthrough.assign(
    count=lambda x: len(x["items"])
)
# 类型系统自动识别: Runnable[dict, dict]
```

### 2. 键冲突警告（LangChain v0.3.15+）

```python
import warnings

chain = RunnablePassthrough.assign(query="new")
result = chain.invoke({"query": "old", "user": "alice"})

# 警告: UserWarning: Key 'query' already exists in input and will be overwritten
```

### 3. 性能优化（LangChain v0.3.18+）

```python
# 批处理性能提升 3-5 倍
results = await chain.abatch([
    {"query": "q1"},
    {"query": "q2"},
    {"query": "q3"}
])
```

---

## 总结

### 核心要点

1. **透传机制**：RunnablePassthrough 实现恒等函数，保持输入不变
2. **assign() 方法**：扩展输入字典，添加新字段而不丢失原有数据
3. **数据保留**：所有原始字段保留，同名字段会被覆盖
4. **性能考虑**：浅拷贝机制，大数据场景下内存友好

### 最佳实践

1. **避免键冲突**：使用唯一的键名
2. **注意执行顺序**：assign 中的字段串行执行
3. **使用类型注解**：提高代码可维护性
4. **添加单元测试**：验证数据完整性

### 常见陷阱

1. ❌ 误以为 assign 会替换整个输入
2. ❌ 忽略键冲突导致数据覆盖
3. ❌ 在 assign 中使用阻塞操作

---

**下一步**：
- 学习 `03_核心概念_02_RunnableLambda自定义函数.md` 了解函数包装机制
- 查看 `07_实战代码_01_基础透传与assign.md` 获取实战示例

---

**参考资料**：
- [RunnablePassthrough 官方文档](https://reference.langchain.com/v0.3/python/core/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html) (2025)
- [RunnablePassthrough 源码](https://reference.langchain.com/v0.3/python/_modules/langchain_core/runnables/passthrough.html) (2025)
- [LangChain's Hidden Gem: RunnablePassthrough.assign()](https://www.linkedin.com/posts/abhishek-rath-48498942_langchain-llm-ai-activity-7404830349609992192-U4uu) (2025)

---

**版本**：v1.0
**最后更新**：2026-02-18
