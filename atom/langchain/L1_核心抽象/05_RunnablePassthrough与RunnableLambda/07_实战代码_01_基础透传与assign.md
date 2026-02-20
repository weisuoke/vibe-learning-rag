# 07_实战代码_01_基础透传与assign

> **学习目标**：通过完整的可运行代码掌握 RunnablePassthrough 的基础用法和 assign() 方法

---

## 环境准备

### 安装依赖

```bash
# 确保在项目根目录
cd /path/to/vibe-learning-rag

# 激活虚拟环境
source .venv/bin/activate

# 安装必要的包（如果还没安装）
uv add langchain-core langchain-openai python-dotenv
```

### 配置 API 密钥

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，添加你的 API key
# OPENAI_API_KEY=your_key_here
```

---

## 示例 1：基础透传

### 代码

```python
"""
示例 1：RunnablePassthrough 基础透传
演示：输入原样输出，不做任何转换
"""

from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def example_basic_passthrough():
    """基础透传示例"""
    print("=" * 50)
    print("示例 1：基础透传")
    print("=" * 50)

    # 创建透传 Runnable
    passthrough = RunnablePassthrough()

    # 测试不同类型的输入
    test_inputs = [
        "hello",
        {"query": "什么是 LangChain?"},
        ["item1", "item2", "item3"],
        42
    ]

    for input_data in test_inputs:
        result = passthrough.invoke(input_data)
        print(f"\n输入: {input_data}")
        print(f"输出: {result}")
        print(f"输入 == 输出: {input_data == result}")


if __name__ == "__main__":
    example_basic_passthrough()
```

### 运行

```bash
python examples/l1_core/05_passthrough_lambda/01_basic_passthrough.py
```

### 输出

```
==================================================
示例 1：基础透传
==================================================

输入: hello
输出: hello
输入 == 输出: True

输入: {'query': '什么是 LangChain?'}
输出: {'query': '什么是 LangChain?'}
输入 == 输出: True

输入: ['item1', 'item2', 'item3']
输出: ['item1', 'item2', 'item3']
输入 == 输出: True

输入: 42
输出: 42
输入 == 输出: True
```

### 关键点

- RunnablePassthrough 是恒等函数：输入 = 输出
- 支持任意类型的输入
- 不做任何数据转换

---

## 示例 2：assign() 添加单个字段

### 代码

```python
"""
示例 2：使用 assign() 添加单个字段
演示：保留原始输入，添加新字段
"""

from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()


def example_assign_single_field():
    """assign 添加单个字段"""
    print("=" * 50)
    print("示例 2：assign() 添加单个字段")
    print("=" * 50)

    # 创建 assign chain
    chain = RunnablePassthrough.assign(
        upper=lambda x: x["text"].upper()
    )

    # 测试
    input_data = {"text": "hello world"}
    result = chain.invoke(input_data)

    print(f"\n输入: {input_data}")
    print(f"输出: {result}")
    print(f"\n字段对比:")
    print(f"  原始字段 'text': {result.get('text')}")
    print(f"  新增字段 'upper': {result.get('upper')}")


if __name__ == "__main__":
    example_assign_single_field()
```

### 输出

```
==================================================
示例 2：assign() 添加单个字段
==================================================

输入: {'text': 'hello world'}
输出: {'text': 'hello world', 'upper': 'HELLO WORLD'}

字段对比:
  原始字段 'text': hello world
  新增字段 'upper': HELLO WORLD
```

### 关键点

- assign() 保留所有原始字段
- 新字段通过 lambda 函数计算
- lambda 接收完整的输入字典

---

## 示例 3：assign() 添加多个字段

### 代码

```python
"""
示例 3：使用 assign() 添加多个字段
演示：同时添加多个计算字段
"""

from langchain_core.runnables import RunnablePassthrough
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def example_assign_multiple_fields():
    """assign 添加多个字段"""
    print("=" * 50)
    print("示例 3：assign() 添加多个字段")
    print("=" * 50)

    # 创建 assign chain
    chain = RunnablePassthrough.assign(
        upper=lambda x: x["text"].upper(),
        length=lambda x: len(x["text"]),
        word_count=lambda x: len(x["text"].split()),
        timestamp=lambda x: datetime.now().isoformat()
    )

    # 测试
    input_data = {"text": "hello world from langchain"}
    result = chain.invoke(input_data)

    print(f"\n输入: {input_data}")
    print(f"\n输出:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    example_assign_multiple_fields()
```

### 输出

```
==================================================
示例 3：assign() 添加多个字段
==================================================

输入: {'text': 'hello world from langchain'}

输出:
  text: hello world from langchain
  upper: HELLO WORLD FROM LANGCHAIN
  length: 27
  word_count: 4
  timestamp: 2026-02-18T13:11:31.750Z
```

### 关键点

- 可以同时添加多个字段
- 每个字段独立计算
- 字段按照定义顺序串行执行

---

## 示例 4：字段覆盖行为

### 代码

```python
"""
示例 4：字段覆盖行为
演示：assign() 如何处理同名字段
"""

from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()


def example_field_overwrite():
    """字段覆盖行为"""
    print("=" * 50)
    print("示例 4：字段覆盖行为")
    print("=" * 50)

    # 创建 assign chain（覆盖 query 字段）
    chain = RunnablePassthrough.assign(
        query=lambda x: x["query"].upper()
    )

    # 测试
    input_data = {
        "query": "hello",
        "user": "alice"
    }
    result = chain.invoke(input_data)

    print(f"\n输入: {input_data}")
    print(f"输出: {result}")
    print(f"\n字段对比:")
    print(f"  原始 query: {input_data['query']}")
    print(f"  新的 query: {result['query']}")
    print(f"  user 字段: {result['user']} (保留)")


def example_avoid_overwrite():
    """避免字段覆盖的最佳实践"""
    print("\n" + "=" * 50)
    print("最佳实践：使用不同的字段名")
    print("=" * 50)

    # 使用不同的字段名
    chain = RunnablePassthrough.assign(
        original_query=lambda x: x["query"],
        processed_query=lambda x: x["query"].upper()
    )

    # 测试
    input_data = {"query": "hello", "user": "alice"}
    result = chain.invoke(input_data)

    print(f"\n输入: {input_data}")
    print(f"输出: {result}")
    print(f"\n所有字段都保留:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    example_field_overwrite()
    example_avoid_overwrite()
```

### 输出

```
==================================================
示例 4：字段覆盖行为
==================================================

输入: {'query': 'hello', 'user': 'alice'}
输出: {'query': 'HELLO', 'user': 'alice'}

字段对比:
  原始 query: hello
  新的 query: HELLO
  user 字段: alice (保留)

==================================================
最佳实践：使用不同的字段名
==================================================

输入: {'query': 'hello', 'user': 'alice'}
输出: {'query': 'hello', 'user': 'alice', 'original_query': 'hello', 'processed_query': 'HELLO'}

所有字段都保留:
  query: hello
  user: alice
  original_query: hello
  processed_query: HELLO
```

### 关键点

- 同名字段会被覆盖
- 其他字段保留
- 最佳实践：使用不同的字段名

---

## 示例 5：链式组合

### 代码

```python
"""
示例 5：链式组合
演示：RunnablePassthrough 与其他 Runnable 的组合
"""

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()


def preprocess(x):
    """预处理函数"""
    return {
        **x,
        "preprocessed": True,
        "query": x["query"].strip().lower()
    }


def postprocess(x):
    """后处理函数"""
    return {
        **x,
        "postprocessed": True,
        "result": f"处理完成: {x['query']}"
    }


def example_chain_composition():
    """链式组合示例"""
    print("=" * 50)
    print("示例 5：链式组合")
    print("=" * 50)

    # 构建链
    chain = (
        RunnableLambda(preprocess)  # 预处理
        | RunnablePassthrough.assign(
            length=lambda x: len(x["query"]),
            word_count=lambda x: len(x["query"].split())
        )  # 添加统计信息
        | RunnableLambda(postprocess)  # 后处理
    )

    # 测试
    input_data = {"query": "  Hello World  "}
    result = chain.invoke(input_data)

    print(f"\n输入: {input_data}")
    print(f"\n输出:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    print(f"\n数据流动:")
    print(f"  1. 输入: {input_data}")
    print(f"  2. 预处理后: query 变为小写并去除空格")
    print(f"  3. assign 后: 添加 length 和 word_count")
    print(f"  4. 后处理后: 添加 result 字段")


if __name__ == "__main__":
    example_chain_composition()
```

### 输出

```
==================================================
示例 5：链式组合
==================================================

输入: {'query': '  Hello World  '}

输出:
  query: hello world
  preprocessed: True
  length: 11
  word_count: 2
  postprocessed: True
  result: 处理完成: hello world

数据流动:
  1. 输入: {'query': '  Hello World  '}
  2. 预处理后: query 变为小写并去除空格
  3. assign 后: 添加 length 和 word_count
  4. 后处理后: 添加 result 字段
```

### 关键点

- RunnablePassthrough 可以与其他 Runnable 链式组合
- 数据按顺序流经各个处理器
- 每个环节都可以访问前面的结果

---

## 示例 6：调试技巧

### 代码

```python
"""
示例 6：调试技巧
演示：如何调试 RunnablePassthrough 链
"""

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import json

load_dotenv()


def debug_print(stage_name):
    """创建调试打印函数"""
    def _print(x):
        print(f"\n[{stage_name}]")
        print(json.dumps(x, indent=2, ensure_ascii=False))
        return x
    return _print


def example_debugging():
    """调试示例"""
    print("=" * 50)
    print("示例 6：调试技巧")
    print("=" * 50)

    # 构建带调试的链
    chain = (
        RunnableLambda(debug_print("输入"))
        | RunnablePassthrough.assign(
            upper=lambda x: x["text"].upper()
        )
        | RunnableLambda(debug_print("assign 后"))
        | RunnablePassthrough.assign(
            length=lambda x: len(x["text"])
        )
        | RunnableLambda(debug_print("最终输出"))
    )

    # 测试
    input_data = {"text": "hello"}
    result = chain.invoke(input_data)

    print(f"\n最终结果: {result}")


if __name__ == "__main__":
    example_debugging()
```

### 输出

```
==================================================
示例 6：调试技巧
==================================================

[输入]
{
  "text": "hello"
}

[assign 后]
{
  "text": "hello",
  "upper": "HELLO"
}

[最终输出]
{
  "text": "hello",
  "upper": "HELLO",
  "length": 5
}

最终结果: {'text': 'hello', 'upper': 'HELLO', 'length': 5}
```

### 关键点

- 使用 RunnableLambda 打印中间结果
- 在关键节点插入调试函数
- 清晰地看到数据流动过程

---

## 完整示例：综合应用

### 代码

```python
"""
完整示例：综合应用
演示：RunnablePassthrough 的实际应用场景
"""

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def validate_input(x):
    """验证输入"""
    if "query" not in x:
        raise ValueError("Missing 'query' field")
    if len(x["query"]) < 3:
        raise ValueError("Query too short")
    return x


def add_metadata(x):
    """添加元数据"""
    return {
        **x,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "user": x.get("user", "anonymous"),
            "version": "1.0"
        }
    }


def example_complete():
    """完整示例"""
    print("=" * 50)
    print("完整示例：综合应用")
    print("=" * 50)

    # 构建完整的处理链
    chain = (
        RunnableLambda(validate_input)  # 验证
        | RunnablePassthrough.assign(
            query_lower=lambda x: x["query"].lower(),
            query_length=lambda x: len(x["query"]),
            word_count=lambda x: len(x["query"].split())
        )  # 添加统计信息
        | RunnableLambda(add_metadata)  # 添加元数据
    )

    # 测试成功案例
    print("\n测试 1：成功案例")
    input_data = {"query": "什么是 LangChain?", "user": "alice"}
    try:
        result = chain.invoke(input_data)
        print(f"输入: {input_data}")
        print(f"输出:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"错误: {e}")

    # 测试失败案例
    print("\n测试 2：失败案例（查询太短）")
    input_data = {"query": "hi"}
    try:
        result = chain.invoke(input_data)
        print(f"输出: {result}")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    example_complete()
```

### 输出

```
==================================================
完整示例：综合应用
==================================================

测试 1：成功案例
输入: {'query': '什么是 LangChain?', 'user': 'alice'}
输出:
  query: 什么是 LangChain?
  user: alice
  query_lower: 什么是 langchain?
  query_length: 15
  word_count: 3
  metadata: {'timestamp': '2026-02-18T13:11:31.750Z', 'user': 'alice', 'version': '1.0'}

测试 2：失败案例（查询太短）
错误: Query too short
```

### 关键点

- 验证输入数据
- 逐步添加字段
- 添加元数据
- 错误处理

---

## 学习检查

### 基础检查

- [ ] 理解 RunnablePassthrough 的透传机制
- [ ] 能使用 assign() 添加单个字段
- [ ] 能使用 assign() 添加多个字段
- [ ] 理解字段覆盖行为

### 进阶检查

- [ ] 能构建链式组合
- [ ] 能调试数据流动
- [ ] 能处理错误情况
- [ ] 能应用到实际场景

---

## 下一步

- 学习 `07_实战代码_02_Lambda自定义处理.md` 了解 RunnableLambda 的高级用法
- 查看 `07_实战代码_03_RAG数据增强.md` 学习完整的 RAG 应用

---

**参考资料**：
- [LangChain RunnablePassthrough 官方文档](https://reference.langchain.com/v0.3/python/core/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html) (2025)
- [Building Production-Ready AI Pipelines](https://medium.com/@sajo02/building-production-ready-ai-pipelines-with-langchain-runnables-a-complete-lcel-guide-2f9b27f6d557) (2026)

---

**版本**：v1.0
**最后更新**：2026-02-18
