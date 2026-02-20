# 核心概念3：RunnableSequence创建

## 概述

RunnableSequence 是管道操作符 `|` 的返回结果，它将多个 Runnable 组件串联成一个新的 Runnable。

**一句话定义**：RunnableSequence 是一个容器，按顺序执行多个 Runnable，将前一个的输出作为后一个的输入。

---

## 1. RunnableSequence 的创建

### 1.1 通过管道操作符创建

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("讲个笑话")
model = ChatOpenAI()
parser = StrOutputParser()

# 使用 | 创建 RunnableSequence
chain = prompt | model | parser

print(type(chain))
# 输出: <class 'langchain_core.runnables.base.RunnableSequence'>
```

### 1.2 直接创建 RunnableSequence

```python
from langchain_core.runnables import RunnableSequence

# 方式1: 使用构造函数
chain = RunnableSequence(first=prompt, last=model)

# 方式2: 使用 from_runnables 类方法
chain = RunnableSequence.from_runnables([prompt, model, parser])

# 方式3: 使用管道操作符（推荐）
chain = prompt | model | parser
```

---

## 2. RunnableSequence 的内部结构

### 2.1 核心属性

```python
class RunnableSequence(Runnable):
    """RunnableSequence 简化实现"""

    def __init__(self, *steps: Runnable):
        self.steps = list(steps)  # 所有步骤
        self.first = steps[0]     # 第一个步骤
        self.last = steps[-1]     # 最后一个步骤

    def invoke(self, input):
        """按顺序执行所有步骤"""
        result = input
        for step in self.steps:
            result = step.invoke(result)
        return result
```

### 2.2 检查 RunnableSequence 结构

```python
chain = prompt | model | parser

# 检查步骤列表
print(chain.steps)
# 输出: [ChatPromptTemplate, ChatOpenAI, StrOutputParser]

# 检查第一个和最后一个
print(chain.first)  # ChatPromptTemplate
print(chain.last)   # StrOutputParser

# 检查步骤数量
print(len(chain.steps))  # 3
```

---

## 3. 嵌套 RunnableSequence 的扁平化

### 3.1 嵌套问题

```python
# 创建两个子链
sub_chain1 = prompt | model
sub_chain2 = parser | validator

# 组合子链
full_chain = sub_chain1 | sub_chain2

# 理论上会产生嵌套结构
# RunnableSequence(
#     RunnableSequence(prompt, model),
#     RunnableSequence(parser, validator)
# )
```

### 3.2 自动扁平化

```python
# LangChain 会自动扁平化嵌套的 RunnableSequence
full_chain = sub_chain1 | sub_chain2

print(full_chain.steps)
# 输出: [ChatPromptTemplate, ChatOpenAI, StrOutputParser, Validator]
# 而不是: [RunnableSequence, RunnableSequence]
```

**扁平化实现**：

```python
class RunnableSequence(Runnable):
    def __or__(self, other):
        """重载 | 运算符，自动扁平化"""
        # 如果 other 也是 RunnableSequence，展开它的步骤
        if isinstance(other, RunnableSequence):
            return RunnableSequence(*self.steps, *other.steps)
        else:
            return RunnableSequence(*self.steps, other)
```

### 3.3 扁平化的好处

```python
# 好处1: 简化结构
chain = (prompt | model) | (parser | validator)
# 扁平化后: [prompt, model, parser, validator]
# 而不是: [[prompt, model], [parser, validator]]

# 好处2: 提高性能
# 避免多层嵌套调用的开销

# 好处3: 便于调试
print(chain.steps)  # 直接看到所有步骤
```

---

## 4. RunnableSequence 的执行流程

### 4.1 同步执行（invoke）

```python
class RunnableSequence(Runnable):
    def invoke(self, input):
        """同步执行所有步骤"""
        result = input
        for step in self.steps:
            result = step.invoke(result)
        return result
```

**执行示例**：

```python
chain = prompt | model | parser

result = chain.invoke({"topic": "AI"})

# 执行流程:
# 1. input = {"topic": "AI"}
# 2. result = prompt.invoke({"topic": "AI"})  → ChatPromptValue
# 3. result = model.invoke(ChatPromptValue)   → AIMessage
# 4. result = parser.invoke(AIMessage)        → str
# 5. return result
```

### 4.2 异步执行（ainvoke）

```python
class RunnableSequence(Runnable):
    async def ainvoke(self, input):
        """异步执行所有步骤"""
        result = input
        for step in self.steps:
            result = await step.ainvoke(result)
        return result
```

**执行示例**：

```python
import asyncio

chain = prompt | model | parser

async def main():
    result = await chain.ainvoke({"topic": "AI"})
    print(result)

asyncio.run(main())
```

### 4.3 批量执行（batch）

```python
class RunnableSequence(Runnable):
    def batch(self, inputs):
        """批量执行"""
        results = inputs
        for step in self.steps:
            results = step.batch(results)
        return results
```

**执行示例**：

```python
chain = prompt | model | parser

results = chain.batch([
    {"topic": "AI"},
    {"topic": "Python"},
    {"topic": "数据库"}
])

# 执行流程:
# 1. inputs = [{"topic": "AI"}, {"topic": "Python"}, {"topic": "数据库"}]
# 2. results = prompt.batch(inputs)  → [ChatPromptValue, ChatPromptValue, ChatPromptValue]
# 3. results = model.batch(results)  → [AIMessage, AIMessage, AIMessage]
# 4. results = parser.batch(results) → [str, str, str]
# 5. return results
```

### 4.4 流式执行（stream）

```python
class RunnableSequence(Runnable):
    def stream(self, input):
        """流式执行"""
        # 只有最后一个步骤支持流式
        result = input
        for step in self.steps[:-1]:
            result = step.invoke(result)

        # 最后一个步骤流式返回
        for chunk in self.steps[-1].stream(result):
            yield chunk
```

**执行示例**：

```python
chain = prompt | model | parser

for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)

# 执行流程:
# 1. result = prompt.invoke({"topic": "AI"})  → ChatPromptValue
# 2. result = model.invoke(ChatPromptValue)   → AIMessage
# 3. for chunk in parser.stream(AIMessage):   → 逐块返回
#        yield chunk
```

---

## 5. 错误传播机制

### 5.1 错误会中断执行

```python
chain = step1 | step2 | step3

try:
    result = chain.invoke(input)
except Exception as e:
    print(f"链执行失败: {e}")

# 如果 step2 抛出异常，step3 不会执行
```

### 5.2 错误传播示例

```python
from langchain_core.runnables import RunnableLambda

def step1(x):
    print("Step 1 执行")
    return x + "_step1"

def step2(x):
    print("Step 2 执行")
    raise ValueError("Step 2 失败")

def step3(x):
    print("Step 3 执行")
    return x + "_step3"

chain = (
    RunnableLambda(step1)
    | RunnableLambda(step2)
    | RunnableLambda(step3)
)

try:
    result = chain.invoke("input")
except ValueError as e:
    print(f"错误: {e}")

# 输出:
# Step 1 执行
# Step 2 执行
# 错误: Step 2 失败
# (Step 3 不会执行)
```

### 5.3 错误处理策略

```python
from langchain_core.runnables import RunnableLambda

def safe_step(func):
    """包装步骤，捕获错误"""
    def wrapper(x):
        try:
            return func(x)
        except Exception as e:
            print(f"步骤失败: {e}")
            return x  # 返回原始输入，继续执行
    return wrapper

chain = (
    RunnableLambda(safe_step(step1))
    | RunnableLambda(safe_step(step2))
    | RunnableLambda(safe_step(step3))
)

result = chain.invoke("input")
# 即使 step2 失败，step3 仍会执行
```

---

## 6. 中间结果的访问

### 6.1 无法直接访问中间结果

```python
chain = prompt | model | parser

result = chain.invoke({"topic": "AI"})

# ❌ 无法直接获取 model 的输出
# 只能获取最终的 parser 输出
```

### 6.2 使用回调访问中间结果

```python
from langchain.callbacks.base import BaseCallbackHandler

class IntermediateResultHandler(BaseCallbackHandler):
    """捕获中间结果的回调"""

    def __init__(self):
        self.results = []

    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"链开始: {inputs}")

    def on_chain_end(self, outputs, **kwargs):
        print(f"链结束: {outputs}")
        self.results.append(outputs)

# 使用
handler = IntermediateResultHandler()
chain = prompt | model | parser

result = chain.invoke(
    {"topic": "AI"},
    config={"callbacks": [handler]}
)

print(handler.results)
```

### 6.3 使用 RunnablePassthrough 保存中间结果

```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# 在管道中保存中间结果
chain = (
    prompt
    | model
    | RunnableParallel(
        final_output=parser,
        raw_message=RunnablePassthrough()  # 保存 AIMessage
    )
)

result = chain.invoke({"topic": "AI"})
print(result)
# {
#     "final_output": "AI 笑话...",
#     "raw_message": AIMessage(content="AI 笑话...")
# }
```

---

## 7. RunnableSequence 的优化

### 7.1 批处理优化

```python
# LangChain 会自动优化批处理
chain = prompt | model | parser

# 批量调用
results = chain.batch([input1, input2, input3])

# 内部优化:
# 1. prompt.batch([input1, input2, input3])  → 一次调用
# 2. model.batch([...])                      → 一次 API 调用（而不是3次）
# 3. parser.batch([...])                     → 一次调用
```

### 7.2 流式优化

```python
# 只有最后一个步骤需要流式
chain = prompt | model | parser

for chunk in chain.stream(input):
    print(chunk, end="")

# 内部优化:
# 1. prompt.invoke(input)  → 同步执行
# 2. model.invoke(...)     → 同步执行
# 3. parser.stream(...)    → 流式执行（只有最后一步）
```

---

## 8. 在 AI Agent 中的应用

### 8.1 构建多步推理链

```python
from langchain_core.runnables import RunnableLambda

# 步骤1: 分析问题
def analyze_question(question):
    return f"分析: {question}"

# 步骤2: 检索信息
def retrieve_info(analysis):
    return f"检索结果: {analysis}"

# 步骤3: 生成答案
def generate_answer(info):
    return f"答案: {info}"

# 创建 RunnableSequence
reasoning_chain = (
    RunnableLambda(analyze_question)
    | RunnableLambda(retrieve_info)
    | RunnableLambda(generate_answer)
)

result = reasoning_chain.invoke("什么是 AI？")
print(result)
# 输出: 答案: 检索结果: 分析: 什么是 AI？
```

### 8.2 构建 RAG 管道

```python
from langchain_core.runnables import RunnablePassthrough

# RAG 管道 = RunnableSequence
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | parser
)

# 检查结构
print(type(rag_chain))  # RunnableSequence
print(rag_chain.steps)
# [RunnableParallel, ChatPromptTemplate, ChatOpenAI, StrOutputParser]
```

### 8.3 构建对话系统

```python
# 对话系统 = 多步 RunnableSequence
conversation_chain = (
    load_history          # 加载历史
    | build_prompt        # 构建提示
    | model               # 生成回复
    | save_history        # 保存历史
    | format_response     # 格式化输出
)

# 每个步骤都是 Runnable
# 组合后形成 RunnableSequence
```

---

## 9. RunnableSequence vs 传统 Chain

### 9.1 传统 Chain（LangChain 0.1.x）

```python
from langchain.chains import LLMChain

# 传统方式
chain = LLMChain(
    prompt=prompt,
    llm=model,
    output_parser=parser
)

result = chain.run({"topic": "AI"})
```

### 9.2 RunnableSequence（LCEL）

```python
# LCEL 方式
chain = prompt | model | parser

result = chain.invoke({"topic": "AI"})
```

### 9.3 对比

| 特性 | 传统 Chain | RunnableSequence |
|------|-----------|------------------|
| **语法** | 配置对象 | 管道操作符 |
| **可读性** | 需要理解配置 | 从左到右，直观 |
| **可组合性** | 有限 | 任意组合 |
| **异步支持** | 需要单独实现 | 自动支持 |
| **批处理** | 需要单独实现 | 自动支持 |
| **流式** | 需要单独实现 | 自动支持 |
| **类型安全** | 弱 | 强（泛型） |

---

## 10. 调试 RunnableSequence

### 10.1 打印步骤信息

```python
chain = prompt | model | parser

# 打印步骤
for i, step in enumerate(chain.steps):
    print(f"步骤 {i+1}: {type(step).__name__}")

# 输出:
# 步骤 1: ChatPromptTemplate
# 步骤 2: ChatOpenAI
# 步骤 3: StrOutputParser
```

### 10.2 逐步执行

```python
chain = prompt | model | parser

# 手动逐步执行
input_data = {"topic": "AI"}

result = input_data
for i, step in enumerate(chain.steps):
    print(f"\n步骤 {i+1}: {type(step).__name__}")
    print(f"输入: {result}")
    result = step.invoke(result)
    print(f"输出: {result}")

print(f"\n最终结果: {result}")
```

### 10.3 使用回调追踪

```python
from langchain.callbacks import StdOutCallbackHandler

chain = prompt | model | parser

result = chain.invoke(
    {"topic": "AI"},
    config={"callbacks": [StdOutCallbackHandler()]}
)

# 自动打印每个步骤的输入输出
```

---

## 11. 高级用法

### 11.1 条件执行

```python
from langchain_core.runnables import RunnableBranch

# 根据条件选择不同的序列
conditional_chain = RunnableBranch(
    (lambda x: len(x) < 10, short_chain),   # 短文本链
    (lambda x: len(x) < 100, medium_chain), # 中等文本链
    long_chain                               # 长文本链（默认）
)

# conditional_chain 本身也是 Runnable
# 可以继续组合
full_chain = preprocessor | conditional_chain | postprocessor
```

### 11.2 循环执行

```python
from langchain_core.runnables import RunnableLambda

def loop_until_done(max_iterations=5):
    """循环执行直到满足条件"""
    def _loop(input):
        result = input
        for i in range(max_iterations):
            result = chain.invoke(result)
            if is_done(result):
                break
        return result
    return RunnableLambda(_loop)

# 使用
iterative_chain = loop_until_done(max_iterations=10)
```

### 11.3 动态构建序列

```python
def build_chain(steps: List[str]):
    """根据配置动态构建链"""
    step_map = {
        "prompt": prompt,
        "model": model,
        "parser": parser,
        "validator": validator,
        "formatter": formatter
    }

    chain = step_map[steps[0]]
    for step_name in steps[1:]:
        chain = chain | step_map[step_name]

    return chain

# 使用
chain1 = build_chain(["prompt", "model", "parser"])
chain2 = build_chain(["prompt", "model", "parser", "validator", "formatter"])
```

---

## 12. 性能考虑

### 12.1 步骤数量的影响

```python
# 步骤越多，执行时间越长
short_chain = step1 | step2  # 快
long_chain = step1 | step2 | step3 | step4 | step5  # 慢

# 但每个步骤的开销很小（主要是函数调用）
# 真正的瓶颈通常是 LLM API 调用
```

### 12.2 批处理的优势

```python
# 单次调用
for input in inputs:
    result = chain.invoke(input)  # N 次 API 调用

# 批量调用
results = chain.batch(inputs)  # 1 次 API 调用（如果 LLM 支持批处理）
```

### 12.3 流式的优势

```python
# 同步调用：等待完整响应
result = chain.invoke(input)  # 等待 5 秒
print(result)

# 流式调用：立即开始输出
for chunk in chain.stream(input):  # 立即开始
    print(chunk, end="")
```

---

## 13. 总结

### RunnableSequence 的核心要点

1. **创建方式**：通过 `|` 运算符自动创建
2. **内部结构**：包含 steps、first、last 属性
3. **自动扁平化**：嵌套的 RunnableSequence 会被展开
4. **执行流程**：按顺序执行所有步骤，前一个的输出是后一个的输入
5. **错误传播**：任何步骤失败，整个链失败
6. **统一接口**：支持 invoke、ainvoke、batch、stream

### 在 AI Agent 中的价值

- **模块化**：每个步骤独立，易于测试和替换
- **可组合**：可以任意组合和嵌套
- **可观测**：可以追踪每个步骤的执行
- **高性能**：自动优化批处理和流式

---

## 14. 学习检查

完成本节后，检查是否掌握：

- [ ] 理解 RunnableSequence 的创建方式
- [ ] 知道 RunnableSequence 的内部结构（steps、first、last）
- [ ] 理解嵌套 RunnableSequence 的扁平化机制
- [ ] 掌握 invoke、ainvoke、batch、stream 的执行流程
- [ ] 理解错误传播机制
- [ ] 能调试 RunnableSequence 的执行过程

---

[Source: Pinecone LCEL Tutorial - https://www.pinecone.io/learn/series/langchain/langchain-expression-language]
[Source: LangChain Official Docs - https://python.langchain.com/docs/concepts/]
