# LangGraph回调处理

## 1. 【30字核心】

**LangGraph回调处理是在状态机工作流中正确追踪和链接LLM调用的技术，解决trace断链和span嵌套问题。**

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题

### LangGraph回调处理的第一性原理

#### 1. 最基础的定义

**LangGraph回调处理 = 在状态机节点中正确传递和管理回调上下文**

仅此而已！没有更基础的了。

#### 2. 为什么需要LangGraph回调处理？

**核心问题：LangGraph的状态机执行模型与传统链式调用不同，导致回调上下文容易丢失**

在传统的LangChain链式调用中，回调上下文会自动沿着调用链传递。但在LangGraph中：
- 节点是独立的函数
- 状态在节点间传递
- 回调上下文不会自动传递到节点内部的LLM调用
- 导致trace断链、span嵌套错误

#### 3. LangGraph回调处理的三层价值

##### 价值1：正确的trace链接

**问题**：使用`@observe`装饰器时，会创建新的trace而不是链接到现有trace

**解决**：
```python
# ❌ 错误：创建新trace
@observe(as_type="generation")
def get_prompt(name: str):
    return langfuse.get_prompt(name).prompt

# ✅ 正确：在节点内部显式传递CallbackHandler
def thinker(state: MessagesState):
    # 从config中获取callbacks
    callbacks = state.get("callbacks", [])
    prompt = get_prompt("test-prompt", callbacks=callbacks)
    return Command(goto="__end__", update={"messages": [prompt]})
```

##### 价值2：正确的span嵌套

**问题**：LangChain的CallbackHandler使用`start_span`而非`start_as_current_span`，导致span不在当前上下文中

**解决**：
```python
# ✅ 使用start_as_current_span确保正确嵌套
from langfuse import get_client
langfuse = get_client()

def call_moderation_model(streaming_response):
    with langfuse.start_as_current_span(name="critic.orchestrator.evaluate") as span:
        # 在span上下文中初始化handler
        langfuse_handler = CallbackHandler()
        # LLM调用会嵌套在这个span下
        result = llm.invoke(streaming_response, callbacks=[langfuse_handler])
        return result
```

##### 价值3：多LLM调用的正确追踪

**问题**：在一个节点中多次调用LLM时，spans可能出现在错误的层级

**解决**：每个LLM调用都需要显式传递CallbackHandler

```python
async def call_bedrock(state: dict):
    langfuse.update_current_trace(tags=["call_bedrock"])

    # 主LLM调用
    main_response = await llm.ainvoke([main_msg])

    # 评估LLM调用 - 需要显式传递handler
    callback_handler = CallbackHandler()
    for msg in messages:
        with langfuse.start_as_current_span(name="evaluation") as span:
            result = await llm.ainvoke(
                [msg],
                config={"callbacks": [callback_handler]}  # 关键：显式传递
            )
    return {"result": result}
```

#### 4. 从第一性原理推导LangGraph回调最佳实践

**推理链：**
```
1. LangGraph节点是独立函数，不自动继承回调上下文
   ↓
2. 需要显式传递CallbackHandler到每个LLM调用
   ↓
3. 使用start_as_current_span而非start_span确保正确嵌套
   ↓
4. 在graph.invoke()时传递callbacks到config
   ↓
5. 在节点内部从config或state中获取callbacks
   ↓
6. 将callbacks传递给所有LLM/Chain调用
   ↓
最终：实现完整的trace链接和正确的span嵌套
```

#### 5. 一句话总结第一性原理

**LangGraph回调处理的本质是显式管理回调上下文在状态机节点间的传递，确保trace完整性和span层级正确。**

---

## 3. 【核心概念】

### 核心概念1：CallbackHandler在LangGraph中的传递机制

**LangGraph不会自动将callbacks传递到节点内部，需要通过config或state显式传递**

```python
from langgraph.graph import StateGraph, MessagesState
from langfuse.callback import CallbackHandler

# 创建handler
langfuse_handler = CallbackHandler()

# 方式1：通过config传递（推荐）
config = {"callbacks": [langfuse_handler]}
response = graph.invoke({"messages": [user_input]}, config)

# 方式2：通过state传递
def node_function(state: MessagesState):
    # 从config中获取callbacks（LangGraph会自动注入）
    callbacks = state.get("callbacks", [])
    # 或者从RunnableConfig中获取
    # callbacks = config.get("callbacks", [])

    # 传递给LLM调用
    result = llm.invoke(messages, callbacks=callbacks)
    return {"messages": [result]}
```

**详细解释**：

LangGraph的执行模型与传统链式调用不同：
- **传统链式调用**：callbacks沿着调用链自动传递
- **LangGraph状态机**：节点是独立函数，callbacks不会自动传递

**关键点**：
1. 在`graph.invoke()`时通过`config`参数传递callbacks
2. LangGraph会将config注入到节点的执行上下文
3. 在节点内部需要显式获取并传递callbacks
4. 每个LLM/Chain调用都需要显式传递callbacks

**在RAG开发中的应用**：
在构建复杂的RAG工作流时，LangGraph常用于编排多个步骤（检索、重排序、生成等）。正确传递callbacks可以追踪每个步骤的性能和成本。

---

### 核心概念2：Trace链接问题与解决方案

**使用@observe装饰器时，langfuse_context不知道当前有trace，会创建新trace而非链接**

```python
# ❌ 问题代码：创建新trace
from langfuse.decorators import observe, langfuse_context

@observe(as_type="generation")
def get_prompt(name: str) -> str:
    langfuse_prompt = langfuse.get_prompt(name)
    langfuse_context.update_current_observation(prompt=langfuse_prompt)
    return langfuse_prompt.prompt

def thinker(state: MessagesState):
    # 这里调用get_prompt会创建新trace，而不是链接到graph的trace
    prompt = get_prompt("test-prompt")
    return Command(goto="__end__", update={"messages": [prompt]})

# ✅ 解决方案1：使用start_as_current_span
from langfuse import get_client
langfuse = get_client()

def get_prompt(name: str, span_context) -> str:
    with span_context:
        langfuse_prompt = langfuse.get_prompt(name)
        return langfuse_prompt.prompt

def thinker(state: MessagesState):
    with langfuse.start_as_current_span(name="get-prompt") as span:
        prompt = get_prompt("test-prompt", span)
    return Command(goto="__end__", update={"messages": [prompt]})

# ✅ 解决方案2：在graph级别使用span包装
with langfuse.start_as_current_span(name="langgraph-request") as span:
    span.update_trace(name=trace_id, input={"user_query": inputs["prompt"]})
    result = await graph.ainvoke(inputs, config=config)
    span.update(output={"response": result["result"]})
```

**为什么会出现这个问题？**

1. **@observe装饰器的工作原理**：
   - 检查当前是否有活跃的trace
   - 如果没有，创建新trace
   - 如果有，作为子observation链接

2. **LangGraph的执行上下文**：
   - CallbackHandler创建的trace在LangGraph的执行上下文中
   - @observe装饰器在节点函数内部执行
   - 两者的上下文不共享，导致@observe检测不到现有trace

3. **解决方案的原理**：
   - `start_as_current_span`显式设置当前span上下文
   - 在这个上下文中的所有操作都会链接到这个span
   - 确保trace的连续性

**在RAG开发中的应用**：
在RAG系统中，经常需要从Langfuse获取prompt模板并追踪使用情况。正确链接trace可以看到prompt版本与最终输出的关系。

---

### 核心概念3：Span嵌套与上下文管理

**CallbackHandler使用start_span而非start_as_current_span，导致span不在当前上下文中**

```python
# 问题场景：多个LLM调用的嵌套追踪
async def call_moderation(user_query: str, response: str):
    messages = [
        HumanMessage(content=f"Analyze safety: {response}"),
        HumanMessage(content=f"Analyze quality: {response}")
    ]

    results = []
    # ❌ 错误：span不会正确嵌套
    for msg in messages:
        with langfuse.start_as_current_span(name="evaluation") as span:
            result = await llm.ainvoke([msg])  # 这个调用的span会出现在外部
            results.append(result.content)
    return results

# ✅ 正确：显式传递CallbackHandler
async def call_moderation(user_query: str, response: str):
    messages = [
        HumanMessage(content=f"Analyze safety: {response}"),
        HumanMessage(content=f"Analyze quality: {response}")
    ]

    results = []
    for msg in messages:
        with langfuse.start_as_current_span(name="evaluation") as span:
            # 在span上下文中创建handler
            callback_handler = CallbackHandler()
            span.update(input=user_query)

            # 显式传递callbacks
            result = await llm.ainvoke(
                [msg],
                config={"callbacks": [callback_handler]}
            )

            span.update(output=[result.content])
            results.append(result.content)
    return results
```

**Span嵌套的三种模式**：

1. **自动嵌套（传统链式调用）**：
```python
# LangChain会自动处理
chain = prompt | llm | output_parser
result = chain.invoke(input, callbacks=[handler])
# span自动嵌套：chain -> prompt -> llm -> output_parser
```

2. **手动嵌套（LangGraph节点）**：
```python
# 需要显式传递callbacks
def node(state):
    callbacks = state.get("callbacks", [])
    result = llm.invoke(messages, callbacks=callbacks)
    return {"result": result}
```

3. **混合嵌套（自定义span + LLM调用）**：
```python
# 使用start_as_current_span + 显式传递callbacks
with langfuse.start_as_current_span(name="custom-step") as span:
    handler = CallbackHandler()
    result = llm.invoke(messages, config={"callbacks": [handler]})
```

**在RAG开发中的应用**：
在RAG系统中，检索、重排序、生成是三个独立步骤，每个步骤可能包含多个LLM调用。正确的span嵌套可以清晰展示每个步骤的耗时和成本。

---

## 4. 【最小可用】

掌握以下内容，就能在LangGraph中正确使用回调处理：

### 4.1 基础配置：在graph.invoke()时传递callbacks

```python
from langgraph.graph import StateGraph, MessagesState
from langfuse.callback import CallbackHandler

# 创建handler
langfuse_handler = CallbackHandler()

# 在invoke时传递
config = {"callbacks": [langfuse_handler]}
result = graph.invoke({"messages": [user_input]}, config)
```

### 4.2 节点内部：显式传递callbacks到LLM调用

```python
def my_node(state: MessagesState):
    # 从state或config中获取callbacks
    callbacks = state.get("callbacks", [])

    # 传递给LLM调用
    result = llm.invoke(state["messages"], callbacks=callbacks)

    return {"messages": [result]}
```

### 4.3 多LLM调用：每个调用都需要传递callbacks

```python
async def complex_node(state: dict):
    callbacks = state.get("callbacks", [])

    # 第一个LLM调用
    result1 = await llm.ainvoke([msg1], config={"callbacks": callbacks})

    # 第二个LLM调用
    result2 = await llm.ainvoke([msg2], config={"callbacks": callbacks})

    return {"result": [result1, result2]}
```

### 4.4 自定义span：使用start_as_current_span + CallbackHandler

```python
from langfuse import get_client
langfuse = get_client()

async def node_with_custom_span(state: dict):
    with langfuse.start_as_current_span(name="custom-step") as span:
        # 在span上下文中创建handler
        handler = CallbackHandler()

        span.update(input=state["input"])
        result = await llm.ainvoke([msg], config={"callbacks": [handler]})
        span.update(output=result.content)

    return {"result": result}
```

**这些知识足以：**
- 在LangGraph中正确追踪LLM调用
- 避免trace断链问题
- 实现基本的span嵌套
- 为后续学习复杂场景打基础

---

## 5. 【双重类比】

### 类比1：CallbackHandler传递

**前端类比：** React Context传递

在React中，Context不会自动传递到所有组件，需要使用Provider和useContext：
```jsx
// Provider在顶层
<CallbackContext.Provider value={handler}>
  <Graph />
</CallbackContext.Provider>

// 子组件中获取
function Node() {
  const handler = useContext(CallbackContext);
  // 使用handler
}
```

**日常生活类比：** 接力赛的接力棒

在接力赛中，接力棒不会自动传递，每个选手必须：
1. 接收上一个选手的接力棒
2. 跑完自己的赛程
3. 将接力棒传递给下一个选手

LangGraph的callbacks传递也是如此：
1. 从config接收callbacks
2. 在节点中使用
3. 传递给LLM调用

```python
# 就像接力赛
def node1(state):
    callbacks = state.get("callbacks", [])  # 接收接力棒
    result = llm.invoke(msg, callbacks=callbacks)  # 传递接力棒
    return {"result": result}
```

---

### 类比2：Trace链接问题

**前端类比：** 异步请求的请求ID传递

在前端中，如果不显式传递请求ID，新的异步请求会生成新ID：
```javascript
// ❌ 错误：每个请求都有新ID
async function fetchData() {
  const response = await fetch('/api/data');  // 新请求ID
  return response.json();
}

// ✅ 正确：传递请求ID
async function fetchData(requestId) {
  const response = await fetch('/api/data', {
    headers: { 'X-Request-ID': requestId }  // 使用现有ID
  });
  return response.json();
}
```

**日常生活类比：** 快递包裹的追踪号

如果你在中途换了一个新包裹（没有使用原来的追踪号），物流系统会认为这是一个新订单：
- ❌ 错误：中途换包裹 = 新追踪号 = 断链
- ✅ 正确：使用原包裹 = 同一追踪号 = 完整链路

```python
# 就像快递追踪号
# ❌ 错误：创建新trace（新追踪号）
@observe()
def get_prompt(name):
    return langfuse.get_prompt(name)

# ✅ 正确：链接到现有trace（使用原追踪号）
with langfuse.start_as_current_span(name="get-prompt") as span:
    prompt = langfuse.get_prompt(name)
```

---

### 类比3：Span嵌套

**前端类比：** 嵌套的性能监控

在前端性能监控中，需要正确嵌套performance marks：
```javascript
// ✅ 正确嵌套
performance.mark('parent-start');
  performance.mark('child1-start');
  // 操作1
  performance.mark('child1-end');

  performance.mark('child2-start');
  // 操作2
  performance.mark('child2-end');
performance.mark('parent-end');
```

**日常生活类比：** 文件夹的层级结构

文件夹的嵌套关系必须正确，否则文件会出现在错误的位置：
```
项目/
  ├── 文档/
  │   ├── 设计文档.md  ✅ 正确位置
  │   └── 技术文档.md  ✅ 正确位置
  └── 代码/
设计文档.md  ❌ 错误：应该在"文档"文件夹内
```

```python
# 就像文件夹嵌套
with langfuse.start_as_current_span(name="parent") as parent_span:
    handler = CallbackHandler()
    # 这个LLM调用会嵌套在parent下
    result = llm.invoke(msg, config={"callbacks": [handler]})
```

---

### 类比总结表

| LangGraph概念 | 前端类比 | 日常生活类比 |
|--------------|---------|-------------|
| CallbackHandler传递 | React Context传递 | 接力赛的接力棒 |
| Trace链接 | 请求ID传递 | 快递追踪号 |
| Span嵌套 | 性能监控嵌套 | 文件夹层级结构 |
| start_as_current_span | 设置当前上下文 | 进入特定房间工作 |
| 显式传递callbacks | 手动传递props | 手动传递工具 |

---

## 6. 【反直觉点】

### 误区1：LangGraph会自动传递callbacks ❌

**为什么错？**
- LangGraph的节点是独立函数，不会自动继承回调上下文
- 即使在`graph.invoke()`时传递了callbacks，节点内部的LLM调用也不会自动获取
- 必须显式从config或state中获取callbacks并传递给LLM调用

**为什么人们容易这样错？**
因为在传统的LangChain链式调用中，callbacks会自动沿着调用链传递。人们习惯了这种自动传递机制，误以为LangGraph也是如此。

**正确理解：**
```python
# ❌ 错误：以为callbacks会自动传递
def my_node(state: MessagesState):
    # 这里的llm调用不会自动获取callbacks
    result = llm.invoke(state["messages"])
    return {"messages": [result]}

# ✅ 正确：显式获取并传递callbacks
def my_node(state: MessagesState):
    # 从state中获取callbacks
    callbacks = state.get("callbacks", [])
    # 显式传递给LLM调用
    result = llm.invoke(state["messages"], callbacks=callbacks)
    return {"messages": [result]}
```

---

### 误区2：@observe装饰器会自动链接到现有trace ❌

**为什么错？**
- @observe装饰器检查当前上下文是否有活跃trace
- LangGraph的CallbackHandler创建的trace在不同的上下文中
- @observe检测不到这个trace，会创建新的独立trace

**为什么人们容易这样错？**
因为@observe装饰器在普通函数中工作良好，人们误以为在LangGraph节点中也会自动链接。实际上，LangGraph的执行上下文与@observe的上下文是隔离的。

**正确理解：**
```python
# ❌ 错误：@observe创建新trace
@observe(as_type="generation")
def get_prompt(name: str):
    return langfuse.get_prompt(name).prompt

def thinker(state: MessagesState):
    # 这里会创建新trace，而不是链接到graph的trace
    prompt = get_prompt("test-prompt")
    return {"messages": [prompt]}

# ✅ 正确：使用start_as_current_span
from langfuse import get_client
langfuse = get_client()

def get_prompt(name: str):
    return langfuse.get_prompt(name).prompt

def thinker(state: MessagesState):
    # 在graph级别使用span包装
    with langfuse.start_as_current_span(name="get-prompt") as span:
        prompt = get_prompt("test-prompt")
        span.update(output=prompt)
    return {"messages": [prompt]}
```

---

### 误区3：一个CallbackHandler实例可以在多个并发请求中重用 ❌

**为什么错？**
- CallbackHandler可能包含状态（如trace_id）
- 并发请求会导致状态冲突
- 不同请求的trace会混在一起

**为什么人们容易这样错？**
为了性能优化，人们习惯重用对象实例。但CallbackHandler是有状态的，重用会导致trace混乱。

**正确理解：**
```python
# ❌ 错误：重用handler导致trace混乱
langfuse_handler = CallbackHandler()  # 全局实例

async def handle_request(user_input):
    # 多个请求共享同一个handler
    result = graph.invoke(
        {"messages": [user_input]},
        config={"callbacks": [langfuse_handler]}
    )
    return result

# ✅ 正确：每个请求创建新handler
async def handle_request(user_input):
    # 每个请求新建handler实例
    handler = CallbackHandler()
    result = graph.invoke(
        {"messages": [user_input]},
        config={"callbacks": [handler]}
    )
    # 获取trace_id并flush
    trace_id = handler.get_trace_id()
    handler.flush()
    return result, trace_id
```

---

## 7. 【实战代码】

```python
"""
LangGraph回调处理实战示例
演示：正确的trace链接、span嵌套、多LLM调用追踪
"""

import os
from typing import Annotated, Dict, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langfuse import get_client
from langfuse.callback import CallbackHandler

load_dotenv()

# ===== 1. 初始化 =====
print("=== 初始化LangGraph和Langfuse ===")

langfuse = get_client()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ===== 2. 定义状态 =====
class CustomState(MessagesState):
    """自定义状态，包含callbacks"""
    user_query: str
    analysis_result: Dict[str, str]

# ===== 3. 定义节点函数 =====

def analyze_query(state: CustomState):
    """分析用户查询"""
    print(f"\n[Node: analyze_query] 分析查询: {state['user_query']}")

    # 关键：从state中获取callbacks
    callbacks = state.get("callbacks", [])

    # 使用start_as_current_span创建自定义span
    with langfuse.start_as_current_span(name="query-analysis") as span:
        span.update(input={"query": state["user_query"]})

        # 在span上下文中创建handler
        handler = CallbackHandler()

        # LLM调用1：分析查询意图
        intent_msg = HumanMessage(
            content=f"Analyze the intent of this query: {state['user_query']}"
        )
        intent_result = llm.invoke(
            [intent_msg],
            config={"callbacks": [handler]}  # 显式传递callbacks
        )

        # LLM调用2：提取关键词
        keywords_msg = HumanMessage(
            content=f"Extract keywords from: {state['user_query']}"
        )
        keywords_result = llm.invoke(
            [keywords_msg],
            config={"callbacks": [handler]}  # 显式传递callbacks
        )

        analysis = {
            "intent": intent_result.content,
            "keywords": keywords_result.content
        }

        span.update(output=analysis)

    return {
        "analysis_result": analysis,
        "messages": [AIMessage(content=f"Analysis complete: {analysis}")]
    }

def generate_response(state: CustomState):
    """生成最终响应"""
    print(f"\n[Node: generate_response] 生成响应")

    callbacks = state.get("callbacks", [])

    # 使用分析结果生成响应
    analysis = state["analysis_result"]
    prompt = f"""
Based on the analysis:
- Intent: {analysis['intent']}
- Keywords: {analysis['keywords']}

Generate a helpful response to: {state['user_query']}
"""

    response = llm.invoke(
        [HumanMessage(content=prompt)],
        callbacks=callbacks  # 传递callbacks
    )

    return {"messages": [response]}

# ===== 4. 构建LangGraph =====
print("\n=== 构建LangGraph ===")

graph_builder = StateGraph(CustomState)

# 添加节点
graph_builder.add_node("analyze", analyze_query)
graph_builder.add_node("generate", generate_response)

# 添加边
graph_builder.add_edge(START, "analyze")
graph_builder.add_edge("analyze", "generate")
graph_builder.add_edge("generate", END)

# 编译
graph = graph_builder.compile()

# ===== 5. 执行示例1：基础追踪 =====
print("\n=== 示例1：基础追踪 ===")

handler1 = CallbackHandler()
config1 = {"callbacks": [handler1]}

result1 = graph.invoke(
    {
        "user_query": "What is LangGraph?",
        "messages": [],
        "analysis_result": {}
    },
    config=config1
)

print(f"\n最终响应: {result1['messages'][-1].content[:100]}...")
print(f"Trace ID: {handler1.get_trace_id()}")
handler1.flush()

# ===== 6. 执行示例2：使用graph级别的span包装 =====
print("\n=== 示例2：Graph级别span包装 ===")

trace_id = f"langgraph-trace-{os.urandom(4).hex()}"

with langfuse.start_as_current_span(name="langgraph-request") as span:
    # 设置trace信息
    span.update_trace(
        name=trace_id,
        input={"user_query": "How does LangGraph handle callbacks?"},
        metadata={"session": "demo"}
    )

    # 创建handler
    handler2 = CallbackHandler()
    config2 = {"callbacks": [handler2]}

    # 执行graph
    result2 = graph.invoke(
        {
            "user_query": "How does LangGraph handle callbacks?",
            "messages": [],
            "analysis_result": {}
        },
        config=config2
    )

    # 更新span输出
    span.update(output={"response": result2['messages'][-1].content})

print(f"\n最终响应: {result2['messages'][-1].content[:100]}...")
print(f"Trace ID: {trace_id}")
handler2.flush()

# ===== 7. 执行示例3：并发请求（正确处理） =====
print("\n=== 示例3：并发请求处理 ===")

import asyncio

async def handle_request(query: str, request_id: str):
    """处理单个请求"""
    print(f"\n[Request {request_id}] 处理查询: {query}")

    # 每个请求创建新的handler
    handler = CallbackHandler()
    config = {"callbacks": [handler]}

    result = await graph.ainvoke(
        {
            "user_query": query,
            "messages": [],
            "analysis_result": {}
        },
        config=config
    )

    trace_id = handler.get_trace_id()
    handler.flush()

    return {
        "request_id": request_id,
        "query": query,
        "response": result['messages'][-1].content,
        "trace_id": trace_id
    }

async def handle_concurrent_requests():
    """并发处理多个请求"""
    queries = [
        ("What is RAG?", "req-1"),
        ("How does vector search work?", "req-2"),
        ("Explain embeddings", "req-3")
    ]

    tasks = [handle_request(query, req_id) for query, req_id in queries]
    results = await asyncio.gather(*tasks)

    for result in results:
        print(f"\n[{result['request_id']}] Trace: {result['trace_id']}")
        print(f"Response: {result['response'][:80]}...")

# 运行并发示例
asyncio.run(handle_concurrent_requests())

print("\n=== 完成 ===")
print("检查Langfuse dashboard查看完整的trace和span结构")
```

**运行输出示例：**
```
=== 初始化LangGraph和Langfuse ===

=== 构建LangGraph ===

=== 示例1：基础追踪 ===

[Node: analyze_query] 分析查询: What is LangGraph?

[Node: generate_response] 生成响应

最终响应: LangGraph is a framework for building stateful, multi-actor applications with LLMs...
Trace ID: 3a7f9b2c-1d4e-4f5a-8b3c-9e2f1a6d7c8b

=== 示例2：Graph级别span包装 ===

[Node: analyze_query] 分析查询: How does LangGraph handle callbacks?

[Node: generate_response] 生成响应

最终响应: LangGraph handles callbacks by allowing you to pass them through the config parameter...
Trace ID: langgraph-trace-a3f2b1c4

=== 示例3：并发请求处理 ===

[Request req-1] 处理查询: What is RAG?
[Request req-2] 处理查询: How does vector search work?
[Request req-3] 处理查询: Explain embeddings

[req-1] Trace: 5b8c9d3e-2f4a-4e6b-9c1d-8a7f6e5d4c3b
Response: RAG (Retrieval-Augmented Generation) is a technique that combines retrieval...

[req-2] Trace: 7d9e1f4a-3b5c-4d6e-8f2a-9b1c7d6e5f4a
Response: Vector search works by converting data into high-dimensional vectors...

[req-3] Trace: 9f1a2b3c-4d5e-6f7a-8b9c-1d2e3f4a5b6c
Response: Embeddings are dense vector representations of data...

=== 完成 ===
检查Langfuse dashboard查看完整的trace和span结构
```

---

## 8. 【面试必问】

### 问题："LangGraph中如何正确追踪LLM调用？"

**普通回答（❌ 不出彩）：**
"在LangGraph中，需要在invoke时传递callbacks参数，然后在节点中使用这些callbacks。"

**出彩回答（✅ 推荐）：**

> **LangGraph的回调处理有三个关键层次：**
>
> 1. **传递层**：在`graph.invoke()`时通过`config={"callbacks": [handler]}`传递CallbackHandler。LangGraph会将config注入到节点的执行上下文中。
>
> 2. **获取层**：在节点函数内部，需要显式从state或config中获取callbacks。这与传统链式调用不同，LangGraph的节点是独立函数，不会自动继承回调上下文。
>
> 3. **应用层**：将获取的callbacks显式传递给每个LLM/Chain调用。对于多个LLM调用，每个都需要传递callbacks以确保正确的span嵌套。
>
> **与传统LangChain的区别**：
> - 传统链式调用：callbacks自动沿调用链传递
> - LangGraph状态机：callbacks需要显式管理和传递
>
> **常见陷阱**：
> - 使用@observe装饰器会创建新trace而非链接现有trace
> - 解决方案是使用`start_as_current_span`并在上下文中创建CallbackHandler
>
> **在生产环境中的最佳实践**：
> - 每个请求创建新的CallbackHandler实例，避免并发冲突
> - 使用`handler.get_trace_id()`获取trace ID用于日志关联
> - 调用`handler.flush()`确保数据持久化

**为什么这个回答出彩？**
1. ✅ 分层次说明了完整的回调处理流程
2. ✅ 对比了与传统LangChain的区别，展示深度理解
3. ✅ 指出了常见陷阱和解决方案
4. ✅ 提供了生产环境的实践建议

---

## 9. 【化骨绵掌】

### 卡片1：LangGraph执行模型的本质

**一句话：** LangGraph是状态机而非调用链，节点是独立函数，上下文不自动传递

**举例：**
```python
# 传统链式调用：上下文自动传递
chain = prompt | llm | parser
result = chain.invoke(input, callbacks=[handler])  # callbacks自动传递

# LangGraph状态机：上下文需要显式管理
def node(state):
    callbacks = state.get("callbacks", [])  # 显式获取
    result = llm.invoke(msg, callbacks=callbacks)  # 显式传递
    return {"result": result}
```

**应用：** 理解这个本质差异是正确使用LangGraph回调的前提

---

### 卡片2：CallbackHandler的传递路径

**一句话：** graph.invoke(config) → LangGraph注入 → state/config → 节点获取 → LLM调用

**举例：**
```python
# 1. 传递给graph
config = {"callbacks": [handler]}
graph.invoke(input, config)

# 2. 节点中获取
def node(state):
    callbacks = state.get("callbacks", [])

# 3. 传递给LLM
result = llm.invoke(msg, callbacks=callbacks)
```

**应用：** 掌握完整的传递路径，避免在任何环节断链

---

### 卡片3：Trace链接问题的根源

**一句话：** @observe装饰器检查当前上下文，但LangGraph的trace在不同上下文中

**举例：**
```python
# 问题：两个独立的上下文
# 上下文1：LangGraph + CallbackHandler
graph.invoke(input, config={"callbacks": [handler]})

# 上下文2：@observe装饰器
@observe()
def get_prompt():  # 检测不到上下文1的trace
    return langfuse.get_prompt("name")
```

**应用：** 理解上下文隔离是解决trace链接问题的关键

---

### 卡片4：start_as_current_span的作用

**一句话：** 显式设置当前span上下文，确保后续操作链接到这个span

**举例：**
```python
with langfuse.start_as_current_span(name="custom-step") as span:
    # 在这个上下文中的所有操作都会链接到这个span
    handler = CallbackHandler()
    result = llm.invoke(msg, config={"callbacks": [handler]})
    span.update(output=result)
```

**应用：** 用于创建自定义span并确保正确的嵌套关系

---

### 卡片5：多LLM调用的span嵌套

**一句话：** 每个LLM调用都需要显式传递CallbackHandler才能正确嵌套

**举例：**
```python
async def node_with_multiple_llms(state):
    # 错误：只有第一个调用会被追踪
    result1 = await llm.ainvoke([msg1])
    result2 = await llm.ainvoke([msg2])

    # 正确：每个调用都传递callbacks
    callbacks = state.get("callbacks", [])
    result1 = await llm.ainvoke([msg1], config={"callbacks": callbacks})
    result2 = await llm.ainvoke([msg2], config={"callbacks": callbacks})
```

**应用：** 在节点中有多个LLM调用时，确保每个都被追踪

---

### 卡片6：并发请求的handler管理

**一句话：** 每个请求必须创建新的CallbackHandler实例，避免状态冲突

**举例：**
```python
# 错误：重用handler导致trace混乱
global_handler = CallbackHandler()

async def handle_request(query):
    result = graph.invoke(input, config={"callbacks": [global_handler]})

# 正确：每个请求新建handler
async def handle_request(query):
    handler = CallbackHandler()  # 新实例
    result = graph.invoke(input, config={"callbacks": [handler]})
    handler.flush()
```

**应用：** 在FastAPI/Django等并发环境中，确保trace不混乱

---

### 卡片7：Graph级别的span包装

**一句话：** 在graph.invoke()外层使用start_as_current_span可以创建顶层span

**举例：**
```python
with langfuse.start_as_current_span(name="langgraph-request") as span:
    span.update_trace(name=trace_id, input=input_data)
    result = graph.invoke(input, config={"callbacks": [handler]})
    span.update(output=result)
```

**应用：** 用于创建完整的请求级别trace，包含graph执行和自定义逻辑

---

### 卡片8：Trace ID的获取和使用

**一句话：** 使用handler.get_trace_id()获取trace ID，用于日志关联和调试

**举例：**
```python
handler = CallbackHandler()
result = graph.invoke(input, config={"callbacks": [handler]})

trace_id = handler.get_trace_id()
print(f"Trace ID: {trace_id}")
# 在日志中记录trace_id，方便后续查询
logger.info(f"Request completed, trace: {trace_id}")
```

**应用：** 将trace ID记录到应用日志，实现端到端的可追溯性

---

### 卡片9：Flush的重要性

**一句话：** handler.flush()确保数据立即发送到Langfuse，避免数据丢失

**举例：**
```python
handler = CallbackHandler()
result = graph.invoke(input, config={"callbacks": [handler]})

# 关键：调用flush确保数据持久化
handler.flush()

# 如果不调用flush，数据可能在进程结束时丢失
```

**应用：** 在每个请求结束时调用flush，特别是在短生命周期的进程中

---

### 卡片10：RAG系统中的应用

**一句话：** 在RAG工作流中，正确的回调处理可以追踪检索、重排序、生成的完整链路

**举例：**
```python
def rag_node(state):
    callbacks = state.get("callbacks", [])

    # 追踪检索步骤
    with langfuse.start_as_current_span(name="retrieval") as span:
        docs = retriever.invoke(query, config={"callbacks": callbacks})
        span.update(output={"doc_count": len(docs)})

    # 追踪生成步骤
    response = llm.invoke(prompt, callbacks=callbacks)

    return {"response": response}
```

**应用：** 在RAG系统中实现完整的性能监控和成本追踪

---

## 10. 【一句话总结】

**LangGraph回调处理的核心是显式管理回调上下文在状态机节点间的传递，通过在graph.invoke()时传递callbacks、在节点中获取并传递给每个LLM调用、使用start_as_current_span确保正确的span嵌套，从而实现完整的trace链接和准确的性能监控，在RAG开发中这对于追踪检索-生成全链路至关重要。**

---

## 附录：学习检查清单

- [ ] 理解LangGraph状态机与传统链式调用的区别
- [ ] 掌握callbacks在graph.invoke()中的传递方式
- [ ] 能够在节点函数中正确获取和传递callbacks
- [ ] 理解@observe装饰器的trace链接问题
- [ ] 掌握start_as_current_span的使用方法
- [ ] 能够处理多LLM调用的span嵌套
- [ ] 理解并发请求中handler的正确管理方式
- [ ] 能够使用graph级别的span包装
- [ ] 掌握trace ID的获取和使用
- [ ] 理解flush()的重要性和使用时机
- [ ] 能够在RAG系统中应用这些技术

## 下一步学习建议

1. **实践练习**：
   - 构建一个包含多个节点的LangGraph工作流
   - 在每个节点中正确传递callbacks
   - 在Langfuse dashboard中查看完整的trace结构

2. **进阶学习**：
   - 学习Agent工具调用追踪（下一个知识点）
   - 探索LangGraph的条件边和循环
   - 研究复杂RAG工作流的回调处理

3. **生产实践**：
   - 在FastAPI中集成LangGraph回调处理
   - 实现请求级别的trace关联
   - 建立性能监控和告警系统

---

**文档版本：** v1.0
**最后更新：** 2026-02-25
**作者：** Claude Code
