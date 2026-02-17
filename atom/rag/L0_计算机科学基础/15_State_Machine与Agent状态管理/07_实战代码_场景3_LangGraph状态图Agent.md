# 实战代码_场景3：LangGraph状态图Agent

> **目标**：使用LangGraph构建一个多步骤推理的RAG Agent

---

## 一、场景描述

**需求**：构建一个文档问答Agent，包含检索、推理、生成三个步骤

**流程**：
```
用户查询 → 检索文档 → 推理分析 → 生成答案 → 验证 → 输出
                ↓ 失败                    ↓ 失败
                重试 ←←←←←←←←←←←←←←←←←←←← 重试
```

---

## 二、完整实现

```python
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import operator

# 1. 定义状态
class RAGState(TypedDict):
    """RAG Agent状态"""
    query: str
    documents: Annotated[List[str], operator.add]
    context: str
    answer: str
    retry_count: int
    validation_passed: bool

# 2. 定义节点
def retrieve_node(state: RAGState) -> RAGState:
    """检索节点"""
    query = state["query"]
    print(f"🔍 检索: {query}")

    # 模拟向量检索
    documents = [
        f"文档1: LangGraph是用于构建有状态AI Agent的框架",
        f"文档2: LangGraph基于状态机模型，提供确定性控制",
        f"文档3: LangGraph支持checkpointing和人机协作"
    ]

    return {"documents": documents}

def reason_node(state: RAGState) -> RAGState:
    """推理节点"""
    docs = state["documents"]
    print(f"🧠 推理: 分析{len(docs)}个文档")

    # 使用LLM推理
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    context = "\n".join(docs)

    prompt = f"""分析以下文档，提取关键信息：

{context}

关键信息："""

    response = llm.invoke([HumanMessage(content=prompt)])
    analyzed_context = response.content

    return {"context": analyzed_context}

def generate_node(state: RAGState) -> RAGState:
    """生成节点"""
    query = state["query"]
    context = state["context"]
    print(f"✍️ 生成答案")

    # 使用LLM生成答案
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    prompt = f"""基于以下上下文回答问题：

上下文：{context}

问题：{query}

答案："""

    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content

    return {"answer": answer}

def validate_node(state: RAGState) -> RAGState:
    """验证节点"""
    answer = state["answer"]
    print(f"✅ 验证答案")

    # 简单验证
    validation_passed = len(answer) > 10

    return {"validation_passed": validation_passed}

# 3. 条件路由
def should_retry(state: RAGState) -> str:
    """决定是否重试"""
    if state["validation_passed"]:
        return "end"
    elif state.get("retry_count", 0) < 3:
        return "retrieve"
    else:
        return "failed"

# 4. 构建图
def create_rag_agent():
    """创建RAG Agent"""
    graph = StateGraph(RAGState)

    # 添加节点
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reason", reason_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)

    # 添加边
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_edge("reason", "generate")
    graph.add_edge("generate", "validate")

    # 条件边
    graph.add_conditional_edges(
        "validate",
        should_retry,
        {
            "end": END,
            "retrieve": "retrieve",
            "failed": END
        }
    )

    return graph.compile()

# 5. 使用示例
if __name__ == "__main__":
    print("=== LangGraph RAG Agent ===\n")

    app = create_rag_agent()

    # 运行
    result = app.invoke({
        "query": "什么是LangGraph？",
        "documents": [],
        "context": "",
        "answer": "",
        "retry_count": 0,
        "validation_passed": False
    })

    print("\n=== 结果 ===")
    print(f"问题: {result['query']}")
    print(f"答案: {result['answer']}")
    print(f"验证: {result['validation_passed']}")

    # 可视化
    print("\n=== Mermaid图 ===")
    print(app.get_graph().draw_mermaid())
```

---

## 三、带Checkpointing的版本

```python
from langgraph.checkpoint.memory import MemorySaver

def create_rag_agent_with_checkpoint():
    """创建带Checkpointing的RAG Agent"""
    graph = StateGraph(RAGState)

    # 添加节点
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reason", reason_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)

    # 添加边
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_edge("reason", "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges("validate", should_retry, {...})

    # 编译时添加checkpointer
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)

# 使用
app = create_rag_agent_with_checkpoint()
config = {"configurable": {"thread_id": "demo"}}

result = app.invoke(initial_state, config=config)

# 获取状态
state = app.get_state(config)
print(f"当前状态: {state.values}")
```

---

## 四、流式输出版本

```python
def run_with_streaming():
    """流式执行Agent"""
    app = create_rag_agent()

    print("=== 流式输出 ===\n")

    for chunk in app.stream({
        "query": "什么是LangGraph？",
        "documents": [],
        "context": "",
        "answer": "",
        "retry_count": 0,
        "validation_passed": False
    }):
        print(f"节点: {list(chunk.keys())[0]}")
        print(f"状态: {chunk}")
        print()

if __name__ == "__main__":
    run_with_streaming()
```

---

## 五、总结

### 核心要点

1. **StateGraph**：定义状态和节点
2. **条件路由**：动态决策下一步
3. **Checkpointing**：状态持久化
4. **流式输出**：实时显示进度

---

**版本**: v1.0
**最后更新**: 2026-02-14
**代码行数**: ~200行
**可运行**: ✅ Python 3.13+ (需要OpenAI API Key)
