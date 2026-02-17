# 实战代码 - 场景1：对话历史管理（Part 1: LangGraph）

## 场景描述

使用 LangGraph 构建一个带记忆的对话 Agent，实现：
- 多轮对话历史管理
- 自动状态持久化
- 消息去重
- 对话历史访问

---

## 完整代码实现

```python
"""
LangGraph 对话历史管理示例
演示：使用 MessagesState 和 MemorySaver 管理对话历史
"""

import os
from typing import Annotated
from dotenv import load_dotenv

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 加载环境变量
load_dotenv()

# ===== 1. 定义 Agent 状态 =====
class AgentState(MessagesState):
    """Agent 状态，继承 MessagesState"""
    # messages: Annotated[list, add_messages]  # 自动继承
    pass


# ===== 2. 创建 LLM =====
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")  # 可选：使用代理
)


# ===== 3. 定义 Agent 节点 =====
def agent_node(state: AgentState) -> AgentState:
    """Agent 节点：调用 LLM 生成回复"""
    messages = state["messages"]

    print(f"\n[Agent] 收到 {len(messages)} 条消息")
    print(f"[Agent] 最新消息: {messages[-1].content}")

    # 调用 LLM
    response = llm.invoke(messages)

    print(f"[Agent] 生成回复: {response.content[:50]}...")

    # 返回新消息（add_messages 会自动合并到历史中）
    return {"messages": [response]}


# ===== 4. 构建图 =====
def create_agent():
    """创建 Agent 图"""
    # 创建图
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("agent", agent_node)

    # 添加边
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)

    # 创建检查点保存器（内存）
    memory = MemorySaver()

    # 编译图
    app = graph.compile(checkpointer=memory)

    return app


# ===== 5. 使用示例 =====
def main():
    """主函数"""
    print("=" * 60)
    print("LangGraph 对话历史管理示例")
    print("=" * 60)

    # 创建 Agent
    app = create_agent()

    # 配置：使用 thread_id 标识会话
    config = {"configurable": {"thread_id": "user_123"}}

    # ===== 第 1 轮对话 =====
    print("\n" + "=" * 60)
    print("第 1 轮对话")
    print("=" * 60)

    result1 = app.invoke({
        "messages": [
            SystemMessage(content="你是一个有帮助的 AI 助手"),
            HumanMessage(content="我叫张三，我在学习 RAG")
        ]
    }, config)

    print(f"\n[结果] 总消息数: {len(result1['messages'])}")
    print(f"[结果] AI 回复: {result1['messages'][-1].content}")

    # ===== 第 2 轮对话 =====
    print("\n" + "=" * 60)
    print("第 2 轮对话")
    print("=" * 60)

    result2 = app.invoke({
        "messages": [HumanMessage(content="我叫什么名字？")]
    }, config)

    print(f"\n[结果] 总消息数: {len(result2['messages'])}")
    print(f"[结果] AI 回复: {result2['messages'][-1].content}")

    # ===== 第 3 轮对话 =====
    print("\n" + "=" * 60)
    print("第 3 轮对话")
    print("=" * 60)

    result3 = app.invoke({
        "messages": [HumanMessage(content="我在学习什么？")]
    }, config)

    print(f"\n[结果] 总消息数: {len(result3['messages'])}")
    print(f"[结果] AI 回复: {result3['messages'][-1].content}")

    # ===== 查看完整对话历史 =====
    print("\n" + "=" * 60)
    print("完整对话历史")
    print("=" * 60)

    for i, msg in enumerate(result3["messages"]):
        role = msg.__class__.__name__
        content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
        print(f"{i}. [{role}] {content}")

    # ===== 访问特定轮次 =====
    print("\n" + "=" * 60)
    print("访问特定轮次")
    print("=" * 60)

    # 第 1 轮（索引 1-2）
    turn_1_user = result3["messages"][1]
    turn_1_ai = result3["messages"][2]
    print(f"第 1 轮:")
    print(f"  用户: {turn_1_user.content}")
    print(f"  AI: {turn_1_ai.content[:50]}...")

    # 最后一轮
    last_user = result3["messages"][-2]
    last_ai = result3["messages"][-1]
    print(f"\n最后一轮:")
    print(f"  用户: {last_user.content}")
    print(f"  AI: {last_ai.content[:50]}...")


# ===== 6. 高级示例：多用户会话隔离 =====
def multi_user_example():
    """多用户会话隔离示例"""
    print("\n" + "=" * 60)
    print("多用户会话隔离示例")
    print("=" * 60)

    app = create_agent()

    # 用户 1
    config_user1 = {"configurable": {"thread_id": "user_001"}}
    result1 = app.invoke({
        "messages": [HumanMessage(content="我叫李四")]
    }, config_user1)
    print(f"\n[用户1] {result1['messages'][-1].content}")

    # 用户 2
    config_user2 = {"configurable": {"thread_id": "user_002"}}
    result2 = app.invoke({
        "messages": [HumanMessage(content="我叫王五")]
    }, config_user2)
    print(f"[用户2] {result2['messages'][-1].content}")

    # 用户 1 继续对话
    result3 = app.invoke({
        "messages": [HumanMessage(content="我叫什么名字？")]
    }, config_user1)
    print(f"\n[用户1] {result3['messages'][-1].content}")

    # 用户 2 继续对话
    result4 = app.invoke({
        "messages": [HumanMessage(content="我叫什么名字？")]
    }, config_user2)
    print(f"[用户2] {result4['messages'][-1].content}")


# ===== 7. 高级示例：消息修剪 =====
def message_trimming_example():
    """消息修剪示例"""
    print("\n" + "=" * 60)
    print("消息修剪示例")
    print("=" * 60)

    # 定义带修剪的状态
    class TrimmedAgentState(MessagesState):
        pass

    def trim_messages(state: TrimmedAgentState) -> TrimmedAgentState:
        """保留最近 10 条消息"""
        messages = state["messages"]

        if len(messages) > 10:
            # 保留系统消息 + 最近 9 条
            system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]
            recent_msgs = messages[-9:]
            trimmed = system_msgs + recent_msgs

            print(f"[修剪] {len(messages)} -> {len(trimmed)} 条消息")
            return {"messages": trimmed}

        return state

    # 构建带修剪的图
    graph = StateGraph(TrimmedAgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("trim", trim_messages)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", "trim")
    graph.add_edge("trim", END)

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)

    # 模拟 20 轮对话
    config = {"configurable": {"thread_id": "trim_test"}}

    for i in range(20):
        result = app.invoke({
            "messages": [HumanMessage(content=f"消息 {i}")]
        }, config)
        print(f"第 {i+1} 轮: {len(result['messages'])} 条消息")


# ===== 8. 性能测试 =====
def performance_test():
    """性能测试"""
    import time

    print("\n" + "=" * 60)
    print("性能测试")
    print("=" * 60)

    app = create_agent()
    config = {"configurable": {"thread_id": "perf_test"}}

    # 测试 100 轮对话
    start = time.perf_counter()

    for i in range(100):
        app.invoke({
            "messages": [HumanMessage(content=f"消息 {i}")]
        }, config)

    elapsed = time.perf_counter() - start

    print(f"\n100 轮对话耗时: {elapsed:.2f}s")
    print(f"平均每轮: {elapsed/100*1000:.2f}ms")

    # 获取最终状态
    final_result = app.invoke({
        "messages": [HumanMessage(content="总结一下")]
    }, config)

    print(f"最终消息数: {len(final_result['messages'])}")


if __name__ == "__main__":
    # 运行主示例
    main()

    # 运行多用户示例
    multi_user_example()

    # 运行消息修剪示例
    message_trimming_example()

    # 运行性能测试
    # performance_test()  # 注释掉，避免调用 API
```

---

## 运行输出示例

```
============================================================
LangGraph 对话历史管理示例
============================================================

============================================================
第 1 轮对话
============================================================

[Agent] 收到 2 条消息
[Agent] 最新消息: 我叫张三，我在学习 RAG
[Agent] 生成回复: 你好，张三！很高兴认识你。RAG（Retrieval-Augmented...

[结果] 总消息数: 3
[结果] AI 回复: 你好，张三！很高兴认识你。RAG（Retrieval-Augmented Generation）是一种...

============================================================
第 2 轮对话
============================================================

[Agent] 收到 4 条消息
[Agent] 最新消息: 我叫什么名字？
[Agent] 生成回复: 你叫张三。

[结果] 总消息数: 5
[结果] AI 回复: 你叫张三。

============================================================
第 3 轮对话
============================================================

[Agent] 收到 6 条消息
[Agent] 最新消息: 我在学习什么？
[Agent] 生成回复: 你在学习 RAG（Retrieval-Augmented Generation）。

[结果] 总消息数: 7
[结果] AI 回复: 你在学习 RAG（Retrieval-Augmented Generation）。

============================================================
完整对话历史
============================================================
0. [SystemMessage] 你是一个有帮助的 AI 助手
1. [HumanMessage] 我叫张三，我在学习 RAG
2. [AIMessage] 你好，张三！很高兴认识你。RAG（Retrieval-Augmented...
3. [HumanMessage] 我叫什么名字？
4. [AIMessage] 你叫张三。
5. [HumanMessage] 我在学习什么？
6. [AIMessage] 你在学习 RAG（Retrieval-Augmented Generation）。

============================================================
访问特定轮次
============================================================
第 1 轮:
  用户: 我叫张三，我在学习 RAG
  AI: 你好，张三！很高兴认识你。RAG（Retrieval-Augmented...

最后一轮:
  用户: 我在学习什么？
  AI: 你在学习 RAG（Retrieval-Augmented Generation）。
```

---

## 关键要点

1. **MessagesState 自动管理**
   - 使用 `Annotated[list, add_messages]` 模式
   - 自动合并和去重消息
   - 无需手动管理列表

2. **MemorySaver 持久化**
   - 基于 thread_id 隔离会话
   - 自动保存和加载状态
   - 支持多用户并发

3. **性能特性**
   - List append：O(1) 摊销
   - 索引访问：O(1)
   - 消息去重：O(n)

4. **最佳实践**
   - 使用 thread_id 隔离用户会话
   - 定期修剪消息历史（避免过长）
   - 系统消息放在开头

---

## 参考来源（2025-2026）

### LangGraph 官方文档
- **LangGraph Memory Overview** (2026)
  - URL: https://langchain-ai.github.io/langgraph/concepts/memory/
  - 描述：LangGraph 官方内存管理文档

- **LangGraph Tutorials** (2026)
  - URL: https://langchain-ai.github.io/langgraph/tutorials/introduction/
  - 描述：LangGraph 入门教程
