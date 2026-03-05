# 实战代码 场景3：多Schema架构实战

> 完整可运行的多Schema架构示例

## 场景描述

构建一个智能客服系统，使用多Schema架构实现输入输出接口隔离，隐藏内部实现细节。

## 完整代码

```python
"""
场景3：多Schema架构实战
功能：智能客服系统with多Schema架构
"""

from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
import time

# ============ 1. 多Schema定义 ============

# 整体状态（内部使用）
class OverallState(TypedDict):
    """完整的内部状态"""
    # 输入字段
    query: str
    user_id: str
    session_id: str

    # 内部处理字段
    intent: str
    entities: dict
    retrieved_docs: Annotated[list, operator.add]
    processing_steps: Annotated[list, operator.add]
    debug_info: dict

    # 输出字段
    answer: str
    confidence: float


# 输入状态（用户提供）
class InputState(TypedDict):
    """用户输入接口"""
    query: str
    user_id: str
    session_id: str


# 输出状态（返回给用户）
class OutputState(TypedDict):
    """用户输出接口"""
    answer: str
    confidence: float
    session_id: str


# ============ 2. 节点函数 ============

def intent_recognition_node(state: OverallState) -> dict:
    """意图识别节点"""
    print(f"\n[意图识别] 分析查询: {state['query']}")

    # 简单的意图识别逻辑
    query_lower = state["query"].lower()
    if "价格" in query_lower or "多少钱" in query_lower:
        intent = "price_inquiry"
    elif "退货" in query_lower or "退款" in query_lower:
        intent = "return_request"
    elif "物流" in query_lower or "快递" in query_lower:
        intent = "logistics_inquiry"
    else:
        intent = "general_inquiry"

    return {
        "intent": intent,
        "processing_steps": [f"意图识别: {intent}"],
        "debug_info": {"intent_confidence": 0.85}
    }


def entity_extraction_node(state: OverallState) -> dict:
    """实体提取节点"""
    print(f"[实体提取] 提取实体...")

    # 简单的实体提取逻辑
    entities = {
        "product": "示例产品",
        "time": time.strftime("%Y-%m-%d")
    }

    return {
        "entities": entities,
        "processing_steps": [f"实体提取: {len(entities)}个实体"]
    }


def knowledge_retrieval_node(state: OverallState) -> dict:
    """知识检索节点"""
    print(f"[知识检索] 检索相关文档...")

    # 模拟检索
    docs = [
        {"title": "产品说明", "content": "产品详细信息..."},
        {"title": "常见问题", "content": "FAQ内容..."}
    ]

    return {
        "retrieved_docs": docs,
        "processing_steps": [f"知识检索: {len(docs)}个文档"]
    }


def answer_generation_node(state: OverallState) -> dict:
    """答案生成节点"""
    print(f"[答案生成] 生成回复...")

    # 根据意图生成答案
    intent = state["intent"]
    if intent == "price_inquiry":
        answer = "该产品价格为99元"
        confidence = 0.9
    elif intent == "return_request":
        answer = "您可以在7天内申请退货"
        confidence = 0.85
    elif intent == "logistics_inquiry":
        answer = "您的订单正在配送中"
        confidence = 0.8
    else:
        answer = "感谢您的咨询，我们会尽快回复"
        confidence = 0.7

    return {
        "answer": answer,
        "confidence": confidence,
        "processing_steps": [f"答案生成: confidence={confidence}"]
    }


# ============ 3. 构建图 ============

def create_customer_service_graph():
    """创建智能客服图"""
    # 使用多Schema创建图
    graph = StateGraph(
        state_schema=OverallState,
        input_schema=InputState,
        output_schema=OutputState
    )

    # 添加节点
    graph.add_node("intent", intent_recognition_node)
    graph.add_node("entity", entity_extraction_node)
    graph.add_node("retrieval", knowledge_retrieval_node)
    graph.add_node("generation", answer_generation_node)

    # 添加边
    graph.add_edge("intent", "entity")
    graph.add_edge("entity", "retrieval")
    graph.add_edge("retrieval", "generation")
    graph.add_edge("generation", END)

    # 设置入口
    graph.set_entry_point("intent")

    return graph.compile()


# ============ 4. 运行示例 ============

def main():
    """主函数"""
    print("=" * 60)
    print("场景3：多Schema架构实战")
    print("=" * 60)

    # 创建图
    app = create_customer_service_graph()

    # 用户输入（只需提供InputState定义的字段）
    user_input = {
        "query": "这个产品多少钱？",
        "user_id": "user123",
        "session_id": "session456"
    }

    print("\n用户输入:")
    print(f"  query: {user_input['query']}")
    print(f"  user_id: {user_input['user_id']}")
    print(f"  session_id: {user_input['session_id']}")

    # 运行图
    result = app.invoke(user_input)

    print("\n" + "=" * 60)
    print("用户输出（只包含OutputState定义的字段）:")
    print("=" * 60)
    print(f"  answer: {result['answer']}")
    print(f"  confidence: {result['confidence']}")
    print(f"  session_id: {result['session_id']}")

    print("\n说明：")
    print("  - 用户看不到内部处理字段（intent, entities, retrieved_docs等）")
    print("  - 内部实现细节被隐藏")
    print("  - 接口清晰简洁")


# ============ 5. 多场景测试 ============

def multi_scenario_test():
    """多场景测试"""
    print("\n" + "=" * 60)
    print("多场景测试")
    print("=" * 60)

    app = create_customer_service_graph()

    scenarios = [
        {
            "query": "这个产品多少钱？",
            "user_id": "user1",
            "session_id": "session1"
        },
        {
            "query": "我想退货",
            "user_id": "user2",
            "session_id": "session2"
        },
        {
            "query": "我的快递到哪了？",
            "user_id": "user3",
            "session_id": "session3"
        }
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- 场景{i} ---")
        print(f"查询: {scenario['query']}")

        result = app.invoke(scenario)

        print(f"回复: {result['answer']}")
        print(f"置信度: {result['confidence']}")


# ============ 6. 内部状态访问示例 ============

def internal_state_access():
    """内部状态访问示例（调试用）"""
    print("\n" + "=" * 60)
    print("内部状态访问示例（调试用）")
    print("=" * 60)

    # 创建图（不使用多Schema）
    graph = StateGraph(OverallState)

    graph.add_node("intent", intent_recognition_node)
    graph.add_node("entity", entity_extraction_node)
    graph.add_node("retrieval", knowledge_retrieval_node)
    graph.add_node("generation", answer_generation_node)

    graph.add_edge("intent", "entity")
    graph.add_edge("entity", "retrieval")
    graph.add_edge("retrieval", "generation")
    graph.add_edge("generation", END)

    graph.set_entry_point("intent")

    app = graph.compile()

    # 运行
    result = app.invoke({
        "query": "这个产品多少钱？",
        "user_id": "user123",
        "session_id": "session456",
        "intent": "",
        "entities": {},
        "retrieved_docs": [],
        "processing_steps": [],
        "debug_info": {},
        "answer": "",
        "confidence": 0.0
    })

    print("\n完整内部状态:")
    print(f"  query: {result['query']}")
    print(f"  intent: {result['intent']}")
    print(f"  entities: {result['entities']}")
    print(f"  retrieved_docs: {len(result['retrieved_docs'])}个文档")
    print(f"  processing_steps: {result['processing_steps']}")
    print(f"  debug_info: {result['debug_info']}")
    print(f"  answer: {result['answer']}")
    print(f"  confidence: {result['confidence']}")


# ============ 7. Schema继承示例 ============

def schema_inheritance_example():
    """Schema继承示例"""
    print("\n" + "=" * 60)
    print("Schema继承示例")
    print("=" * 60)

    # 基础状态
    class BaseState(TypedDict):
        user_id: str
        timestamp: float

    # 输入继承
    class InputState(BaseState):
        query: str

    # 整体继承
    class OverallState(BaseState):
        query: str
        intent: str
        answer: str

    # 输出继承
    class OutputState(BaseState):
        answer: str

    print("\nBaseState字段:")
    print("  - user_id")
    print("  - timestamp")

    print("\nInputState字段（继承BaseState）:")
    print("  - user_id")
    print("  - timestamp")
    print("  - query")

    print("\nOverallState字段（继承BaseState）:")
    print("  - user_id")
    print("  - timestamp")
    print("  - query")
    print("  - intent")
    print("  - answer")

    print("\nOutputState字段（继承BaseState）:")
    print("  - user_id")
    print("  - timestamp")
    print("  - answer")


# ============ 8. 运行所有示例 ============

if __name__ == "__main__":
    # 基础示例
    main()

    # 多场景测试
    multi_scenario_test()

    # 内部状态访问
    internal_state_access()

    # Schema继承
    schema_inheritance_example()

    print("\n" + "=" * 60)
    print("所有示例运行完成")
    print("=" * 60)
```

## 运行结果

```
============================================================
场景3：多Schema架构实战
============================================================

用户输入:
  query: 这个产品多少钱？
  user_id: user123
  session_id: session456

[意图识别] 分析查询: 这个产品多少钱？
[实体提取] 提取实体...
[知识检索] 检索相关文档...
[答案生成] 生成回复...

============================================================
用户输出（只包含OutputState定义的字段）:
============================================================
  answer: 该产品价格为99元
  confidence: 0.9
  session_id: session456

说明：
  - 用户看不到内部处理字段（intent, entities, retrieved_docs等）
  - 内部实现细节被隐藏
  - 接口清晰简洁

============================================================
多场景测试
============================================================

--- 场景1 ---
查询: 这个产品多少钱？
回复: 该产品价格为99元
置信度: 0.9

--- 场景2 ---
查询: 我想退货
回复: 您可以在7天内申请退货
置信度: 0.85

--- 场景3 ---
查询: 我的快递到哪了？
回复: 您的订单正在配送中
置信度: 0.8
```

## 关键知识点

### 1. 多Schema定义

```python
# 整体状态
class OverallState(TypedDict):
    query: str
    intent: str  # 内部字段
    answer: str

# 输入状态
class InputState(TypedDict):
    query: str

# 输出状态
class OutputState(TypedDict):
    answer: str

# 创建图
graph = StateGraph(
    state_schema=OverallState,
    input_schema=InputState,
    output_schema=OutputState
)
```

### 2. 接口隔离

```python
# 用户只需提供InputState定义的字段
user_input = {
    "query": "问题",
    "user_id": "user123"
}

# 用户只能看到OutputState定义的字段
result = app.invoke(user_input)
# result = {"answer": "...", "confidence": 0.9}
```

### 3. 内部字段隐藏

```python
# 内部字段不暴露给用户
class OverallState(TypedDict):
    query: str
    intent: str  # 内部
    entities: dict  # 内部
    debug_info: dict  # 内部
    answer: str

class OutputState(TypedDict):
    answer: str  # 只暴露answer
```

### 4. Schema继承

```python
class BaseState(TypedDict):
    user_id: str
    timestamp: float

class InputState(BaseState):
    query: str

class OutputState(BaseState):
    answer: str
```

## 扩展练习

### 练习1：添加错误处理

```python
class OutputState(TypedDict):
    answer: str
    confidence: float
    error: str | None  # 新增错误字段

def node(state: OverallState) -> dict:
    try:
        result = process()
        return {"answer": result, "confidence": 0.9, "error": None}
    except Exception as e:
        return {"answer": "处理失败", "confidence": 0.0, "error": str(e)}
```

### 练习2：添加元数据

```python
class OutputState(TypedDict):
    answer: str
    confidence: float
    metadata: dict  # 新增元数据

def node(state: OverallState) -> dict:
    return {
        "answer": "...",
        "confidence": 0.9,
        "metadata": {
            "processing_time": 0.5,
            "model_version": "v1.0"
        }
    }
```

### 练习3：多版本API

```python
# API v1
class OutputStateV1(TypedDict):
    answer: str

# API v2
class OutputStateV2(TypedDict):
    answer: str
    confidence: float
    metadata: dict

# 根据版本选择Schema
def create_graph(api_version: str):
    if api_version == "v1":
        return StateGraph(
            state_schema=OverallState,
            output_schema=OutputStateV1
        )
    else:
        return StateGraph(
            state_schema=OverallState,
            output_schema=OutputStateV2
        )
```

## 总结

**本场景展示了**：
1. 多Schema架构的完整实现
2. 输入输出接口隔离
3. 内部实现细节隐藏
4. Schema继承模式
5. 多场景测试
6. 调试模式下的内部状态访问

**关键要点**：
- state_schema：内部完整状态
- input_schema：用户输入接口
- output_schema：用户输出接口
- 接口隔离提高安全性和可维护性

## 参考资料

- 多Schema架构：`03_核心概念_05_多Schema架构.md`
- TypedDict基础：`03_核心概念_01_TypedDict状态定义.md`
- 最小可用：`04_最小可用.md`
