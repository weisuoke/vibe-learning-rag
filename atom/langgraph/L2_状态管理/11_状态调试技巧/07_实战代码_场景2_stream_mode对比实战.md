# 实战代码 - 场景2：stream_mode 对比实战

## 场景说明

本场景使用同一个 3 节点图（分析 -> 处理 -> 输出），分别用 5 种 stream_mode 执行，对比输出差异：

1. `values` —— 每步输出完整状态快照
2. `updates` —— 只输出节点的增量更新
3. `debug` —— 输出最详细的调试事件（任务 + 检查点）
4. `custom` —— 通过 `get_stream_writer()` 发送自定义数据
5. 多模式组合 —— 同时使用多种模式，输出格式变为 `(mode, data)` 元组

**核心目标：** 同一个图、同一份输入，不同 stream_mode 看到的"世界"完全不同。

[来源: reference/source_状态调试_01.md | LangGraph 源码分析]
[来源: reference/context7_langgraph_01.md | LangGraph 官方文档]

---

## 完整代码

```python
"""
LangGraph 状态调试实战 - 场景2：stream_mode 对比实战
演示：同一个图在不同 stream_mode 下的输出差异

核心知识点：
- values 模式：完整状态快照（像拍全家福）
- updates 模式：增量更新（像看变更日志）
- debug 模式：最详细的调试事件（像看监控录像）
- custom 模式：自定义数据流（像自己写日记）
- 多模式组合：同时使用多种模式

运行环境：Python 3.13+, langgraph
安装依赖：uv add langgraph
"""

from typing import TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# 第一部分：定义状态和节点
# ============================================================

class State(TypedDict):
    """三节点流水线的状态"""
    messages: Annotated[list[str], add]  # 消息列表，使用 add reducer 累积
    step_count: int                      # 已执行的步骤数
    result: str                          # 最终输出结果


def analyze(state: State) -> dict:
    """节点1：分析阶段"""
    writer = get_stream_writer()
    writer({"node": "analyze", "status": "开始分析输入数据..."})

    query = state["messages"][-1] if state["messages"] else "无输入"
    writer({"node": "analyze", "detail": f"分析内容: '{query}'"})

    return {
        "messages": [f"[分析完成] 输入'{query}'已解析"],
        "step_count": state["step_count"] + 1,
    }


def process(state: State) -> dict:
    """节点2：处理阶段"""
    writer = get_stream_writer()
    writer({"node": "process", "status": "正在处理数据..."})

    doc_count = 3  # 模拟检索到 3 篇文档
    writer({"node": "process", "detail": f"检索到 {doc_count} 篇相关文档"})

    return {
        "messages": [f"[处理完成] 基于 {doc_count} 篇文档生成上下文"],
        "step_count": state["step_count"] + 1,
    }


def output(state: State) -> dict:
    """节点3：输出阶段"""
    writer = get_stream_writer()
    writer({"node": "output", "status": "正在生成最终回答..."})

    answer = f"综合 {state['step_count']} 步处理，生成最终回答"
    writer({"node": "output", "detail": f"回答长度: {len(answer)} 字符"})

    return {
        "messages": [f"[输出完成] {answer}"],
        "step_count": state["step_count"] + 1,
        "result": answer,
    }


# ============================================================
# 第二部分：构建图
# ============================================================

def build_graph():
    """构建 3 节点线性图：分析 -> 处理 -> 输出"""
    graph = (
        StateGraph(State)
        .add_node("analyze", analyze)
        .add_node("process", process)
        .add_node("output", output)
        .add_edge(START, "analyze")
        .add_edge("analyze", "process")
        .add_edge("process", "output")
        .add_edge("output", END)
        .compile(checkpointer=InMemorySaver())
    )
    return graph


# 初始输入（所有模式共用）
initial_input = {
    "messages": ["什么是 RAG？"],
    "step_count": 0,
    "result": "",
}


def separator(title: str) -> None:
    """打印分隔线"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ============================================================
# 第三部分：values 模式 —— 完整状态快照
# ============================================================

def demo_values_mode():
    """values 模式：每个节点执行后，输出完整状态（像拍全家福）"""
    separator("模式1: stream_mode='values' — 完整状态快照")

    graph = build_graph()
    config = {"configurable": {"thread_id": "demo-values"}}

    print("每个 chunk 是执行某个节点后的【完整状态】：\n")

    chunk_index = 0
    for chunk in graph.stream(initial_input, config, stream_mode="values"):
        print(f"--- chunk {chunk_index} ---")
        print(f"  messages:   {chunk['messages']}")
        print(f"  step_count: {chunk['step_count']}")
        print(f"  result:     '{chunk['result']}'")
        chunk_index += 1

    print(f"\n共输出 {chunk_index} 个 chunk（初始状态 + 每个节点执行后各一个）")


# ============================================================
# 第四部分：updates 模式 —— 增量更新
# ============================================================

def demo_updates_mode():
    """updates 模式：只输出节点名和返回值（像看变更日志）"""
    separator("模式2: stream_mode='updates' — 增量更新")

    graph = build_graph()
    config = {"configurable": {"thread_id": "demo-updates"}}

    print("每个 chunk 是 {节点名: 节点返回值}，只看变化部分：\n")

    for chunk in graph.stream(initial_input, config, stream_mode="updates"):
        # chunk 格式: {"node_name": {"field": value, ...}}
        for node_name, update in chunk.items():
            print(f"--- 节点: {node_name} ---")
            for key, value in update.items():
                print(f"  {key}: {value}")


# ============================================================
# 第五部分：debug 模式 —— 最详细的调试事件
# ============================================================

def demo_debug_mode():
    """debug 模式：输出 task + checkpoint 事件（像看监控录像）"""
    separator("模式3: stream_mode='debug' — 最详细调试")

    graph = build_graph()
    config = {"configurable": {"thread_id": "demo-debug"}}

    print("每个事件包含 type + step + timestamp + payload：\n")

    event_counts = {"task": 0, "task_result": 0, "checkpoint": 0}

    for event in graph.stream(initial_input, config, stream_mode="debug"):
        event_type = event["type"]
        step = event["step"]
        event_counts[event_type] = event_counts.get(event_type, 0) + 1

        if event_type == "task":
            payload = event["payload"]
            print(f"[step={step}] TASK_START: {payload['name']}")
            # 只打印输入的 messages 字段，避免输出过长
            input_msgs = payload["input"].get("messages", [])
            print(f"  输入 messages: {input_msgs}")

        elif event_type == "task_result":
            payload = event["payload"]
            print(f"[step={step}] TASK_DONE:  {payload['name']}")
            print(f"  结果: {payload['result']}")
            if payload.get("error"):
                print(f"  错误: {payload['error']}")

        elif event_type == "checkpoint":
            payload = event["payload"]
            print(f"[step={step}] CHECKPOINT")
            print(f"  next: {payload['next']}")
            print(f"  step_count: {payload['values'].get('step_count', 'N/A')}")

        print()

    print("事件统计:")
    for etype, count in event_counts.items():
        print(f"  {etype}: {count} 个")


# ============================================================
# 第六部分：custom 模式 —— 自定义数据流
# ============================================================

def demo_custom_mode():
    """custom 模式：只输出 get_stream_writer() 发送的自定义数据"""
    separator("模式4: stream_mode='custom' — 自定义数据流")

    graph = build_graph()
    config = {"configurable": {"thread_id": "demo-custom"}}

    print("只输出节点内 writer() 发送的数据，框架不自动输出任何状态：\n")

    for chunk in graph.stream(initial_input, config, stream_mode="custom"):
        # chunk 就是 writer() 的参数，格式完全由你定义
        node = chunk.get("node", "unknown")
        status = chunk.get("status", "")
        detail = chunk.get("detail", "")
        if status:
            print(f"  [{node}] {status}")
        if detail:
            print(f"  [{node}] {detail}")


# ============================================================
# 第七部分：多模式组合 —— 同时使用多种模式
# ============================================================

def demo_multi_mode():
    """多模式组合：传入列表，chunk 变成 (mode, data) 元组"""
    separator("模式5: 多模式组合 ['updates', 'custom'] — 增量 + 自定义")

    graph = build_graph()
    config = {"configurable": {"thread_id": "demo-multi"}}

    print("每个 chunk 变成 (mode_name, data) 元组：\n")

    for mode, data in graph.stream(
        initial_input, config,
        stream_mode=["updates", "custom"]
    ):
        if mode == "custom":
            # 自定义数据
            print(f"  [custom]  {data}")
        elif mode == "updates":
            # 增量更新
            for node_name, update in data.items():
                print(f"  [updates] {node_name} -> {update}")
        print()


# ============================================================
# 第八部分：运行所有演示
# ============================================================

if __name__ == "__main__":
    demo_values_mode()
    demo_updates_mode()
    demo_debug_mode()
    demo_custom_mode()
    demo_multi_mode()

    # 最后打印模式选择决策树
    separator("模式选择决策树")
    print("""    你想调试什么？
    |
    +-- "每步后状态对不对？"     -> values（最直观，初学者首选）
    +-- "哪个节点改了什么？"     -> updates（只看增量，状态大时省流量）
    +-- "任务什么时候开始/结束？" -> debug（最详细，含 task + checkpoint）
    +-- "输出自定义中间数据"     -> custom（配合 get_stream_writer()）
    +-- "同时看多种信息"         -> 传入列表，如 ["updates", "custom"]
    """)
```

---

## 预期输出

运行上述代码后，你会看到以下输出（时间戳等动态值会不同）：

```
============================================================
  模式1: stream_mode='values' — 完整状态快照
============================================================

每个 chunk 是执行某个节点后的【完整状态】：

--- chunk 0 ---
  messages:   ['什么是 RAG？']
  step_count: 0
  result:     ''
--- chunk 1 ---
  messages:   ['什么是 RAG？', "[分析完成] 输入'什么是 RAG？'已解析"]
  step_count: 1
  result:     ''
--- chunk 2 ---
  messages:   ['什么是 RAG？', "[分析完成] 输入'什么是 RAG？'已解析", '[处理完成] 基于 3 篇文档生成上下文']
  step_count: 2
  result:     ''
--- chunk 3 ---
  messages:   [... 4条消息]
  step_count: 3
  result:     '综合 2 步处理，生成最终回答'

============================================================
  模式2: stream_mode='updates' — 增量更新
============================================================

每个 chunk 是 {节点名: 节点返回值}，只看变化部分：

--- 节点: analyze ---
  messages: ["[分析完成] 输入'什么是 RAG？'已解析"]
  step_count: 1
--- 节点: process ---
  messages: ['[处理完成] 基于 3 篇文档生成上下文']
  step_count: 2
--- 节点: output ---
  messages: ['[输出完成] 综合 2 步处理，生成最终回答']
  step_count: 3
  result: 综合 2 步处理，生成最终回答

============================================================
  模式3: stream_mode='debug' — 最详细调试
============================================================

每个事件包含 type + step + timestamp + payload：

[step=-1] CHECKPOINT
  next: ['analyze']
  step_count: 0
[step=0] TASK_START: analyze
  输入 messages: ['什么是 RAG？']
[step=0] TASK_DONE:  analyze
  结果: {'messages': ["[分析完成] 输入'什么是 RAG？'已解析"], 'step_count': 1}
[step=0] CHECKPOINT
  next: ['process']
  step_count: 1
[step=1] TASK_START: process
  输入 messages: ['什么是 RAG？', "[分析完成] 输入'什么是 RAG？'已解析"]
[step=1] TASK_DONE:  process
  结果: {'messages': ['[处理完成] 基于 3 篇文档生成上下文'], 'step_count': 2}
[step=1] CHECKPOINT
  next: ['output']
  step_count: 2
[step=2] TASK_START: output
  输入 messages: [... 3条消息]
[step=2] TASK_DONE:  output
  结果: {'messages': ['[输出完成] 综合 2 步处理，生成最终回答'], 'step_count': 3, 'result': '...'}
[step=2] CHECKPOINT
  next: []
  step_count: 3

事件统计:
  task: 3 个
  task_result: 3 个
  checkpoint: 4 个

============================================================
  模式4: stream_mode='custom' — 自定义数据流
============================================================

只输出节点内 writer() 发送的数据，框架不自动输出任何状态：

  [analyze] 开始分析输入数据...
  [analyze] 分析内容: '什么是 RAG？'
  [process] 正在处理数据...
  [process] 检索到 3 篇相关文档
  [output] 正在生成最终回答...
  [output] 回答长度: 18 字符

============================================================
  模式5: 多模式组合 ['updates', 'custom'] — 增量 + 自定义
============================================================

每个 chunk 变成 (mode_name, data) 元组：

  [custom]  {'node': 'analyze', 'status': '开始分析输入数据...'}
  [custom]  {'node': 'analyze', 'detail': "分析内容: '什么是 RAG？'"}
  [updates] analyze -> {'messages': ["[分析完成] 输入'什么是 RAG？'已解析"], 'step_count': 1}
  [custom]  {'node': 'process', 'status': '正在处理数据...'}
  [custom]  {'node': 'process', 'detail': '检索到 3 篇相关文档'}
  [updates] process -> {'messages': ['[处理完成] 基于 3 篇文档生成上下文'], 'step_count': 2}
  [custom]  {'node': 'output', 'status': '正在生成最终回答...'}
  [custom]  {'node': 'output', 'detail': '回答长度: 18 字符'}
  [updates] output -> {'messages': ['[输出完成] 综合 2 步处理，生成最终回答'], 'step_count': 3, 'result': '...'}
```

---

## 输出对比表

| 对比维度 | values | updates | debug | custom | 多模式组合 |
|---------|--------|---------|-------|--------|-----------|
| 每个 chunk 是什么 | 完整状态字典 | `{节点名: 返回值}` | `{type, step, timestamp, payload}` | writer() 的参数 | `(mode, data)` 元组 |
| 输出次数（3节点图） | 4 次（初始+3节点） | 3 次（每节点1次） | 10 次（4检查点+3任务开始+3任务完成） | 取决于 writer() 调用次数 | 各模式之和 |
| 包含节点名 | 不包含 | 包含（作为 key） | 包含（在 payload 中） | 不包含（需自己写入） | 取决于子模式 |
| 包含时间戳 | 不包含 | 不包含 | 包含 | 不包含 | 取决于子模式 |
| 状态大时的开销 | 大（每次输出完整状态） | 小（只输出变化部分） | 大（包含完整检查点） | 可控（你决定写什么） | 取决于子模式 |
| 初学者友好度 | 最高 | 高 | 中（信息量大） | 中（需要写代码） | 中 |

---

## 关键差异图解

```
各模式输出的 chunk 数量（3 节点图）：

values:   [初始] [analyze后] [process后] [output后]        = 4 个
updates:  [analyze更新] [process更新] [output更新]          = 3 个
debug:    [CP][T][TR][CP][T][TR][CP][T][TR][CP]            = 10 个
custom:   [w1][w2][w3][w4][w5][w6]                         = 6 个（取决于代码）

CP=checkpoint  T=task  TR=task_result  w=writer()调用
```

---

## 模式选择决策树

```
                    你在调试什么问题？
                          |
          +---------------+---------------+
          |               |               |
     "状态不对"      "不知道哪里出错"    "需要进度反馈"
          |               |               |
     values 或        debug            custom
     updates              |
          |          （看 task_result
    +-----+-----+    中的 error 字段）
    |           |
"想看全貌"  "想看变化"
    |           |
  values     updates
```

### 速查表

| 场景 | 推荐模式 | 理由 |
|------|---------|------|
| 初学者第一次调试 | `values` | 最直观，看到完整状态 |
| 定位"谁改了什么" | `updates` | 只看增量，信息精准 |
| 排查复杂 bug | `debug` | 信息最全，包含时序 |
| 输出处理进度 | `custom` | 完全自定义 |
| 生产环境监控 | `["updates", "custom"]` | 增量 + 自定义日志 |
| 验证检查点持久化 | `debug` 或 `checkpoints` | 能看到每个检查点的详情 |

---

## 常见问题

### Q1: values 和 updates 该选哪个？

状态小（< 10 个字段）用 `values`，状态大（包含长对话历史、大文档列表）用 `updates`。

### Q2: debug 模式的事件顺序是什么？

每个节点产生 3 个事件：`task` -> `task_result` -> `checkpoint`。加上初始 checkpoint（step=-1），3 节点图总共 10 个事件。

### Q3: custom 模式不写 writer() 会怎样？

什么都不输出。框架不会自动输出任何东西，只有 `get_stream_writer()` 主动发送的数据才会出现。

### Q4: 多模式组合时怎么区分数据来源？

每个 chunk 变成 `(mode_name, data)` 元组：

```python
for mode, data in graph.stream(input, config, stream_mode=["values", "custom"]):
    if mode == "values":
        pass  # 完整状态
    elif mode == "custom":
        pass  # 自定义数据
```

---

## 下一步学习

- **07_实战代码_场景1_基础状态打印与检查.md** — 基础的 print/logging + get_state 调试方法
- **07_实战代码_场景3_时间旅行调试.md** — 使用 checkpoint_id 回溯与分叉执行
- **03_核心概念_1_stream_mode调试模式.md** — stream_mode 的完整理论讲解和源码分析
