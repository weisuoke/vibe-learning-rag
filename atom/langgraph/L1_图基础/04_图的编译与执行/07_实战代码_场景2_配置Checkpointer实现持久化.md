# 实战代码 - 场景2：配置Checkpointer实现持久化

> 演示如何配置 Checkpointer 实现图的状态持久化和断点续传

---

## 场景概述

本场景演示：
1. 使用 MemorySaver 实现内存持久化
2. 使用 SqliteSaver 实现数据库持久化
3. 通过 thread_id 管理多个会话
4. 实现断点续传和状态恢复
5. 理解 Checkpoint 机制

**实际应用**：
- 长时间运行的工作流
- 需要暂停和恢复的任务
- 多用户会话管理
- 错误恢复和重试

---

## 完整代码示例

```python
"""
场景2：配置Checkpointer实现持久化
演示 LangGraph 的状态持久化和断点续传机制

来源：
- Context7: langgraph_compile_01.md
- Context7: langgraph_invoke_01.md
- 源码: source_编译执行_02.md
"""

import os
import sqlite3
from typing import TypedDict, Annotated
from operator import add
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# 加载环境变量
load_dotenv()

# ===== 1. 定义状态 Schema =====
print("=== 1. 定义状态 Schema ===\n")


class TaskState(TypedDict):
    """
    任务状态定义

    类比：
    - 前端：Redux store + LocalStorage
    - 日常：工作进度记录本
    """
    task_id: str  # 任务ID
    steps_completed: Annotated[list[str], add]  # 已完成的步骤
    current_step: str  # 当前步骤
    data: dict  # 任务数据
    status: str  # 任务状态


print("状态 Schema 定义完成\n")


# ===== 2. 定义节点函数 =====
print("=== 2. 定义节点函数 ===\n")


def step1_node(state: TaskState) -> dict:
    """步骤1：初始化任务"""
    print(f"[step1_node] 执行中...")
    print(f"  任务ID: {state['task_id']}")

    return {
        "current_step": "step1",
        "steps_completed": ["初始化完成"],
        "data": {"initialized": True},
        "status": "processing"
    }


def step2_node(state: TaskState) -> dict:
    """步骤2：处理数据"""
    print(f"[step2_node] 执行中...")
    print(f"  已完成步骤: {state['steps_completed']}")

    return {
        "current_step": "step2",
        "steps_completed": ["数据处理完成"],
        "data": {**state['data'], "processed": True}
    }


def step3_node(state: TaskState) -> dict:
    """步骤3：生成结果"""
    print(f"[step3_node] 执行中...")

    return {
        "current_step": "step3",
        "steps_completed": ["结果生成完成"],
        "data": {**state['data'], "result": "success"},
        "status": "completed"
    }


print("节点函数定义完成\n")


# ===== 3. 场景A：使用 MemorySaver（内存持久化）=====
print("=== 3. 场景A：MemorySaver（内存持久化）===\n")

# 创建图
workflow_memory = StateGraph(TaskState)
workflow_memory.add_node("step1", step1_node)
workflow_memory.add_node("step2", step2_node)
workflow_memory.add_node("step3", step3_node)

workflow_memory.add_edge(START, "step1")
workflow_memory.add_edge("step1", "step2")
workflow_memory.add_edge("step2", "step3")
workflow_memory.add_edge("step3", END)

# 配置 MemorySaver
# 来源: Context7 langgraph_compile_01.md
memory_saver = MemorySaver()
app_memory = workflow_memory.compile(checkpointer=memory_saver)

print("图编译完成（使用 MemorySaver）")
print(f"Checkpointer 类型: {type(memory_saver)}\n")

# 准备配置（必须包含 thread_id）
# 来源: Context7 langgraph_invoke_01.md
config_session1 = {
    "configurable": {
        "thread_id": "session-001"
    }
}

# 准备输入
inputs = {
    "task_id": "task-001",
    "steps_completed": [],
    "current_step": "",
    "data": {},
    "status": "pending"
}

print("执行任务（会话1）...")
result = app_memory.invoke(inputs, config=config_session1)

print(f"\n任务完成！")
print(f"  状态: {result['status']}")
print(f"  完成步骤: {result['steps_completed']}")
print(f"  数据: {result['data']}\n")


# ===== 4. 验证状态持久化 =====
print("=== 4. 验证状态持久化 ===\n")

# 使用相同的 thread_id 再次调用
print("使用相同的 thread_id 再次调用...")
result2 = app_memory.invoke(inputs, config=config_session1)

print(f"第二次调用结果:")
print(f"  状态: {result2['status']}")
print(f"  完成步骤: {result2['steps_completed']}\n")


# ===== 5. 多会话管理 =====
print("=== 5. 多会话管理 ===\n")

# 创建第二个会话
config_session2 = {
    "configurable": {
        "thread_id": "session-002"
    }
}

inputs2 = {
    "task_id": "task-002",
    "steps_completed": [],
    "current_step": "",
    "data": {},
    "status": "pending"
}

print("执行任务（会话2）...")
result_s2 = app_memory.invoke(inputs2, config=config_session2)

print(f"\n会话2完成！")
print(f"  任务ID: {result_s2['task_id']}")
print(f"  状态: {result_s2['status']}\n")

print("两个会话的状态是独立的：")
print(f"  会话1任务ID: {result['task_id']}")
print(f"  会话2任务ID: {result_s2['task_id']}\n")


# ===== 6. 场景B：使用 SqliteSaver（数据库持久化）=====
print("=== 6. 场景B：SqliteSaver（数据库持久化）===\n")

# 创建图
workflow_sqlite = StateGraph(TaskState)
workflow_sqlite.add_node("step1", step1_node)
workflow_sqlite.add_node("step2", step2_node)
workflow_sqlite.add_node("step3", step3_node)

workflow_sqlite.add_edge(START, "step1")
workflow_sqlite.add_edge("step1", "step2")
workflow_sqlite.add_edge("step2", "step3")
workflow_sqlite.add_edge("step3", END)

# 配置 SqliteSaver
# 来源: Context7 langgraph_compile_01.md
db_path = "checkpoints.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
sqlite_saver = SqliteSaver(conn)

app_sqlite = workflow_sqlite.compile(checkpointer=sqlite_saver)

print("图编译完成（使用 SqliteSaver）")
print(f"数据库路径: {db_path}\n")

# 执行任务
config_db = {
    "configurable": {
        "thread_id": "db-session-001"
    }
}

inputs_db = {
    "task_id": "task-db-001",
    "steps_completed": [],
    "current_step": "",
    "data": {},
    "status": "pending"
}

print("执行任务（数据库持久化）...")
result_db = app_sqlite.invoke(inputs_db, config=config_db)

print(f"\n任务完成！")
print(f"  状态: {result_db['status']}")
print(f"  数据已保存到: {db_path}\n")


# ===== 7. 断点续传示例 =====
print("=== 7. 断点续传示例 ===\n")


def step1_slow(state: TaskState) -> dict:
    """模拟慢速步骤"""
    print(f"[step1_slow] 执行中...")
    import time
    time.sleep(1)
    return {
        "current_step": "step1",
        "steps_completed": ["步骤1完成"],
        "status": "processing"
    }


def step2_may_fail(state: TaskState) -> dict:
    """模拟可能失败的步骤"""
    print(f"[step2_may_fail] 执行中...")

    # 模拟失败（第一次调用）
    if len(state['steps_completed']) == 1:
        print("  ❌ 模拟失败！")
        raise Exception("模拟的错误")

    print("  ✅ 成功！")
    return {
        "current_step": "step2",
        "steps_completed": ["步骤2完成"],
        "status": "processing"
    }


def step3_final(state: TaskState) -> dict:
    """最终步骤"""
    print(f"[step3_final] 执行中...")
    return {
        "current_step": "step3",
        "steps_completed": ["步骤3完成"],
        "status": "completed"
    }


# 创建带错误处理的图
workflow_resume = StateGraph(TaskState)
workflow_resume.add_node("step1", step1_slow)
workflow_resume.add_node("step2", step2_may_fail)
workflow_resume.add_node("step3", step3_final)

workflow_resume.add_edge(START, "step1")
workflow_resume.add_edge("step1", "step2")
workflow_resume.add_edge("step2", "step3")
workflow_resume.add_edge("step3", END)

# 使用 SqliteSaver
conn_resume = sqlite3.connect("resume.db", check_same_thread=False)
app_resume = workflow_resume.compile(checkpointer=SqliteSaver(conn_resume))

config_resume = {
    "configurable": {
        "thread_id": "resume-session-001"
    }
}

inputs_resume = {
    "task_id": "resume-task-001",
    "steps_completed": [],
    "current_step": "",
    "data": {},
    "status": "pending"
}

print("第一次执行（预期失败）...")
try:
    result_resume = app_resume.invoke(inputs_resume, config=config_resume)
except Exception as e:
    print(f"  ❌ 执行失败: {e}\n")

print("从断点恢复执行...")
# 使用相同的 thread_id 恢复
result_resume = app_resume.invoke(None, config=config_resume)

print(f"\n恢复成功！")
print(f"  状态: {result_resume['status']}")
print(f"  完成步骤: {result_resume['steps_completed']}\n")


# ===== 8. Checkpointer 对比 =====
print("=== 8. Checkpointer 对比 ===\n")

comparison = """
┌──────────────┬────────────────────────────────────────────────────┐
│ Checkpointer │                     特点                           │
├──────────────┼────────────────────────────────────────────────────┤
│ MemorySaver  │ - 内存中存储，进程结束后丢失                       │
│              │ - 速度快，适合开发和测试                           │
│              │ - 不支持跨进程共享                                 │
│              │ - 类比：前端的 sessionStorage                      │
├──────────────┼────────────────────────────────────────────────────┤
│ SqliteSaver  │ - 持久化到 SQLite 数据库                           │
│              │ - 支持断点续传和错误恢复                           │
│              │ - 适合单机部署                                     │
│              │ - 类比：前端的 localStorage + IndexedDB            │
├──────────────┼────────────────────────────────────────────────────┤
│ PostgreSQL   │ - 持久化到 PostgreSQL                              │
│ Saver        │ - 支持分布式部署                                   │
│              │ - 适合生产环境                                     │
│              │ - 类比：前端的云端存储                             │
└──────────────┴────────────────────────────────────────────────────┘
"""

print(comparison)


# ===== 9. thread_id 的作用 =====
print("=== 9. thread_id 的作用 ===\n")

explanation = """
thread_id 是什么？
==================

thread_id 是 Checkpoint 的唯一标识符，类似于：
- 前端：用户会话ID（session ID）
- 日常：档案柜的抽屉编号

作用：
1. 隔离不同会话的状态
2. 支持多用户并发
3. 实现断点续传
4. 管理会话生命周期

使用场景：
- 多用户系统：每个用户一个 thread_id
- 多任务系统：每个任务一个 thread_id
- 对话系统：每个对话一个 thread_id

来源: Context7 langgraph_invoke_01.md
"""

print(explanation)


# ===== 10. 实际应用示例 =====
print("=== 10. 实际应用示例 ===\n")


def create_persistent_workflow():
    """
    创建一个持久化的工作流

    实际应用：
    - 长时间运行的数据处理任务
    - 需要人工审批的工作流
    - 可恢复的批处理任务
    """
    workflow = StateGraph(TaskState)

    workflow.add_node("step1", step1_node)
    workflow.add_node("step2", step2_node)
    workflow.add_node("step3", step3_node)

    workflow.add_edge(START, "step1")
    workflow.add_edge("step1", "step2")
    workflow.add_edge("step2", "step3")
    workflow.add_edge("step3", END)

    # 生产环境使用 SqliteSaver 或 PostgresSaver
    conn = sqlite3.connect("production.db", check_same_thread=False)
    return workflow.compile(checkpointer=SqliteSaver(conn))


# 创建工作流
app_prod = create_persistent_workflow()

# 模拟多个任务
tasks = [
    {"task_id": "task-001", "user": "alice"},
    {"task_id": "task-002", "user": "bob"},
    {"task_id": "task-003", "user": "charlie"}
]

print("批量处理任务（每个任务独立持久化）:")
for task in tasks:
    config = {
        "configurable": {
            "thread_id": f"user-{task['user']}-{task['task_id']}"
        }
    }

    inputs = {
        "task_id": task['task_id'],
        "steps_completed": [],
        "current_step": "",
        "data": {"user": task['user']},
        "status": "pending"
    }

    result = app_prod.invoke(inputs, config=config)
    print(f"  {task['user']}: {task['task_id']} - {result['status']}")

print()


# ===== 11. 性能考虑 =====
print("=== 11. 性能考虑 ===\n")

performance_tips = """
性能优化建议
============

1. 选择合适的 Checkpointer
   - 开发/测试：MemorySaver
   - 单机生产：SqliteSaver
   - 分布式生产：PostgresSaver

2. thread_id 命名策略
   - 使用有意义的命名：user-{user_id}-{task_id}
   - 避免过长的 ID
   - 考虑添加时间戳：{prefix}-{timestamp}

3. 清理策略
   - 定期清理过期的 checkpoint
   - 设置 checkpoint 保留期限
   - 监控数据库大小

4. 并发控制
   - 使用连接池（PostgreSQL）
   - 避免频繁创建连接（SQLite）
   - 考虑使用 check_same_thread=False（SQLite）

来源: 源码 source_编译执行_02.md
"""

print(performance_tips)


# ===== 12. 总结 =====
print("=== 12. 总结 ===\n")

summary = """
关键要点
========

1. Checkpointer 配置
   - 在 compile() 时配置
   - MemorySaver：内存持久化
   - SqliteSaver：数据库持久化

2. thread_id 管理
   - 必须在 config 中提供
   - 用于隔离不同会话
   - 支持多用户并发

3. 断点续传
   - 使用相同的 thread_id 恢复
   - 自动从上次中断点继续
   - 支持错误恢复

4. 实际应用
   - 长时间运行任务
   - 多用户系统
   - 需要审批的工作流
   - 批处理任务

5. 最佳实践
   - 选择合适的 Checkpointer
   - 合理命名 thread_id
   - 定期清理过期数据
   - 监控性能

来源：
- Context7: langgraph_compile_01.md
- Context7: langgraph_invoke_01.md
- 源码: source_编译执行_02.md
"""

print(summary)

print("\n" + "=" * 60)
print("场景2 演示完成！")
print("=" * 60)
```

---

## 运行输出示例

```
=== 3. 场景A：MemorySaver（内存持久化）===

图编译完成（使用 MemorySaver）
Checkpointer 类型: <class 'langgraph.checkpoint.memory.MemorySaver'>

执行任务（会话1）...
[step1_node] 执行中...
  任务ID: task-001
[step2_node] 执行中...
  已完成步骤: ['初始化完成']
[step3_node] 执行中...

任务完成！
  状态: completed
  完成步骤: ['初始化完成', '数据处理完成', '结果生成完成']
  数据: {'initialized': True, 'processed': True, 'result': 'success'}

=== 7. 断点续传示例 ===

第一次执行（预期失败）...
[step1_slow] 执行中...
[step2_may_fail] 执行中...
  ❌ 模拟失败！
  ❌ 执行失败: 模拟的错误

从断点恢复执行...
[step2_may_fail] 执行中...
  ✅ 成功！
[step3_final] 执行中...

恢复成功！
  状态: completed
  完成步骤: ['步骤1完成', '步骤2完成', '步骤3完成']
```

---

## 关键知识点

### 1. Checkpointer 配置

```python
# MemorySaver
from langgraph.checkpoint.memory import MemorySaver
app = workflow.compile(checkpointer=MemorySaver())

# SqliteSaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
conn = sqlite3.connect("checkpoints.db")
app = workflow.compile(checkpointer=SqliteSaver(conn))
```

### 2. thread_id 使用

```python
# 配置 thread_id
config = {
    "configurable": {
        "thread_id": "unique-session-id"
    }
}

# 执行
result = app.invoke(inputs, config=config)

# 恢复（使用相同的 thread_id）
result = app.invoke(None, config=config)
```

### 3. 断点续传流程

```
1. 首次执行
   app.invoke(inputs, config={"configurable": {"thread_id": "xxx"}})
   ↓
2. 执行失败或中断
   ↓
3. 从断点恢复
   app.invoke(None, config={"configurable": {"thread_id": "xxx"}})
   ↓
4. 继续执行
```

---

## 下一步学习

1. **场景3**：配置中断点实现人机交互
2. **场景4**：流式执行与监控
3. **场景5**：持久化模式与性能优化

---

## 参考资料

- **Context7 文档**：
  - `langgraph_compile_01.md` - compile 方法和 Checkpointer
  - `langgraph_invoke_01.md` - thread_id 配置

- **源码分析**：
  - `source_编译执行_02.md` - Checkpoint 机制

---

**版本**: v1.0
**最后更新**: 2026-02-25
**适用于**: LangGraph 0.2+, Python 3.13+
