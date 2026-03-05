# 实战代码_场景5_生产级HITL系统

> 结合 FastAPI 的完整 Human-in-the-loop 服务：API 端点 + 中断管理 + 超时处理

---

## 导航

- 上一篇：[07_实战代码_场景4_用户反馈循环](./07_实战代码_场景4_用户反馈循环.md)
- 下一篇：[08_面试必问](./08_面试必问.md)

---

## 引用来源

本文档基于以下资料：

1. **参考资料**：
   - `reference/source_hitl_01.md` - interrupt/Command/Interrupt 源码分析
   - `reference/source_hitl_02.md` - should_interrupt/PregelLoop 源码分析
   - `reference/context7_langgraph_01.md` - LangGraph HITL 官方文档
   - `reference/search_hitl_01.md` - 生产级 HITL 项目模板

2. **关键来源**：
   - [LangGraph 官方文档 - Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
   - [Production-ready LangGraph interrupt template](https://github.com/KirtiJha/langgraph-interrupt-workflow-template)

---

## 场景说明

前面的场景都是"脚本式"运行——在一个 Python 文件里依次调用 `invoke` 和 `Command(resume=...)`。

但在真实项目中，HITL 系统通常是这样的：

```
前端/客户端                          后端 API
    │                                  │
    ├─ POST /tasks ──────────────────→ 创建任务，运行图
    │                                  图遇到 interrupt 暂停
    │                                  │
    ├─ GET /tasks/{id} ──────────────→ 查询任务状态（是否有中断）
    │  ← 返回中断信息                   │
    │                                  │
    ├─ POST /tasks/{id}/resume ──────→ 提交人类决策，恢复执行
    │  ← 返回最终结果                   │
```

本场景构建一个完整的 FastAPI + LangGraph HITL 服务。

---

## 架构概览

```
┌─────────────────────────────────────────────────┐
│                  FastAPI 应用                      │
│                                                   │
│  POST /tasks          → 创建任务，启动图执行        │
│  GET  /tasks/{id}     → 查询任务状态和中断信息      │
│  POST /tasks/{id}/resume → 恢复中断，提交人类决策   │
│  GET  /tasks          → 列出所有任务               │
│  DELETE /tasks/{id}   → 取消超时任务               │
│                                                   │
│  ┌─────────────────────────────────────────────┐ │
│  │           InterruptManager                   │ │
│  │  - 跟踪所有 pending interrupts               │ │
│  │  - 超时自动标记过期                           │ │
│  │  - 任务状态管理                               │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  ┌─────────────────────────────────────────────┐ │
│  │           LangGraph 图                       │ │
│  │  - MemorySaver（开发环境）                    │ │
│  │  - 生产环境替换为 PostgresSaver               │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

---

## 完整代码

> 以下是一个完整的 FastAPI 应用，可以直接保存为 `hitl_server.py` 运行。

```python
"""
生产级 HITL 系统：FastAPI + LangGraph
运行方式：uvicorn hitl_server:app --reload --port 8000

注意：
- 开发环境使用 MemorySaver（内存存储，重启丢失）
- 生产环境应替换为 PostgresSaver：
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    checkpointer = AsyncPostgresSaver.from_conn_string("postgresql://...")
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import TypedDict, Annotated, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
import operator


# ============================================================
# 第 1 部分：数据模型（Pydantic）
# ============================================================

class TaskStatus(str, Enum):
    """任务状态枚举"""
    RUNNING = "running"           # 图正在执行
    WAITING_INPUT = "waiting_input"  # 等待人类输入（有中断）
    COMPLETED = "completed"       # 已完成
    FAILED = "failed"             # 执行失败
    TIMEOUT = "timeout"           # 中断超时
    CANCELLED = "cancelled"       # 已取消


class CreateTaskRequest(BaseModel):
    """创建任务请求"""
    topic: str = Field(..., description="要处理的主题")
    timeout_minutes: int = Field(
        default=30, description="中断超时时间（分钟），超时后自动标记"
    )


class ResumeRequest(BaseModel):
    """恢复中断请求"""
    value: Any = Field(..., description="人类决策值（传递给 Command(resume=...)）")


class InterruptInfo(BaseModel):
    """中断信息"""
    interrupt_id: str
    value: Any
    node_name: str
    created_at: str


class TaskResponse(BaseModel):
    """任务响应"""
    task_id: str
    status: TaskStatus
    state: Optional[dict] = None
    interrupts: list[InterruptInfo] = []
    created_at: str
    updated_at: str
    timeout_at: Optional[str] = None


class TaskListResponse(BaseModel):
    """任务列表响应"""
    tasks: list[TaskResponse]
    total: int


# ============================================================
# 第 2 部分：中断管理器（InterruptManager）
# ============================================================

class InterruptManager:
    """
    管理所有任务的中断状态。

    职责：
    1. 跟踪每个任务的状态和中断信息
    2. 处理超时逻辑
    3. 提供任务查询接口
    """

    def __init__(self):
        # task_id → 任务元数据
        self._tasks: dict[str, dict] = {}

    def create_task(self, task_id: str, timeout_minutes: int = 30) -> dict:
        """注册新任务"""
        now = datetime.now()
        task_meta = {
            "task_id": task_id,
            "status": TaskStatus.RUNNING,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "timeout_minutes": timeout_minutes,
            "timeout_at": None,
            "interrupts": [],
        }
        self._tasks[task_id] = task_meta
        return task_meta

    def mark_interrupted(
        self, task_id: str, interrupts: list[dict]
    ) -> None:
        """标记任务为等待输入状态"""
        if task_id not in self._tasks:
            return
        now = datetime.now()
        meta = self._tasks[task_id]
        meta["status"] = TaskStatus.WAITING_INPUT
        meta["updated_at"] = now.isoformat()
        meta["interrupts"] = interrupts
        meta["timeout_at"] = (
            now + timedelta(minutes=meta["timeout_minutes"])
        ).isoformat()

    def mark_completed(self, task_id: str) -> None:
        """标记任务为已完成"""
        if task_id not in self._tasks:
            return
        meta = self._tasks[task_id]
        meta["status"] = TaskStatus.COMPLETED
        meta["updated_at"] = datetime.now().isoformat()
        meta["timeout_at"] = None
        meta["interrupts"] = []

    def mark_failed(self, task_id: str, error: str) -> None:
        """标记任务为失败"""
        if task_id not in self._tasks:
            return
        meta = self._tasks[task_id]
        meta["status"] = TaskStatus.FAILED
        meta["updated_at"] = datetime.now().isoformat()
        meta["error"] = error

    def mark_cancelled(self, task_id: str) -> None:
        """取消任务"""
        if task_id not in self._tasks:
            return
        meta = self._tasks[task_id]
        meta["status"] = TaskStatus.CANCELLED
        meta["updated_at"] = datetime.now().isoformat()
        meta["timeout_at"] = None

    def is_timeout(self, task_id: str) -> bool:
        """检查任务是否超时"""
        meta = self._tasks.get(task_id)
        if not meta or not meta.get("timeout_at"):
            return False
        timeout_at = datetime.fromisoformat(meta["timeout_at"])
        return datetime.now() > timeout_at

    def get_task(self, task_id: str) -> Optional[dict]:
        """获取任务元数据"""
        meta = self._tasks.get(task_id)
        if meta and self.is_timeout(task_id):
            meta["status"] = TaskStatus.TIMEOUT
            meta["updated_at"] = datetime.now().isoformat()
        return meta

    def list_tasks(
        self, status: Optional[TaskStatus] = None
    ) -> list[dict]:
        """列出任务（可按状态过滤）"""
        # 先检查超时
        for task_id in list(self._tasks.keys()):
            self.get_task(task_id)

        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t["status"] == status]
        return tasks


# ============================================================
# 第 3 部分：LangGraph 图定义
# ============================================================

class DocumentState(TypedDict):
    """文档处理状态"""
    topic: str
    draft: str
    feedback: str
    status: str
    revision_count: int
    history: Annotated[list[str], operator.add]


def generate_document(state: DocumentState):
    """生成文档（实际项目中调用 LLM）"""
    revision = state.get("revision_count", 0)

    if revision == 0:
        draft = f"关于「{state['topic']}」的文档初稿。\n这里是详细内容..."
    else:
        draft = (
            f"关于「{state['topic']}」的第 {revision} 次修改版。\n"
            f"根据反馈「{state.get('feedback', '')}」进行了改进。"
        )

    return {
        "draft": draft,
        "revision_count": revision + 1,
        "status": "draft_ready",
        "history": [f"v{revision + 1}: {draft[:50]}..."],
    }


def review_document(state: DocumentState):
    """审阅文档：interrupt 等待人类反馈"""
    feedback = interrupt({
        "type": "document_review",
        "draft": state["draft"],
        "revision_count": state["revision_count"],
        "instruction": "请审阅文档。回复 'ok' 表示通过，或输入修改意见。",
    })
    return {"feedback": feedback}


def check_feedback(state: DocumentState) -> str:
    """条件路由：继续修改或完成"""
    if state.get("feedback", "").lower().strip() == "ok":
        return "finalize"
    return "generate_document"


def finalize_document(state: DocumentState):
    """完成文档"""
    return {"status": "completed"}


# 构建图
builder = StateGraph(DocumentState)
builder.add_node("generate_document", generate_document)
builder.add_node("review_document", review_document)
builder.add_node("finalize", finalize_document)

builder.add_edge(START, "generate_document")
builder.add_edge("generate_document", "review_document")
builder.add_conditional_edges("review_document", check_feedback)
builder.add_edge("finalize", END)

# 使用 MemorySaver（开发环境）
# 生产环境替换：
#   from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
#   checkpointer = AsyncPostgresSaver.from_conn_string(DATABASE_URL)
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# ============================================================
# 第 4 部分：FastAPI 应用
# ============================================================

app = FastAPI(
    title="HITL Document Service",
    description="基于 LangGraph 的人机循环文档处理服务",
    version="1.0.0",
)

# 全局中断管理器
manager = InterruptManager()


def _extract_interrupts(task_id: str, config: dict) -> list[dict]:
    """从图状态中提取中断信息"""
    snapshot = graph.get_state(config)
    interrupts = []
    for task in snapshot.tasks:
        for intr in task.interrupts:
            interrupts.append({
                "interrupt_id": str(intr.id) if hasattr(intr, "id") else "",
                "value": intr.value,
                "node_name": task.name,
                "created_at": datetime.now().isoformat(),
            })
    return interrupts


def _build_task_response(task_id: str) -> TaskResponse:
    """构建任务响应对象"""
    meta = manager.get_task(task_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

    return TaskResponse(
        task_id=meta["task_id"],
        status=meta["status"],
        interrupts=[InterruptInfo(**i) for i in meta.get("interrupts", [])],
        created_at=meta["created_at"],
        updated_at=meta["updated_at"],
        timeout_at=meta.get("timeout_at"),
    )


# ---------- API 端点 ----------


@app.post("/tasks", response_model=TaskResponse, status_code=201)
async def create_task(request: CreateTaskRequest):
    """
    创建新任务并启动图执行。

    图会运行到第一个 interrupt 处暂停，返回任务 ID 和中断信息。
    """
    task_id = f"task-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": task_id}}

    # 注册任务
    manager.create_task(task_id, request.timeout_minutes)

    try:
        # 启动图执行（会在 interrupt 处暂停）
        result = await asyncio.to_thread(
            graph.invoke,
            {
                "topic": request.topic,
                "draft": "",
                "feedback": "",
                "status": "pending",
                "revision_count": 0,
                "history": [],
            },
            config,
        )

        # 检查是否有中断
        snapshot = graph.get_state(config)
        if snapshot.next:
            # 图暂停了，提取中断信息
            interrupts = _extract_interrupts(task_id, config)
            manager.mark_interrupted(task_id, interrupts)
        else:
            # 图直接完成了（没有中断）
            manager.mark_completed(task_id)

    except Exception as e:
        manager.mark_failed(task_id, str(e))
        raise HTTPException(status_code=500, detail=f"任务执行失败: {e}")

    return _build_task_response(task_id)


@app.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """
    查询任务状态。

    返回当前状态、中断信息（如果有）、超时时间等。
    """
    meta = manager.get_task(task_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

    # 如果任务在等待输入，刷新中断信息
    if meta["status"] == TaskStatus.WAITING_INPUT:
        config = {"configurable": {"thread_id": task_id}}
        snapshot = graph.get_state(config)

        # 附加当前状态值
        response = _build_task_response(task_id)
        response.state = dict(snapshot.values) if snapshot.values else None
        return response

    return _build_task_response(task_id)


@app.post("/tasks/{task_id}/resume", response_model=TaskResponse)
async def resume_task(task_id: str, request: ResumeRequest):
    """
    恢复中断的任务。

    提交人类决策值，图从中断处继续执行。
    如果图再次遇到 interrupt，会再次暂停并返回新的中断信息。
    """
    meta = manager.get_task(task_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

    # 检查任务状态
    if meta["status"] == TaskStatus.TIMEOUT:
        raise HTTPException(status_code=410, detail="任务已超时，请创建新任务")

    if meta["status"] == TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="任务已完成，无需恢复")

    if meta["status"] == TaskStatus.CANCELLED:
        raise HTTPException(status_code=400, detail="任务已取消")

    if meta["status"] != TaskStatus.WAITING_INPUT:
        raise HTTPException(
            status_code=400,
            detail=f"任务状态为 {meta['status']}，无法恢复",
        )

    config = {"configurable": {"thread_id": task_id}}

    try:
        # 恢复执行
        result = await asyncio.to_thread(
            graph.invoke,
            Command(resume=request.value),
            config,
        )

        # 检查是否还有中断
        snapshot = graph.get_state(config)
        if snapshot.next:
            interrupts = _extract_interrupts(task_id, config)
            manager.mark_interrupted(task_id, interrupts)
        else:
            manager.mark_completed(task_id)

    except Exception as e:
        manager.mark_failed(task_id, str(e))
        raise HTTPException(status_code=500, detail=f"恢复执行失败: {e}")

    response = _build_task_response(task_id)
    # 附加最终状态
    if meta["status"] == TaskStatus.COMPLETED:
        snapshot = graph.get_state(config)
        response.state = dict(snapshot.values) if snapshot.values else None
    return response


@app.get("/tasks", response_model=TaskListResponse)
async def list_tasks(status: Optional[TaskStatus] = None):
    """
    列出所有任务（可按状态过滤）。

    查询参数：
    - status: 按状态过滤（running, waiting_input, completed, failed, timeout）
    """
    tasks = manager.list_tasks(status)
    return TaskListResponse(
        tasks=[
            TaskResponse(
                task_id=t["task_id"],
                status=t["status"],
                interrupts=[InterruptInfo(**i) for i in t.get("interrupts", [])],
                created_at=t["created_at"],
                updated_at=t["updated_at"],
                timeout_at=t.get("timeout_at"),
            )
            for t in tasks
        ],
        total=len(tasks),
    )


@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """取消任务"""
    meta = manager.get_task(task_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

    manager.mark_cancelled(task_id)
    return {"message": f"任务 {task_id} 已取消", "task_id": task_id}


# ============================================================
# 第 5 部分：健康检查和启动
# ============================================================

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "checkpointer": "MemorySaver",
        "note": "生产环境请替换为 PostgresSaver",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## API 使用示例

### 用 curl 测试完整流程

```bash
# 1. 创建任务
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"topic": "LangGraph 入门", "timeout_minutes": 30}'

# 响应示例：
# {
#   "task_id": "task-a1b2c3d4e5f6",
#   "status": "waiting_input",
#   "interrupts": [{
#     "interrupt_id": "...",
#     "value": {
#       "type": "document_review",
#       "draft": "关于「LangGraph 入门」的文档初稿...",
#       "revision_count": 1,
#       "instruction": "请审阅文档..."
#     },
#     "node_name": "review_document",
#     "created_at": "2026-02-28T10:00:00"
#   }],
#   "created_at": "2026-02-28T10:00:00",
#   "updated_at": "2026-02-28T10:00:00",
#   "timeout_at": "2026-02-28T10:30:00"
# }


# 2. 查询任务状态
curl http://localhost:8000/tasks/task-a1b2c3d4e5f6

# 3. 提交反馈（不满意，要求修改）
curl -X POST http://localhost:8000/tasks/task-a1b2c3d4e5f6/resume \
  -H "Content-Type: application/json" \
  -d '{"value": "请增加代码示例"}'

# 4. 提交反馈（满意）
curl -X POST http://localhost:8000/tasks/task-a1b2c3d4e5f6/resume \
  -H "Content-Type: application/json" \
  -d '{"value": "ok"}'

# 响应：status 变为 "completed"


# 5. 列出所有等待输入的任务
curl "http://localhost:8000/tasks?status=waiting_input"

# 6. 取消超时任务
curl -X DELETE http://localhost:8000/tasks/task-a1b2c3d4e5f6
```

### 用 Python requests 测试

```python
"""
客户端测试脚本：模拟完整的 HITL 交互流程
"""

import requests
import time

BASE_URL = "http://localhost:8000"


def test_full_flow():
    """测试完整的创建→反馈→完成流程"""

    # 1. 创建任务
    print("1. 创建任务...")
    resp = requests.post(f"{BASE_URL}/tasks", json={
        "topic": "Python 最佳实践",
        "timeout_minutes": 5,
    })
    task = resp.json()
    task_id = task["task_id"]
    print(f"   任务 ID: {task_id}")
    print(f"   状态: {task['status']}")

    # 2. 查看中断信息
    if task["interrupts"]:
        intr = task["interrupts"][0]
        print(f"   中断节点: {intr['node_name']}")
        print(f"   草稿预览: {intr['value']['draft'][:50]}...")

    # 3. 第一次反馈：要求修改
    print("\n2. 提交反馈：要求增加示例...")
    resp = requests.post(f"{BASE_URL}/tasks/{task_id}/resume", json={
        "value": "请增加实际代码示例",
    })
    task = resp.json()
    print(f"   状态: {task['status']}")

    # 4. 第二次反馈：满意
    print("\n3. 提交反馈：满意...")
    resp = requests.post(f"{BASE_URL}/tasks/{task_id}/resume", json={
        "value": "ok",
    })
    task = resp.json()
    print(f"   状态: {task['status']}")

    # 5. 查看最终状态
    print("\n4. 查看最终状态...")
    resp = requests.get(f"{BASE_URL}/tasks/{task_id}")
    task = resp.json()
    print(f"   最终状态: {task['status']}")

    print("\n完成！")


def test_timeout():
    """测试超时场景"""

    # 创建一个 1 分钟超时的任务
    print("创建短超时任务（1 分钟）...")
    resp = requests.post(f"{BASE_URL}/tasks", json={
        "topic": "超时测试",
        "timeout_minutes": 1,
    })
    task = resp.json()
    task_id = task["task_id"]
    print(f"任务 ID: {task_id}")
    print(f"超时时间: {task['timeout_at']}")

    # 等待超时后尝试恢复
    print("等待超时...")
    time.sleep(65)

    resp = requests.post(f"{BASE_URL}/tasks/{task_id}/resume", json={
        "value": "ok",
    })
    print(f"恢复结果: {resp.status_code} - {resp.json()}")
    # 预期：410 Gone - 任务已超时


def test_list_tasks():
    """测试任务列表"""
    resp = requests.get(f"{BASE_URL}/tasks")
    data = resp.json()
    print(f"总任务数: {data['total']}")
    for t in data["tasks"]:
        print(f"  {t['task_id']}: {t['status']}")


if __name__ == "__main__":
    test_full_flow()
```

---

## 关键设计解析

### 1. 为什么用 `asyncio.to_thread`？

```python
result = await asyncio.to_thread(graph.invoke, input, config)
```

LangGraph 的 `invoke` 是同步方法。在 FastAPI 的异步端点中直接调用会阻塞事件循环。`asyncio.to_thread` 把同步调用放到线程池中执行，不阻塞其他请求。

如果 LangGraph 图本身支持异步（`ainvoke`），可以直接用：

```python
# 如果图支持异步
result = await graph.ainvoke(input, config)
```

### 2. 中断管理器的职责

InterruptManager 是一个轻量级的状态跟踪层，它不替代 LangGraph 的 checkpointer，而是补充：

| 职责 | Checkpointer | InterruptManager |
|------|-------------|-----------------|
| 保存图状态 | 是 | 否 |
| 跟踪中断信息 | 是（通过 get_state） | 是（缓存副本） |
| 超时管理 | 否 | 是 |
| 任务列表 | 否 | 是 |
| 状态枚举 | 否 | 是 |

### 3. 超时处理策略

```python
def is_timeout(self, task_id: str) -> bool:
    timeout_at = datetime.fromisoformat(meta["timeout_at"])
    return datetime.now() > timeout_at
```

超时是"惰性检查"——不用后台定时器，而是在每次查询时检查。这样更简单，适合中小规模场景。

大规模场景可以加一个后台任务定期清理：

```python
@app.on_event("startup")
async def start_timeout_checker():
    async def check_timeouts():
        while True:
            for task_id in list(manager._tasks.keys()):
                if manager.is_timeout(task_id):
                    manager.get_task(task_id)  # 触发状态更新
            await asyncio.sleep(60)  # 每分钟检查一次
    asyncio.create_task(check_timeouts())
```

### 4. 生产环境替换清单

从开发环境迁移到生产环境，需要替换以下组件：

```python
# ❌ 开发环境
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

# ✅ 生产环境
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
checkpointer = AsyncPostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/langgraph"
)
```

完整替换清单：

| 组件 | 开发环境 | 生产环境 |
|------|---------|---------|
| Checkpointer | `MemorySaver` | `PostgresSaver` / `AsyncPostgresSaver` |
| InterruptManager | 内存字典 | Redis / PostgreSQL |
| 日志 | print | structlog / logging |
| 认证 | 无 | JWT / API Key |
| 部署 | uvicorn | gunicorn + uvicorn workers |

---

## 常见问题

### Q1：MemorySaver 重启后数据会丢失吗？

会。`MemorySaver` 把所有状态存在内存中，进程重启后全部丢失。这就是为什么生产环境必须用 `PostgresSaver`——数据持久化到数据库。

### Q2：多个 worker 进程能共享 MemorySaver 吗？

不能。每个进程有自己的 `MemorySaver` 实例，互相看不到对方的状态。生产环境用 `PostgresSaver` 就没这个问题，因为所有 worker 共享同一个数据库。

### Q3：如何处理并发恢复同一个任务？

LangGraph 的 checkpointer 有内置的并发控制。如果两个请求同时尝试恢复同一个 thread_id，第二个会失败。在 API 层可以加锁：

```python
import asyncio

_task_locks: dict[str, asyncio.Lock] = {}

async def get_task_lock(task_id: str) -> asyncio.Lock:
    if task_id not in _task_locks:
        _task_locks[task_id] = asyncio.Lock()
    return _task_locks[task_id]

@app.post("/tasks/{task_id}/resume")
async def resume_task(task_id: str, request: ResumeRequest):
    lock = await get_task_lock(task_id)
    async with lock:
        # ... 恢复逻辑
        pass
```

### Q4：如何添加认证？

```python
from fastapi import Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    token = credentials.credentials
    # 验证 token 逻辑
    if not is_valid_token(token):
        raise HTTPException(status_code=401, detail="无效的认证令牌")
    return token

# 在端点中使用
@app.post("/tasks", dependencies=[Depends(verify_token)])
async def create_task(request: CreateTaskRequest):
    ...
```

---

## 总结

| 组件 | 作用 | 关键代码 |
|------|------|---------|
| FastAPI 端点 | HTTP API 接口 | `@app.post("/tasks")` |
| InterruptManager | 任务状态跟踪 + 超时管理 | `manager.mark_interrupted()` |
| LangGraph 图 | 业务逻辑 + 中断机制 | `interrupt()` + `Command(resume=)` |
| Checkpointer | 状态持久化 | `MemorySaver()` → `PostgresSaver()` |

核心模式：**API 创建任务 → 图执行到 interrupt 暂停 → API 返回中断信息 → 客户端提交决策 → API 恢复图执行**。

这个架构可以直接用于生产环境，只需把 MemorySaver 替换为 PostgresSaver，并添加认证和日志。

---

**版本**: v1.0
**最后更新**: 2026-02-28
**维护者**: Claude Code
