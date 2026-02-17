# 实战代码：AI Agent任务调度

## 概述

**本文展示2025-2026最新的AI Agent任务调度实现,包括LangGraph异步队列、Celery集成、多Agent编排等生产级方案。**

---

## 环境准备

```bash
# 安装依赖
pip install langchain langgraph langchain-openai celery redis

# 配置环境变量
export OPENAI_API_KEY="your-key"
export REDIS_URL="redis://localhost:6379/0"
```

---

## 方案1:LangGraph异步任务队列(2025-2026)

### 基础异步任务

根据LangGraph最新文档,使用StateGraph实现异步任务队列:

```python
import asyncio
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

class TaskState(TypedDict):
    """任务状态"""
    task_id: str
    task_type: str
    input_data: dict
    result: str
    status: str

class AsyncTaskQueue:
    """LangGraph异步任务队列"""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4")
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建状态图"""
        workflow = StateGraph(TaskState)

        # 添加节点
        workflow.add_node("process", self._process_task)
        workflow.add_node("complete", self._complete_task)

        # 添加边
        workflow.set_entry_point("process")
        workflow.add_edge("process", "complete")
        workflow.add_edge("complete", END)

        return workflow.compile()

    async def _process_task(self, state: TaskState):
        """处理任务"""
        print(f"处理任务: {state['task_id']}")

        # 调用LLM
        response = await self.llm.ainvoke(state['input_data']['prompt'])

        return {
            **state,
            "result": response.content,
            "status": "processing"
        }

    async def _complete_task(self, state: TaskState):
        """完成任务"""
        print(f"完成任务: {state['task_id']}")
        return {**state, "status": "completed"}

    async def submit_task(self, task_id: str, task_type: str, input_data: dict):
        """提交任务"""
        initial_state = {
            "task_id": task_id,
            "task_type": task_type,
            "input_data": input_data,
            "result": "",
            "status": "pending"
        }

        result = await self.graph.ainvoke(initial_state)
        return result

# 使用
async def main():
    queue = AsyncTaskQueue()

    # 提交多个任务
    tasks = [
        queue.submit_task(f"task_{i}", "llm_call", {"prompt": f"问题{i}"})
        for i in range(5)
    ]

    results = await asyncio.gather(*tasks)
    for result in results:
        print(f"任务{result['task_id']}: {result['status']}")

asyncio.run(main())
```

### LangGraph后台作业(2025-2026新特性)

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

class BackgroundJobQueue:
    """LangGraph后台作业队列"""

    def __init__(self):
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建带检查点的图"""
        workflow = StateGraph(TaskState)

        workflow.add_node("process", self._process_task)
        workflow.add_node("save_checkpoint", self._save_checkpoint)

        workflow.set_entry_point("process")
        workflow.add_edge("process", "save_checkpoint")
        workflow.add_edge("save_checkpoint", END)

        # 启用检查点
        return workflow.compile(checkpointer=self.checkpointer)

    async def _process_task(self, state: TaskState):
        """处理任务"""
        # 长时间任务
        await asyncio.sleep(2)
        return {**state, "result": "处理完成"}

    async def _save_checkpoint(self, state: TaskState):
        """保存检查点"""
        print(f"保存检查点: {state['task_id']}")
        return state

    async def submit_background_job(self, task_id: str, input_data: dict):
        """提交后台作业"""
        config = {"configurable": {"thread_id": task_id}}

        initial_state = {
            "task_id": task_id,
            "task_type": "background",
            "input_data": input_data,
            "result": "",
            "status": "pending"
        }

        # 异步执行
        result = await self.graph.ainvoke(initial_state, config)
        return result

    async def get_job_status(self, task_id: str):
        """获取作业状态"""
        config = {"configurable": {"thread_id": task_id}}
        state = self.checkpointer.get(config)
        return state

# 使用
async def main():
    queue = BackgroundJobQueue()

    # 提交后台作业
    result = await queue.submit_background_job("bg_task_1", {"data": "test"})
    print(f"后台作业状态: {result['status']}")

asyncio.run(main())
```

---

## 方案2:Celery + LangGraph集成

### Celery任务队列

```python
from celery import Celery
from langchain_openai import ChatOpenAI

# 创建Celery应用
app = Celery('agent_tasks', broker='redis://localhost:6379/0')

# 配置
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

@app.task(bind=True, max_retries=3)
def process_llm_task(self, task_id: str, prompt: str):
    """处理LLM任务"""
    try:
        llm = ChatOpenAI(model="gpt-4")
        response = llm.invoke(prompt)

        return {
            "task_id": task_id,
            "result": response.content,
            "status": "completed"
        }

    except Exception as e:
        # 重试
        raise self.retry(exc=e, countdown=5)

@app.task
def process_agent_workflow(task_id: str, workflow_data: dict):
    """处理Agent工作流"""
    # 使用LangGraph处理
    from langgraph.graph import StateGraph

    # 构建工作流
    workflow = StateGraph(TaskState)
    # ... 添加节点和边

    result = workflow.invoke(workflow_data)
    return result

# 提交任务
if __name__ == "__main__":
    # 异步提交
    result = process_llm_task.delay("task_1", "你好")

    # 获取结果
    print(result.get(timeout=10))
```

### Celery Beat定时任务

```python
from celery import Celery
from celery.schedules import crontab

app = Celery('scheduled_tasks', broker='redis://localhost:6379/0')

# 配置定时任务
app.conf.beat_schedule = {
    'daily-report': {
        'task': 'tasks.generate_daily_report',
        'schedule': crontab(hour=9, minute=0),  # 每天9点
    },
    'hourly-check': {
        'task': 'tasks.check_agent_health',
        'schedule': crontab(minute=0),  # 每小时
    },
}

@app.task
def generate_daily_report():
    """生成每日报告"""
    # 调用Agent生成报告
    pass

@app.task
def check_agent_health():
    """检查Agent健康状态"""
    # 健康检查逻辑
    pass

# 启动Beat调度器
# celery -A tasks beat --loglevel=info

# 启动Worker
# celery -A tasks worker --loglevel=info
```

---

## 方案3:优先级任务调度

### 多优先级队列

```python
import asyncio
import heapq
from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass(order=True)
class PriorityTask:
    priority: int
    timestamp: float = field(compare=True)
    task_id: str = field(compare=False)
    func: Callable = field(compare=False)
    args: tuple = field(compare=False)
    kwargs: dict = field(compare=False)

class PriorityTaskScheduler:
    """优先级任务调度器"""

    def __init__(self, num_workers: int = 3):
        self.heap = []
        self.num_workers = num_workers
        self.counter = 0

    async def add_task(self, func: Callable, priority: int, *args, **kwargs):
        """添加任务"""
        task = PriorityTask(
            priority=priority,
            timestamp=asyncio.get_event_loop().time(),
            task_id=f"task_{self.counter}",
            func=func,
            args=args,
            kwargs=kwargs
        )
        heapq.heappush(self.heap, task)
        self.counter += 1

    async def worker(self, worker_id: int):
        """工作协程"""
        while True:
            if not self.heap:
                await asyncio.sleep(0.1)
                continue

            task = heapq.heappop(self.heap)
            print(f"Worker {worker_id} 执行任务 {task.task_id} (优先级{task.priority})")

            try:
                if asyncio.iscoroutinefunction(task.func):
                    await task.func(*task.args, **task.kwargs)
                else:
                    task.func(*task.args, **task.kwargs)
            except Exception as e:
                print(f"任务失败: {e}")

    async def start(self):
        """启动调度器"""
        workers = [
            asyncio.create_task(self.worker(i))
            for i in range(self.num_workers)
        ]
        await asyncio.gather(*workers)

# 使用
async def llm_task(prompt: str):
    """LLM任务"""
    await asyncio.sleep(1)
    print(f"处理: {prompt}")

async def main():
    scheduler = PriorityTaskScheduler(num_workers=3)

    # 添加不同优先级的任务
    await scheduler.add_task(llm_task, priority=5, prompt="普通任务1")
    await scheduler.add_task(llm_task, priority=1, prompt="紧急任务")
    await scheduler.add_task(llm_task, priority=5, prompt="普通任务2")

    # 启动调度器
    await asyncio.wait_for(scheduler.start(), timeout=10)

asyncio.run(main())
```

### SLO感知调度(2025-2026模式)

```python
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class SLOTask:
    """SLO感知任务"""
    task_id: str
    priority: int
    deadline: datetime
    func: Callable
    args: tuple
    kwargs: dict

class SLOAwareScheduler:
    """SLO感知调度器"""

    def __init__(self):
        self.tasks = []

    async def add_task(self, func: Callable, priority: int, slo_seconds: int, *args, **kwargs):
        """添加任务"""
        task = SLOTask(
            task_id=f"task_{len(self.tasks)}",
            priority=priority,
            deadline=datetime.now() + timedelta(seconds=slo_seconds),
            func=func,
            args=args,
            kwargs=kwargs
        )
        self.tasks.append(task)

    async def schedule(self):
        """调度任务"""
        while self.tasks:
            # 按截止时间和优先级排序
            self.tasks.sort(key=lambda t: (t.deadline, t.priority))

            task = self.tasks.pop(0)

            # 检查是否超时
            if datetime.now() > task.deadline:
                print(f"任务{task.task_id}超时")
                continue

            # 执行任务
            await task.func(*task.args, **task.kwargs)

# 使用
async def main():
    scheduler = SLOAwareScheduler()

    # 添加任务(5秒SLO)
    await scheduler.add_task(llm_task, priority=5, slo_seconds=5, prompt="任务1")
    await scheduler.add_task(llm_task, priority=1, slo_seconds=2, prompt="紧急任务")

    await scheduler.schedule()

asyncio.run(main())
```

---

## 方案4:多Agent编排

### Agent池管理

```python
import asyncio
from typing import List, Dict
from langchain_openai import ChatOpenAI

class AgentPool:
    """Agent池"""

    def __init__(self, pool_size: int = 5):
        self.pool_size = pool_size
        self.agents = [ChatOpenAI(model="gpt-4") for _ in range(pool_size)]
        self.available = asyncio.Queue()
        self.busy = set()

        # 初始化可用Agent
        for agent in self.agents:
            self.available.put_nowait(agent)

    async def acquire(self):
        """获取Agent"""
        agent = await self.available.get()
        self.busy.add(agent)
        return agent

    async def release(self, agent):
        """释放Agent"""
        self.busy.remove(agent)
        await self.available.put(agent)

    async def execute_task(self, prompt: str):
        """执行任务"""
        agent = await self.acquire()

        try:
            response = await agent.ainvoke(prompt)
            return response.content
        finally:
            await self.release(agent)

# 使用
async def main():
    pool = AgentPool(pool_size=3)

    # 并发执行多个任务
    tasks = [
        pool.execute_task(f"问题{i}")
        for i in range(10)
    ]

    results = await asyncio.gather(*tasks)
    for i, result in enumerate(results):
        print(f"任务{i}: {result[:50]}...")

asyncio.run(main())
```

### 多Agent协作

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class MultiAgentState(TypedDict):
    """多Agent状态"""
    query: str
    search_result: str
    analysis: str
    summary: str

class MultiAgentOrchestrator:
    """多Agent编排器"""

    def __init__(self):
        self.search_agent = ChatOpenAI(model="gpt-4")
        self.analysis_agent = ChatOpenAI(model="gpt-4")
        self.summary_agent = ChatOpenAI(model="gpt-4")
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建工作流"""
        workflow = StateGraph(MultiAgentState)

        # 添加Agent节点
        workflow.add_node("search", self._search)
        workflow.add_node("analyze", self._analyze)
        workflow.add_node("summarize", self._summarize)

        # 定义流程
        workflow.set_entry_point("search")
        workflow.add_edge("search", "analyze")
        workflow.add_edge("analyze", "summarize")
        workflow.add_edge("summarize", END)

        return workflow.compile()

    async def _search(self, state: MultiAgentState):
        """搜索Agent"""
        response = await self.search_agent.ainvoke(f"搜索: {state['query']}")
        return {**state, "search_result": response.content}

    async def _analyze(self, state: MultiAgentState):
        """分析Agent"""
        response = await self.analysis_agent.ainvoke(f"分析: {state['search_result']}")
        return {**state, "analysis": response.content}

    async def _summarize(self, state: MultiAgentState):
        """总结Agent"""
        response = await self.summary_agent.ainvoke(f"总结: {state['analysis']}")
        return {**state, "summary": response.content}

    async def run(self, query: str):
        """运行工作流"""
        initial_state = {
            "query": query,
            "search_result": "",
            "analysis": "",
            "summary": ""
        }

        result = await self.graph.ainvoke(initial_state)
        return result

# 使用
async def main():
    orchestrator = MultiAgentOrchestrator()
    result = await orchestrator.run("什么是量子计算?")
    print(f"最终总结: {result['summary']}")

asyncio.run(main())
```

---

## 方案5:限流与背压

### 令牌桶限流

```python
import asyncio
from datetime import datetime

class TokenBucketLimiter:
    """令牌桶限流器"""

    def __init__(self, rate_limit: int = 10):
        self.rate_limit = rate_limit
        self.tokens = rate_limit
        self.last_update = datetime.now()

    async def acquire(self):
        """获取令牌"""
        now = datetime.now()
        elapsed = (now - self.last_update).total_seconds()

        # 补充令牌
        self.tokens = min(
            self.rate_limit,
            self.tokens + elapsed * self.rate_limit
        )
        self.last_update = now

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        else:
            # 等待令牌
            wait_time = (1 - self.tokens) / self.rate_limit
            await asyncio.sleep(wait_time)
            self.tokens = 0
            return True

class RateLimitedQueue:
    """限流队列"""

    def __init__(self, rate_limit: int = 10):
        self.queue = asyncio.Queue()
        self.limiter = TokenBucketLimiter(rate_limit)

    async def add_task(self, task):
        """添加任务"""
        await self.queue.put(task)

    async def process(self):
        """处理任务"""
        while True:
            task = await self.queue.get()

            # 限流
            await self.limiter.acquire()

            # 执行任务
            await task()

            self.queue.task_done()

# 使用
async def main():
    queue = RateLimitedQueue(rate_limit=5)

    # 启动处理器
    asyncio.create_task(queue.process())

    # 添加任务
    for i in range(20):
        await queue.add_task(lambda i=i: llm_task(f"任务{i}"))

    await queue.queue.join()

asyncio.run(main())
```

---

## 完整示例:生产级Agent调度系统

```python
import asyncio
import heapq
from typing import Dict, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from langchain_openai import ChatOpenAI

@dataclass(order=True)
class AgentTask:
    priority: int
    timestamp: float = field(compare=True)
    task_id: str = field(compare=False)
    agent_type: str = field(compare=False)
    func: Callable = field(compare=False)
    args: tuple = field(compare=False)
    kwargs: dict = field(compare=False)
    max_retries: int = field(compare=False, default=3)
    retry_count: int = field(compare=False, default=0)

class ProductionAgentScheduler:
    """生产级Agent调度器"""

    def __init__(self, num_workers: int = 5, rate_limit: int = 10):
        self.heap = []
        self.num_workers = num_workers
        self.limiter = TokenBucketLimiter(rate_limit)
        self.counter = 0
        self.metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0
        }

    async def add_task(self, func: Callable, priority: int, agent_type: str, *args, **kwargs):
        """添加任务"""
        task = AgentTask(
            priority=priority,
            timestamp=asyncio.get_event_loop().time(),
            task_id=f"task_{self.counter}",
            agent_type=agent_type,
            func=func,
            args=args,
            kwargs=kwargs
        )
        heapq.heappush(self.heap, task)
        self.counter += 1
        self.metrics['total_tasks'] += 1

    async def worker(self, worker_id: int):
        """工作协程"""
        while True:
            if not self.heap:
                await asyncio.sleep(0.1)
                continue

            task = heapq.heappop(self.heap)

            # 限流
            await self.limiter.acquire()

            print(f"Worker {worker_id} 执行 {task.task_id} (优先级{task.priority})")

            try:
                if asyncio.iscoroutinefunction(task.func):
                    await task.func(*task.args, **task.kwargs)
                else:
                    task.func(*task.args, **task.kwargs)

                self.metrics['completed_tasks'] += 1

            except Exception as e:
                print(f"任务失败: {e}")

                # 重试
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    heapq.heappush(self.heap, task)
                else:
                    self.metrics['failed_tasks'] += 1

    async def start(self):
        """启动调度器"""
        workers = [
            asyncio.create_task(self.worker(i))
            for i in range(self.num_workers)
        ]
        await asyncio.gather(*workers)

    def get_metrics(self):
        """获取指标"""
        return self.metrics

# 使用
async def main():
    scheduler = ProductionAgentScheduler(num_workers=5, rate_limit=10)

    # 添加任务
    for i in range(20):
        priority = 1 if i % 5 == 0 else 5
        await scheduler.add_task(
            llm_task,
            priority=priority,
            agent_type="llm",
            prompt=f"任务{i}"
        )

    # 启动调度器
    await asyncio.wait_for(scheduler.start(), timeout=30)

    # 打印指标
    print(scheduler.get_metrics())

asyncio.run(main())
```

---

## 总结

### 方案对比

| 方案 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| LangGraph异步队列 | 单机Agent编排 | 简单,内置检查点 | 不支持分布式 |
| Celery集成 | 分布式任务 | 成熟,可靠 | 配置复杂 |
| 优先级调度 | 多优先级任务 | 灵活 | 需要防饥饿 |
| Agent池 | 高并发 | 资源复用 | 需要管理 |

### 选择指南

1. **单机简单场景** → LangGraph异步队列
2. **分布式生产环境** → Celery + Redis
3. **需要优先级** → 优先级调度器
4. **高并发** → Agent池 + 限流

### 关键洞察

- **2025-2026趋势**:LangGraph成为主流Agent编排工具
- **检查点机制**:长时间任务必须支持检查点
- **限流保护**:生产环境必须实现限流
- **监控指标**:任务成功率、延迟、吞吐量

---

**下一步**:学习[07_实战代码_03_优先级调度系统](./07_实战代码_03_优先级调度系统.md),深入了解防饥饿机制和公平调度。
