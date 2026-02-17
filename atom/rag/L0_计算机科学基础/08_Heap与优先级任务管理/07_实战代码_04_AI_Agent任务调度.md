# 实战代码04: AI Agent任务调度

## 核心目标

**实现一个完整的AI Agent任务调度系统,支持多级优先级、Aging机制、任务依赖和监控指标。**

---

## 1. 完整实现

```python
import heapq
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Optional
import time

class AgentPriority(IntEnum):
    """Agent任务优先级"""
    USER_QUERY = 0      # 用户查询
    TOOL_CALL = 1       # 工具调用
    MEMORY_UPDATE = 2   # 记忆更新
    LOGGING = 3         # 日志记录

@dataclass(order=True)
class AgentTask:
    """Agent任务"""
    priority: int
    timestamp: float = field(compare=False)
    counter: int = field(compare=False)
    task_id: str = field(compare=False)
    action: Callable = field(compare=False)
    dependencies: List[str] = field(default_factory=list, compare=False)

class AgentTaskScheduler:
    """
    AI Agent任务调度器

    特性:
    - 多级优先级
    - FIFO保证(相同优先级)
    - Aging机制(防止饥饿)
    - 任务依赖
    - 监控指标
    """

    def __init__(self, age_interval=10):
        self.task_queue = []
        self.counter = 0
        self.completed_tasks = set()
        self.age_interval = age_interval
        self.metrics = {
            'submitted': 0,
            'executed': 0,
            'wait_times': [],
            'priority_distribution': {p.name: 0 for p in AgentPriority}
        }

    def submit_task(
        self,
        priority: AgentPriority,
        task_id: str,
        action: Callable,
        dependencies: List[str] = None
    ):
        """提交任务"""
        task = AgentTask(
            priority=priority.value,
            timestamp=time.time(),
            counter=self.counter,
            task_id=task_id,
            action=action,
            dependencies=dependencies or []
        )

        heapq.heappush(self.task_queue, task)
        self.counter += 1
        self.metrics['submitted'] += 1
        self.metrics['priority_distribution'][priority.name] += 1

    def can_execute(self, task: AgentTask) -> bool:
        """检查任务是否可执行(依赖已完成)"""
        return all(dep in self.completed_tasks for dep in task.dependencies)

    def execute_next(self):
        """执行下一个任务"""
        if not self.task_queue:
            return None

        # 找到第一个可执行的任务
        temp_queue = []
        task = None

        while self.task_queue:
            candidate = heapq.heappop(self.task_queue)

            if self.can_execute(candidate):
                task = candidate
                break
            else:
                temp_queue.append(candidate)

        # 将未执行的任务放回队列
        for t in temp_queue:
            heapq.heappush(self.task_queue, t)

        if task is None:
            return None

        # 执行任务
        wait_time = time.time() - task.timestamp
        self.metrics['wait_times'].append(wait_time)
        self.metrics['executed'] += 1

        print(f"[{time.strftime('%H:%M:%S')}] Executing {task.task_id} "
              f"(priority={task.priority}, wait={wait_time:.2f}s)")

        result = task.action()
        self.completed_tasks.add(task.task_id)

        return result

    def apply_aging(self):
        """应用Aging机制"""
        current_time = time.time()
        aged_queue = []

        while self.task_queue:
            task = heapq.heappop(self.task_queue)
            wait_time = current_time - task.timestamp
            age_boost = int(wait_time / self.age_interval)
            new_priority = max(0, task.priority - age_boost)

            aged_task = AgentTask(
                priority=new_priority,
                timestamp=task.timestamp,
                counter=task.counter,
                task_id=task.task_id,
                action=task.action,
                dependencies=task.dependencies
            )
            aged_queue.append(aged_task)

        self.task_queue = aged_queue
        heapq.heapify(self.task_queue)

    def get_metrics(self) -> Dict:
        """获取监控指标"""
        if not self.metrics['wait_times']:
            return self.metrics

        return {
            **self.metrics,
            'avg_wait_time': sum(self.metrics['wait_times']) / len(self.metrics['wait_times']),
            'max_wait_time': max(self.metrics['wait_times']),
            'min_wait_time': min(self.metrics['wait_times']),
            'pending_tasks': len(self.task_queue)
        }

    def execute_all(self):
        """执行所有任务"""
        while self.task_queue:
            result = self.execute_next()
            if result is None:
                # 所有剩余任务都有未完成的依赖
                print("Warning: Circular dependency or missing dependencies detected")
                break

# 使用示例
scheduler = AgentTaskScheduler(age_interval=10)

# 提交任务
scheduler.submit_task(
    AgentPriority.USER_QUERY,
    "query_1",
    lambda: "User query processed"
)

scheduler.submit_task(
    AgentPriority.TOOL_CALL,
    "tool_1",
    lambda: "Tool executed",
    dependencies=["query_1"]
)

scheduler.submit_task(
    AgentPriority.MEMORY_UPDATE,
    "memory_1",
    lambda: "Memory updated",
    dependencies=["tool_1"]
)

scheduler.submit_task(
    AgentPriority.LOGGING,
    "log_1",
    lambda: "Log written"
)

# 执行所有任务
scheduler.execute_all()

# 查看指标
print("\nMetrics:")
for key, value in scheduler.get_metrics().items():
    print(f"  {key}: {value}")
```

---

## 2. LangChain风格实现

```python
from typing import Any, Dict
import heapq
import time

class LangChainStyleScheduler:
    """
    LangChain风格的Agent调度器

    参考:
    - "10 LangChain Priority Queues for Fair, Fast Agents" (Medium, 2025)
    - "Priority Queues That Make LangChain Agents Feel Fair" (Medium, 2025)
    """

    def __init__(self):
        self.queue = []
        self.counter = 0

    def schedule(self, task: Dict[str, Any]):
        """调度任务"""
        priority = task.get('priority', 2)
        heapq.heappush(
            self.queue,
            (priority, self.counter, task)
        )
        self.counter += 1

    def run(self):
        """运行调度器"""
        while self.queue:
            priority, _, task = heapq.heappop(self.queue)
            print(f"Running task: {task['name']} (priority={priority})")
            task['action']()

# 使用示例
scheduler = LangChainStyleScheduler()

scheduler.schedule({
    'name': 'user_query',
    'priority': 0,
    'action': lambda: print("Processing user query")
})

scheduler.schedule({
    'name': 'tool_call',
    'priority': 1,
    'action': lambda: print("Calling tool")
})

scheduler.run()
```

---

## 3. 2025-2026最新应用

### OrcaLoca框架集成

```python
# 参考: arXiv:2502.00350v2 (2025)
# OrcaLoca: Heap-based priority queue for dynamic LLM-guided action scheduling

class OrcaLocaScheduler:
    """
    OrcaLoca风格的动态优先级调度

    特点:
    - LLM动态评估任务优先级
    - Heap维护优先级队列
    - 实现SWE-bench Lite SOTA
    """

    def __init__(self, llm_client):
        self.queue = []
        self.llm_client = llm_client
        self.counter = 0

    def submit_with_llm_priority(self, task_description: str, action: Callable):
        """使用LLM评估优先级"""
        # LLM评估优先级
        priority = self.llm_client.evaluate_priority(task_description)

        heapq.heappush(
            self.queue,
            (priority, self.counter, task_description, action)
        )
        self.counter += 1

    def execute_next(self):
        """执行下一个任务"""
        if not self.queue:
            return None

        priority, _, description, action = heapq.heappop(self.queue)
        print(f"Executing: {description} (LLM priority={priority})")
        return action()
```

---

## 4. 性能监控

```python
import time
from collections import defaultdict

class MonitoredScheduler(AgentTaskScheduler):
    """带性能监控的调度器"""

    def __init__(self, age_interval=10):
        super().__init__(age_interval)
        self.execution_times = defaultdict(list)

    def execute_next(self):
        """执行任务并记录性能"""
        if not self.task_queue:
            return None

        task = heapq.heappop(self.task_queue)

        start_time = time.time()
        result = super().execute_next()
        execution_time = time.time() - start_time

        self.execution_times[task.task_id].append(execution_time)

        return result

    def get_performance_report(self):
        """获取性能报告"""
        report = {}
        for task_id, times in self.execution_times.items():
            report[task_id] = {
                'count': len(times),
                'avg_time': sum(times) / len(times),
                'max_time': max(times),
                'min_time': min(times)
            }
        return report
```

---

## 5. 一句话总结

**AI Agent任务调度使用heap实现多级优先级队列,支持任务依赖、Aging机制防止饥饿、性能监控,参考2025年LangChain和OrcaLoca框架的最新实践,实现公平高效的Agent任务管理。**
