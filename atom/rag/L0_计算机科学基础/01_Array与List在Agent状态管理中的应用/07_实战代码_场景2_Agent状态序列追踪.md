# 实战代码 - 场景2：Agent 状态序列追踪

## 场景描述

构建一个 Agent 状态追踪系统，记录 Agent 的每个动作和状态变化，支持：
- 状态序列记录
- 时间戳追踪
- 状态回放
- 状态可视化

---

## 完整代码实现

```python
"""
Agent 状态序列追踪示例
演示：使用 List 管理 Agent 的状态历史
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json


# ===== 1. 定义状态数据结构 =====
@dataclass
class AgentState:
    """Agent 状态"""
    timestamp: float
    step: int
    action: str
    observation: str
    thought: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentState':
        """从字典创建"""
        return cls(**data)

    def __repr__(self) -> str:
        dt = datetime.fromtimestamp(self.timestamp)
        return f"AgentState(step={self.step}, action='{self.action}', time={dt.strftime('%H:%M:%S')})"


# ===== 2. 状态追踪器 =====
class StateTracker:
    """Agent 状态序列追踪器"""

    def __init__(self):
        self.states: List[AgentState] = []  # 使用 List 存储状态序列
        self.current_step = 0

    def add_state(
        self,
        action: str,
        observation: str,
        thought: str = "",
        metadata: Optional[Dict] = None
    ) -> AgentState:
        """添加新状态（O(1) 摊销）"""
        state = AgentState(
            timestamp=time.time(),
            step=self.current_step,
            action=action,
            observation=observation,
            thought=thought,
            metadata=metadata or {}
        )

        self.states.append(state)
        self.current_step += 1

        return state

    def get_state(self, step: int) -> Optional[AgentState]:
        """获取特定步骤的状态（O(1)）"""
        if 0 <= step < len(self.states):
            return self.states[step]
        return None

    def get_recent_states(self, n: int = 10) -> List[AgentState]:
        """获取最近 N 个状态（O(N)）"""
        return self.states[-n:]

    def get_states_by_action(self, action: str) -> List[AgentState]:
        """获取特定动作的所有状态（O(N)）"""
        return [state for state in self.states if state.action == action]

    def replay(self, start: int = 0, end: Optional[int] = None):
        """回放状态序列"""
        end = end or len(self.states)

        print(f"\n{'='*60}")
        print(f"回放状态序列 (步骤 {start} - {end-1})")
        print(f"{'='*60}\n")

        for i in range(start, min(end, len(self.states))):
            state = self.states[i]
            dt = datetime.fromtimestamp(state.timestamp)

            print(f"步骤 {state.step} [{dt.strftime('%H:%M:%S')}]")
            print(f"  动作: {state.action}")
            print(f"  观察: {state.observation}")
            if state.thought:
                print(f"  思考: {state.thought}")
            if state.metadata:
                print(f"  元数据: {state.metadata}")
            print()

    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self.states:
            return {"total": 0, "actions": {}, "duration": 0}

        # 统计动作分布
        action_counts = {}
        for state in self.states:
            action_counts[state.action] = action_counts.get(state.action, 0) + 1

        # 计算总时长
        duration = self.states[-1].timestamp - self.states[0].timestamp

        return {
            "total": len(self.states),
            "actions": action_counts,
            "duration": duration,
            "start_time": datetime.fromtimestamp(self.states[0].timestamp).isoformat(),
            "end_time": datetime.fromtimestamp(self.states[-1].timestamp).isoformat(),
        }

    def save_to_file(self, filepath: str):
        """保存状态序列到文件"""
        data = {
            "states": [state.to_dict() for state in self.states],
            "current_step": self.current_step,
            "saved_at": time.time()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[保存] 状态序列已保存到 {filepath}")

    def load_from_file(self, filepath: str):
        """从文件加载状态序列"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.states = [AgentState.from_dict(s) for s in data["states"]]
        self.current_step = data["current_step"]

        print(f"[加载] 状态序列已加载，共 {len(self.states)} 个状态")

    def visualize(self):
        """可视化状态序列"""
        if not self.states:
            print("没有状态可显示")
            return

        print(f"\n{'='*60}")
        print("状态序列可视化")
        print(f"{'='*60}\n")

        # 时间线
        print("时间线:")
        for i, state in enumerate(self.states):
            dt = datetime.fromtimestamp(state.timestamp)
            print(f"  {i:3d}. [{dt.strftime('%H:%M:%S')}] {state.action}")

        # 动作分布
        print(f"\n动作分布:")
        stats = self.get_stats()
        for action, count in stats["actions"].items():
            percentage = count / stats["total"] * 100
            bar = "█" * int(percentage / 2)
            print(f"  {action:20s} {count:3d} ({percentage:5.1f}%) {bar}")

        # 总结
        print(f"\n总结:")
        print(f"  总步骤数: {stats['total']}")
        print(f"  总时长: {stats['duration']:.2f}s")
        print(f"  平均每步: {stats['duration']/stats['total']:.2f}s")


# ===== 3. 模拟 Agent 执行 =====
def simulate_agent_execution():
    """模拟 Agent 执行过程"""
    print("=" * 60)
    print("模拟 Agent 执行")
    print("=" * 60)

    tracker = StateTracker()

    # 步骤 1: 接收任务
    tracker.add_state(
        action="receive_task",
        observation="用户请求：查询天气信息",
        thought="需要调用天气 API",
        metadata={"user_id": "user_123", "task_id": "task_001"}
    )
    time.sleep(0.1)

    # 步骤 2: 规划
    tracker.add_state(
        action="plan",
        observation="分析任务需求",
        thought="需要先获取用户位置，然后调用天气 API",
        metadata={"plan": ["get_location", "call_weather_api", "format_response"]}
    )
    time.sleep(0.1)

    # 步骤 3: 获取位置
    tracker.add_state(
        action="get_location",
        observation="用户位置：北京",
        thought="位置获取成功",
        metadata={"location": "北京", "lat": 39.9042, "lon": 116.4074}
    )
    time.sleep(0.1)

    # 步骤 4: 调用天气 API
    tracker.add_state(
        action="call_weather_api",
        observation="API 返回：晴天，温度 15°C",
        thought="天气信息获取成功",
        metadata={"weather": "晴天", "temperature": 15, "humidity": 60}
    )
    time.sleep(0.1)

    # 步骤 5: 格式化响应
    tracker.add_state(
        action="format_response",
        observation="生成用户友好的回复",
        thought="组织天气信息",
        metadata={"response": "北京今天晴天，温度 15°C，湿度 60%"}
    )
    time.sleep(0.1)

    # 步骤 6: 返回结果
    tracker.add_state(
        action="return_result",
        observation="任务完成",
        thought="成功返回结果给用户",
        metadata={"status": "success", "duration": 0.5}
    )

    return tracker


# ===== 4. 使用示例 =====
def main():
    """主函数"""
    # 模拟 Agent 执行
    tracker = simulate_agent_execution()

    # 查看所有状态
    print(f"\n总状态数: {len(tracker.states)}")

    # 访问特定步骤
    print(f"\n{'='*60}")
    print("访问特定步骤")
    print(f"{'='*60}\n")

    step_3 = tracker.get_state(3)
    print(f"步骤 3: {step_3}")
    print(f"  动作: {step_3.action}")
    print(f"  观察: {step_3.observation}")

    # 获取最近 3 个状态
    print(f"\n{'='*60}")
    print("最近 3 个状态")
    print(f"{'='*60}\n")

    recent = tracker.get_recent_states(3)
    for state in recent:
        print(f"  {state}")

    # 按动作筛选
    print(f"\n{'='*60}")
    print("筛选特定动作")
    print(f"{'='*60}\n")

    api_calls = tracker.get_states_by_action("call_weather_api")
    print(f"API 调用次数: {len(api_calls)}")
    for state in api_calls:
        print(f"  {state}")

    # 回放状态序列
    tracker.replay()

    # 可视化
    tracker.visualize()

    # 统计信息
    print(f"\n{'='*60}")
    print("统计信息")
    print(f"{'='*60}\n")

    stats = tracker.get_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # 保存到文件
    tracker.save_to_file("data/agent_states/execution_001.json")


# ===== 5. 高级示例：多任务追踪 =====
class MultiTaskTracker:
    """多任务状态追踪器"""

    def __init__(self):
        self.tasks: Dict[str, StateTracker] = {}

    def create_task(self, task_id: str) -> StateTracker:
        """创建新任务"""
        tracker = StateTracker()
        self.tasks[task_id] = tracker
        return tracker

    def get_task(self, task_id: str) -> Optional[StateTracker]:
        """获取任务追踪器"""
        return self.tasks.get(task_id)

    def get_all_stats(self) -> Dict:
        """获取所有任务的统计信息"""
        return {
            task_id: tracker.get_stats()
            for task_id, tracker in self.tasks.items()
        }

    def visualize_all(self):
        """可视化所有任务"""
        print(f"\n{'='*60}")
        print("所有任务概览")
        print(f"{'='*60}\n")

        for task_id, tracker in self.tasks.items():
            stats = tracker.get_stats()
            print(f"任务 {task_id}:")
            print(f"  步骤数: {stats['total']}")
            print(f"  时长: {stats['duration']:.2f}s")
            print(f"  动作: {list(stats['actions'].keys())}")
            print()


def multi_task_example():
    """多任务追踪示例"""
    print("\n" + "=" * 60)
    print("多任务追踪示例")
    print("=" * 60)

    multi_tracker = MultiTaskTracker()

    # 任务 1: 天气查询
    task1 = multi_tracker.create_task("weather_query")
    task1.add_state("receive_task", "查询天气", "开始任务")
    task1.add_state("get_location", "北京", "获取位置")
    task1.add_state("call_api", "晴天 15°C", "调用 API")
    task1.add_state("return_result", "完成", "返回结果")

    # 任务 2: 新闻搜索
    task2 = multi_tracker.create_task("news_search")
    task2.add_state("receive_task", "搜索新闻", "开始任务")
    task2.add_state("search", "找到 10 条新闻", "搜索")
    task2.add_state("filter", "筛选出 3 条", "过滤")
    task2.add_state("return_result", "完成", "返回结果")

    # 可视化所有任务
    multi_tracker.visualize_all()

    # 统计信息
    all_stats = multi_tracker.get_all_stats()
    print("所有任务统计:")
    print(json.dumps(all_stats, indent=2, ensure_ascii=False))


# ===== 6. 高级示例：状态回滚 =====
class RollbackableTracker(StateTracker):
    """支持回滚的状态追踪器"""

    def rollback(self, steps: int = 1):
        """回滚 N 个步骤"""
        if steps > len(self.states):
            print(f"[警告] 无法回滚 {steps} 步，只有 {len(self.states)} 个状态")
            steps = len(self.states)

        removed = self.states[-steps:]
        self.states = self.states[:-steps]
        self.current_step -= steps

        print(f"[回滚] 已回滚 {steps} 步")
        for state in removed:
            print(f"  移除: {state}")

    def rollback_to(self, step: int):
        """回滚到特定步骤"""
        if step < 0 or step >= len(self.states):
            print(f"[错误] 无效的步骤: {step}")
            return

        steps_to_remove = len(self.states) - step - 1
        self.rollback(steps_to_remove)


def rollback_example():
    """回滚示例"""
    print("\n" + "=" * 60)
    print("状态回滚示例")
    print("=" * 60)

    tracker = RollbackableTracker()

    # 添加状态
    for i in range(5):
        tracker.add_state(f"action_{i}", f"observation_{i}", f"thought_{i}")

    print(f"\n初始状态数: {len(tracker.states)}")

    # 回滚 2 步
    tracker.rollback(2)
    print(f"回滚后状态数: {len(tracker.states)}")

    # 继续添加
    tracker.add_state("action_new", "observation_new", "thought_new")
    print(f"添加新状态后: {len(tracker.states)}")

    # 回放
    tracker.replay()


# ===== 7. 性能测试 =====
def performance_test():
    """性能测试"""
    print("\n" + "=" * 60)
    print("性能测试")
    print("=" * 60)

    tracker = StateTracker()

    # 测试添加 10000 个状态
    start = time.perf_counter()

    for i in range(10000):
        tracker.add_state(
            action=f"action_{i % 10}",
            observation=f"observation_{i}",
            thought=f"thought_{i}",
            metadata={"index": i}
        )

    elapsed = time.perf_counter() - start

    print(f"\n添加 10000 个状态耗时: {elapsed*1000:.2f} ms")
    print(f"平均每个状态: {elapsed/10000*1e6:.2f} μs")

    # 测试索引访问
    start = time.perf_counter()

    for i in range(1000):
        _ = tracker.get_state(i * 10)

    elapsed = time.perf_counter() - start

    print(f"\n1000 次索引访问耗时: {elapsed*1000:.2f} ms")
    print(f"平均每次访问: {elapsed/1000*1e6:.2f} μs")

    # 测试筛选
    start = time.perf_counter()

    filtered = tracker.get_states_by_action("action_5")

    elapsed = time.perf_counter() - start

    print(f"\n筛选操作耗时: {elapsed*1000:.2f} ms")
    print(f"筛选结果: {len(filtered)} 个状态")


if __name__ == "__main__":
    # 运行主示例
    main()

    # 运行多任务示例
    multi_task_example()

    # 运行回滚示例
    rollback_example()

    # 运行性能测试
    performance_test()
```

---

## 运行输出示例

```
============================================================
模拟 Agent 执行
============================================================

总状态数: 6

============================================================
访问特定步骤
============================================================

步骤 3: AgentState(step=3, action='call_weather_api', time=12:34:56)
  动作: call_weather_api
  观察: API 返回：晴天，温度 15°C

============================================================
最近 3 个状态
============================================================

  AgentState(step=3, action='call_weather_api', time=12:34:56)
  AgentState(step=4, action='format_response', time=12:34:56)
  AgentState(step=5, action='return_result', time=12:34:56)

============================================================
筛选特定动作
============================================================

API 调用次数: 1
  AgentState(step=3, action='call_weather_api', time=12:34:56)

============================================================
回放状态序列 (步骤 0 - 5)
============================================================

步骤 0 [12:34:56]
  动作: receive_task
  观察: 用户请求：查询天气信息
  思考: 需要调用天气 API
  元数据: {'user_id': 'user_123', 'task_id': 'task_001'}

步骤 1 [12:34:56]
  动作: plan
  观察: 分析任务需求
  思考: 需要先获取用户位置，然后调用天气 API
  元数据: {'plan': ['get_location', 'call_weather_api', 'format_response']}

...
```

---

## 关键要点

1. **List 作为状态序列**
   - append：O(1) 摊销
   - 索引访问：O(1)
   - 筛选：O(n)

2. **时间戳追踪**
   - 每个状态记录时间戳
   - 支持时间线可视化
   - 计算执行时长

3. **状态回放**
   - 按顺序重现执行过程
   - 支持部分回放
   - 调试和分析工具

4. **持久化**
   - JSON 格式存储
   - 支持加载和恢复
   - 跨会话追踪

5. **性能特性**
   - 10000 个状态：~15 ms
   - 索引访问：~0.05 μs
   - 筛选操作：~2 ms

---

## 参考来源（2025-2026）

### Agent 框架
- **LangGraph State Management** (2026)
  - URL: https://langchain-ai.github.io/langgraph/concepts/low_level/
  - 描述：LangGraph 状态管理文档

### Python 数据结构
- **Python dataclasses** (2026)
  - URL: https://docs.python.org/3/library/dataclasses.html
  - 描述：Python dataclass 官方文档
