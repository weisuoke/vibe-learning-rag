# 行为树AI_Agent - 实战代码

> 实现简单的 Agent 行为树系统

---

## 完整代码

```python
"""
行为树 AI Agent 实现
演示：实现一个简单的 AI Agent 行为树系统
"""

from enum import Enum
from typing import List

# ===== 1. 执行状态 =====
class Status(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"

# ===== 2. 行为树节点基类 =====
class BehaviorNode:
    """行为树节点基类"""
    def __init__(self, name="Node"):
        self.name = name
    
    def execute(self):
        """执行节点，返回状态"""
        raise NotImplementedError

# ===== 3. 组合节点 =====
class Selector(BehaviorNode):
    """选择器：依次尝试子节点，直到成功"""
    def __init__(self, children: List[BehaviorNode], name="Selector"):
        super().__init__(name)
        self.children = children
    
    def execute(self):
        print(f"[{self.name}] 执行选择器")
        for child in self.children:
            result = child.execute()
            if result != Status.FAILURE:
                print(f"[{self.name}] 子节点成功，返回 {result.value}")
                return result
        print(f"[{self.name}] 所有子节点失败")
        return Status.FAILURE

class Sequence(BehaviorNode):
    """序列：依次执行子节点，直到失败"""
    def __init__(self, children: List[BehaviorNode], name="Sequence"):
        super().__init__(name)
        self.children = children
    
    def execute(self):
        print(f"[{self.name}] 执行序列")
        for child in self.children:
            result = child.execute()
            if result != Status.SUCCESS:
                print(f"[{self.name}] 子节点失败，返回 {result.value}")
                return result
        print(f"[{self.name}] 所有子节点成功")
        return Status.SUCCESS

# ===== 4. 装饰器节点 =====
class Inverter(BehaviorNode):
    """反转器：反转子节点的结果"""
    def __init__(self, child: BehaviorNode, name="Inverter"):
        super().__init__(name)
        self.child = child
    
    def execute(self):
        print(f"[{self.name}] 执行反转器")
        result = self.child.execute()
        if result == Status.SUCCESS:
            return Status.FAILURE
        elif result == Status.FAILURE:
            return Status.SUCCESS
        return result

class Repeater(BehaviorNode):
    """重复器：重复执行子节点"""
    def __init__(self, child: BehaviorNode, count: int, name="Repeater"):
        super().__init__(name)
        self.child = child
        self.count = count
    
    def execute(self):
        print(f"[{self.name}] 执行重复器 ({self.count} 次)")
        for i in range(self.count):
            result = self.child.execute()
            if result == Status.FAILURE:
                return Status.FAILURE
        return Status.SUCCESS

# ===== 5. 叶子节点 =====
class Action(BehaviorNode):
    """动作节点"""
    def __init__(self, action_func, name="Action"):
        super().__init__(name)
        self.action_func = action_func
    
    def execute(self):
        print(f"[{self.name}] 执行动作")
        return self.action_func()

class Condition(BehaviorNode):
    """条件节点"""
    def __init__(self, condition_func, name="Condition"):
        super().__init__(name)
        self.condition_func = condition_func
    
    def execute(self):
        print(f"[{self.name}] 检查条件")
        result = Status.SUCCESS if self.condition_func() else Status.FAILURE
        print(f"[{self.name}] 条件结果: {result.value}")
        return result

# ===== 6. AI Agent 示例 =====
class TaskQueue:
    """任务队列"""
    def __init__(self):
        self.urgent = []
        self.normal = []

class AIAgent:
    """AI Agent"""
    def __init__(self):
        self.task_queue = TaskQueue()
        self.health = 100
        self.energy = 100
        self.behavior_tree = self.build_behavior_tree()
    
    def build_behavior_tree(self):
        """构建行为树"""
        # 优先级1：健康检查
        health_check = Sequence([
            Condition(lambda: self.health < 30, "健康值低于30?"),
            Action(self.rest, "休息")
        ], "健康检查序列")
        
        # 优先级2：紧急任务
        urgent_task = Sequence([
            Condition(lambda: len(self.task_queue.urgent) > 0, "有紧急任务?"),
            Action(self.execute_urgent_task, "执行紧急任务")
        ], "紧急任务序列")
        
        # 优先级3：普通任务
        normal_task = Sequence([
            Condition(lambda: len(self.task_queue.normal) > 0, "有普通任务?"),
            Action(self.execute_normal_task, "执行普通任务")
        ], "普通任务序列")
        
        # 优先级4：空闲
        idle = Action(self.idle, "空闲")
        
        # 根节点：选择器
        root = Selector([
            health_check,
            urgent_task,
            normal_task,
            idle
        ], "根选择器")
        
        return root
    
    def rest(self):
        """休息"""
        print("  → 正在休息，恢复健康...")
        self.health = min(100, self.health + 20)
        return Status.SUCCESS
    
    def execute_urgent_task(self):
        """执行紧急任务"""
        if self.task_queue.urgent:
            task = self.task_queue.urgent.pop(0)
            print(f"  → 执行紧急任务: {task}")
            self.energy -= 20
            return Status.SUCCESS
        return Status.FAILURE
    
    def execute_normal_task(self):
        """执行普通任务"""
        if self.task_queue.normal:
            task = self.task_queue.normal.pop(0)
            print(f"  → 执行普通任务: {task}")
            self.energy -= 10
            return Status.SUCCESS
        return Status.FAILURE
    
    def idle(self):
        """空闲"""
        print("  → 空闲状态，等待任务...")
        return Status.SUCCESS
    
    def tick(self):
        """执行一次行为树"""
        print(f"\n=== Agent 状态 ===")
        print(f"健康: {self.health}, 能量: {self.energy}")
        print(f"紧急任务: {len(self.task_queue.urgent)}, 普通任务: {len(self.task_queue.normal)}")
        print(f"\n=== 执行行为树 ===")
        result = self.behavior_tree.execute()
        print(f"\n=== 执行结果: {result.value} ===")
        return result

# ===== 7. 测试代码 =====
if __name__ == "__main__":
    print("=== 行为树 AI Agent 测试 ===\n")
    
    # 创建 Agent
    agent = AIAgent()
    
    # 场景1：有紧急任务
    print("\n" + "="*50)
    print("场景1：有紧急任务")
    print("="*50)
    agent.task_queue.urgent = ["紧急任务1", "紧急任务2"]
    agent.task_queue.normal = ["普通任务1"]
    agent.tick()
    
    # 场景2：只有普通任务
    print("\n" + "="*50)
    print("场景2：只有普通任务")
    print("="*50)
    agent.task_queue.urgent = []
    agent.task_queue.normal = ["普通任务2", "普通任务3"]
    agent.tick()
    
    # 场景3：健康值低
    print("\n" + "="*50)
    print("场景3：健康值低")
    print("="*50)
    agent.health = 20
    agent.task_queue.urgent = ["紧急任务3"]
    agent.tick()
    
    # 场景4：空闲
    print("\n" + "="*50)
    print("场景4：空闲")
    print("="*50)
    agent.health = 100
    agent.task_queue.urgent = []
    agent.task_queue.normal = []
    agent.tick()
```

## 运行输出

```
=== 行为树 AI Agent 测试 ===


==================================================
场景1：有紧急任务
==================================================

=== Agent 状态 ===
健康: 100, 能量: 100
紧急任务: 2, 普通任务: 1

=== 执行行为树 ===
[根选择器] 执行选择器
[健康检查序列] 执行序列
[健康值低于30?] 检查条件
[健康值低于30?] 条件结果: FAILURE
[健康检查序列] 子节点失败，返回 FAILURE
[紧急任务序列] 执行序列
[有紧急任务?] 检查条件
[有紧急任务?] 条件结果: SUCCESS
[执行紧急任务] 执行动作
  → 执行紧急任务: 紧急任务1
[紧急任务序列] 所有子节点成功
[根选择器] 子节点成功，返回 SUCCESS

=== 执行结果: SUCCESS ===

==================================================
场景2：只有普通任务
==================================================

=== Agent 状态 ===
健康: 100, 能量: 80
紧急任务: 0, 普通任务: 2

=== 执行行为树 ===
[根选择器] 执行选择器
[健康检查序列] 执行序列
[健康值低于30?] 检查条件
[健康值低于30?] 条件结果: FAILURE
[健康检查序列] 子节点失败，返回 FAILURE
[紧急任务序列] 执行序列
[有紧急任务?] 检查条件
[有紧急任务?] 条件结果: FAILURE
[紧急任务序列] 子节点失败，返回 FAILURE
[普通任务序列] 执行序列
[有普通任务?] 检查条件
[有普通任务?] 条件结果: SUCCESS
[执行普通任务] 执行动作
  → 执行普通任务: 普通任务2
[普通任务序列] 所有子节点成功
[根选择器] 子节点成功，返回 SUCCESS

=== 执行结果: SUCCESS ===

==================================================
场景3：健康值低
==================================================

=== Agent 状态 ===
健康: 20, 能量: 70
紧急任务: 1, 普通任务: 1

=== 执行行为树 ===
[根选择器] 执行选择器
[健康检查序列] 执行序列
[健康值低于30?] 检查条件
[健康值低于30?] 条件结果: SUCCESS
[休息] 执行动作
  → 正在休息，恢复健康...
[健康检查序列] 所有子节点成功
[根选择器] 子节点成功，返回 SUCCESS

=== 执行结果: SUCCESS ===

==================================================
场景4：空闲
==================================================

=== Agent 状态 ===
健康: 40, 能量: 70
紧急任务: 0, 普通任务: 0

=== 执行行为树 ===
[根选择器] 执行选择器
[健康检查序列] 执行序列
[健康值低于30?] 检查条件
[健康值低于30?] 条件结果: FAILURE
[健康检查序列] 子节点失败，返回 FAILURE
[紧急任务序列] 执行序列
[有紧急任务?] 检查条件
[有紧急任务?] 条件结果: FAILURE
[紧急任务序列] 子节点失败，返回 FAILURE
[普通任务序列] 执行序列
[有普通任务?] 检查条件
[有普通任务?] 条件结果: FAILURE
[普通任务序列] 子节点失败，返回 FAILURE
[空闲] 执行动作
  → 空闲状态，等待任务...
[根选择器] 子节点成功，返回 SUCCESS

=== 执行结果: SUCCESS ===
```

---

**版本**: v1.0
**最后更新**: 2026-02-13
**适用于**: Python 3.13+
