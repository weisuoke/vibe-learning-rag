# 核心概念01：TypedDict状态定义

> LangGraph 状态定义的基础方式

## 概念定义

**TypedDict 是 Python 的类型化字典，用于定义 LangGraph 图的状态结构，提供编译时类型检查和清晰的数据契约。**

## 为什么使用 TypedDict

### 1. 轻量级

```python
from typing import TypedDict

# TypedDict - 无运行时开销
class State(TypedDict):
    messages: list
    user_id: str
    count: int

# 等价于普通字典，但有类型提示
state = {"messages": [], "user_id": "123", "count": 0}
```

**优势**：
- 无需额外依赖
- 无运行时性能损耗
- 与标准 Python 类型系统集成

### 2. 类型安全

```python
from typing import TypedDict

class State(TypedDict):
    count: int

def node(state: State) -> dict:
    # mypy 会检查类型
    return {"count": "not an int"}  # ❌ 类型错误
    return {"count": 42}  # ✅ 正确
```

**优势**：
- 编译时发现错误
- IDE 自动补全
- 重构更安全

### 3. 清晰的契约

```python
class State(TypedDict):
    """图的状态定义"""
    messages: list  # 对话历史
    user_id: str    # 用户ID
    count: int      # 计数器
```

**优势**：
- 一眼看清状态结构
- 文档即代码
- 团队协作更清晰

## 基础语法

### 1. 定义状态

```python
from typing import TypedDict

class State(TypedDict):
    # 字段名: 类型
    messages: list
    user_id: str
    count: int
    metadata: dict
```

### 2. 使用状态

```python
from langgraph.graph import StateGraph

# 创建图
graph = StateGraph(State)

# 定义节点
def my_node(state: State) -> dict:
    # 读取状态
    messages = state["messages"]
    user_id = state["user_id"]

    # 返回部分更新
    return {
        "messages": messages + ["new message"],
        "count": state["count"] + 1
    }

graph.add_node("my_node", my_node)
```

### 3. 初始化状态

```python
# 运行图时提供初始状态
result = graph.invoke({
    "messages": [],
    "user_id": "user123",
    "count": 0,
    "metadata": {}
})
```

## 高级特性

### 1. Required 与 NotRequired

```python
from typing import TypedDict
from typing_extensions import Required, NotRequired

class State(TypedDict):
    # 必需字段
    user_id: Required[str]

    # 可选字段
    count: NotRequired[int]
    metadata: NotRequired[dict]
```

**使用场景**：
- 必需字段：图运行必须提供
- 可选字段：可以省略，节点内部处理默认值

### 2. 继承与组合

```python
# 基础状态
class BaseState(TypedDict):
    user_id: str
    timestamp: float

# 扩展状态
class ChatState(BaseState):
    messages: list
    context: str

# 多继承
class FullState(ChatState, BaseState):
    metadata: dict
```

### 3. 泛型支持

```python
from typing import TypedDict, Generic, TypeVar

T = TypeVar('T')

class State(TypedDict, Generic[T]):
    data: T
    user_id: str

# 使用
class MyState(State[list]):
    pass
```

## 与 Pydantic 对比

### TypedDict 方式

```python
from typing import TypedDict

class State(TypedDict):
    count: int
    name: str

# 优点：
# - 轻量级，无依赖
# - 性能好
# - 类型检查在编译时

# 缺点：
# - 无运行时验证
# - 无默认值（需要手动处理）
# - 无复杂验证逻辑
```

### Pydantic 方式

```python
from pydantic import BaseModel, Field

class State(BaseModel):
    count: int = Field(default=0, ge=0)
    name: str = Field(min_length=1)

# 优点：
# - 运行时验证
# - 默认值支持
# - 复杂验证逻辑
# - 序列化/反序列化

# 缺点：
# - 需要依赖 pydantic
# - 性能稍慢
# - 更重量级
```

### 选择建议

| 场景 | 推荐 |
|------|------|
| 简单状态，无需验证 | TypedDict |
| 需要默认值 | TypedDict + 手动处理 或 Pydantic |
| 需要验证用户输入 | Pydantic |
| 性能敏感 | TypedDict |
| 复杂业务逻辑 | Pydantic |

## 在 LangGraph 中的应用

### 1. 聊天机器人状态

```python
from typing import TypedDict, Annotated
import operator

class ChatState(TypedDict):
    # 对话历史（追加）
    messages: Annotated[list, operator.add]

    # 用户信息（覆盖）
    user_id: str
    user_name: str

    # 上下文（覆盖）
    context: str

    # 元数据（覆盖）
    metadata: dict
```

### 2. 工作流状态

```python
class WorkflowState(TypedDict):
    # 任务ID
    task_id: str

    # 任务状态
    status: str  # "pending", "running", "completed", "failed"

    # 执行结果
    results: Annotated[list, operator.add]

    # 错误信息
    errors: Annotated[list, operator.add]

    # 重试次数
    retry_count: int
```

### 3. 数据处理管道状态

```python
class PipelineState(TypedDict):
    # 原始数据
    raw_data: list

    # 处理后的数据
    processed_data: Annotated[list, operator.add]

    # 处理步骤
    steps: Annotated[list, operator.add]

    # 统计信息
    stats: dict
```

## 常见模式

### 1. 消息累积模式

```python
class State(TypedDict):
    messages: Annotated[list, operator.add]

def node1(state: State) -> dict:
    return {"messages": ["msg1"]}

def node2(state: State) -> dict:
    return {"messages": ["msg2"]}

# 最终 messages = ["msg1", "msg2"]
```

### 2. 状态覆盖模式

```python
class State(TypedDict):
    current_step: str
    result: str

def node1(state: State) -> dict:
    return {"current_step": "step1", "result": "result1"}

def node2(state: State) -> dict:
    return {"current_step": "step2", "result": "result2"}

# 最终 current_step = "step2", result = "result2"
```

### 3. 混合模式

```python
class State(TypedDict):
    # 累积
    logs: Annotated[list, operator.add]

    # 覆盖
    current_status: str
    last_update: float

def node(state: State) -> dict:
    return {
        "logs": ["new log"],  # 追加
        "current_status": "running",  # 覆盖
        "last_update": time.time()  # 覆盖
    }
```

## 最佳实践

### 1. 命名规范

```python
# ✅ 清晰的命名
class ChatState(TypedDict):
    messages: list
    user_id: str
    context: str

# ❌ 模糊的命名
class State(TypedDict):
    m: list
    u: str
    c: str
```

### 2. 类型注解

```python
# ✅ 具体的类型
from typing import TypedDict, List, Dict

class State(TypedDict):
    messages: List[str]
    metadata: Dict[str, str]

# ❌ 泛泛的类型
class State(TypedDict):
    messages: list
    metadata: dict
```

### 3. 文档注释

```python
class State(TypedDict):
    """聊天机器人状态

    Attributes:
        messages: 对话历史，按时间顺序排列
        user_id: 用户唯一标识符
        context: 当前对话上下文
    """
    messages: Annotated[list, operator.add]
    user_id: str
    context: str
```

### 4. 字段分组

```python
class State(TypedDict):
    # === 用户信息 ===
    user_id: str
    user_name: str
    user_email: str

    # === 对话数据 ===
    messages: Annotated[list, operator.add]
    context: str

    # === 元数据 ===
    created_at: float
    updated_at: float
```

## 常见错误

### 1. 忘记导入 TypedDict

```python
# ❌ 错误
class State(TypedDict):  # NameError
    messages: list

# ✅ 正确
from typing import TypedDict

class State(TypedDict):
    messages: list
```

### 2. 使用可变默认值

```python
# ❌ 错误 - TypedDict 不支持默认值
class State(TypedDict):
    messages: list = []  # SyntaxError

# ✅ 正确 - 在初始化时提供
graph.invoke({"messages": []})
```

### 3. 类型不匹配

```python
class State(TypedDict):
    count: int

def node(state: State) -> dict:
    # ❌ 返回错误类型
    return {"count": "123"}  # 应该是 int

    # ✅ 正确
    return {"count": 123}
```

## 调试技巧

### 1. 使用 mypy 检查

```bash
# 安装 mypy
pip install mypy

# 检查类型
mypy your_script.py
```

### 2. 打印状态结构

```python
from typing import get_type_hints

class State(TypedDict):
    messages: list
    user_id: str

# 查看字段类型
hints = get_type_hints(State)
print(hints)
# {'messages': <class 'list'>, 'user_id': <class 'str'>}
```

### 3. 运行时验证

```python
def validate_state(state: dict, state_class: type) -> bool:
    """验证状态是否符合 TypedDict 定义"""
    hints = get_type_hints(state_class)
    for key, expected_type in hints.items():
        if key not in state:
            print(f"Missing key: {key}")
            return False
        if not isinstance(state[key], expected_type):
            print(f"Wrong type for {key}: expected {expected_type}, got {type(state[key])}")
            return False
    return True

# 使用
state = {"messages": [], "user_id": "123"}
validate_state(state, State)
```

## 总结

**TypedDict 是 LangGraph 状态定义的推荐方式**：
- 轻量级，无运行时开销
- 类型安全，编译时检查
- 清晰的数据契约
- 与 Python 类型系统原生集成

**何时使用 Pydantic**：
- 需要运行时验证
- 需要复杂默认值
- 需要序列化/反序列化

**最佳实践**：
- 清晰的命名
- 具体的类型注解
- 完善的文档注释
- 合理的字段分组

## 参考资料

- 详细实战：`07_实战代码_场景1_基础TypedDict状态定义.md`
- Reducer 使用：`03_核心概念_02_Annotated与Reducer.md`
- 多 Schema 架构：`03_核心概念_05_多Schema架构.md`
