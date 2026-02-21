# 实战代码4：LangGraph错误处理

## 概述

本文提供LangGraph错误处理的完整可运行示例，涵盖：
- 节点级重试策略
- 图级异常处理
- 状态管理与错误
- 2026新特性

所有代码都可以直接复制运行。

---

## 示例1：节点级重试

**场景：** 为LangGraph节点添加重试机制

```python
"""
节点级重试
演示：在LangGraph节点中使用with_retry()
"""

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

print("=== LangGraph节点级重试 ===\n")

# ===== 1. 定义状态 =====
class AgentState(TypedDict):
    """Agent状态"""
    messages: list
    current_step: str
    error_count: int

# ===== 2. 创建带重试的LLM节点 =====
llm = ChatOpenAI(model="gpt-4").with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True
)

def llm_node(state: AgentState) -> AgentState:
    """LLM节点（带重试）"""
    print(f"  [LLM节点] 当前步骤: {state['current_step']}")

    try:
        # 调用LLM（自动重试）
        response = llm.invoke(state['messages'][-1])

        return {
            **state,
            "messages": state['messages'] + [response.content],
            "current_step": "completed"
        }
    except Exception as e:
        print(f"  ❌ LLM节点失败: {type(e).__name__}")
        return {
            **state,
            "error_count": state.get('error_count', 0) + 1,
            "current_step": "failed"
        }

# ===== 3. 构建图 =====
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("llm", llm_node)

# 设置入口
workflow.set_entry_point("llm")

# 添加边
workflow.add_edge("llm", END)

# 编译图
app = workflow.compile()

print("✅ LangGraph构建完成\n")

# ===== 4. 测试 =====
initial_state = {
    "messages": ["你好，请介绍一下自己"],
    "current_step": "start",
    "error_count": 0
}

print("=== 执行图 ===")
try:
    result = app.invoke(initial_state)
    print(f"✅ 成功")
    print(f"最终状态: {result['current_step']}")
    print(f"错误次数: {result['error_count']}")
    print(f"消息数: {len(result['messages'])}")
except Exception as e:
    print(f"❌ 失败: {type(e).__name__}: {e}")
```

---

## 示例2：图级异常处理

**场景：** 在图级别捕获和处理异常

```python
"""
图级异常处理
演示：使用try-except包装图的执行
"""

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

print("=== 图级异常处理 ===\n")

# ===== 1. 定义状态 =====
class AgentState(TypedDict):
    input: str
    output: str
    status: str
    error: str

# ===== 2. 创建节点 =====
llm = ChatOpenAI(model="gpt-4").with_retry(stop_after_attempt=3)

def process_node(state: AgentState) -> AgentState:
    """处理节点"""
    print(f"  [处理节点] 输入: {state['input'][:30]}...")

    try:
        response = llm.invoke(state['input'])
        return {
            **state,
            "output": response.content,
            "status": "success"
        }
    except Exception as e:
        print(f"  ❌ 节点失败: {type(e).__name__}")
        return {
            **state,
            "status": "failed",
            "error": str(e)
        }

# ===== 3. 构建图 =====
workflow = StateGraph(AgentState)
workflow.add_node("process", process_node)
workflow.set_entry_point("process")
workflow.add_edge("process", END)

app = workflow.compile()

print("✅ 图构建完成\n")

# ===== 4. 安全执行包装器 =====
def safe_invoke(app, input_data):
    """安全执行图"""
    initial_state = {
        "input": input_data,
        "output": "",
        "status": "pending",
        "error": ""
    }

    try:
        result = app.invoke(initial_state)

        if result['status'] == "success":
            return {
                "success": True,
                "output": result['output']
            }
        else:
            return {
                "success": False,
                "error": result['error']
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"图执行失败: {type(e).__name__}: {e}"
        }

# ===== 5. 测试 =====
test_inputs = [
    "什么是Python？",
    "什么是JavaScript？",
    "什么是Go？"
]

print("=== 测试 ===")
for i, input_data in enumerate(test_inputs):
    print(f"\n[{i+1}/{len(test_inputs)}] 输入: {input_data}")
    result = safe_invoke(app, input_data)

    if result['success']:
        print(f"✅ 成功: {result['output'][:50]}...")
    else:
        print(f"❌ 失败: {result['error']}")
```

---

## 示例3：状态管理与错误

**场景：** 在状态中追踪错误信息

```python
"""
状态管理与错误
演示：在状态中记录错误历史
"""

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, List
from dotenv import load_dotenv
import time

load_dotenv()

print("=== 状态管理与错误 ===\n")

# ===== 1. 定义状态（包含错误历史）=====
class AgentState(TypedDict):
    input: str
    output: str
    error_history: List[dict]
    retry_count: int
    status: str

# ===== 2. 创建节点 =====
llm = ChatOpenAI(model="gpt-4").with_retry(stop_after_attempt=3)

def process_with_error_tracking(state: AgentState) -> AgentState:
    """带错误追踪的处理节点"""
    print(f"  [处理] 重试次数: {state['retry_count']}")

    try:
        response = llm.invoke(state['input'])

        return {
            **state,
            "output": response.content,
            "status": "success"
        }
    except Exception as e:
        # 记录错误
        error_record = {
            "timestamp": time.time(),
            "error_type": type(e).__name__,
            "error_message": str(e),
            "retry_count": state['retry_count']
        }

        new_error_history = state['error_history'] + [error_record]

        print(f"  ❌ 错误: {type(e).__name__}")

        # 检查是否应该重试
        if state['retry_count'] < 3:
            return {
                **state,
                "error_history": new_error_history,
                "retry_count": state['retry_count'] + 1,
                "status": "retry"
            }
        else:
            return {
                **state,
                "error_history": new_error_history,
                "status": "failed"
            }

# ===== 3. 构建图（带条件路由）=====
workflow = StateGraph(AgentState)
workflow.add_node("process", process_with_error_tracking)
workflow.set_entry_point("process")

# 条件路由
def should_retry(state: AgentState) -> str:
    """决定是否重试"""
    if state['status'] == "success":
        return "end"
    elif state['status'] == "retry":
        return "retry"
    else:
        return "end"

workflow.add_conditional_edges(
    "process",
    should_retry,
    {
        "end": END,
        "retry": "process"
    }
)

app = workflow.compile()

print("✅ 图构建完成（带重试逻辑）\n")

# ===== 4. 测试 =====
initial_state = {
    "input": "你好，请介绍一下自己",
    "output": "",
    "error_history": [],
    "retry_count": 0,
    "status": "pending"
}

print("=== 执行图 ===")
result = app.invoke(initial_state)

print(f"\n=== 结果 ===")
print(f"状态: {result['status']}")
print(f"重试次数: {result['retry_count']}")
print(f"错误历史: {len(result['error_history'])} 个错误")

if result['status'] == "success":
    print(f"输出: {result['output'][:50]}...")
else:
    print("错误详情:")
    for i, error in enumerate(result['error_history']):
        print(f"  {i+1}. {error['error_type']}: {error['error_message'][:50]}...")
```

---

## 示例4：多节点错误处理

**场景：** 多个节点的协同错误处理

```python
"""
多节点错误处理
演示：多个节点的错误处理策略
"""

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

print("=== 多节点错误处理 ===\n")

# ===== 1. 定义状态 =====
class AgentState(TypedDict):
    input: str
    step1_output: str
    step2_output: str
    final_output: str
    failed_steps: list

# ===== 2. 创建多个节点 =====
llm = ChatOpenAI(model="gpt-4").with_retry(stop_after_attempt=3)

def step1_node(state: AgentState) -> AgentState:
    """步骤1：分析输入"""
    print("  [步骤1] 分析输入...")

    try:
        response = llm.invoke(f"分析以下输入: {state['input']}")
        return {
            **state,
            "step1_output": response.content
        }
    except Exception as e:
        print(f"  ❌ 步骤1失败: {type(e).__name__}")
        return {
            **state,
            "step1_output": "",
            "failed_steps": state.get('failed_steps', []) + ["step1"]
        }

def step2_node(state: AgentState) -> AgentState:
    """步骤2：生成响应"""
    print("  [步骤2] 生成响应...")

    # 检查步骤1是否成功
    if "step1" in state.get('failed_steps', []):
        print("  ⚠️  步骤1失败，跳过步骤2")
        return {
            **state,
            "step2_output": "",
            "failed_steps": state.get('failed_steps', []) + ["step2"]
        }

    try:
        response = llm.invoke(f"基于分析生成响应: {state['step1_output']}")
        return {
            **state,
            "step2_output": response.content
        }
    except Exception as e:
        print(f"  ❌ 步骤2失败: {type(e).__name__}")
        return {
            **state,
            "step2_output": "",
            "failed_steps": state.get('failed_steps', []) + ["step2"]
        }

def final_node(state: AgentState) -> AgentState:
    """最终节点：汇总结果"""
    print("  [最终] 汇总结果...")

    failed_steps = state.get('failed_steps', [])

    if not failed_steps:
        # 所有步骤成功
        return {
            **state,
            "final_output": state['step2_output']
        }
    else:
        # 有步骤失败，返回降级响应
        return {
            **state,
            "final_output": f"部分步骤失败: {', '.join(failed_steps)}"
        }

# ===== 3. 构建图 =====
workflow = StateGraph(AgentState)

workflow.add_node("step1", step1_node)
workflow.add_node("step2", step2_node)
workflow.add_node("final", final_node)

workflow.set_entry_point("step1")
workflow.add_edge("step1", "step2")
workflow.add_edge("step2", "final")
workflow.add_edge("final", END)

app = workflow.compile()

print("✅ 多节点图构建完成\n")

# ===== 4. 测试 =====
initial_state = {
    "input": "什么是LangChain？",
    "step1_output": "",
    "step2_output": "",
    "final_output": "",
    "failed_steps": []
}

print("=== 执行图 ===")
result = app.invoke(initial_state)

print(f"\n=== 结果 ===")
print(f"失败步骤: {result['failed_steps']}")
print(f"最终输出: {result['final_output'][:100]}...")
```

---

## 示例5：检查点与错误恢复

**场景：** 使用检查点实现错误恢复

```python
"""
检查点与错误恢复
演示：使用LangGraph检查点功能
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

print("=== 检查点与错误恢复 ===\n")

# ===== 1. 定义状态 =====
class AgentState(TypedDict):
    messages: list
    current_step: int
    completed_steps: list

# ===== 2. 创建节点 =====
llm = ChatOpenAI(model="gpt-4").with_retry(stop_after_attempt=3)

def step_node(state: AgentState) -> AgentState:
    """处理步骤节点"""
    current_step = state['current_step']
    print(f"  [步骤{current_step}] 处理中...")

    try:
        message = state['messages'][current_step]
        response = llm.invoke(message)

        return {
            **state,
            "messages": state['messages'] + [response.content],
            "current_step": current_step + 1,
            "completed_steps": state['completed_steps'] + [current_step]
        }
    except Exception as e:
        print(f"  ❌ 步骤{current_step}失败: {type(e).__name__}")
        # 保持当前状态，等待重试
        return state

# ===== 3. 构建图（带检查点）=====
workflow = StateGraph(AgentState)
workflow.add_node("step", step_node)
workflow.set_entry_point("step")

# 条件路由
def should_continue(state: AgentState) -> str:
    """决定是否继续"""
    if state['current_step'] >= len(state['messages']):
        return "end"
    else:
        return "continue"

workflow.add_conditional_edges(
    "step",
    should_continue,
    {
        "end": END,
        "continue": "step"
    }
)

# 使用内存检查点
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

print("✅ 图构建完成（带检查点）\n")

# ===== 4. 测试（带恢复）=====
initial_state = {
    "messages": [
        "什么是Python？",
        "什么是JavaScript？",
        "什么是Go？"
    ],
    "current_step": 0,
    "completed_steps": []
}

config = {"configurable": {"thread_id": "test-thread"}}

print("=== 执行图 ===")
try:
    result = app.invoke(initial_state, config)
    print(f"✅ 成功")
    print(f"完成步骤: {result['completed_steps']}")
    print(f"消息数: {len(result['messages'])}")
except Exception as e:
    print(f"❌ 失败: {type(e).__name__}")

    # 从检查点恢复
    print("\n=== 从检查点恢复 ===")
    try:
        result = app.invoke(initial_state, config)
        print(f"✅ 恢复成功")
        print(f"完成步骤: {result['completed_steps']}")
    except Exception as e:
        print(f"❌ 恢复失败: {type(e).__name__}")
```

---

## 示例6：生产级LangGraph错误处理

**场景：** 完整的生产级配置

```python
"""
生产级LangGraph错误处理
演示：完整的错误处理策略
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from typing import TypedDict, List
from dotenv import load_dotenv
import logging
import time

# ===== 1. 配置日志 =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

print("=== 生产级LangGraph错误处理 ===\n")

# ===== 2. 定义状态 =====
class ProductionState(TypedDict):
    input: str
    output: str
    error_count: int
    error_history: List[dict]
    retry_count: int
    status: str
    execution_time: float

# ===== 3. 创建带完整错误处理的节点 =====
primary_llm = ChatOpenAI(model="gpt-4").with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True
)

fallback_llm = ChatOpenAI(model="gpt-3.5-turbo").with_retry(
    stop_after_attempt=2
)

llm = primary_llm.with_fallbacks([fallback_llm])

def production_node(state: ProductionState) -> ProductionState:
    """生产级处理节点"""
    start_time = time.time()
    logger.info(f"处理输入: {state['input'][:30]}...")

    try:
        response = llm.invoke(state['input'])
        execution_time = time.time() - start_time

        logger.info(f"成功，耗时: {execution_time:.2f}秒")

        return {
            **state,
            "output": response.content,
            "status": "success",
            "execution_time": execution_time
        }

    except Exception as e:
        execution_time = time.time() - start_time

        # 记录错误
        error_record = {
            "timestamp": time.time(),
            "error_type": type(e).__name__,
            "error_message": str(e),
            "execution_time": execution_time
        }

        logger.error(f"失败: {type(e).__name__}, 耗时: {execution_time:.2f}秒")

        return {
            **state,
            "error_count": state['error_count'] + 1,
            "error_history": state['error_history'] + [error_record],
            "retry_count": state['retry_count'] + 1,
            "status": "failed",
            "execution_time": execution_time
        }

# ===== 4. 构建图 =====
workflow = StateGraph(ProductionState)
workflow.add_node("process", production_node)
workflow.set_entry_point("process")
workflow.add_edge("process", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

logger.info("图构建完成\n")

# ===== 5. 生产级执行包装器 =====
def production_invoke(input_data, thread_id="default"):
    """生产级执行"""
    initial_state = {
        "input": input_data,
        "output": "",
        "error_count": 0,
        "error_history": [],
        "retry_count": 0,
        "status": "pending",
        "execution_time": 0.0
    }

    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = app.invoke(initial_state, config)

        # 记录指标
        logger.info(f"执行完成 - 状态: {result['status']}, "
                   f"错误次数: {result['error_count']}, "
                   f"耗时: {result['execution_time']:.2f}秒")

        return result

    except Exception as e:
        logger.error(f"图执行失败: {type(e).__name__}: {e}")
        raise

# ===== 6. 测试 =====
print("=== 测试 ===\n")

test_inputs = [
    "什么是LangChain？",
    "什么是LCEL？",
    "什么是Runnable？"
]

for i, input_data in enumerate(test_inputs):
    print(f"[{i+1}/{len(test_inputs)}] 输入: {input_data}")

    try:
        result = production_invoke(input_data, thread_id=f"test-{i}")

        if result['status'] == "success":
            print(f"✅ 成功: {result['output'][:50]}...")
        else:
            print(f"❌ 失败，错误次数: {result['error_count']}")

        print(f"   耗时: {result['execution_time']:.2f}秒\n")

    except Exception as e:
        print(f"❌ 异常: {type(e).__name__}\n")
```

---

## 运行环境要求

### 依赖安装

```bash
uv add langchain langchain-openai langgraph python-dotenv
```

### 环境变量配置

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 最佳实践总结

### 1. 节点级错误处理

- 为每个节点添加try-except
- 使用with_retry()为LLM调用添加重试
- 在状态中记录错误信息

### 2. 图级错误处理

- 使用safe_invoke包装器
- 实现条件路由处理错误
- 使用检查点实现错误恢复

### 3. 状态管理

- 在状态中追踪错误历史
- 记录重试次数
- 保存执行时间等指标

### 4. 生产级配置

- 重试 + 降级组合
- 完整的日志记录
- 检查点持久化
- 监控和告警

---

**记住：LangGraph的错误处理需要在节点级和图级同时考虑。**
