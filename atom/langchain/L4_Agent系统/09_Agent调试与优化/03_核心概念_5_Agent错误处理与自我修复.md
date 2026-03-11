# 核心概念5：Agent 错误处理与自我修复

> Agent 在生产环境中一定会出错——通过错误回调捕获异常、重试策略恢复执行、优雅降级保底输出、自我修复循环自动纠正，构建真正可靠的 Agent 系统。

---

## 概述

### Agent 错误的特殊性

传统程序的错误是确定性的：同样的输入，同样的 bug。但 Agent 的错误是**不确定性的**：

```
Agent 可能出错的地方：

🧠 LLM 层面
├── API 超时 / 限流（Rate Limit）
├── 输出格式错误（JSON 解析失败）
├── 幻觉（给出错误信息）
└── 拒绝回答（安全过滤触发）

🔧 工具层面
├── 工具调用参数错误
├── 外部 API 不可用
├── 返回数据格式异常
└── 超时

🔄 推理层面
├── 无限循环（Agent 反复调用同一工具）
├── 选错工具
├── 推理偏离目标
└── 上下文窗口溢出
```

### 错误处理的四个层次

```
Layer 4: 自我修复循环（Agent 自己发现并修正错误）    ← 最高级
Layer 3: 优雅降级（出错后提供替代方案）
Layer 2: 重试策略（自动重试可恢复的错误）
Layer 1: 错误捕获与日志（知道出了什么错）            ← 最基础
```

---

## 1. 错误回调捕获（Layer 1）

### LangChain 的错误回调钩子

LangChain 的回调系统为每个组件提供了专门的错误回调（源码：`callbacks/base.py`）：

```python
# LLM 错误
def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs) -> None: ...

# Chain/Agent 错误
def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs) -> None: ...

# 工具错误
def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs) -> None: ...

# 检索器错误
def on_retriever_error(self, error: BaseException, *, run_id: UUID, **kwargs) -> None: ...
```

### 完整的错误监控回调

```python
import logging
from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("agent_errors")

class ErrorMonitorCallback(BaseCallbackHandler):
    """生产级错误监控回调"""
    
    def __init__(self):
        self.errors = []
        self.raise_error = False  # 不中断执行，只记录
    
    def on_llm_error(self, error, *, run_id, **kwargs):
        error_info = {
            "type": "llm_error",
            "error": str(error),
            "error_class": type(error).__name__,
            "run_id": str(run_id),
            "timestamp": datetime.now().isoformat(),
        }
        self.errors.append(error_info)
        logger.error(f"🧠 LLM 错误: {error}")
    
    def on_tool_error(self, error, *, run_id, **kwargs):
        error_info = {
            "type": "tool_error",
            "error": str(error),
            "error_class": type(error).__name__,
            "run_id": str(run_id),
            "timestamp": datetime.now().isoformat(),
        }
        self.errors.append(error_info)
        logger.error(f"🔧 工具错误: {error}")
    
    def on_chain_error(self, error, *, run_id, **kwargs):
        error_info = {
            "type": "chain_error",
            "error": str(error),
            "error_class": type(error).__name__,
            "run_id": str(run_id),
            "timestamp": datetime.now().isoformat(),
        }
        self.errors.append(error_info)
        logger.error(f"🔗 Chain 错误: {error}")
    
    def report(self):
        if not self.errors:
            print("✅ 无错误")
            return
        
        print(f"\n❌ 错误报告（共 {len(self.errors)} 个错误）")
        print("=" * 50)
        for i, err in enumerate(self.errors, 1):
            print(f"  #{i} [{err['type']}] {err['error_class']}: {err['error']}")
            print(f"      时间: {err['timestamp']}")
```

### raise_error 标志的作用

源码中，`CallbackManager.handle_event()` 会根据 `raise_error` 决定是否传播异常：

```python
# 源码简化 (callbacks/manager.py)
def handle_event(handlers, event_name, *args, **kwargs):
    for handler in handlers:
        try:
            getattr(handler, event_name)(*args, **kwargs)
        except Exception as e:
            if handler.raise_error:
                raise  # 传播异常，中断执行
            else:
                logger.warning(f"Error in callback: {e}")  # 仅记录，继续执行
```

**最佳实践**：
- **开发环境**：`raise_error = True`（发现问题立即中断）
- **生产环境**：`raise_error = False`（记录错误，继续服务）

---

## 2. 重试策略（Layer 2）

### 工具级别的重试

```python
import time
from langchain_core.tools import tool

def retry_on_failure(max_retries=3, delay=1.0, backoff=2.0):
    """带指数退避的重试装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_error = None
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        print(f"  ⚠️ 重试 {attempt + 1}/{max_retries}，等待 {current_delay:.1f}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_error
        return wrapper
    return decorator

@tool
@retry_on_failure(max_retries=3, delay=1.0)
def search_api(query: str) -> str:
    """搜索 API（带自动重试）"""
    import requests
    response = requests.get(f"https://api.example.com/search?q={query}", timeout=10)
    response.raise_for_status()
    return response.text
```

### LLM 级别的重试

```python
from langchain_openai import ChatOpenAI

# ChatOpenAI 内置了重试机制
llm = ChatOpenAI(
    model="gpt-4o-mini",
    max_retries=3,           # API 失败时最多重试 3 次
    request_timeout=30,      # 单次请求超时 30 秒
)
```

### 可重试 vs 不可重试的错误

| 错误类型 | 是否可重试 | 处理方式 |
|---------|-----------|---------|
| API 超时 | ✅ 可重试 | 指数退避重试 |
| Rate Limit (429) | ✅ 可重试 | 等待后重试 |
| 网络错误 | ✅ 可重试 | 短暂等待后重试 |
| 认证错误 (401) | ❌ 不可重试 | 检查 API Key |
| 参数错误 (400) | ❌ 不可重试 | 修复参数 |
| 模型不存在 (404) | ❌ 不可重试 | 修复模型名 |
| 内容过滤 | ⚠️ 视情况 | 重写 Prompt |

---

## 3. 优雅降级（Layer 3）

### 核心思想

> 当首选方案失败时，不是直接报错，而是自动切换到备选方案。

### 模型降级

```python
from langchain_openai import ChatOpenAI

def invoke_with_fallback(prompt, primary_model="gpt-4o", fallback_model="gpt-4o-mini"):
    """主模型失败时自动降级到备选模型"""
    try:
        llm = ChatOpenAI(model=primary_model, request_timeout=15)
        return llm.invoke(prompt)
    except Exception as e:
        print(f"⚠️ {primary_model} 失败: {e}")
        print(f"🔄 降级到 {fallback_model}")
        llm = ChatOpenAI(model=fallback_model, request_timeout=30)
        return llm.invoke(prompt)
```

### 使用 RunnableWithFallbacks

```python
from langchain_openai import ChatOpenAI

primary_llm = ChatOpenAI(model="gpt-4o", request_timeout=10)
fallback_llm = ChatOpenAI(model="gpt-4o-mini", request_timeout=30)

# LangChain 内置的 fallback 机制
llm_with_fallback = primary_llm.with_fallbacks([fallback_llm])

# 自动降级：gpt-4o 超时 → 自动切换到 gpt-4o-mini
result = llm_with_fallback.invoke("你好")
```

### 工具降级

```python
from langchain_core.tools import tool

@tool
def search_with_fallback(query: str) -> str:
    """搜索信息（带降级策略）"""
    
    # 首选：实时搜索 API
    try:
        return real_search_api(query)
    except Exception:
        pass
    
    # 备选：本地知识库
    try:
        return local_knowledge_base.search(query)
    except Exception:
        pass
    
    # 兜底：返回提示信息
    return f"抱歉，暂时无法搜索 '{query}'。请稍后重试或换个关键词。"
```

---

## 4. 自我修复循环（Layer 4）

### 核心思想

> 让 Agent 自己发现错误、分析原因、修正行为——从「被动报错」升级为「主动修复」。

### 基础自我修复：输出解析重试

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
parser = JsonOutputParser()

def invoke_with_self_repair(prompt, max_repairs=2):
    """LLM 输出解析失败时，让 LLM 自己修正"""
    
    result = llm.invoke(prompt)
    
    for attempt in range(max_repairs):
        try:
            parsed = parser.parse(result.content)
            return parsed
        except Exception as parse_error:
            print(f"⚠️ 解析失败（尝试 {attempt + 1}），让 LLM 自我修复...")
            
            repair_prompt = f"""你之前的输出格式有误，请修正。

之前的输出：
{result.content}

错误信息：
{str(parse_error)}

请重新输出正确的 JSON 格式："""
            
            result = llm.invoke(repair_prompt)
    
    raise ValueError(f"自我修复失败，{max_repairs} 次尝试后仍无法解析")
```

### 进阶自我修复：工具调用纠错

```python
from langchain_core.callbacks import BaseCallbackHandler

class SelfHealingCallback(BaseCallbackHandler):
    """自我修复回调：检测并修正 Agent 的常见错误模式"""
    
    def __init__(self):
        self.tool_call_history = []
        self.error_count = 0
        self.max_consecutive_errors = 3
    
    def on_agent_action(self, action, **kwargs):
        self.tool_call_history.append(action.tool)
        
        # 检测重复调用（可能是死循环）
        if len(self.tool_call_history) >= 3:
            last_3 = self.tool_call_history[-3:]
            if len(set(last_3)) == 1:
                print(f"⚠️ 检测到重复调用 {last_3[0]}，可能是死循环")
    
    def on_tool_error(self, error, **kwargs):
        self.error_count += 1
        if self.error_count >= self.max_consecutive_errors:
            print(f"❌ 连续 {self.error_count} 次工具错误，建议中断")
```

### LangGraph 中的自我修复模式

```python
# LangGraph 提供了更结构化的自我修复支持
# 通过条件边实现错误检测和重试路由

"""
自我修复状态图：

  ┌──────────┐
  │  开始    │
  └────┬─────┘
       │
  ┌────▼─────┐
  │ 执行工具  │◄────────┐
  └────┬─────┘         │
       │               │
  ┌────▼─────┐    ┌────┴─────┐
  │ 检查结果  │───►│ 修复错误  │
  └────┬─────┘    └──────────┘
       │ 成功
  ┌────▼─────┐
  │  完成    │
  └──────────┘
"""
```

---

## 5. 生产环境错误处理清单

### 必须实现的

- [ ] **错误回调**：记录所有错误到日志系统
- [ ] **LLM 重试**：设置 `max_retries=3` 和合理的 `request_timeout`
- [ ] **模型降级**：使用 `.with_fallbacks()` 配置备选模型
- [ ] **递归限制**：设置 `recursion_limit` 防止无限循环
- [ ] **超时控制**：所有外部调用设置超时

### 推荐实现的

- [ ] **工具重试**：关键工具添加重试装饰器
- [ ] **输出修复**：解析失败时让 LLM 自我修正
- [ ] **循环检测**：监控重复的工具调用模式
- [ ] **成本熔断**：单次请求超过成本阈值时中断
- [ ] **告警通知**：错误率超阈值时通知团队

---

## 6. 常见错误与解决方案速查表

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `RateLimitError` | API 调用频率超限 | 指数退避重试 + 降级 |
| `TimeoutError` | LLM/工具响应超时 | 增加 timeout + 降级 |
| `OutputParserException` | LLM 输出格式错误 | 自我修复 + 更明确的 Prompt |
| `RecursionError` | Agent 无限循环 | 设置 `recursion_limit` |
| `ToolException` | 工具执行失败 | 重试 + 降级工具 |
| `ContextWindowExceeded` | 上下文窗口溢出 | 压缩历史 + 选择性记忆 |
| `AuthenticationError` | API Key 无效 | 检查环境变量 |
| `ContentFilterError` | 内容安全过滤 | 调整 Prompt |

---

## 小结

| 层次 | 策略 | 一句话描述 |
|------|------|-----------|
| **Layer 1** | 错误捕获 | 知道出了什么错（ErrorMonitorCallback） |
| **Layer 2** | 重试策略 | 可恢复的错误自动重试（指数退避） |
| **Layer 3** | 优雅降级 | 不可恢复时切换备选方案（.with_fallbacks()） |
| **Layer 4** | 自我修复 | Agent 自己发现并纠正错误（修复循环） |

**一句话记忆**：错误处理四层塔——捕获→重试→降级→自愈，每一层都是生产环境的安全网。

---

[来源: sourcecode/langchain/libs/core/langchain_core/callbacks/manager.py, reference/search_agent_debug_01.md]
