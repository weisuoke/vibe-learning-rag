# 核心概念2：LangSmith 追踪与可观测性

> LangSmith 是 LangChain 官方的可观测性平台——将回调事件持久化为可视化的追踪树，让你真正「看见」Agent 在想什么、做什么、花了多少钱。

---

## 概述

### 为什么 LangSmith 是 2025-2026 年的首选调试工具？

截至 2026 年，LangChain 官方文档的调试页面已经从传统的 `set_debug(True)` 转向以 **LangSmith** 为核心的调试方案。原因很简单：

```
传统调试（set_debug / print）       LangSmith 追踪
┌──────────────────────┐         ┌──────────────────────┐
│ ✗ 只能看文字日志        │         │ ✓ 可视化追踪树          │
│ ✗ 日志刷屏、难以定位     │         │ ✓ 层级关系一目了然       │
│ ✗ 无法持久化保存        │         │ ✓ 历史记录可回放        │
│ ✗ 无法对比不同运行      │         │ ✓ 两次运行并排对比       │
│ ✗ 不知道花了多少钱      │         │ ✓ Token 用量 + 成本追踪  │
│ ✗ 无法分享给团队       │         │ ✓ 分享链接一键协作       │
└──────────────────────┘         └──────────────────────┘
```

**一句话理解：** 如果 `set_debug(True)` 是「听 Agent 自言自语」，那 LangSmith 追踪就是「给 Agent 装上行车记录仪」——完整记录、可回放、可分析。

### 与回调系统的关系

LangSmith 追踪并不是一套全新的系统，而是**构建在回调系统之上**的：

```
回调系统（底层）               LangSmith（上层）
┌────────────────┐           ┌─────────────────────┐
│ on_llm_start   │           │                     │
│ on_llm_end     │ ──事件──▶ │  LangChainTracer    │
│ on_tool_start  │   流入    │  (特殊的回调处理器)   │
│ on_tool_end    │           │       │              │
│ on_chain_start │           │       ▼              │
│ on_chain_end   │           │  持久化到 LangSmith  │
│ on_agent_action│           │  服务器              │
│ on_agent_finish│           │                     │
└────────────────┘           └─────────────────────┘
```

核心概念1介绍的回调系统产生**临时事件**（打印完就没了），而 LangSmith 的 `LangChainTracer` 是一个特殊的回调处理器，它把这些临时事件**持久化**为可查看、可分析的追踪记录。

---

## 1. 从回调到追踪

### 回调 vs 追踪：本质区别

| 维度 | 回调（Callback） | 追踪（Trace） |
|------|------------------|---------------|
| **生命周期** | 临时的，用完即弃 | 持久化，永久保存 |
| **结构** | 扁平的事件流 | 树状的层级结构 |
| **关联性** | 独立事件 | 父子关系清晰 |
| **查看方式** | 控制台文字 | Web UI 可视化 |
| **分析能力** | 无 | 过滤、对比、统计 |
| **协作** | 不支持 | 分享链接 |

**前端类比：** 回调就像 `console.log()`，追踪就像 Chrome DevTools 的 Performance 面板——都是记录信息，但后者结构化、可视化、可分析。

### 追踪树的概念

每一次 Agent 执行会生成一棵**追踪树（Trace Tree）**，树的每个节点是一个 **Run（运行）**：

```
追踪树示例：
═══════════════════════════════════════════════
🏠 Agent Run (root)                         [总计 3.4s, $0.002]
 │
 ├── 🧠 LLM Call #1 (planning)
 │    ├── input: "用户问了关于LangChain调试的问题..."
 │    ├── output: "我需要先搜索相关资料..."
 │    ├── tokens: 150 in / 80 out
 │    └── latency: 1.2s
 │
 ├── 🔧 Tool Call: search_docs
 │    ├── input: "LangChain debugging best practices"
 │    ├── output: [3 documents found...]
 │    └── latency: 0.8s
 │
 ├── 🧠 LLM Call #2 (reasoning with context)
 │    ├── input: "根据搜索结果，回答用户问题..."
 │    ├── output: "LangChain 调试有三种方式..."
 │    ├── tokens: 500 in / 200 out
 │    └── latency: 0.9s
 │
 └── ✅ Final Output
      ├── answer: "LangChain 调试有三种方式：1...."
      └── total_cost: $0.002
═══════════════════════════════════════════════
```

**关键概念：**
- **根节点（Root Run）**：整个 Agent 调用，记录总耗时和总成本
- **子节点（Child Run）**：每次 LLM 调用、Tool 调用都是子节点
- **父子关系**：通过 `run_id` 和 `parent_run_id` 关联
- **每个节点包含**：输入、输出、耗时、Token 数、错误信息（如果有）

### 源码中的层级追踪

在 LangChain 源码 `tracers/core.py` 中，`_TracerCore` 类维护了这种层级关系：

```python
class _TracerCore:
    """追踪核心，维护运行的层级关系"""

    def __init__(self):
        self.run_map: dict[str, Run] = {}      # run_id -> Run 对象
        self.order_map: dict[str, int] = {}     # run_id -> 执行顺序

    def _create_chain_run(self, run):
        """创建新的链运行，自动建立父子关系"""
        if run.parent_run_id:
            parent_run = self.run_map.get(str(run.parent_run_id))
            if parent_run:
                self._add_child_run(parent_run, run)
        self.run_map[str(run.id)] = run

    def _add_child_run(self, parent_run, child_run):
        """将子运行添加到父运行的 child_runs 列表"""
        parent_run.child_runs.append(child_run)

    def _get_stacktrace(self, run):
        """获取从根节点到当前节点的完整路径"""
        # 用于错误报告和调试
        ...
```

**日常生活类比：** 这就像一个家族族谱——每个人（Run）知道自己的父母（parent_run_id），也知道自己的孩子（child_runs）。`_get_stacktrace()` 就是追溯到「太爷爷」。

---

## 2. LangSmith 快速配置

### 方式一：环境变量（推荐）

最简单的方式，只需设置 4 个环境变量：

```python
import os

# ===== 必需配置 =====
os.environ["LANGCHAIN_TRACING_V2"] = "true"      # 开启追踪
os.environ["LANGCHAIN_API_KEY"] = "ls-your-key"    # LangSmith API Key

# ===== 可选配置 =====
os.environ["LANGCHAIN_PROJECT"] = "my-agent-debug"  # 项目名（默认 "default"）
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"  # 端点地址
```

**配置完成后，所有 LangChain 操作自动被追踪**——不需要修改任何业务代码！

```python
# 你的业务代码完全不需要改
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor

llm = ChatOpenAI(model="gpt-4o-mini")
agent = AgentExecutor(agent=create_react_agent(llm, tools, prompt), tools=tools)

# 直接调用，追踪自动生效
result = agent.invoke({"input": "北京今天天气怎么样？"})
# 打开 LangSmith 网页 → 就能看到追踪树！
```

**推荐使用 `.env` 文件管理：**

```bash
# .env 文件
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls-your-key
LANGCHAIN_PROJECT=my-agent-debug
```

```python
from dotenv import load_dotenv
load_dotenv()  # 自动加载 .env 文件中的环境变量
```

### 方式二：编程式启用（精细控制）

当你只想追踪**部分代码**时，使用上下文管理器：

```python
from langchain_core.tracers.context import tracing_v2_enabled

# 只有 with 块内的代码会被追踪
with tracing_v2_enabled(project_name="debug-session") as tracer:
    result = agent.invoke({"input": "帮我分析这个错误"})

    # 获取追踪的 URL，可以直接在浏览器打开
    run_url = tracer.get_run_url()
    print(f"🔗 查看追踪: {run_url}")

# with 块外的代码不会被追踪
result2 = agent.invoke({"input": "这个调用不会被追踪"})
```

**使用场景：**
- 只追踪可疑的代码段
- 临时调试特定问题
- 不想全局开启追踪（节省成本）

### 方式三：直接实例化 Tracer

最底层的方式，完全手动控制：

```python
from langchain_core.tracers.langchain import LangChainTracer

# 创建 tracer 实例
tracer = LangChainTracer(
    project_name="manual-debug",
    tags=["v2", "production-test"],
)

# 通过 config 传入
result = agent.invoke(
    {"input": "手动控制追踪"},
    config={"callbacks": [tracer]}
)

# 获取运行 URL
print(f"🔗 追踪链接: {tracer.get_run_url()}")
```

### 三种方式对比

| 方式 | 作用范围 | 代码侵入性 | 适用场景 |
|------|---------|-----------|---------|
| 环境变量 | 全局 | 零侵入 | 开发阶段、全量追踪 |
| `tracing_v2_enabled` | 代码块 | 低侵入 | 临时调试、部分追踪 |
| 手动实例化 | 单次调用 | 中侵入 | 精细控制、自定义标签 |

---

## 3. 追踪系统架构（源码解析）

### 3.1 BaseTracer 与 _TracerCore

追踪系统的类继承结构：

```
_TracerCore                    BaseCallbackHandler
   │                              │
   │ (追踪核心逻辑)                │ (回调接口)
   │                              │
   └──────────┬───────────────────┘
              │
         BaseTracer
              │
              │ (继承两者)
              │
    ┌─────────┼──────────┐
    │         │          │
LangChain  EventStream  LogStream
 Tracer     Tracer       Tracer
(→ LangSmith)(→ astream_events)(→ astream_log)
```

**`_TracerCore` 的核心职责：**

```python
# 源码简化版 - tracers/core.py

class _TracerCore:
    """所有 Tracer 的核心逻辑"""

    run_map: dict[str, Run]    # 保存所有活跃的 Run

    def _start_trace(self, run: Run) -> None:
        """开始追踪一个 Run"""
        # 1. 如果有父 Run，建立父子关系
        if run.parent_run_id and str(run.parent_run_id) in self.run_map:
            parent = self.run_map[str(run.parent_run_id)]
            parent.child_runs.append(run)

        # 2. 注册到 run_map
        self.run_map[str(run.id)] = run

    def _end_trace(self, run: Run) -> None:
        """结束追踪一个 Run"""
        # 1. 从 run_map 移除
        self.run_map.pop(str(run.id), None)

        # 2. 如果是根 Run（没有父节点），触发持久化
        if not run.parent_run_id:
            self._persist_run(run)  # 由子类实现
```

### 3.2 LangChainTracer（LangSmith 集成）

`LangChainTracer` 是连接 LangChain 和 LangSmith 的桥梁：

```python
# 源码简化版 - tracers/langchain.py

class LangChainTracer(BaseTracer):
    """将运行数据持久化到 LangSmith 平台"""

    def __init__(
        self,
        project_name: str | None = None,    # 项目名
        example_id: UUID | None = None,      # 关联的示例 ID（用于评估）
        tags: list[str] | None = None,       # 标签
        client: Client | None = None,        # LangSmith 客户端
    ):
        self.project_name = project_name or os.environ.get(
            "LANGCHAIN_PROJECT", "default"
        )
        self.client = client or get_langsmith_client()

    def _persist_run(self, run: Run) -> None:
        """将完整的运行树发送到 LangSmith"""
        self.client.create_run(
            name=run.name,
            inputs=run.inputs,
            outputs=run.outputs,
            run_type=run.run_type,
            start_time=run.start_time,
            end_time=run.end_time,
            error=run.error,
            tags=run.tags,
            extra=run.extra,      # 包含 token 用量等
            child_runs=run.child_runs,
            project_name=self.project_name,
        )

    def get_run_url(self) -> str | None:
        """返回 LangSmith 上查看该运行的 URL"""
        if self.latest_run:
            return self.client.get_run_url(
                run=self.latest_run,
                project_name=self.project_name,
            )
        return None
```

**关键流程：**

```
Agent.invoke()
    │
    ▼
CallbackManager 触发事件
    │
    ├── on_chain_start()  ──▶ LangChainTracer._start_trace()
    ├── on_llm_start()    ──▶ 创建子 Run，记录输入
    ├── on_llm_end()      ──▶ 记录输出、Token 数
    ├── on_tool_start()   ──▶ 创建工具子 Run
    ├── on_tool_end()     ──▶ 记录工具输出
    └── on_chain_end()    ──▶ LangChainTracer._end_trace()
                                     │
                                     ▼
                              _persist_run()
                                     │
                                     ▼
                          发送到 LangSmith 服务器
```

### 3.3 运行数据模型（Run 对象）

每个 Run 包含以下关键信息：

```python
# Run 对象的核心字段
{
    "id": "uuid-xxxx",                    # 唯一标识
    "parent_run_id": "uuid-parent",       # 父运行 ID
    "name": "AgentExecutor",              # 运行名称
    "run_type": "chain",                  # 类型: chain/llm/tool/retriever
    "inputs": {"input": "用户问题"},       # 输入
    "outputs": {"output": "Agent回答"},    # 输出
    "error": None,                        # 错误信息（如果失败）
    "start_time": "2026-03-06T10:00:00",  # 开始时间
    "end_time": "2026-03-06T10:00:03",    # 结束时间
    "extra": {
        "tokens": {
            "prompt_tokens": 150,         # 输入 Token 数
            "completion_tokens": 80,      # 输出 Token 数
            "total_tokens": 230,          # 总 Token 数
        },
        "model_name": "gpt-4o-mini",      # 模型名
    },
    "tags": ["debug", "v2"],              # 标签
    "metadata": {"user_id": "u123"},      # 元数据
    "child_runs": [...]                   # 子运行列表
}
```

---

## 4. 事件流与日志流

除了 LangSmith 的 Web UI，LangChain 还提供了**编程式**的追踪数据获取方式。

### 4.1 `astream_events()` 实时事件流

异步流式获取每个执行事件，适合**实时监控**：

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search_docs(query: str) -> str:
    """搜索文档"""
    return f"找到关于 '{query}' 的 3 篇文档"

llm = ChatOpenAI(model="gpt-4o-mini")

async def monitor_agent():
    """实时监控 Agent 执行的每个事件"""
    async for event in agent.astream_events(
        {"input": "帮我搜索 LangChain 调试方法"},
        version="v2",  # 使用 v2 事件格式
    ):
        kind = event["event"]
        name = event["name"]

        if kind == "on_chain_start":
            print(f"🔗 链开始: {name}")
        elif kind == "on_llm_start":
            print(f"🧠 LLM 开始思考: {name}")
        elif kind == "on_llm_end":
            output = event["data"]["output"]
            print(f"🧠 LLM 思考完成: {name}")
        elif kind == "on_tool_start":
            tool_input = event["data"].get("input", "")
            print(f"🔧 调用工具: {name}({tool_input})")
        elif kind == "on_tool_end":
            tool_output = event["data"].get("output", "")
            print(f"🔧 工具返回: {name} → {tool_output[:100]}")
        elif kind == "on_chain_end":
            print(f"✅ 链结束: {name}")

asyncio.run(monitor_agent())
```

**预期输出：**
```
🔗 链开始: AgentExecutor
🧠 LLM 开始思考: ChatOpenAI
🧠 LLM 思考完成: ChatOpenAI
🔧 调用工具: search_docs(LangChain 调试方法)
🔧 工具返回: search_docs → 找到关于 'LangChain 调试方法' 的 3 篇文档
🧠 LLM 开始思考: ChatOpenAI
🧠 LLM 思考完成: ChatOpenAI
✅ 链结束: AgentExecutor
```

**源码中的 RunInfo 类型（event_stream.py）：**

```python
class RunInfo(TypedDict):
    """每个事件携带的运行信息"""
    name: str                    # 运行名称
    tags: list[str]              # 标签
    metadata: dict[str, Any]     # 元数据
    run_type: str                # 类型: chain/llm/tool
    inputs: Any                  # 输入数据
    parent_run_id: UUID | None   # 父运行 ID
    tool_call_id: str | None     # 工具调用 ID
```

### 4.2 `collect_runs()` 编程式收集

当你想**在代码中分析运行数据**而不是在 Web UI 中查看时：

```python
from langchain_core.tracers.context import collect_runs

def analyze_agent_run():
    """收集并分析 Agent 的运行数据"""
    with collect_runs() as runs_cb:
        result = agent.invoke({"input": "什么是 RAG？"})

        # 遍历所有收集到的运行
        for run in runs_cb.traced_runs:
            print(f"📊 运行: {run.name}")
            print(f"   类型: {run.run_type}")
            print(f"   耗时: {run.end_time - run.start_time}")

            if run.error:
                print(f"   ❌ 错误: {run.error}")

            # 分析 Token 用量
            if hasattr(run, 'extra') and run.extra:
                tokens = run.extra.get('tokens', {})
                if tokens:
                    print(f"   Token: {tokens.get('total_tokens', 'N/A')}")

            # 递归打印子运行
            for child in (run.child_runs or []):
                print(f"   └── 子运行: {child.name} ({child.run_type})")

analyze_agent_run()
```

**预期输出：**
```
📊 运行: AgentExecutor
   类型: chain
   耗时: 0:00:03.421000
   └── 子运行: ChatOpenAI (llm)
   └── 子运行: search_docs (tool)
   └── 子运行: ChatOpenAI (llm)
```

### 4.3 `astream_log()` JSON Patch 日志流

基于 JSON Patch 的增量更新流，适合需要**精细追踪状态变化**的场景：

```python
async def stream_log_demo():
    """使用 astream_log 获取增量日志"""
    async for log_patch in agent.astream_log(
        {"input": "解释一下 Embedding"},
        include_names=["ChatOpenAI"],  # 只追踪 LLM 调用
    ):
        # log_patch 是 JSON Patch 格式
        for op in log_patch.ops:
            print(f"操作: {op['op']} 路径: {op['path']}")
            if op['op'] == 'add' and 'value' in op:
                print(f"  值: {str(op['value'])[:100]}")
```

---

## 5. LangSmith 核心功能

### 5.1 追踪可视化

在 LangSmith Web UI 中，追踪以**树状结构**呈现：

```
LangSmith UI 布局：
┌─────────────────────────────────────────────────────┐
│ 项目: my-agent-debug                     [过滤] [搜索]│
├─────────────────────────────────────────────────────┤
│ 运行列表                                             │
│ ┌─────────────────────────────────────────────────┐ │
│ │ ✅ AgentExecutor  3.4s  $0.002  [debug][v2]    │ │
│ │ ✅ AgentExecutor  2.1s  $0.001  [debug]        │ │
│ │ ❌ AgentExecutor  5.2s  ERROR   [prod]         │ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
│ 选中运行的追踪树                                      │
│ ┌─────────────────────────────────────────────────┐ │
│ │ 🏠 AgentExecutor (3.4s)                        │ │
│ │  ├─ 🧠 ChatOpenAI (1.2s, 230 tokens)          │ │
│ │  ├─ 🔧 search_docs (0.8s)                     │ │
│ │  └─ 🧠 ChatOpenAI (0.9s, 700 tokens)          │ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
│ 选中节点的详情                                        │
│ ┌─────────────────────────────────────────────────┐ │
│ │ Input:  "用户问了关于LangChain调试..."           │ │
│ │ Output: "我建议使用 LangSmith..."                │ │
│ │ Tokens: 150 in / 80 out                        │ │
│ │ Latency: 1.2s                                  │ │
│ │ Model: gpt-4o-mini                             │ │
│ └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

**你能从追踪中看到什么：**
- Agent 每一步的**思考过程**（LLM 的完整输入/输出）
- 选择了哪个 **Tool**，传了什么参数
- 每步的**耗时和 Token 用量**
- 如果出错，**具体在哪一步出的错**

### 5.2 追踪过滤与搜索

用标签和元数据组织追踪，方便后续查找：

```python
# 添加标签和元数据，便于在 LangSmith 中过滤
result = agent.invoke(
    {"input": "帮我分析这个 Bug"},
    config={
        "tags": ["debug", "bug-fix", "v2.1"],          # 标签过滤
        "metadata": {
            "user_id": "user_123",                      # 按用户过滤
            "session_id": "sess_abc",                   # 按会话过滤
            "feature": "code-analysis",                 # 按功能过滤
            "environment": "staging",                   # 按环境过滤
        },
        "run_name": "BugAnalysis-2026-03-06",           # 自定义运行名称
    }
)
```

**在 LangSmith UI 中你可以按以下条件过滤：**
- 标签：`tag:debug AND tag:v2.1`
- 元数据：`metadata.user_id = "user_123"`
- 运行类型：`run_type:chain / llm / tool`
- 时间范围：最近 1 小时 / 24 小时 / 7 天
- 状态：成功 / 失败 / 运行中
- 延迟：`latency > 5s`（找出慢请求）

### 5.3 追踪对比

当 Agent 对同一问题给出不同答案时，**对比两次运行**找出差异：

```
对比视图：
┌──────────────────────┬──────────────────────┐
│ 运行 A (正确)         │ 运行 B (错误)         │
├──────────────────────┼──────────────────────┤
│ 🧠 LLM: "我应该使用   │ 🧠 LLM: "我直接回答   │
│    search 工具查找"   │    用户的问题"         │
│                      │                      │
│ 🔧 Tool: search_docs │ ⚠️ 没有调用 Tool!     │
│    → 3 results       │                      │
│                      │                      │
│ 🧠 LLM: "根据搜索    │ 🧠 LLM: "RAG就是..."  │
│    结果，RAG 是..."   │    (幻觉！)            │
│                      │                      │
│ ✅ 答案正确           │ ❌ 答案有幻觉           │
└──────────────────────┴──────────────────────┘
```

通过对比可以发现：运行 B 中 LLM 跳过了 Tool 调用，直接「编造」答案。

### 5.4 成本追踪

LangSmith 自动追踪每次 API 调用的 Token 用量和成本：

```
项目: my-agent-debug
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
过去 24 小时:
  总调用次数: 1,247
  总 Token 数: 2,345,678
  总成本: $4.56
  平均每次成本: $0.0037

按模型分布:
  gpt-4o:      $3.20 (70%)
  gpt-4o-mini: $1.36 (30%)

成本趋势: ↗️ 比昨天增加 15%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 5.5 Polly AI 助手（2025-2026 新功能）

LangSmith 内置的 **AI 助手 Polly**，用 AI 分析 AI 的执行痕迹：

```
你可以问 Polly：
- "为什么这次运行比平时慢 3 倍？"
- "最近失败率上升的原因是什么？"
- "哪些 Tool 调用消耗了最多 Token？"
- "对比这两次运行，找出关键差异"

Polly 的回答：
"我分析了最近 100 次运行，发现：
 1. 慢请求集中在 search_docs 工具返回超过 5000 字的情况
 2. 当搜索结果过长时，后续 LLM 调用的 Token 数增加 3x
 3. 建议：对搜索结果进行截断或摘要处理"
```

### 5.6 监控与告警

生产环境的监控配置：

```python
# LangSmith 支持通过 SDK 查询追踪数据
from langsmith import Client

client = Client()

# 查询最近的失败运行
failed_runs = client.list_runs(
    project_name="production-agent",
    is_error=True,
    start_time=datetime.now() - timedelta(hours=1),
)

for run in failed_runs:
    print(f"❌ 失败: {run.name}")
    print(f"   错误: {run.error}")
    print(f"   时间: {run.start_time}")

# 统计成功率
all_runs = list(client.list_runs(
    project_name="production-agent",
    start_time=datetime.now() - timedelta(hours=24),
))
success_count = sum(1 for r in all_runs if not r.error)
total_count = len(all_runs)
print(f"📊 24h 成功率: {success_count}/{total_count} = {success_count/total_count:.1%}")
```

---

## 6. 替代方案

### 6.1 Langfuse（开源替代）

| 维度 | LangSmith | Langfuse |
|------|-----------|----------|
| **开源** | ❌ 闭源（可自托管） | ✅ 完全开源 |
| **集成** | LangChain 原生 | 需额外配置 |
| **功能** | 最全面 | 核心功能齐全 |
| **成本** | 免费额度有限 | 自托管免费 |
| **社区** | 官方支持 | 社区驱动 |
| **Polly AI** | ✅ 有 | ❌ 没有 |

```python
# Langfuse 集成示例
from langfuse.callback import CallbackHandler as LangfuseHandler

langfuse_handler = LangfuseHandler(
    public_key="pk-xxx",
    secret_key="sk-xxx",
    host="https://cloud.langfuse.com",
)

# 作为回调传入
result = agent.invoke(
    {"input": "测试问题"},
    config={"callbacks": [langfuse_handler]}
)
```

### 6.2 自建追踪（Custom Tracer）

如果你有特殊需求（如写入内部数据库），可以继承 `BaseTracer`：

```python
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
import json
from datetime import datetime

class MyCustomTracer(BaseTracer):
    """自定义追踪器：将运行数据保存到本地文件"""

    def __init__(self, log_file: str = "agent_traces.jsonl"):
        super().__init__()
        self.log_file = log_file

    def _persist_run(self, run: Run) -> None:
        """将运行数据追加写入 JSONL 文件"""
        run_data = {
            "id": str(run.id),
            "name": run.name,
            "run_type": run.run_type,
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "end_time": run.end_time.isoformat() if run.end_time else None,
            "inputs": str(run.inputs)[:500],      # 截断防止过大
            "outputs": str(run.outputs)[:500],
            "error": run.error,
            "tags": run.tags,
            "child_count": len(run.child_runs) if run.child_runs else 0,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(run_data, ensure_ascii=False) + "\n")

        print(f"📝 追踪已保存: {run.name} → {self.log_file}")


# 使用自定义追踪器
custom_tracer = MyCustomTracer(log_file="my_agent_traces.jsonl")

result = agent.invoke(
    {"input": "使用自定义追踪器"},
    config={"callbacks": [custom_tracer]}
)
```

**适用场景：**
- 公司不允许数据外传到 LangSmith
- 需要写入内部的 Elasticsearch / ClickHouse
- 只想追踪特定字段，减少数据量
- 自定义告警逻辑

---

## 7. 最佳实践

### 7.1 开发阶段

```python
import os
from dotenv import load_dotenv

load_dotenv()

# ===== 开发环境推荐配置 =====
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "dev-agent-debug"

# 给每次运行打标签，方便回溯
def debug_agent(user_input: str, feature_tag: str):
    """带标签的调试运行"""
    result = agent.invoke(
        {"input": user_input},
        config={
            "tags": [
                f"feature:{feature_tag}",
                f"dev:{os.getenv('USER', 'unknown')}",
            ],
            "metadata": {
                "debug_reason": "testing new tool",
                "git_branch": "feature/new-search",
            },
        }
    )
    return result
```

**开发阶段清单：**
- ✅ 全量追踪（不采样）
- ✅ 每个功能用不同标签
- ✅ 元数据中记录开发者和分支
- ✅ 频繁查看追踪树，理解 Agent 行为
- ✅ 使用追踪对比验证代码修改的效果

### 7.2 生产阶段

```python
import random

# ===== 生产环境推荐配置 =====
SAMPLING_RATE = 0.1  # 只追踪 10% 的请求

def should_trace() -> bool:
    """采样决策：10% 的请求被追踪"""
    return random.random() < SAMPLING_RATE

def production_agent_call(user_input: str, user_id: str):
    """生产环境的 Agent 调用"""
    config = {
        "tags": ["production", "v2.1"],
        "metadata": {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
        },
    }

    if should_trace():
        # 被采样到的请求，启用追踪
        from langchain_core.tracers.context import tracing_v2_enabled
        with tracing_v2_enabled(project_name="production-agent"):
            return agent.invoke({"input": user_input}, config=config)
    else:
        # 未被采样的请求，正常执行
        return agent.invoke({"input": user_input}, config=config)
```

**生产阶段清单：**
- ✅ 采样追踪（10%-20%），避免成本过高
- ✅ 错误请求 100% 追踪（出错时自动开启）
- ✅ 设置成本告警阈值
- ✅ 监控延迟 P99
- ✅ 定期检查失败率趋势
- ✅ 敏感数据脱敏后再追踪

### 7.3 标签命名规范

```python
# 推荐的标签命名规范
tags = [
    "env:production",          # 环境
    "version:v2.1",            # 版本
    "feature:code-analysis",   # 功能
    "model:gpt-4o-mini",       # 模型
    "priority:high",           # 优先级
]

# 推荐的元数据规范
metadata = {
    "user_id": "u_123",            # 用户 ID（脱敏）
    "session_id": "sess_abc",      # 会话 ID
    "request_id": "req_xyz",       # 请求 ID（关联其他日志系统）
    "ab_test_group": "control",    # A/B 测试分组
}
```

---

## 小结

### 核心要点

| 要点 | 说明 |
|------|------|
| **LangSmith = 持久化的回调** | 本质是一个特殊的回调处理器，把临时事件变成可查看的追踪记录 |
| **追踪树 = Agent 的行车记录仪** | 完整记录每一步的输入、输出、耗时、Token、错误 |
| **零侵入配置** | 设置环境变量即可，不需要修改业务代码 |
| **编程式访问** | `collect_runs()` 和 `astream_events()` 让你在代码中分析追踪数据 |
| **开发全量、生产采样** | 开发阶段追踪所有请求，生产阶段采样 10%-20% |

### 一句话记忆

> **LangSmith 追踪 = Chrome DevTools for AI Agent。回调产生事件，追踪器持久化事件，LangSmith 可视化事件。**

### 下一步学习

- **核心概念3**：性能分析与瓶颈定位——学会用追踪数据找出 Agent 的性能瓶颈
- **核心概念4**：成本优化策略——学会用缓存和模型路由降低 LLM 调用成本
- **实战代码 场景3**：生产环境监控系统——从零搭建基于 LangSmith 的监控告警

---

## 学习检查

- [ ] 能说出回调（Callback）和追踪（Trace）的三个核心区别
- [ ] 能用环境变量方式配置 LangSmith 追踪
- [ ] 能用 `tracing_v2_enabled()` 局部启用追踪
- [ ] 理解 `LangChainTracer` 在源码中的工作流程
- [ ] 能用 `collect_runs()` 编程式收集运行数据
- [ ] 知道 `astream_events()` 和 `astream_log()` 的区别和适用场景
- [ ] 理解开发阶段和生产阶段的追踪策略差异
- [ ] 能说出至少一种 LangSmith 的替代方案

---

[来源: sourcecode/langchain/libs/core/langchain_core/tracers/, reference/fetch_langsmith_observability_01.md, reference/fetch_langchain_debugging_01.md, reference/search_agent_debug_01.md]
