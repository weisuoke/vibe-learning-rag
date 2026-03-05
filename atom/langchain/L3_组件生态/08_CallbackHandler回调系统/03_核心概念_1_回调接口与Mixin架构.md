# CallbackHandler回调系统 - 核心概念1：回调接口与Mixin架构

## 概述

**本文讲解 LangChain CallbackHandler 的核心设计：Mixin 架构模式**

LangChain 采用 **Mixin 模式** 设计回调系统，将不同组件的回调方法分散到不同的 Mixin 类中，通过组合实现灵活的回调处理。

---

## 核心概念1：BaseCallbackHandler 设计

**BaseCallbackHandler 是所有回调处理器的基类，通过继承多个 Mixin 提供完整的回调接口。**

### 设计哲学

```python
# LangChain 的设计思路
class BaseCallbackHandler(
    LLMManagerMixin,        # LLM 相关回调
    ChainManagerMixin,      # Chain 相关回调
    ToolManagerMixin,       # Tool 相关回调
    RetrieverManagerMixin,  # Retriever 相关回调
    CallbackManagerMixin,   # 启动相关回调
    RunManagerMixin,        # 运行时相关回调
):
    """所有回调处理器的基类"""
    
    raise_error: bool = False  # 是否抛出异常
    run_inline: bool = False   # 是否内联运行
```

**核心特性**：
1. **多重继承**：继承 6 个 Mixin，获得所有回调方法
2. **可选实现**：只需实现需要的回调方法
3. **统一接口**：所有回调方法使用一致的参数模式

### 关键属性

```python
class BaseCallbackHandler:
    # 控制属性
    raise_error: bool = False  # 回调错误时是否抛出异常
    run_inline: bool = False   # 是否内联运行（同步执行）
    
    # 忽略特定类型回调
    @property
    def ignore_llm(self) -> bool:
        """是否忽略 LLM 回调"""
        return False
    
    @property
    def ignore_chain(self) -> bool:
        """是否忽略 Chain 回调"""
        return False
    
    @property
    def ignore_agent(self) -> bool:
        """是否忽略 Agent 回调"""
        return False
    
    @property
    def ignore_retriever(self) -> bool:
        """是否忽略 Retriever 回调"""
        return False
```

**在 RAG 开发中的应用**：
```python
class RAGCallbackHandler(BaseCallbackHandler):
    """只监控 Retriever 和 LLM，忽略其他组件"""
    
    @property
    def ignore_chain(self) -> bool:
        return True  # 忽略 Chain 回调
    
    @property
    def ignore_agent(self) -> bool:
        return True  # 忽略 Agent 回调
    
    def on_retriever_end(self, documents, **kwargs):
        print(f"检索到 {len(documents)} 个文档")
    
    def on_llm_end(self, response, **kwargs):
        tokens = response.llm_output["token_usage"]["total_tokens"]
        print(f"LLM 使用 {tokens} tokens")
```

---

## 核心概念2：6个 Mixin 的职责划分

**LangChain 将回调方法按组件类型分散到 6 个 Mixin 中，每个 Mixin 负责一类组件的回调。**

### Mixin 1: LLMManagerMixin

**职责**：处理 LLM 相关的回调事件

```python
class LLMManagerMixin:
    """LLM 回调方法"""
    
    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: GenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """流式输出时，每个新 token 触发"""
        pass
    
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """LLM 完成时触发"""
        pass
    
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """LLM 错误时触发"""
        pass
```

**RAG 应用场景**：
- 追踪 LLM 生成过程
- 统计 token 用量
- 实时显示流式输出
- 捕获 LLM 错误

### Mixin 2: ChainManagerMixin

**职责**：处理 Chain 和 Agent 相关的回调事件

```python
class ChainManagerMixin:
    """Chain 和 Agent 回调方法"""
    
    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Chain 结束时触发"""
        pass
    
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Chain 错误时触发"""
        pass
    
    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Agent 执行动作时触发"""
        pass
    
    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Agent 完成时触发"""
        pass
```

**RAG 应用场景**：
- 追踪 RAG 管道执行
- 监控 Agent 推理过程
- 记录中间步骤输出

### Mixin 3: ToolManagerMixin

**职责**：处理 Tool 相关的回调事件

```python
class ToolManagerMixin:
    """Tool 回调方法"""
    
    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Tool 结束时触发"""
        pass
    
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Tool 错误时触发"""
        pass
```

**RAG 应用场景**：
- 监控工具调用（如搜索工具、计算工具）
- 追踪工具参数和返回值
- 调试 Agent 工具使用

### Mixin 4: RetrieverManagerMixin

**职责**：处理 Retriever 相关的回调事件

```python
class RetrieverManagerMixin:
    """Retriever 回调方法"""
    
    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Retriever 结束时触发"""
        pass
    
    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Retriever 错误时触发"""
        pass
```

**RAG 应用场景**：
- 监控文档检索过程
- 记录检索结果和相似度
- 调试检索失败问题

### Mixin 5: CallbackManagerMixin

**职责**：处理所有组件的启动事件

```python
class CallbackManagerMixin:
    """启动相关回调方法"""
    
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """LLM 开始时触发"""
        pass
    
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Chat Model 开始时触发"""
        pass
    
    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Retriever 开始时触发"""
        pass
    
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Chain 开始时触发"""
        pass
    
    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Tool 开始时触发"""
        pass
```

**RAG 应用场景**：
- 记录每个组件的启动时间
- 追踪输入参数
- 生成执行链路追踪

### Mixin 6: RunManagerMixin

**职责**：处理运行时的通用事件

```python
class RunManagerMixin:
    """运行时相关回调方法"""
    
    def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """任意文本输出时触发"""
        pass
    
    def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """重试时触发"""
        pass
    
    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Any:
        """自定义事件时触发"""
        pass
```

**RAG 应用场景**：
- 监控重试机制
- 处理自定义事件
- 记录任意文本输出

---

## 核心概念3：回调接口的组合模式

**通过组合 Mixin，可以灵活地选择需要的回调方法。**

### 组合模式示例

```python
# 示例1：只监控 LLM
class LLMOnlyHandler(BaseCallbackHandler):
    """只监控 LLM，忽略其他组件"""
    
    @property
    def ignore_chain(self) -> bool:
        return True
    
    @property
    def ignore_retriever(self) -> bool:
        return True
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM 开始: {prompts[0][:50]}...")
    
    def on_llm_end(self, response, **kwargs):
        print(f"LLM 完成: {response.generations[0][0].text[:50]}...")

# 示例2：监控 RAG 管道
class RAGPipelineHandler(BaseCallbackHandler):
    """监控完整的 RAG 管道"""
    
    def on_retriever_start(self, serialized, query, **kwargs):
        print(f"🔍 开始检索: {query}")
    
    def on_retriever_end(self, documents, **kwargs):
        print(f"✅ 检索到 {len(documents)} 个文档")
        for i, doc in enumerate(documents[:3]):
            print(f"  文档 {i+1}: {doc.page_content[:50]}...")
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"🤖 开始生成答案")
    
    def on_llm_end(self, response, **kwargs):
        answer = response.generations[0][0].text
        print(f"✅ 生成完成: {answer[:100]}...")

# 示例3：成本追踪
class CostTrackingHandler(BaseCallbackHandler):
    """追踪 token 用量和成本"""
    
    def __init__(self):
        super().__init__()
        self.total_tokens = 0
        self.total_cost = 0.0
    
    def on_llm_end(self, response, **kwargs):
        usage = response.llm_output.get("token_usage", {})
        tokens = usage.get("total_tokens", 0)
        self.total_tokens += tokens
        
        # 假设 $0.002 / 1K tokens
        cost = (tokens / 1000) * 0.002
        self.total_cost += cost
        
        print(f"Token 用量: {tokens}, 成本: ${cost:.4f}")
    
    def get_summary(self):
        return {
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost
        }
```

### 统一参数模式

**所有回调方法都遵循统一的参数模式**：

```python
def on_*_start/end/error(
    self,
    # 特定参数（根据回调类型不同）
    ...,
    *,
    # 统一参数
    run_id: UUID,                    # 当前运行的唯一标识
    parent_run_id: UUID | None = None,  # 父运行的标识（支持嵌套）
    tags: list[str] | None = None,   # 标签（用于分类和过滤）
    metadata: dict[str, Any] | None = None,  # 元数据
    **kwargs: Any,                   # 额外参数
) -> Any:
    pass
```

**参数说明**：
- `run_id`：每次运行的唯一标识，用于追踪
- `parent_run_id`：父运行的标识，支持嵌套回调追踪
- `tags`：标签列表，用于分类和过滤回调
- `metadata`：元数据字典，包含额外信息
- `**kwargs`：额外参数，不同回调类型可能包含不同的额外信息

---

## 实战代码示例

### 示例1：完整的 RAG 监控回调

```python
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from uuid import UUID
from typing import Any

class RAGMonitoringHandler(BaseCallbackHandler):
    """完整的 RAG 监控回调处理器"""
    
    def __init__(self):
        super().__init__()
        self.retrieval_time = 0
        self.generation_time = 0
        self.total_tokens = 0
    
    # Retriever 回调
    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        import time
        self.retrieval_start = time.time()
        print(f"\n🔍 开始检索: {query}")
    
    def on_retriever_end(self, documents, **kwargs):
        import time
        self.retrieval_time = time.time() - self.retrieval_start
        print(f"✅ 检索完成 ({self.retrieval_time:.2f}s)")
        print(f"   检索到 {len(documents)} 个文档")
        
        for i, doc in enumerate(documents[:2]):
            print(f"   文档 {i+1}: {doc.page_content[:80]}...")
    
    # LLM 回调
    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs):
        import time
        self.generation_start = time.time()
        print(f"\n🤖 开始生成答案")
    
    def on_llm_end(self, response, **kwargs):
        import time
        self.generation_time = time.time() - self.generation_start
        
        # 获取 token 用量
        usage = response.llm_output.get("token_usage", {})
        self.total_tokens = usage.get("total_tokens", 0)
        
        # 计算成本
        cost = (self.total_tokens / 1000) * 0.002
        
        print(f"✅ 生成完成 ({self.generation_time:.2f}s)")
        print(f"   Token 用量: {self.total_tokens}")
        print(f"   成本: ${cost:.4f}")
    
    def get_report(self):
        """获取完整报告"""
        return {
            "retrieval_time": self.retrieval_time,
            "generation_time": self.generation_time,
            "total_time": self.retrieval_time + self.generation_time,
            "total_tokens": self.total_tokens,
            "total_cost": (self.total_tokens / 1000) * 0.002
        }

# 使用示例
if __name__ == "__main__":
    # 初始化回调处理器
    handler = RAGMonitoringHandler()
    
    # 创建 RAG 管道
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        collection_name="demo",
        embedding_function=embeddings
    )
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        callbacks=[handler]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(callbacks=[handler]),
        callbacks=[handler]
    )
    
    # 执行查询
    result = qa_chain.invoke({"query": "什么是 RAG？"})
    
    # 打印报告
    print("\n" + "="*50)
    print("执行报告:")
    report = handler.get_report()
    for key, value in report.items():
        print(f"  {key}: {value}")
```

### 示例2：选择性监控

```python
class SelectiveHandler(BaseCallbackHandler):
    """选择性监控特定组件"""
    
    def __init__(self, monitor_llm=True, monitor_retriever=True):
        super().__init__()
        self._monitor_llm = monitor_llm
        self._monitor_retriever = monitor_retriever
    
    @property
    def ignore_llm(self) -> bool:
        return not self._monitor_llm
    
    @property
    def ignore_retriever(self) -> bool:
        return not self._monitor_retriever
    
    @property
    def ignore_chain(self) -> bool:
        return True  # 总是忽略 Chain
    
    def on_llm_end(self, response, **kwargs):
        if self._monitor_llm:
            tokens = response.llm_output["token_usage"]["total_tokens"]
            print(f"LLM: {tokens} tokens")
    
    def on_retriever_end(self, documents, **kwargs):
        if self._monitor_retriever:
            print(f"Retriever: {len(documents)} docs")

# 使用示例
handler1 = SelectiveHandler(monitor_llm=True, monitor_retriever=False)
handler2 = SelectiveHandler(monitor_llm=False, monitor_retriever=True)
```

---

## 与 RAG 开发的联系

### 1. RAG 管道的完整监控

```python
class RAGPipelineMonitor(BaseCallbackHandler):
    """监控 RAG 管道的每个步骤"""
    
    def on_retriever_start(self, serialized, query, **kwargs):
        print(f"步骤1: 检索文档 - 查询: {query}")
    
    def on_retriever_end(self, documents, **kwargs):
        print(f"步骤1: 完成 - 检索到 {len(documents)} 个文档")
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"步骤2: 生成答案 - Prompt 长度: {len(prompts[0])} 字符")
    
    def on_llm_end(self, response, **kwargs):
        answer = response.generations[0][0].text
        print(f"步骤2: 完成 - 答案: {answer[:100]}...")
```

### 2. 成本和性能追踪

```python
class RAGCostTracker(BaseCallbackHandler):
    """追踪 RAG 系统的成本和性能"""
    
    def __init__(self):
        super().__init__()
        self.metrics = {
            "retrieval_count": 0,
            "llm_calls": 0,
            "total_tokens": 0,
            "total_cost": 0.0
        }
    
    def on_retriever_end(self, documents, **kwargs):
        self.metrics["retrieval_count"] += 1
    
    def on_llm_end(self, response, **kwargs):
        self.metrics["llm_calls"] += 1
        tokens = response.llm_output["token_usage"]["total_tokens"]
        self.metrics["total_tokens"] += tokens
        self.metrics["total_cost"] += (tokens / 1000) * 0.002
```

### 3. 调试和错误追踪

```python
class RAGDebugHandler(BaseCallbackHandler):
    """调试 RAG 系统"""
    
    def on_retriever_end(self, documents, **kwargs):
        if len(documents) == 0:
            print("⚠️ 警告: 没有检索到任何文档!")
    
    def on_llm_error(self, error, **kwargs):
        print(f"❌ LLM 错误: {error}")
        print(f"   运行 ID: {kwargs.get('run_id')}")
    
    def on_retriever_error(self, error, **kwargs):
        print(f"❌ Retriever 错误: {error}")
```

---

## 核心要点总结

1. **Mixin 架构**：LangChain 使用 6 个 Mixin 组织回调方法
2. **BaseCallbackHandler**：通过多重继承获得所有回调接口
3. **6 个 Mixin**：
   - LLMManagerMixin：LLM 回调
   - ChainManagerMixin：Chain 和 Agent 回调
   - ToolManagerMixin：Tool 回调
   - RetrieverManagerMixin：Retriever 回调
   - CallbackManagerMixin：启动回调
   - RunManagerMixin：运行时回调
4. **组合模式**：通过 `ignore_*` 属性选择性监控
5. **统一参数**：所有回调方法使用一致的参数模式

---

## 学习检查清单

- [ ] 理解 BaseCallbackHandler 的设计哲学
- [ ] 掌握 6 个 Mixin 的职责划分
- [ ] 理解回调方法的统一参数模式
- [ ] 能够使用 `ignore_*` 属性选择性监控
- [ ] 能够实现自定义回调处理器
- [ ] 理解 Mixin 架构的优势

---

## 下一步学习

- **核心概念2**：内置回调处理器 - 学习 LangChain 提供的开箱即用的回调处理器
- **核心概念3**：回调管理器系统 - 理解回调的传递和继承机制
- **核心概念4**：回调生命周期 - 深入理解回调方法的触发时机

---

**记住**：Mixin 架构通过组合而非继承实现灵活的回调接口，这是 LangChain 回调系统的核心设计！
