---
type: context7_documentation
library: LangChain
version: latest (2026-02-17)
fetched_at: 2026-02-25
knowledge_point: CallbackHandler回调系统
context7_query: callback handlers on_llm_start on_chain_start on_tool_start custom callback implementation
---

# Context7 文档：LangChain 自定义回调处理器

## 文档来源
- 库名称：LangChain
- 版本：latest (2026-02-17)
- 官方文档链接：https://docs.langchain.com/
- Context7 Library ID：/websites/langchain

## 关键信息提取

### 1. 自定义异步回调处理器

#### Bedrock Guardrails 回调示例

```python
from typing import Any
from langchain_core.callbacks import AsyncCallbackHandler

class BedrockAsyncCallbackHandler(AsyncCallbackHandler):
    # Async callback handler that can be used to handle callbacks from langchain.

    async def on_llm_error(self, error: BaseException, **kwargs: Any) -> Any:
        reason = kwargs.get("reason")
        if reason == "GUARDRAIL_INTERVENED":
            print(f"Guardrails: {kwargs}")

# Guardrails for Amazon Bedrock with trace
llm = BedrockLLM(
    credentials_profile_name="bedrock-admin",
    model_id="<Model_ID>",
    model_kwargs={},
    guardrails={"id": "<Guardrail_ID>", "version": "<Version>", "trace": True},
    callbacks=[BedrockAsyncCallbackHandler()],
)
```

**关键特性**：
- 继承 `AsyncCallbackHandler`
- 实现 `on_llm_error()` 方法
- 通过 `kwargs` 获取额外信息（如 `reason`）
- 在 LLM 初始化时传入 `callbacks` 参数

### 2. 回调处理器与 LLMChain 集成

#### Context Callback 示例

```python
import os
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

token = os.environ["CONTEXT_API_TOKEN"]

human_message_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template="What is a good name for a company that makes {product}?",
        input_variables=["product"],
    )
)
chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
callback = ContextCallbackHandler(token)
chat = ChatOpenAI(temperature=0.9, callbacks=[callback])
chain = LLMChain(llm=chat, prompt=chat_prompt_template, callbacks=[callback])
print(chain.run("colorful socks"))
```

**关键特性**：
- 同一个回调实例可以传递给多个组件（chat model 和 chain）
- 确保整个管道的一致追踪
- 回调处理器可以接收配置参数（如 token）

### 3. 第三方回调处理器集成

#### LLMonitor 回调示例

```python
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI

handler = LLMonitorCallbackHandler()

llm = OpenAI(
    callbacks=[handler],
)

chat = ChatOpenAI(callbacks=[handler])

llm("Tell me a joke")
```

**关键特性**：
- 简单的初始化和使用
- 同一个处理器可用于不同类型的模型（LLM 和 ChatModel）

#### Wandb 回调示例

```python
from datetime import datetime

session_group = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
wandb_callback = WandbCallbackHandler(
    job_type="inference",
    project="langchain_callback_demo",
    group=f"minimal_{session_group}",
    name="llm",
    tags=["test"]
)
callbacks = [StdOutCallbackHandler(), wandb_callback]
llm = OpenAI(temperature=0, callbacks=callbacks)
```

**关键特性**：
- 支持多个回调处理器同时使用（列表形式）
- 可以配置详细的元数据（job_type, project, group, name, tags）
- 可以与内置回调处理器（如 `StdOutCallbackHandler`）组合使用

### 4. Agent 回调处理器

#### LLMonitor Agent 回调示例

```python
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.llmonitor_callback import LLMonitorCallbackHandler
from langchain.messages import SystemMessage, HumanMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor, tool

llm = ChatOpenAI(temperature=0)
handler = LLMonitorCallbackHandler()

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

tools = [get_word_length]

prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=SystemMessage(
        content="You are very powerful assistant, but bad at calculating lengths of words."
    )
)

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt, verbose=True)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    metadata={"agent_name": "WordCount"}  # <- recommended, assign a custom name
)
agent_executor.run("how many letters in the word educa?", callbacks=[handler])
```

**关键特性**：
- 回调处理器传递给 `AgentExecutor.run()` 方法
- 可以通过 `metadata` 参数设置自定义名称
- 支持追踪 Agent 的工具调用和推理过程

## 回调处理器使用模式总结

### 1. 初始化时传入

```python
llm = OpenAI(callbacks=[handler])
chain = LLMChain(llm=llm, callbacks=[handler])
```

### 2. 运行时传入

```python
agent_executor.run("query", callbacks=[handler])
chain.invoke({"input": "query"}, config={"callbacks": [handler]})
```

### 3. 多个回调处理器组合

```python
callbacks = [
    StdOutCallbackHandler(),
    WandbCallbackHandler(...),
    CustomCallbackHandler()
]
llm = OpenAI(callbacks=callbacks)
```

### 4. 异步回调处理器

```python
class CustomAsyncCallbackHandler(AsyncCallbackHandler):
    async def on_llm_start(self, serialized, prompts, **kwargs):
        # 异步处理逻辑
        pass

    async def on_llm_end(self, response, **kwargs):
        # 异步处理逻辑
        pass
```

## 常见第三方回调处理器

从文档中识别出的第三方回调处理器：

1. **LLMonitorCallbackHandler** - LLM 监控和可观测性
2. **WandbCallbackHandler** - Weights & Biases 集成
3. **ContextCallbackHandler** - Context.ai 集成
4. **BedrockAsyncCallbackHandler** - Amazon Bedrock Guardrails

## 回调处理器的关键方法

从示例中可以看出，回调处理器可以实现以下方法：

- `on_llm_start()` - LLM 开始时触发
- `on_llm_end()` - LLM 结束时触发
- `on_llm_error()` - LLM 错误时触发
- `on_chain_start()` - Chain 开始时触发
- `on_chain_end()` - Chain 结束时触发
- `on_tool_start()` - Tool 开始时触发
- `on_tool_end()` - Tool 结束时触发
- `on_agent_action()` - Agent 执行动作时触发
- `on_agent_finish()` - Agent 完成时触发

## 与 CallbackHandler 的关系

从这些文档中可以看出：

1. **回调处理器是可插拔的**：
   - 可以在初始化时传入
   - 可以在运行时传入
   - 可以组合多个回调处理器

2. **回调处理器支持异步**：
   - `AsyncCallbackHandler` 提供异步版本
   - 所有方法都是 `async def`

3. **回调处理器可以访问详细信息**：
   - 通过 `**kwargs` 获取额外信息
   - 可以访问 `metadata`、`tags` 等

4. **回调处理器用于可观测性**：
   - 追踪 LLM 调用
   - 监控 Agent 执行
   - 记录错误和异常
   - 集成第三方监控平台

## 相关依赖

- `langchain-core` - 核心回调接口
- `langchain-openai` - OpenAI 集成
- `langchain-community` - 社区回调处理器
- 第三方库：`wandb`, `llmonitor`, `context-ai`, `boto3` (Bedrock)
