---
type: context7_documentation
library: LangChain
version: latest (2026-02-17)
fetched_at: 2026-02-25
knowledge_point: CallbackHandler回调系统
context7_query: CallbackHandler BaseCallbackHandler CallbackManager custom callbacks streaming async observability
---

# Context7 文档：LangChain 可观测性与追踪

## 文档来源
- 库名称：LangChain
- 版本：latest (2026-02-17)
- 官方文档链接：https://docs.langchain.com/
- Context7 Library ID：/websites/langchain
- 总文档片段：26795
- 信任分数：10/10
- 质量分数：83/100

## 关键信息提取

### 1. LangSmith 追踪系统

LangSmith 是 LangChain 的官方可观测性平台，通过回调系统实现自动追踪。

#### @traceable 装饰器（Python）

```python
from langsmith import traceable

@traceable(
    tags=["openai", "chat"],
    metadata={"foo": "bar"}
)
def invoke_runnnable(question, context):
    result = chain.invoke({"question": question, "context": context})
    return "The response is: " + result

invoke_runnnable("Can you summarize this morning's meetings?", "During this morning's meeting, we solved all world conflict.")
```

**关键特性**：
- 自动追踪函数执行
- 支持自定义 tags 和 metadata
- 可配置 run_type（llm, chain, tool, retriever）
- 支持流式输出的聚合（reduce_fn）

#### 流式模型追踪

```python
def _reduce_chunks(chunks: list):
    all_text = "".join([chunk["choices"][0]["message"]["content"] for chunk in chunks])
    return {"choices": [{"message": {"content": all_text, "role": "assistant"}}]}

@traceable(
    run_type="llm",
    reduce_fn=_reduce_chunks,
    metadata={"ls_provider": "my_provider", "ls_model_name": "my_model"}
)
def my_streaming_chat_model(messages: list):
    for chunk in ["Hello, " + messages[1]["content"]]:
        yield {
            "choices": [
                {
                    "message": {
                        "content": chunk,
                        "role": "assistant",
                    }
                }
            ]
        }
```

**关键特性**：
- `reduce_fn` 用于聚合流式输出
- `metadata` 中的 `ls_provider` 和 `ls_model_name` 用于标识模型
- 支持生成器函数

### 2. OpenAI 客户端包装

#### Python 包装

```python
from openai import OpenAI
from langsmith.wrappers import wrap_openai

client = wrap_openai(OpenAI())  # 自动追踪所有 OpenAI 调用

def rag(question: str) -> str:
    docs = retriever(question)
    system_message = (
        "Answer the user's question using only the provided information below:\n"
        + "\n".join(docs)
    )
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]
    )
    return resp.choices[0].message.content
```

#### TypeScript 包装

```typescript
import OpenAI from "openai";
import { wrapOpenAI } from "langsmith/wrappers";

const client = wrapOpenAI(new OpenAI());  // 自动追踪所有 OpenAI 调用
```

**关键特性**：
- 自动追踪所有 LLM 调用
- 记录输入、输出和 token 使用量
- 无需修改现有代码逻辑

### 3. TypeScript traceable 函数

```typescript
import { traceable } from "langsmith/traceable";
import OpenAI from "openai";

const openai = new OpenAI();

const formatPrompt = traceable((subject: string) => {
  return [
    {
      role: "system" as const,
      content: "You are a helpful assistant.",
    },
    {
      role: "user" as const,
      content: `What's a good name for a store that sells ${subject}?`,
    },
  ];
},{ name: "formatPrompt" });

const invokeLLM = traceable(
  async ({ messages }: { messages: { role: string; content: string }[] }) => {
      return openai.chat.completions.create({
          model: "gpt-4.1-mini",
          messages: messages,
          temperature: 0,
      });
  },
  { run_type: "llm", name: "invokeLLM" }
);

const parseOutput = traceable(
  (response: any) => {
      return response.choices[0].message.content;
  },
  { name: "parseOutput" }
);

const runPipeline = traceable(
  async () => {
      const messages = await formatPrompt("colorful socks");
      const response = await invokeLLM({ messages });
      return parseOutput(response);
  },
  { name: "runPipeline" }
);

await runPipeline();
```

**关键特性**：
- 每个函数都可以单独追踪
- 支持嵌套追踪（父子关系）
- 可配置 run_type 和 name

### 4. Next.js API 路由追踪

```typescript
import { NextRequest, NextResponse } from "next/server";
import { OpenAI } from "openai";
import { traceable } from "langsmith/traceable";
import { wrapOpenAI } from "langsmith/wrappers";

export const runtime = "edge";

const handler = traceable(
  async function () {
    const openai = wrapOpenAI(new OpenAI());

    const completion = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [{ content: "Why is the sky blue?", role: "user" }],
    });

    const response1 = completion.choices[0].message.content;

    const completion2 = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [
        { content: "Why is the sky blue?", role: "user" },
        { content: response1, role: "assistant" },
        { content: "Cool thank you!", role: "user" },
      ],
    });

    const response2 = completion2.choices[0].message.content;

    return {
      text: response2,
    };
  },
  {
    name: "Simple Next.js handler",
  }
);

export async function POST(req: NextRequest) {
  const result = await handler();
  return NextResponse.json(result);
}
```

**关键特性**：
- 整个 API 路由作为一个追踪单元
- 所有子调用（OpenAI）自动作为子运行追踪
- 支持 Edge Runtime

### 5. astream_events - 事件流

```python
async for event in runnable.astream_events("input_data", version="v2"):
    print(event)
```

**事件类型**：
- `on_chain_start` - Chain 开始
- `on_chain_stream` - Chain 流式输出
- `on_chain_end` - Chain 结束

**过滤选项**：
- `include_names` - 只包含特定名称的事件
- `include_types` - 只包含特定类型的事件
- `include_tags` - 只包含特定标签的事件
- `exclude_names` - 排除特定名称的事件
- `exclude_types` - 排除特定类型的事件
- `exclude_tags` - 排除特定标签的事件

### 6. astream_log - 回调日志流

```python
astream_log(
    input: Any,
    config: RunnableConfig | None = None,
    *,
    diff: bool = True,
    with_streamed_output_list: bool = True,
    include_names: Sequence[str] | None = None,
    include_types: Sequence[str] | None = None,
    include_tags: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    exclude_types: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    **kwargs: Any,
) -> AsyncIterator[RunLogPatch] | AsyncIterator[RunLog]
```

**关键特性**：
- 流式输出所有内部运行（LLMs, Retrievers, Tools）
- 返回 JSONPatch 操作描述状态变化
- 支持 diff 模式或完整状态输出

## 与 CallbackHandler 的关系

从这些文档中可以看出：

1. **LangSmith 追踪基于回调系统**：
   - `@traceable` 装饰器内部使用回调系统
   - `wrap_openai` 通过回调系统记录 LLM 调用

2. **事件流是回调系统的高级接口**：
   - `astream_events` 提供了访问回调事件的流式接口
   - `astream_log` 通过回调系统记录所有内部运行

3. **可观测性的核心是回调**：
   - 所有追踪、日志、监控都通过回调系统实现
   - 回调系统是 LangChain 可观测性的基础设施

## 环境变量配置

```bash
# 启用 LangSmith 追踪
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=your_api_key_here

# 可选：指定项目名称
export LANGSMITH_PROJECT=my_project
```

## 相关依赖

- `langsmith` - LangSmith SDK（追踪和可观测性）
- `openai` - OpenAI SDK
- `langchain-core` - LangChain 核心库（包含回调系统）
