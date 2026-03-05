---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/graph/message.py
analyzed_at: 2026-02-26
knowledge_point: Reducer函数与状态更新
---

# 源码分析：add_messages Reducer 实现

## 分析的文件

### `libs/langgraph/langgraph/graph/message.py`
LangGraph 内置的消息列表 Reducer,用于聊天机器人等场景。

## 关键发现

### 1. add_messages 函数签名 (message.py:60-244)

```python
@_add_messages_wrapper
def add_messages(
    left: Messages,
    right: Messages,
    *,
    format: Literal["langchain-openai"] | None = None,
) -> Messages:
    """Merges two lists of messages, updating existing messages by ID.

    By default, this ensures the state is "append-only", unless the
    new message has the same ID as an existing message.

    Args:
        left: The base list of `Messages`.
        right: The list of `Messages` (or single `Message`) to merge
            into the base list.
        format: The format to return messages in. If `None` then `Messages` will be
            returned as is. If `langchain-openai` then `Messages` will be returned as
            `BaseMessage` objects with their contents formatted to match OpenAI message
            format, meaning contents can be string, `'text'` blocks, or `'image_url'` blocks
            and tool responses are returned as their own `ToolMessage` objects.

    Returns:
        A new list of messages with the messages from `right` merged into `left`.
        If a message in `right` has the same ID as a message in `left`, the
            message from `right` will replace the message from `left`.
    """
```

**核心特性**:
- **按 ID 合并**: 相同 ID 的消息会被替换,而非追加
- **Append-only**: 默认行为是追加新消息
- **格式化支持**: 可选的 OpenAI 格式转换

### 2. 实现逻辑 (message.py:187-244)

```python
def add_messages(left: Messages, right: Messages, *, format: Literal["langchain-openai"] | None = None) -> Messages:
    remove_all_idx = None

    # 1. 转换为列表
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]

    # 2. 转换为消息对象
    left = [
        message_chunk_to_message(cast(BaseMessageChunk, m))
        for m in convert_to_messages(left)
    ]
    right = [
        message_chunk_to_message(cast(BaseMessageChunk, m))
        for m in convert_to_messages(right)
    ]

    # 3. 分配 ID
    for m in left:
        if m.id is None:
            m.id = str(uuid.uuid4())
    for idx, m in enumerate(right):
        if m.id is None:
            m.id = str(uuid.uuid4())
        if isinstance(m, RemoveMessage) and m.id == REMOVE_ALL_MESSAGES:
            remove_all_idx = idx

    # 4. 处理 REMOVE_ALL_MESSAGES
    if remove_all_idx is not None:
        return right[remove_all_idx + 1 :]

    # 5. 按 ID 合并
    merged = left.copy()
    merged_by_id = {m.id: i for i, m in enumerate(merged)}
    ids_to_remove = set()

    for m in right:
        if (existing_idx := merged_by_id.get(m.id)) is not None:
            if isinstance(m, RemoveMessage):
                ids_to_remove.add(m.id)
            else:
                ids_to_remove.discard(m.id)
                merged[existing_idx] = m
        else:
            if isinstance(m, RemoveMessage):
                raise ValueError(
                    f"Attempting to delete a message with an ID that doesn't exist ('{m.id}')"
                )
            merged_by_id[m.id] = len(merged)
            merged.append(m)

    merged = [m for m in merged if m.id not in ids_to_remove]

    # 6. 格式化输出
    if format == "langchain-openai":
        merged = _format_messages(merged)
    elif format:
        msg = f"Unrecognized {format=}. Expected one of 'langchain-openai', None."
        raise ValueError(msg)

    return merged
```

**实现步骤**:
1. **类型转换**: 确保 left 和 right 都是列表
2. **消息转换**: 转换为 `BaseMessage` 对象
3. **ID 分配**: 为没有 ID 的消息生成 UUID
4. **REMOVE_ALL_MESSAGES**: 如果遇到,清空所有历史消息
5. **按 ID 合并**:
   - 如果 ID 已存在且是 `RemoveMessage`,标记删除
   - 如果 ID 已存在且是普通消息,替换旧消息
   - 如果 ID 不存在,追加新消息
6. **格式化**: 可选的 OpenAI 格式转换

### 3. 使用示例

#### 基础用法 (message.py:91-99)
```python
from langchain_core.messages import AIMessage, HumanMessage

msgs1 = [HumanMessage(content="Hello", id="1")]
msgs2 = [AIMessage(content="Hi there!", id="2")]
add_messages(msgs1, msgs2)
# [HumanMessage(content='Hello', id='1'), AIMessage(content='Hi there!', id='2')]
```

#### 覆盖已有消息 (message.py:101-107)
```python
msgs1 = [HumanMessage(content="Hello", id="1")]
msgs2 = [HumanMessage(content="Hello again", id="1")]
add_messages(msgs1, msgs2)
# [HumanMessage(content='Hello again', id='1')]
```

#### 在 StateGraph 中使用 (message.py:109-127)
```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph

class State(TypedDict):
    messages: Annotated[list, add_messages]

builder = StateGraph(State)
builder.add_node("chatbot", lambda state: {"messages": [("assistant", "Hello")]})
builder.set_entry_point("chatbot")
builder.set_finish_point("chatbot")
graph = builder.compile()
graph.invoke({})
# {'messages': [AIMessage(content='Hello', id=...)]}
```

#### 使用 OpenAI 格式 (message.py:129-184)
```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages(format="langchain-openai")]

def chatbot_node(state: State) -> list:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here's an image:",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "1234",
                        },
                    },
                ],
            },
        ]
    }

builder = StateGraph(State)
builder.add_node("chatbot", chatbot_node)
builder.set_entry_point("chatbot")
builder.set_finish_point("chatbot")
graph = builder.compile()
graph.invoke({"messages": []})
# {
#     'messages': [
#         HumanMessage(
#             content=[
#                 {"type": "text", "text": "Here's an image:"},
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": "data:image/jpeg;base64,1234"},
#                 },
#             ],
#         ),
#     ]
# }
```

### 4. 特殊功能

#### RemoveMessage (message.py:209-234)
```python
# 删除特定消息
from langchain_core.messages import RemoveMessage

# 标记删除 ID 为 "1" 的消息
remove_msg = RemoveMessage(id="1")

# 在 add_messages 中处理
# 如果 ID 存在,该消息会被删除
# 如果 ID 不存在,抛出 ValueError
```

#### REMOVE_ALL_MESSAGES (message.py:38, 212-213)
```python
REMOVE_ALL_MESSAGES = "__remove_all__"

# 清空所有历史消息
remove_all = RemoveMessage(id=REMOVE_ALL_MESSAGES)

# 在 add_messages 中处理
# 返回 remove_all 之后的所有消息
# 之前的所有消息都被丢弃
```

### 5. _add_messages_wrapper 装饰器 (message.py:41-57)

```python
def _add_messages_wrapper(func: Callable) -> Callable[[Messages, Messages], Messages]:
    def _add_messages(
        left: Messages | None = None, right: Messages | None = None, **kwargs: Any
    ) -> Messages | Callable[[Messages, Messages], Messages]:
        if left is not None and right is not None:
            return func(left, right, **kwargs)
        elif left is not None or right is not None:
            msg = (
                f"Must specify non-null arguments for both 'left' and 'right'. Only "
                f"received: '{'left' if left else 'right'}'."
            )
            raise ValueError(msg)
        else:
            return partial(func, **kwargs)

    _add_messages.__doc__ = func.__doc__
    return cast(Callable[[Messages, Messages], Messages], _add_messages)
```

**功能**:
- 支持部分应用 (partial application)
- 可以预先绑定 `format` 参数
- 示例: `add_messages(format="langchain-openai")` 返回一个新的 Reducer 函数

## 技术要点总结

### 1. 消息合并策略
- **按 ID 合并**: 相同 ID 的消息会被替换
- **追加新消息**: 新 ID 的消息会被追加到列表末尾
- **自动生成 ID**: 没有 ID 的消息会自动分配 UUID

### 2. 删除机制
- **RemoveMessage**: 删除特定 ID 的消息
- **REMOVE_ALL_MESSAGES**: 清空所有历史消息

### 3. 格式化支持
- **langchain-openai**: 转换为 OpenAI 消息格式
- 支持 text blocks 和 image_url blocks
- 需要 `langchain-core>=0.3.11`

### 4. 使用场景
- **聊天机器人**: 维护对话历史
- **多轮对话**: 累积用户和 AI 的消息
- **消息更新**: 替换或删除特定消息
- **格式转换**: 适配不同 LLM 的消息格式

## 源码位置索引

| 功能 | 文件 | 行号 |
|------|------|------|
| add_messages 定义 | message.py | 60-244 |
| _add_messages_wrapper | message.py | 41-57 |
| REMOVE_ALL_MESSAGES | message.py | 38 |
| 基础用法示例 | message.py | 91-99 |
| 覆盖消息示例 | message.py | 101-107 |
| StateGraph 集成示例 | message.py | 109-127 |
| OpenAI 格式示例 | message.py | 129-184 |
