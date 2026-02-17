# 实战代码2：AI Agent LLM 集成

完整的 AI Agent + Pydantic 集成示例，展示 LLM 请求验证和流式响应处理。

---

## 场景说明

构建一个 AI 聊天 API，集成 OpenAI/Anthropic LLM，展示 Pydantic 在 AI Agent 开发中的实际应用。

---

## 完整代码

```python
"""
AI Agent + Pydantic 集成示例
演示：LLM 聊天 API（支持流式和非流式）
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Literal, Optional, AsyncIterator
from datetime import datetime
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ===== 1. 创建 FastAPI 应用 =====
app = FastAPI(
    title="AI Chat API",
    description="AI 聊天 API，支持多种 LLM 模型",
    version="1.0.0"
)

# ===== 2. 定义 Pydantic 模型 =====

class Message(BaseModel):
    """聊天消息模型"""
    role: Literal["user", "assistant", "system"] = Field(
        ...,
        description="消息角色",
        example="user"
    )
    content: str = Field(
        ...,
        min_length=1,
        description="消息内容",
        example="Hello, how are you?"
    )

    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v):
        """验证消息内容不为空"""
        if not v.strip():
            raise ValueError('message content cannot be empty')
        return v.strip()


class ChatRequest(BaseModel):
    """聊天请求模型"""
    messages: List[Message] = Field(
        ...,
        min_length=1,
        description="消息列表",
        example=[
            {"role": "user", "content": "Hello"}
        ]
    )

    model: Literal["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"] = Field(
        "gpt-3.5-turbo",
        description="LLM 模型名称"
    )

    temperature: float = Field(
        0.7,
        ge=0,
        le=2,
        description="温度参数，控制输出随机性"
    )

    max_tokens: Optional[int] = Field(
        None,
        ge=1,
        le=4096,
        description="最大生成 token 数"
    )

    stream: bool = Field(
        False,
        description="是否使用流式输出"
    )

    @field_validator('messages')
    @classmethod
    def messages_not_empty(cls, v):
        """验证消息列表不为空"""
        if not v:
            raise ValueError('messages list cannot be empty')
        return v

    @field_validator('messages')
    @classmethod
    def first_message_must_be_user(cls, v):
        """验证第一条消息必须来自用户"""
        if v and v[0].role == 'system':
            # system 消息可以在第一位
            return v
        if v and v[0].role != 'user':
            raise ValueError('first non-system message must be from user')
        return v

    @model_validator(mode='after')
    def validate_temperature_for_model(self):
        """验证不同模型的温度范围"""
        if 'gpt' in self.model and not (0 <= self.temperature <= 2):
            raise ValueError('GPT temperature must be 0-2')
        if 'claude' in self.model and not (0 <= self.temperature <= 1):
            raise ValueError('Claude temperature must be 0-1')
        return self


class ChatResponse(BaseModel):
    """聊天响应模型"""
    id: str = Field(..., description="响应 ID")
    model: str = Field(..., description="使用的模型")
    message: Message = Field(..., description="AI 回复消息")
    usage: dict = Field(..., description="Token 使用情况")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="创建时间"
    )


class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误消息")
    details: Optional[dict] = Field(None, description="错误详情")


# ===== 3. LLM 客户端封装 =====

class LLMClient:
    """LLM 客户端（模拟实现）"""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("警告: OPENAI_API_KEY 未设置，使用模拟响应")

    async def chat(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> dict:
        """
        调用 LLM API（非流式）

        实际项目中应该调用真实的 LLM API
        """
        # 模拟 API 调用
        import asyncio
        await asyncio.sleep(0.5)  # 模拟网络延迟

        # 模拟响应
        return {
            "id": "chatcmpl-123456",
            "model": model,
            "message": {
                "role": "assistant",
                "content": f"这是来自 {model} 的模拟回复。你说：{messages[-1].content}"
            },
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }

    async def chat_stream(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[str]:
        """
        调用 LLM API（流式）

        实际项目中应该调用真实的 LLM API
        """
        # 模拟流式响应
        import asyncio

        response_text = f"这是来自 {model} 的模拟流式回复。你说：{messages[-1].content}"

        for char in response_text:
            await asyncio.sleep(0.05)  # 模拟流式输出延迟
            yield char


# 创建 LLM 客户端实例
llm_client = LLMClient()


# ===== 4. API 端点 =====

@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        422: {"model": ErrorResponse, "description": "验证错误"},
        500: {"model": ErrorResponse, "description": "服务器错误"}
    },
    summary="AI 聊天",
    description="发送消息给 AI 并获取回复"
)
async def chat(request: ChatRequest):
    """
    AI 聊天（非流式）

    Pydantic 自动验证：
    - messages 不为空
    - 第一条消息来自用户
    - model 只能是指定的值
    - temperature 在有效范围内
    - max_tokens 在有效范围内
    """
    try:
        # 调用 LLM API
        response = await llm_client.chat(
            messages=request.messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        # 返回响应
        return ChatResponse(
            id=response["id"],
            model=response["model"],
            message=Message(**response["message"]),
            usage=response["usage"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/chat/stream",
    summary="AI 聊天（流式）",
    description="发送消息给 AI 并获取流式回复"
)
async def chat_stream(request: ChatRequest):
    """
    AI 聊天（流式）

    返回 Server-Sent Events (SSE) 格式的流式响应
    """
    # 强制使用流式模式
    request.stream = True

    async def generate():
        """生成流式响应"""
        try:
            # 发送开始标记
            yield f"data: {{'event': 'start', 'model': '{request.model}'}}\n\n"

            # 流式输出内容
            async for chunk in llm_client.chat_stream(
                messages=request.messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ):
                yield f"data: {{'event': 'content', 'content': '{chunk}'}}\n\n"

            # 发送结束标记
            yield f"data: {{'event': 'end'}}\n\n"

        except Exception as e:
            yield f"data: {{'event': 'error', 'message': '{str(e)}'}}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


# ===== 5. 高级功能：对话历史管理 =====

class ConversationHistory(BaseModel):
    """对话历史模型"""
    conversation_id: str = Field(..., description="对话 ID")
    messages: List[Message] = Field(default_factory=list, description="消息列表")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @field_validator('messages')
    @classmethod
    def limit_message_count(cls, v):
        """限制消息数量（避免上下文过长）"""
        max_messages = 50
        if len(v) > max_messages:
            # 保留最近的消息
            return v[-max_messages:]
        return v


# 模拟对话历史存储
conversation_store: dict[str, ConversationHistory] = {}


@app.post(
    "/conversations",
    response_model=ConversationHistory,
    summary="创建对话",
    description="创建一个新的对话会话"
)
async def create_conversation():
    """创建新对话"""
    import uuid

    conversation_id = str(uuid.uuid4())
    conversation = ConversationHistory(conversation_id=conversation_id)
    conversation_store[conversation_id] = conversation

    return conversation


@app.post(
    "/conversations/{conversation_id}/messages",
    response_model=ChatResponse,
    summary="发送消息到对话",
    description="向指定对话发送消息"
)
async def send_message_to_conversation(
    conversation_id: str,
    message: Message,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7
):
    """向对话发送消息"""
    # 获取对话历史
    conversation = conversation_store.get(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found"
        )

    # 添加用户消息
    conversation.messages.append(message)

    # 调用 LLM
    response = await llm_client.chat(
        messages=conversation.messages,
        model=model,
        temperature=temperature
    )

    # 添加 AI 回复
    ai_message = Message(**response["message"])
    conversation.messages.append(ai_message)
    conversation.updated_at = datetime.now()

    return ChatResponse(
        id=response["id"],
        model=response["model"],
        message=ai_message,
        usage=response["usage"]
    )


@app.get(
    "/conversations/{conversation_id}",
    response_model=ConversationHistory,
    summary="获取对话历史",
    description="获取指定对话的完整历史"
)
async def get_conversation(conversation_id: str):
    """获取对话历史"""
    conversation = conversation_store.get(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found"
        )

    return conversation


# ===== 6. 高级功能：函数调用（Tool Use） =====

class ToolParameter(BaseModel):
    """工具参数模型"""
    name: str = Field(..., description="参数名")
    type: str = Field(..., description="参数类型")
    description: str = Field(..., description="参数描述")
    required: bool = Field(True, description="是否必填")


class Tool(BaseModel):
    """工具定义模型"""
    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具描述")
    parameters: List[ToolParameter] = Field(..., description="工具参数")


class ToolCall(BaseModel):
    """工具调用模型"""
    tool_name: str = Field(..., description="工具名称")
    parameters: dict = Field(..., description="工具参数")

    @field_validator('tool_name')
    @classmethod
    def tool_name_valid(cls, v):
        """验证工具名称"""
        valid_tools = ["search", "calculator", "weather"]
        if v not in valid_tools:
            raise ValueError(f'tool_name must be one of {valid_tools}')
        return v


class ChatWithToolsRequest(ChatRequest):
    """带工具的聊天请求"""
    tools: List[Tool] = Field(default_factory=list, description="可用工具列表")


class ChatWithToolsResponse(ChatResponse):
    """带工具的聊天响应"""
    tool_calls: List[ToolCall] = Field(default_factory=list, description="工具调用列表")


@app.post(
    "/chat/tools",
    response_model=ChatWithToolsResponse,
    summary="AI 聊天（支持工具调用）",
    description="发送消息给 AI，AI 可以调用工具"
)
async def chat_with_tools(request: ChatWithToolsRequest):
    """
    AI 聊天（支持工具调用）

    实际项目中，LLM 会根据对话内容决定是否调用工具
    """
    # 调用 LLM（实际项目中传入 tools）
    response = await llm_client.chat(
        messages=request.messages,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )

    # 模拟工具调用（实际项目中由 LLM 决定）
    tool_calls = []
    if "天气" in request.messages[-1].content:
        tool_calls.append(ToolCall(
            tool_name="weather",
            parameters={"location": "北京"}
        ))

    return ChatWithToolsResponse(
        id=response["id"],
        model=response["model"],
        message=Message(**response["message"]),
        usage=response["usage"],
        tool_calls=tool_calls
    )


# ===== 7. 健康检查 =====

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ===== 8. 运行应用 =====

if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("AI Agent + Pydantic 集成示例")
    print("=" * 50)
    print("\n功能：")
    print("- ✅ LLM 聊天（非流式）")
    print("- ✅ LLM 聊天（流式）")
    print("- ✅ 对话历史管理")
    print("- ✅ 工具调用（Tool Use）")
    print("\n启动服务器...")
    print("API 文档: http://localhost:8000/docs")
    print("\n按 Ctrl+C 停止服务器\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 运行示例

### 1. 安装依赖

```bash
# 安装依赖
uv add fastapi uvicorn[standard] pydantic python-dotenv

# 可选：安装 OpenAI SDK（如果使用真实 API）
uv add openai
```

### 2. 配置环境变量

```bash
# 创建 .env 文件
cat > .env << EOF
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
EOF
```

### 3. 运行服务器

```bash
python examples/ai_agent_llm.py
```

### 4. 测试 API

**非流式聊天：**

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "model": "gpt-3.5-turbo",
    "temperature": 0.7
  }'
```

**响应：**

```json
{
  "id": "chatcmpl-123456",
  "model": "gpt-3.5-turbo",
  "message": {
    "role": "assistant",
    "content": "这是来自 gpt-3.5-turbo 的模拟回复。你说：Hello, how are you?"
  },
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  },
  "created_at": "2026-02-11T06:20:29.095Z"
}
```

**流式聊天：**

```bash
curl -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "model": "gpt-4",
    "stream": true
  }'
```

**响应（SSE 格式）：**

```
data: {'event': 'start', 'model': 'gpt-4'}

data: {'event': 'content', 'content': '这'}

data: {'event': 'content', 'content': '是'}

data: {'event': 'content', 'content': '来'}

...

data: {'event': 'end'}
```

**创建对话：**

```bash
curl -X POST "http://localhost:8000/conversations"
```

**响应：**

```json
{
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "messages": [],
  "created_at": "2026-02-11T06:20:29.095Z",
  "updated_at": "2026-02-11T06:20:29.095Z"
}
```

**发送消息到对话：**

```bash
curl -X POST "http://localhost:8000/conversations/550e8400-e29b-41d4-a716-446655440000/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "role": "user",
    "content": "Hello"
  }'
```

---

## 验证失败示例

### 1. 消息列表为空

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [],
    "model": "gpt-3.5-turbo"
  }'
```

**响应（422 错误）：**

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "messages"],
      "msg": "Value error, messages list cannot be empty"
    }
  ]
}
```

### 2. 温度超出范围

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "model": "claude-3-sonnet",
    "temperature": 1.5
  }'
```

**响应（422 错误）：**

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body"],
      "msg": "Value error, Claude temperature must be 0-1"
    }
  ]
}
```

### 3. 无效的工具名称

```bash
curl -X POST "http://localhost:8000/chat/tools" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "tools": [
      {
        "name": "invalid_tool",
        "description": "Invalid tool",
        "parameters": []
      }
    ]
  }'
```

---

## 核心要点

### 1. 复杂验证逻辑

```python
@field_validator('messages')
@classmethod
def first_message_must_be_user(cls, v):
    if v and v[0].role != 'user':
        raise ValueError('first message must be from user')
    return v

@model_validator(mode='after')
def validate_temperature_for_model(self):
    if 'gpt' in self.model and not (0 <= self.temperature <= 2):
        raise ValueError('GPT temperature must be 0-2')
    return self
```

### 2. 嵌套模型

```python
class ChatRequest(BaseModel):
    messages: List[Message]  # 嵌套 Message 模型

class ChatResponse(BaseModel):
    message: Message  # 嵌套 Message 模型
    usage: dict
```

### 3. 流式响应

```python
async def generate():
    async for chunk in llm_client.chat_stream(...):
        yield f"data: {{'content': '{chunk}'}}\n\n"

return StreamingResponse(generate(), media_type="text/event-stream")
```

### 4. 对话历史管理

```python
class ConversationHistory(BaseModel):
    conversation_id: str
    messages: List[Message]

    @field_validator('messages')
    @classmethod
    def limit_message_count(cls, v):
        # 限制消息数量
        return v[-50:]
```

---

## 最佳实践

### 1. 模型验证

```python
# ✅ 好的做法：多层验证
@field_validator('messages')
@classmethod
def messages_not_empty(cls, v):
    if not v:
        raise ValueError('messages cannot be empty')
    return v

@model_validator(mode='after')
def validate_model_specific_params(self):
    # 跨字段验证
    return self
```

### 2. 错误处理

```python
try:
    response = await llm_client.chat(...)
    return ChatResponse(...)
except Exception as e:
    raise HTTPException(
        status_code=500,
        detail=str(e)
    )
```

### 3. 流式输出

```python
async def generate():
    try:
        yield "data: {'event': 'start'}\n\n"
        async for chunk in stream:
            yield f"data: {{'content': '{chunk}'}}\n\n"
        yield "data: {'event': 'end'}\n\n"
    except Exception as e:
        yield f"data: {{'event': 'error', 'message': '{e}'}}\n\n"
```

### 4. 对话管理

```python
# 限制消息数量（避免上下文过长）
@field_validator('messages')
@classmethod
def limit_message_count(cls, v):
    return v[-50:]  # 只保留最近50条
```

---

## 扩展功能

### 1. 真实 LLM 集成

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def chat(messages, model, temperature):
    response = await client.chat.completions.create(
        model=model,
        messages=[m.model_dump() for m in messages],
        temperature=temperature
    )
    return response
```

### 2. 缓存机制

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_response(messages_hash, model):
    # 缓存相同请求的响应
    pass
```

### 3. 速率限制

```python
from slowapi import Limiter

limiter = Limiter(key_func=lambda: "global")

@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: ChatRequest):
    pass
```

---

**版本：** v1.0
**最后更新：** 2026-02-11
