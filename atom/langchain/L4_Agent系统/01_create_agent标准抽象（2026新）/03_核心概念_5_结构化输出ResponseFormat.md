# 核心概念 5：结构化输出 ResponseFormat

> Agent 不只会说话，还能返回你定义好的数据结构——三种策略，一套统一接口

---

## 为什么 Agent 需要结构化输出？

默认情况下，Agent 返回的是自由文本。但在生产环境中，你经常需要 Agent 返回**可编程消费的数据**：

```python
# ❌ 自由文本 —— 下游代码无法可靠解析
"北京今天 25°C，晴，湿度 40%。"

# ✅ 结构化输出 —— 直接拿到 Python 对象
WeatherReport(city="北京", temperature=25.0, condition="晴", humidity=40)
```

典型场景：
- **数据提取**：从非结构化文本中提取实体（人名、地址、金额）
- **决策输出**：Agent 返回分类标签、置信度分数
- **API 对接**：Agent 输出直接作为下游 API 的请求体
- **多 Agent 协作**：Agent 之间通过结构化数据通信

LangChain 1.0 的关键改进：**将结构化输出集成到 Agent 的 model-tools 主循环中**，不再需要额外的 LLM 调用来格式化输出，减少延迟和成本。

---

## 三种策略总览

LangChain 1.0 提供三种结构化输出策略，统一为 `ResponseFormat` 类型：

```
┌─────────────────────────────────────────────────────────┐
│                    ResponseFormat                        │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ AutoStrategy │  │ ToolStrategy │  │ProviderStrategy│ │
│  │  （推荐）     │  │  （兼容性好） │  │ （性能更好）    │ │
│  └──────┬──────┘  └──────────────┘  └────────────────┘ │
│         │                                               │
│         ├─ 模型支持原生？──Yes──▶ ProviderStrategy       │
│         └─ 不支持？──────No───▶ ToolStrategy             │
└─────────────────────────────────────────────────────────┘
```

| 策略 | 原理 | 兼容性 | 性能 | 推荐场景 |
|------|------|--------|------|----------|
| AutoStrategy | 自动检测模型能力，选最优 | 最广 | 自适应 | 默认首选 |
| ToolStrategy | 把 schema 伪装成工具调用 | 所有支持 tool_call 的模型 | 一般 | 需要显式控制 |
| ProviderStrategy | 用模型原生结构化输出 | 仅支持的模型（如 OpenAI） | 最好 | 确定模型支持时 |

---

## 1. AutoStrategy（推荐用法）

你只需要传一个 schema，LangChain 自动帮你选最优策略：

```python
from pydantic import BaseModel
from langchain.agents import create_agent

class WeatherReport(BaseModel):
    """天气预报结果"""
    city: str
    temperature: float
    condition: str
    humidity: int

# 最简用法：直接传 Pydantic model 类
agent = create_agent(
    model="openai:gpt-4o",
    tools=[get_weather_tool],
    response_format=WeatherReport,  # 自动包装为 AutoStrategy
)

result = agent.invoke({"messages": [("user", "北京今天天气怎么样？")]})
report = result["structured_response"]  # WeatherReport 实例
print(f"{report.city}: {report.temperature}°C, {report.condition}")
```

内部逻辑：当你传入裸的 Pydantic 类，`create_agent` 自动包装为 `AutoStrategy`，然后在 `_get_bound_model` 中检测模型能力：

```python
# factory.py 中的自动检测逻辑（简化）
if isinstance(response_format, AutoStrategy):
    if _supports_provider_strategy(request.model, tools=request.tools):
        effective_response_format = ProviderStrategy(schema=response_format.schema)
    else:
        effective_response_format = ToolStrategy(schema=response_format.schema)
```

---

## 2. ToolStrategy（兼容性最好）

核心思路：**把你的 schema 伪装成一个"工具"，让模型通过 tool_call 返回结构化数据**。

```
WeatherReport schema → 转换为工具定义（OutputToolBinding）
  → 模型看到名为 "WeatherReport" 的工具
  → 模型通过 tool_call 返回 {"city": "北京", ...}
  → LangChain 拦截这个 tool_call，解析为 WeatherReport 实例
  → 写入 state["structured_response"]
```

这不是真正的工具调用——模型以为自己在调用工具，实际上 LangChain 在"截胡"。

```python
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool],
    response_format=ToolStrategy(
        schema=WeatherReport,
        tool_message_content="天气查询完成",  # 自定义工具返回消息
        handle_errors=True,                    # 解析失败时自动重试
    ),
)
```

### handle_errors 详解

`handle_errors` 控制当模型返回的数据不符合 schema 时的行为：

```python
ToolStrategy(schema=MySchema, handle_errors=True)       # 默认：捕获错误，提示模型重试
ToolStrategy(schema=MySchema, handle_errors="请严格按照 schema 格式返回")  # 自定义错误消息
ToolStrategy(schema=MySchema, handle_errors=ValueError)  # 只捕获特定异常
ToolStrategy(schema=MySchema, handle_errors=lambda e: f"解析失败: {e}")  # 自定义处理函数
ToolStrategy(schema=MySchema, handle_errors=False)       # 不重试，直接抛异常
```

### Union 类型支持

ToolStrategy 支持 Union 类型，让模型从多个 schema 中选择：

```python
class SuccessResult(BaseModel):
    """操作成功"""
    data: str
    confidence: float

class ErrorResult(BaseModel):
    """操作失败"""
    error_code: int
    message: str

# 模型会看到两个"工具"，选择合适的那个
agent = create_agent(
    model="openai:gpt-4o",
    response_format=ToolStrategy(schema=SuccessResult | ErrorResult),
)
```

---

## 3. ProviderStrategy（性能最好）

直接使用模型提供商的原生结构化输出能力（如 OpenAI 的 `response_format={"type": "json_schema", ...}`）。模型从 token 生成层面就被约束为 JSON 格式，不需要"伪装成工具"。

```python
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool],
    response_format=ProviderStrategy(
        schema=WeatherReport,
        strict=True,  # 启用 OpenAI 严格模式
    ),
)
```

内部通过 `to_model_kwargs()` 将 schema 转换为模型 API 参数：

```python
def to_model_kwargs(self) -> dict[str, Any]:
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": self.schema_spec.name,
                "schema": self.schema_spec.json_schema,
                "strict": True,
            },
        }
    }
```

### ProviderStrategy vs ToolStrategy

| 维度 | ProviderStrategy | ToolStrategy |
|------|-----------------|--------------|
| 原理 | 模型 API 层面约束输出格式 | 伪装成工具调用 |
| 兼容性 | 仅支持的模型（OpenAI gpt-4o 等） | 所有支持 tool_call 的模型 |
| 输出质量 | 更高（token 级别约束） | 依赖模型遵循工具 schema |
| 错误重试 | 不支持（通常不需要） | 支持 handle_errors |
| Union 类型 | 不支持 | 支持 |

---

## 4. 支持的 Schema 类型

`response_format` 支持四种 schema 定义方式：

```python
SchemaKind = Literal["pydantic", "dataclass", "typeddict", "json_schema"]
```

### Pydantic BaseModel（推荐）

功能最完整，支持验证、默认值、嵌套模型：

```python
from pydantic import BaseModel, Field

class ExtractedEntity(BaseModel):
    """从文本中提取的实体"""
    name: str = Field(description="实体名称")
    entity_type: str = Field(description="实体类型：人名/地名/组织")
    confidence: float = Field(ge=0, le=1, description="置信度")
```

### dataclass / TypedDict / JSON Schema dict

```python
# dataclass — 标准库风格，返回类实例
@dataclass
class SearchResult:
    title: str
    url: str
    relevance_score: float

# TypedDict — 最轻量，返回普通字典
class SentimentResult(TypedDict):
    sentiment: str
    score: float

# JSON Schema dict — 最灵活，适合动态生成
schema = {
    "title": "TaskOutput",
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["success", "failure"]},
        "result": {"type": "string"},
    },
    "required": ["status", "result"],
}
```

**选择决策：** 需要验证 → Pydantic；想要类实例 → dataclass；想要字典 → TypedDict；动态 schema → dict。

---

## 5. 内部机制：OutputToolBinding

理解内部机制有助于排查问题。

无论传入哪种 schema，内部都会被转换为 `_SchemaSpec`（统一描述），然后根据策略生成对应的 Binding：

```python
# ToolStrategy 的执行单元
@dataclass
class OutputToolBinding(Generic[SchemaT]):
    schema: type[SchemaT] | dict[str, Any]
    schema_kind: SchemaKind
    tool: BaseTool                  # 对应的 LangChain 工具

    def parse(self, tool_args: dict[str, Any]) -> SchemaT:
        """将工具调用参数解析为 schema 实例"""
        return _parse_with_schema(self.schema, self.schema_kind, tool_args)

# ProviderStrategy 的执行单元
@dataclass
class ProviderStrategyBinding(Generic[SchemaT]):
    schema: type[SchemaT] | dict[str, Any]
    schema_kind: SchemaKind

    def parse(self, response: AIMessage) -> SchemaT:
        """从 AIMessage 文本中解析 JSON 并转换为 schema 实例"""
        data = json.loads(self._extract_text_content_from_message(response))
        return _parse_with_schema(self.schema, self.schema_kind, data)
```

---

## 6. 结构化输出在 Agent 循环中的位置

```
模型调用 → AIMessage (可能包含 tool_calls)
  ↓
检查 tool_calls 中是否有结构化输出工具
  ├─ 是 → 解析 → structured_response 存入 AgentState → 结束循环
  └─ 否 → 正常工具调用流程 → 继续循环
```

关键点：结构化输出工具的 tool_call 不会被当作普通工具执行，而是被 LangChain 拦截并解析。这意味着**结构化输出和普通工具可以共存**——模型可以先调用普通工具获取信息，最后通过结构化输出工具返回结果。

### 访问结构化响应

```python
result = agent.invoke({"messages": [("user", "查询北京天气")]})

weather = result["structured_response"]  # WeatherReport 实例
print(f"城市: {weather.city}, 温度: {weather.temperature}°C")

# 消息历史仍然在 messages 字段
for msg in result["messages"]:
    print(f"[{msg.type}] {msg.content[:50]}...")
```

---

## 7. 错误处理体系

结构化输出可能失败——模型返回的 JSON 不合法、字段缺失、类型不匹配。LangChain 提供了专门的异常层次：

```
StructuredOutputError (基类)
  ├── StructuredOutputValidationError  # 解析/验证失败
  └── MultipleStructuredOutputsError   # 模型返回了多个结构化输出
```

**StructuredOutputValidationError** — 模型返回的数据无法解析为目标 schema：

```python
class StructuredOutputValidationError(StructuredOutputError):
    tool_name: str        # 哪个 schema 解析失败
    source: Exception     # 原始异常（如 ValidationError）
    ai_message: AIMessage # 原始 AI 消息（用于调试）
```

**MultipleStructuredOutputsError** — 模型在一次响应中返回了多个结构化输出 tool_call（期望只有一个）：

```python
class MultipleStructuredOutputsError(StructuredOutputError):
    tool_names: list[str]  # 返回了哪些工具调用
    ai_message: AIMessage  # 原始 AI 消息
```

### 错误处理实战

```python
from langchain.agents.structured_output import (
    StructuredOutputValidationError,
    MultipleStructuredOutputsError,
    ToolStrategy,
)

# 方式一：让 ToolStrategy 自动重试（推荐）
agent = create_agent(
    model="openai:gpt-4o",
    response_format=ToolStrategy(schema=WeatherReport, handle_errors=True),
)

# 方式二：手动捕获异常
agent = create_agent(
    model="openai:gpt-4o",
    response_format=ToolStrategy(schema=WeatherReport, handle_errors=False),
)

try:
    result = agent.invoke({"messages": [("user", "查天气")]})
except StructuredOutputValidationError as e:
    print(f"解析失败: {e.source}")
    print(f"原始消息: {e.ai_message.content}")  # 调试关键
except MultipleStructuredOutputsError as e:
    print(f"模型返回了多个输出: {e.tool_names}")
```

所有异常都携带 `ai_message` 字段——你可以看到模型实际返回了什么，判断是 schema 定义问题还是模型能力问题。

---

## 8. 实战代码示例

### 场景一：带工具的结构化输出

Agent 先调用工具获取数据，再以结构化格式返回：

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.tools import tool

class TravelPlan(BaseModel):
    """旅行计划"""
    destination: str = Field(description="目的地")
    duration_days: int = Field(description="天数")
    budget: float = Field(description="预算（元）")
    highlights: list[str] = Field(description="推荐景点")

@tool
def search_flights(destination: str) -> str:
    """搜索航班信息"""
    return f"北京→{destination}：往返 2400 元，飞行 3 小时"

@tool
def search_hotels(destination: str, days: int) -> str:
    """搜索酒店信息"""
    return f"{destination} {days}晚住宿：均价 500 元/晚，总计 {days * 500} 元"

agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_flights, search_hotels],
    response_format=TravelPlan,
)

result = agent.invoke({"messages": [("user", "帮我规划一个去成都的 3 天旅行")]})
plan = result["structured_response"]
# TravelPlan(destination="成都", duration_days=3, budget=3900.0,
#            highlights=["宽窄巷子", "锦里", "大熊猫基地"])
```

### 场景二：Union 类型实现条件输出

```python
class AnswerFound(BaseModel):
    """找到了答案"""
    answer: str
    confidence: float
    sources: list[str]

class AnswerNotFound(BaseModel):
    """无法回答"""
    reason: str
    suggestions: list[str]

agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_knowledge_base],
    response_format=ToolStrategy(schema=AnswerFound | AnswerNotFound),
)

result = agent.invoke({"messages": [("user", "量子纠缠的速度是多少？")]})
response = result["structured_response"]

if isinstance(response, AnswerFound):
    print(f"答案: {response.answer} (置信度: {response.confidence})")
elif isinstance(response, AnswerNotFound):
    print(f"无法回答: {response.reason}")
```

### 场景三：自定义验证 + 自动重试

```python
from pydantic import BaseModel, Field, field_validator

class SentimentAnalysis(BaseModel):
    """情感分析结果"""
    text: str = Field(description="原始文本")
    sentiment: str = Field(description="情感倾向")
    score: float = Field(ge=-1.0, le=1.0, description="情感分数")

    @field_validator("sentiment")
    @classmethod
    def validate_sentiment(cls, v):
        allowed = {"positive", "negative", "neutral", "mixed"}
        if v not in allowed:
            raise ValueError(f"sentiment 必须是 {allowed} 之一")
        return v

agent = create_agent(
    model="openai:gpt-4o",
    response_format=ToolStrategy(
        schema=SentimentAnalysis,
        handle_errors=True,  # 验证失败时自动重试
    ),
)
```

---

## 9. 最佳实践

### 策略选择

1. **大多数情况直接传 Pydantic 类**（AutoStrategy），让框架自动选择
2. 需要 Union 类型时，显式使用 `ToolStrategy`
3. 确定模型支持且追求性能时，显式使用 `ProviderStrategy`

### Schema 设计

1. **给每个字段加 `Field(description=...)`**——这是模型理解字段含义的关键
2. **给类加 docstring**——会被用作工具描述
3. **用 Pydantic 验证器**保证输出质量，配合 `handle_errors=True` 自动重试
4. **避免过于复杂的嵌套**——模型对深层嵌套的遵循度会下降

### 错误处理

1. 生产环境始终开启 `handle_errors=True`
2. 对关键业务逻辑，额外用 `try/except` 捕获 `StructuredOutputError`
3. 利用异常中的 `ai_message` 字段做日志和调试

---

## 速记口诀

> **结构化输出三策略，Auto 自动最省力，**
> **Tool 伪装兼容广，Provider 原生最给力。**
> **四种 Schema 随你选，Pydantic 验证最靠谱。**

---

**来源：** `sourcecode/langchain/libs/langchain_v1/langchain/agents/factory.py`

**上一篇**: [03_核心概念_4_ModelRequest与ModelResponse数据流.md](./03_核心概念_4_ModelRequest与ModelResponse数据流.md)
**下一篇**: [04_最小可用.md](./04_最小可用.md)
