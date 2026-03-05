# 实战代码 场景3：结构化输出与动态 Prompt

> Agent 不只会说话，还能返回你定义好的数据结构；Prompt 不再写死，而是根据运行时上下文动态生成

---

## 场景概述

生产环境中，下游系统需要**可编程消费的结构化数据**，Prompt 也需要根据用户角色、业务场景**动态调整**。

本文用 5 个完整示例，覆盖结构化输出三种策略 + 动态 Prompt + 两者组合的实战模式。

---

## 示例1：AutoStrategy 基础结构化输出

**解决的问题：** Agent 返回自由文本，下游代码无法可靠解析。直接传 Pydantic 类，框架自动选最优策略。

```python
"""
基础结构化输出 - AutoStrategy
演示：直接传 Pydantic 类作为 response_format，框架自动选择最优策略
"""
import json
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.tools import tool


# ===== 1. 定义结构化输出 Schema =====
class WeatherReport(BaseModel):
    """天气报告结构"""
    city: str = Field(description="城市名称")
    temperature: float = Field(description="温度（摄氏度）")
    condition: str = Field(description="天气状况：晴/多云/阴/雨/雪")
    humidity: int = Field(description="湿度百分比")
    suggestion: str = Field(description="穿衣建议")


# ===== 2. 定义工具 =====
@tool
def get_weather(city: str) -> str:
    """获取指定城市的实时天气数据"""
    weather_data = {
        "北京": {"temp": 22, "condition": "晴", "humidity": 45},
        "上海": {"temp": 26, "condition": "多云", "humidity": 72},
        "广州": {"temp": 31, "condition": "雨", "humidity": 88},
    }
    data = weather_data.get(city, {"temp": 20, "condition": "未知", "humidity": 50})
    return json.dumps({"city": city, **data}, ensure_ascii=False)


# ===== 3. 创建带结构化输出的 Agent =====
agent = create_agent(
    model="openai:gpt-4o",
    tools=[get_weather],
    # 直接传 Pydantic 类 → 自动包装为 AutoStrategy
    response_format=WeatherReport,
    system_prompt="你是天气助手。查询天气后，返回结构化的天气报告，包含穿衣建议。",
)

# ===== 4. 调用并获取结构化响应 =====
result = agent.invoke(
    {"messages": [{"role": "user", "content": "北京今天天气怎么样？"}]}
)

# structured_response 是 WeatherReport 实例，不是字符串
weather = result["structured_response"]
print(f"城市: {weather.city}")
print(f"温度: {weather.temperature}°C")
print(f"天气: {weather.condition}")
print(f"湿度: {weather.humidity}%")
print(f"建议: {weather.suggestion}")
```

### 预期输出

```
城市: 北京
温度: 22.0°C
天气: 晴
湿度: 45%
建议: 天气晴朗，温度适宜，建议穿薄外套或长袖衬衫
```

---

## 示例2：ToolStrategy + Union 类型

**解决的问题：** 需要模型从多种输出格式中选择（成功/失败），或需要自定义错误重试。

```python
"""
ToolStrategy 结构化输出
演示：Union 类型让模型根据情况选择不同的输出结构
"""
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool


# ===== 1. 定义两种可能的输出 =====
class AnswerFound(BaseModel):
    """成功找到答案"""
    answer: str = Field(description="回答内容")
    confidence: float = Field(ge=0, le=1, description="置信度 0-1")
    sources: list[str] = Field(description="参考来源")


class AnswerNotFound(BaseModel):
    """无法回答"""
    reason: str = Field(description="无法回答的原因")
    suggestions: list[str] = Field(description="建议用户尝试的替代问题")


# ===== 2. 定义知识库搜索工具 =====
@tool
def search_knowledge_base(query: str) -> str:
    """搜索内部知识库"""
    knowledge = {
        "RAG": "RAG（检索增强生成）是将外部知识检索与LLM生成结合的技术架构",
        "Embedding": "Embedding 是将文本映射到高维向量空间的技术",
    }
    for key, value in knowledge.items():
        if key.lower() in query.lower():
            return f"找到相关知识: {value}"
    return "未找到相关信息"


# ===== 3. 使用 ToolStrategy + Union 类型 =====
agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_knowledge_base],
    response_format=ToolStrategy(
        schema=AnswerFound | AnswerNotFound,  # 模型会看到两个"工具"，选合适的
        handle_errors=True,                    # 解析失败自动重试
    ),
    system_prompt="你是知识库问答助手。找到答案返回答案和置信度；找不到说明原因并给建议。",
)

# ===== 4. 测试两种情况 =====
# 能找到答案
result = agent.invoke(
    {"messages": [{"role": "user", "content": "什么是 RAG？"}]}
)
response = result["structured_response"]

if isinstance(response, AnswerFound):
    print(f"[找到] {response.answer} (置信度: {response.confidence})")
elif isinstance(response, AnswerNotFound):
    print(f"[未找到] {response.reason}")

# 找不到答案
result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "量子纠缠的传播速度是多少？"}]}
)
response2 = result2["structured_response"]

if isinstance(response2, AnswerNotFound):
    print(f"[未找到] {response2.reason}")
    print(f"  建议: {response2.suggestions}")
```

### 预期输出

```
[找到] RAG（检索增强生成）是将外部知识检索与LLM生成结合的技术架构 (置信度: 0.95)
[未找到] 量子纠缠相关知识不在当前知识库范围内
  建议: ['尝试搜索物理学专业数据库', '咨询物理学领域专家']
```

---

## 示例3：ProviderStrategy 原生结构化输出

**解决的问题：** 确定模型支持原生结构化输出时，用 ProviderStrategy 获得 token 级别的格式保证。

```python
"""
ProviderStrategy 结构化输出
演示：使用模型提供商的原生 JSON Schema 约束
"""
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.tools import tool


class CodeReview(BaseModel):
    """代码审查结果"""
    file_name: str = Field(description="被审查的文件名")
    score: int = Field(ge=1, le=10, description="代码质量评分 1-10")
    issues: list[str] = Field(description="发现的问题列表")
    suggestions: list[str] = Field(description="改进建议列表")
    summary: str = Field(description="一句话总结")


@tool
def read_code_file(file_name: str) -> str:
    """读取代码文件内容"""
    code_samples = {
        "utils.py": 'import os\ndef get_config():\n    api_key = "sk-1234567890abcdef"\n    return {"key": api_key}',
    }
    return code_samples.get(file_name, f"文件 {file_name} 不存在")


agent = create_agent(
    model="openai:gpt-4o",
    tools=[read_code_file],
    response_format=ProviderStrategy(
        schema=CodeReview,
        strict=True,  # OpenAI 严格模式，100% 保证格式正确
    ),
    system_prompt="你是代码审查专家。评分标准：1-3差，4-6中等，7-9良好，10优秀。",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "审查一下 utils.py"}]}
)

review = result["structured_response"]
print(f"文件: {review.file_name} | 评分: {review.score}/10")
print(f"问题: {review.issues}")
print(f"总结: {review.summary}")
```

### 三种策略速查

| 策略 | 代码 | 适用场景 |
|------|------|----------|
| Auto（推荐） | `response_format=MyModel` | 大多数场景 |
| Tool | `ToolStrategy(schema=A \| B)` | 需要 Union 类型 |
| Provider | `ProviderStrategy(schema=MyModel, strict=True)` | 追求输出质量 |

---

## 示例4：@dynamic_prompt 动态 Prompt

**解决的问题：** 系统提示不能写死——不同用户角色需要不同的 Prompt。

```python
"""
动态 Prompt
演示：根据运行时 context 动态生成系统提示，同一个 Agent 服务不同角色
"""
from typing import TypedDict
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_core.tools import tool


# ===== 1. 定义运行时上下文类型 =====
class UserContext(TypedDict):
    user_role: str       # "beginner" | "expert"
    language: str        # "zh" | "en"
    department: str      # 所属部门


# ===== 2. 定义动态 Prompt 中间件 =====
@dynamic_prompt
def role_based_prompt(request: ModelRequest) -> str:
    """根据用户角色动态生成系统提示"""
    ctx = request.runtime.context
    role = ctx.get("user_role", "beginner")
    lang = ctx.get("language", "zh")
    dept = ctx.get("department", "通用")

    role_prompts = {
        "beginner": "你是耐心的技术导师。用简单语言解释概念，多用类比。",
        "expert": "你是高级技术顾问。直接给技术细节和最佳实践。",
    }

    prompt = role_prompts.get(role, role_prompts["beginner"])
    prompt += " Please respond in English." if lang == "en" else " 请用中文回答。"
    prompt += f"\n用户部门: {dept}"
    return prompt


# ===== 3. 定义工具 =====
@tool
def search_docs(query: str) -> str:
    """搜索技术文档"""
    return f"找到关于 '{query}' 的文档: RAG 是检索增强生成的缩写，核心思路是先检索再生成..."


# ===== 4. 创建 Agent =====
agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_docs],
    middleware=[role_based_prompt],
    context_schema=UserContext,
)

# ===== 5. 同一个 Agent，不同角色体验 =====
print("=== 初学者模式 ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "什么是 RAG？"}]},
    context={"user_role": "beginner", "language": "zh", "department": "产品部"},
)
print(result["messages"][-1].content[:200])

print("\n=== 专家模式 ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "什么是 RAG？"}]},
    context={"user_role": "expert", "language": "zh", "department": "技术部"},
)
print(result["messages"][-1].content[:200])
```

### 预期输出

```
=== 初学者模式 ===
RAG 就像是给 AI 配了一个"参考书"。想象你在考试，RAG 就是允许你翻书找答案：
先在书里找到相关内容（检索），然后用自己的话组织答案（生成）...

=== 专家模式 ===
RAG（Retrieval-Augmented Generation）架构核心：
1. Indexing: 文档分块 → Embedding → 向量存储
2. Retrieval: Query Embedding → ANN 检索 → Top-K 召回
3. Generation: 检索结果注入 Context → LLM 生成...
```

`@dynamic_prompt` 本质上创建了一个 `wrap_model_call` 中间件，内部用 `request.override(system_message=...)` 替换系统消息后传给下一层。

---

## 示例5：动态 Prompt + 结构化输出（组合实战）

**解决的问题：** 生产环境中两者往往同时使用——根据领域动态调整 Prompt，同时返回结构化数据。

```python
"""
动态 Prompt + 结构化输出 + 错误处理
演示：完整的生产级组合模式
"""
from typing import TypedDict
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents.structured_output import (
    ToolStrategy,
    StructuredOutputValidationError,
)
from langchain_core.tools import tool


# ===== 1. 定义上下文和输出类型 =====
class AnalysisContext(TypedDict):
    domain: str         # "tech" | "finance" | "health"
    detail_level: str   # "brief" | "detailed"


class AnalysisResult(BaseModel):
    """结构化分析结果"""
    question: str = Field(description="用户的原始问题")
    answer: str = Field(description="分析回答")
    confidence: float = Field(ge=0, le=1, description="置信度 0-1")
    key_points: list[str] = Field(description="关键要点列表")
    sources: list[str] = Field(description="参考来源")
    follow_up_questions: list[str] = Field(description="推荐追问问题")


# ===== 2. 动态 Prompt =====
@dynamic_prompt
def domain_expert_prompt(request: ModelRequest) -> str:
    """根据领域动态生成专家 Prompt"""
    ctx = request.runtime.context
    domain = ctx.get("domain", "tech")
    detail = ctx.get("detail_level", "brief")

    domain_configs = {
        "tech": ("资深技术架构师", "关注技术实现细节和最佳实践"),
        "finance": ("金融分析师", "关注市场趋势和风险评估"),
        "health": ("健康顾问", "关注科学依据，强调就医建议"),
    }

    role, focus = domain_configs.get(domain, domain_configs["tech"])
    prompt = f"你是一位{role}。{focus}。"

    if detail == "detailed":
        prompt += "请提供详细分析，包含具体数据和案例。"
    else:
        prompt += "请简洁回答，突出核心要点。"

    return prompt


# ===== 3. 定义工具 =====
@tool
def search_domain_knowledge(query: str, domain: str = "tech") -> str:
    """搜索特定领域的知识库"""
    results = {
        "tech": f"技术资料: {query} 相关的最新技术文档和实践案例...",
        "finance": f"金融数据: {query} 相关的市场报告和分析数据...",
        "health": f"医学文献: {query} 相关的临床研究和指南...",
    }
    return results.get(domain, f"通用搜索结果: {query}")


# ===== 4. 组装 Agent =====
agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_domain_knowledge],
    middleware=[domain_expert_prompt],
    response_format=ToolStrategy(
        schema=AnalysisResult,
        handle_errors=True,
    ),
    context_schema=AnalysisContext,
)

# ===== 5. 技术领域 - 详细模式 =====
print("=== 技术领域分析 ===")
try:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "RAG 系统如何优化检索质量？"}]},
        context={"domain": "tech", "detail_level": "detailed"},
    )
    analysis = result["structured_response"]
    print(f"问题: {analysis.question}")
    print(f"置信度: {analysis.confidence}")
    print(f"关键要点:")
    for point in analysis.key_points:
        print(f"  - {point}")

except StructuredOutputValidationError as e:
    print(f"解析失败: {e.source}")
    print(f"模型原始回复: {e.ai_message.content[:200]}")

# ===== 6. 金融领域 - 简洁模式 =====
print("\n=== 金融领域分析 ===")
result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "2026年AI行业投资前景如何？"}]},
    context={"domain": "finance", "detail_level": "brief"},
)
analysis2 = result2["structured_response"]
print(f"回答: {analysis2.answer[:100]}...")
print(f"置信度: {analysis2.confidence}")
```

### 预期输出

```
=== 技术领域分析 ===
问题: RAG 系统如何优化检索质量？
置信度: 0.88
关键要点:
  - 使用混合检索（向量 + BM25）提升召回率
  - 引入 ReRank 模型对检索结果重排序
  - 优化 Chunking 策略，保持语义完整性

=== 金融领域分析 ===
回答: AI行业2026年投资前景整体乐观，企业级AI应用落地加速...
置信度: 0.72
```

---

## 关键要点

### 动态 Prompt 核心模式

```python
# 四步走：定义 context → 定义 @dynamic_prompt → 传入 middleware → 调用时传 context
class MyContext(TypedDict):
    role: str

@dynamic_prompt
def my_prompt(request: ModelRequest) -> str:
    return f"你是 {request.runtime.context['role']}"

agent = create_agent(model="openai:gpt-4o", middleware=[my_prompt], context_schema=MyContext)
agent.invoke({"messages": [...]}, context={"role": "专家"})
```

### 常见陷阱

1. `response_format` 传**类**不是实例：`WeatherReport` 而不是 `WeatherReport()`
2. 结构化响应在 `result["structured_response"]` 中，不在 `result["messages"]` 中
3. `@dynamic_prompt` 返回**字符串**（新的 system prompt），不是 dict
4. `context_schema` 和 `context={}` 要配套使用
5. ProviderStrategy 不支持 Union 类型，需要 Union 时用 ToolStrategy

---

[来源: reference/context7_langchain_01.md]
[来源: reference/fetch_1dot0_blog_02.md]
[来源: reference/source_create_agent_01.md]

**上一篇**: [07_实战代码_场景2_自定义中间件开发.md](./07_实战代码_场景2_自定义中间件开发.md)
**下一篇**: [07_实战代码_场景4_从AgentExecutor迁移.md](./07_实战代码_场景4_从AgentExecutor迁移.md)
