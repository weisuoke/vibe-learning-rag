---
type: source_code_analysis
source: sourcecode/langchain/libs/core/langchain_core/prompts/chat.py
analyzed_files: [sourcecode/langchain/libs/core/langchain_core/prompts/chat.py]
analyzed_at: 2026-02-26
knowledge_point: 11_ChatPromptTemplate对话模板
---

# 源码分析：ChatPromptTemplate 核心实现

## 分析的文件
- `sourcecode/langchain/libs/core/langchain_core/prompts/chat.py` (1484 lines) - ChatPromptTemplate 核心实现

## 关键发现

### 1. 核心类架构

#### 1.1 MessagesPlaceholder (lines 52-217)
**功能**: 对话历史占位符，用于在模板中插入消息列表

**关键特性**:
```python
class MessagesPlaceholder(BaseMessagePromptTemplate):
    variable_name: str  # 变量名
    optional: bool = False  # 是否可选
    n_messages: PositiveInt | None = None  # 限制消息数量
```

**核心方法**:
- `format_messages(**kwargs)`: 格式化消息列表
- 支持 `optional=True` 时可以不传递参数
- 支持 `n_messages` 限制返回的消息数量（取最后 N 条）

**使用示例** (from docstring):
```python
# 基础用法
prompt = MessagesPlaceholder("history")
prompt.format_messages(history=[("human", "Hi!"), ("ai", "Hello!")])

# 可选占位符
prompt = MessagesPlaceholder("history", optional=True)
prompt.format_messages()  # 返回空列表

# 限制消息数量
prompt = MessagesPlaceholder("history", n_messages=1)
prompt.format_messages(history=[("system", "..."), ("human", "Hello!")])
# 只返回最后1条消息
```

#### 1.2 BaseStringMessagePromptTemplate (lines 225-351)
**功能**: 基于字符串模板的消息模板基类

**关键特性**:
```python
class BaseStringMessagePromptTemplate(BaseMessagePromptTemplate, ABC):
    prompt: StringPromptTemplate  # 字符串模板
    additional_kwargs: dict = Field(default_factory=dict)  # 额外参数
```

**核心方法**:
- `from_template(template, template_format, partial_variables)`: 从模板字符串创建
- `from_template_file(template_file)`: 从文件加载模板
- `format(**kwargs)`: 格式化为单个消息
- `format_messages(**kwargs)`: 格式化为消息列表

#### 1.3 ChatMessagePromptTemplate (lines 353-386)
**功能**: 自定义角色的消息模板

**关键特性**:
```python
class ChatMessagePromptTemplate(BaseStringMessagePromptTemplate):
    role: str  # 自定义角色名称
```

**使用场景**: 当需要使用 system/human/ai 之外的自定义角色时

#### 1.4 _StringImageMessagePromptTemplate (lines 396-662)
**功能**: 支持多模态（文本+图片）的消息模板基类

**关键特性**:
- 支持纯文本模板: `"Hello {name}"`
- 支持文本+图片列表: `[{"text": "..."}, {"image_url": "..."}]`
- 图片支持 URL 或本地路径
- 图片支持 detail 参数（low/high/auto）

**from_template 支持的格式**:
```python
# 纯文本
template = "Hello {name}"

# 文本+图片
template = [
    {"text": "Describe this image: {description}"},
    {"image_url": "{image_path}"}
]
```

#### 1.5 HumanMessagePromptTemplate (lines 663-670)
**功能**: Human 消息模板（继承自 _StringImageMessagePromptTemplate）

**特点**: 支持多模态（文本+图片）

#### 1.6 AIMessagePromptTemplate (lines 672-679)
**功能**: AI 消息模板（继承自 _StringImageMessagePromptTemplate）

**特点**: 支持多模态（文本+图片）

#### 1.7 SystemMessagePromptTemplate (lines 681-688)
**功能**: System 消息模板（继承自 _StringImageMessagePromptTemplate）

**特点**: 支持多模态（文本+图片）

#### 1.8 ChatPromptTemplate (lines 789-1484)
**功能**: 对话模板的核心类

**关键特性**:
```python
class ChatPromptTemplate(BaseChatPromptTemplate):
    messages: list[MessageLike]  # 消息列表
    validate_template: bool = False  # 是否验证模板
```

**核心方法**:

1. **构造方法**:
```python
def __init__(
    self,
    messages: Sequence[MessageLikeRepresentation],
    *,
    template_format: PromptTemplateFormat = "f-string",
    **kwargs
)
```

支持的消息格式:
- `BaseMessagePromptTemplate` - 消息模板对象
- `BaseMessage` - 消息对象
- `(message_type, template)` - 元组格式，如 `("human", "{input}")`
- `(message_class, template)` - 类+模板
- `str` - 字符串（默认为 human 消息）

2. **类方法**:
- `from_template(template)`: 从单个模板字符串创建（默认为 human 消息）
- `from_messages(messages, template_format)`: 从消息列表创建

3. **格式化方法**:
- `format_messages(**kwargs)`: 格式化为消息列表
- `aformat_messages(**kwargs)`: 异步格式化

4. **部分变量**:
```python
def partial(**kwargs) -> ChatPromptTemplate:
    """预填充部分变量"""
```

5. **模板组合**:
```python
def __add__(self, other) -> ChatPromptTemplate:
    """使用 + 操作符组合模板"""
```

支持的组合方式:
- `ChatPromptTemplate + ChatPromptTemplate`
- `ChatPromptTemplate + BaseMessagePromptTemplate`
- `ChatPromptTemplate + BaseMessage`
- `ChatPromptTemplate + list/tuple`
- `ChatPromptTemplate + str`

6. **动态修改**:
- `append(message)`: 追加单个消息
- `extend(messages)`: 追加多个消息
- `__getitem__(index)`: 索引访问或切片

7. **变量自动推断**:
- 自动从消息模板中提取 `input_variables`
- 自动处理 `optional_variables`（来自 optional MessagesPlaceholder）
- 自动合并 `partial_variables`

### 2. 关键特性总结

#### 2.1 消息模板类型
- **SystemMessagePromptTemplate**: 系统消息
- **HumanMessagePromptTemplate**: 用户消息
- **AIMessagePromptTemplate**: AI 消息
- **ChatMessagePromptTemplate**: 自定义角色消息
- **MessagesPlaceholder**: 对话历史占位符

#### 2.2 多模态支持
- 所有消息模板（除 ChatMessagePromptTemplate）都支持文本+图片
- 图片支持 URL 和本地路径
- 图片支持 detail 参数控制分辨率

#### 2.3 模板格式
- 支持 f-string（默认）: `"{variable}"`
- 支持 mustache: `"{{variable}}"`
- 支持 jinja2: `"{{ variable }}"`

#### 2.4 Partial Variables
- 支持字符串固定值: `partial(name="Alice")`
- 支持函数动态值: `partial(date=lambda: datetime.now())`
- 在 `format_messages` 时自动合并

#### 2.5 模板组合
- 使用 `+` 操作符组合多个模板
- 自动合并 input_variables（取并集）
- 自动合并 partial_variables（检查冲突）

#### 2.6 MessagesPlaceholder 特性
- 支持 optional 参数（可选占位符）
- 支持 n_messages 限制消息数量
- 自动转换元组为消息对象

#### 2.7 模板验证
- 通过 `validate_template=True` 启用
- 检查模板变量与 input_variables 是否匹配
- 在构造时验证

#### 2.8 动态修改
- `append()` / `extend()`: 动态添加消息
- `__getitem__()`: 索引访问和切片
- 支持运行时修改模板结构

### 3. 与 PromptTemplate 的关系

#### 3.1 继承关系
```
BasePromptTemplate (基类)
├── PromptTemplate (文本模板)
└── BaseChatPromptTemplate (对话模板基类)
    └── ChatPromptTemplate (对话模板)
```

#### 3.2 共享特性
- 都支持 partial_variables
- 都支持 template_format (f-string/mustache/jinja2)
- 都支持 validate_template
- 都支持 input_variables 自动推断

#### 3.3 独特特性
ChatPromptTemplate 独有:
- 消息列表管理（messages）
- 多种消息类型（System/Human/AI/Chat）
- MessagesPlaceholder（对话历史）
- 多模态支持（文本+图片）
- 消息级别的模板组合

### 4. 实际应用场景

#### 4.1 基础对话模板
```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])
```

#### 4.2 带历史记录的对话
```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])
```

#### 4.3 Few-shot 学习
```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are a translator."),
    ("human", "Hello"),
    ("ai", "Bonjour"),
    ("human", "Goodbye"),
    ("ai", "Au revoir"),
    ("human", "{input}")
])
```

#### 4.4 多模态对话
```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are a vision assistant."),
    ("human", [
        {"text": "Describe this image:"},
        {"image_url": "{image_path}"}
    ])
])
```

#### 4.5 动态模板组合
```python
base = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant.")
])
extended = base + ("human", "{input}")
```

### 5. 性能与优化考虑

#### 5.1 变量自动推断
- 在构造时自动提取所有 input_variables
- 避免手动指定变量列表
- 减少人为错误

#### 5.2 Optional MessagesPlaceholder
- 使用 optional=True 避免必须传递空列表
- 自动设置 partial_variables 为空列表
- 简化调用代码

#### 5.3 消息数量限制
- 使用 n_messages 限制历史消息数量
- 避免上下文窗口溢出
- 自动取最后 N 条消息

### 6. 常见陷阱与注意事项

#### 6.1 Partial Variables 兼容性
- 某些版本中 ChatPromptTemplate 的 partial_variables 可能有兼容性问题
- 参考 GitHub Issue #17560
- 建议在使用前测试

#### 6.2 模板格式一致性
- 组合模板时，template_format 必须一致
- 不能混用 f-string 和 mustache

#### 6.3 变量冲突
- partial_variables 中的同名变量会导致冲突
- input_variables 会自动合并（取并集）

#### 6.4 图片模板限制
- 每个图片模板只能有一个变量
- 不支持 partial_variables 用于图片列表模板

### 7. 源码位置参考

- **MessagesPlaceholder**: lines 52-217
- **BaseStringMessagePromptTemplate**: lines 225-351
- **ChatMessagePromptTemplate**: lines 353-386
- **_StringImageMessagePromptTemplate**: lines 396-662
- **HumanMessagePromptTemplate**: lines 663-670
- **AIMessagePromptTemplate**: lines 672-679
- **SystemMessagePromptTemplate**: lines 681-688
- **ChatPromptTemplate**: lines 789-1484

### 8. 依赖库识别

从源码中识别的依赖库:
- `langchain_core.messages`: 消息类型定义
- `langchain_core.prompt_values`: ChatPromptValue, ImageURL
- `langchain_core.prompts.base`: BasePromptTemplate
- `langchain_core.prompts.prompt`: PromptTemplate
- `langchain_core.prompts.string`: StringPromptTemplate, PromptTemplateFormat
- `pydantic`: 数据验证

### 9. 下一步调研方向

基于源码分析，需要进一步调研:
1. **Context7 官方文档**: LangChain ChatPromptTemplate 官方文档
2. **社区实践**: ChatPromptTemplate 实际应用案例
3. **多模态支持**: 图片消息的实际使用
4. **MessagesPlaceholder**: 对话历史管理最佳实践
5. **Partial Variables**: 兼容性问题和解决方案
6. **模板组合**: 复杂场景的组合策略
