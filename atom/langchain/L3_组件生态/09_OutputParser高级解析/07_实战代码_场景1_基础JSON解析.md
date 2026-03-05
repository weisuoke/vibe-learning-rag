# 实战代码 - 场景1：基础 JSON 解析

> **数据来源**：
> - `reference/source_outputparser_02_JSON解析器.md` - JsonOutputParser 源码分析
> - `reference/context7_langchain_02_Pydantic模型.md` - LangChain 官方文档

---

## 场景概述

JsonOutputParser 是 LangChain 中最可靠的结构化输出解析器，用于将 LLM 的文本输出解析为 JSON 对象。它支持：
- 自动从 Markdown 代码块中提取 JSON
- 与 LCEL 无缝集成
- 流式解析（partial 模式）
- JSON Patch diff 模式

**适用场景**：
- 需要灵活的 JSON 输出（schema 不固定）
- 模型不支持原生结构化输出
- 需要完整的流式支持
- 需要自定义解析逻辑

---

## 示例 1：基础 JSON 解析

**目标**：使用 JsonOutputParser 解析 LLM 输出的 JSON 对象

**完整代码**：

```python
"""
示例 1：基础 JSON 解析
演示 JsonOutputParser 的基本使用
"""

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# 1. 初始化 LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",  # 使用更便宜的模型
    temperature=0,  # 确保输出稳定
)

# 2. 创建 JsonOutputParser
parser = JsonOutputParser()

# 3. 创建 Prompt（包含格式指令）
prompt = PromptTemplate(
    template="""提取以下文本中的人物信息，返回 JSON 格式：

文本：{text}

请返回包含以下字段的 JSON：
- name: 人物姓名
- age: 年龄（数字）
- occupation: 职业

{format_instructions}
""",
    input_variables=["text"],
    partial_variables={
        "format_instructions": "请确保输出是有效的 JSON 格式。"
    },
)

# 4. 构建 LCEL 链
chain = prompt | llm | parser

# 5. 执行解析
text = "张三是一位 30 岁的软件工程师，他在一家科技公司工作。"

try:
    result = chain.invoke({"text": text})
    print("✓ 解析成功！")
    print(f"类型: {type(result)}")
    print(f"结果: {result}")
    print(f"\n访问字段:")
    print(f"  姓名: {result['name']}")
    print(f"  年龄: {result['age']}")
    print(f"  职业: {result['occupation']}")
except Exception as e:
    print(f"✗ 解析失败: {e}")

"""
预期输出：
✓ 解析成功！
类型: <class 'dict'>
结果: {'name': '张三', 'age': 30, 'occupation': '软件工程师'}

访问字段:
  姓名: 张三
  年龄: 30
  职业: 软件工程师
"""
```

**关键点**：
1. **JsonOutputParser 无需参数**：直接创建即可使用
2. **返回 Python dict**：可以直接访问字段
3. **与 LCEL 集成**：使用 `|` 操作符连接
4. **自动解析**：LLM 输出的 JSON 字符串自动转换为 dict

---

## 示例 2：处理 Markdown 代码块

**目标**：自动从 Markdown 代码块中提取 JSON

**完整代码**：

```python
"""
示例 2：处理 Markdown 代码块
演示 JsonOutputParser 自动处理 Markdown 格式
"""

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# 1. 初始化组件
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = JsonOutputParser()

# 2. 创建 Prompt（明确要求 Markdown 格式）
prompt = PromptTemplate(
    template="""分析以下产品评论，提取关键信息。

评论：{review}

请返回 JSON 格式，包含：
- sentiment: 情感（positive/negative/neutral）
- rating: 评分（1-5）
- keywords: 关键词列表

请使用 Markdown 代码块格式返回 JSON。
""",
    input_variables=["review"],
)

# 3. 构建链
chain = prompt | llm | parser

# 4. 测试不同的 Markdown 格式
test_cases = [
    "这个产品非常好用，质量很棒，强烈推荐！",
    "价格有点贵，但是功能确实不错。",
    "完全不值这个价格，质量太差了。",
]

print("=" * 60)
print("测试 Markdown 代码块解析")
print("=" * 60)

for i, review in enumerate(test_cases, 1):
    print(f"\n【测试 {i}】")
    print(f"评论: {review}")

    try:
        result = chain.invoke({"review": review})
        print(f"✓ 解析成功")
        print(f"  情感: {result['sentiment']}")
        print(f"  评分: {result['rating']}")
        print(f"  关键词: {', '.join(result['keywords'])}")
    except Exception as e:
        print(f"✗ 解析失败: {e}")

"""
预期输出：
============================================================
测试 Markdown 代码块解析
============================================================

【测试 1】
评论: 这个产品非常好用，质量很棒，强烈推荐！
✓ 解析成功
  情感: positive
  评分: 5
  关键词: 好用, 质量, 推荐

【测试 2】
评论: 价格有点贵，但是功能确实不错。
✓ 解析成功
  情感: neutral
  评分: 3
  关键词: 价格, 功能

【测试 3】
评论: 完全不值这个价格，质量太差了。
✓ 解析成功
  情感: negative
  评分: 1
  关键词: 价格, 质量
"""
```

**关键点**：
1. **自动处理 Markdown**：即使 LLM 返回 ` ```json ... ``` ` 格式，也能正确解析
2. **parse_json_markdown() 工具**：内部使用此函数提取 JSON
3. **容错性强**：支持多种 Markdown 格式

---

## 示例 3：与 LCEL 高级集成

**目标**：在复杂的 LCEL 链中使用 JsonOutputParser

**完整代码**：

```python
"""
示例 3：与 LCEL 高级集成
演示 JsonOutputParser 在复杂链中的使用
"""

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()

# 1. 初始化组件
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = JsonOutputParser()

# 2. 创建多步骤 Prompt
system_prompt = """你是一个数据分析助手。
你的任务是分析用户输入的文本，提取结构化信息。
始终返回有效的 JSON 格式。"""

user_prompt = """分析以下文本：

{text}

提取以下信息：
1. 主题（topic）
2. 实体列表（entities）：人名、地名、组织名
3. 时间信息（time_info）：如果有的话
4. 摘要（summary）：一句话概括

返回 JSON 格式。"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", user_prompt),
])

# 3. 构建复杂链（包含预处理和后处理）
def preprocess(inputs):
    """预处理：清理文本"""
    text = inputs["text"].strip()
    return {"text": text}

def postprocess(result):
    """后处理：添加元数据"""
    result["processed_at"] = "2026-02-26"
    result["parser_version"] = "1.0"
    return result

# 完整链
chain = (
    RunnablePassthrough()
    | preprocess
    | prompt
    | llm
    | parser
    | postprocess
)

# 4. 测试
text = """
2026年2月26日，OpenAI 在旧金山发布了 GPT-5 模型。
CEO Sam Altman 表示这是公司历史上最重要的里程碑。
新模型在推理能力上有显著提升。
"""

print("=" * 60)
print("复杂 LCEL 链测试")
print("=" * 60)

try:
    result = chain.invoke({"text": text})
    print("\n✓ 解析成功！")
    print(f"\n完整结果:")
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))
except Exception as e:
    print(f"\n✗ 解析失败: {e}")

"""
预期输出：
============================================================
复杂 LCEL 链测试
============================================================

✓ 解析成功！

完整结果:
{
  "topic": "GPT-5 模型发布",
  "entities": [
    "OpenAI",
    "旧金山",
    "Sam Altman"
  ],
  "time_info": "2026年2月26日",
  "summary": "OpenAI 发布 GPT-5 模型，推理能力显著提升",
  "processed_at": "2026-02-26",
  "parser_version": "1.0"
}
"""
```

**关键点**：
1. **RunnablePassthrough**：传递输入到下一步
2. **预处理和后处理**：可以在解析前后添加自定义逻辑
3. **链式组合**：多个步骤无缝连接
4. **元数据添加**：后处理可以添加额外信息

---

## 示例 4：流式解析（Partial 模式）

**目标**：在流式场景中逐步解析 JSON

**完整代码**：

```python
"""
示例 4：流式解析（Partial 模式）
演示 JsonOutputParser 的流式解析能力
"""

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# 1. 初始化组件
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = JsonOutputParser()

# 2. 创建 Prompt
prompt = PromptTemplate(
    template="""生成一个包含 5 个字段的 JSON 对象，描述一本书：

书名：{book_title}

字段：
- title: 书名
- author: 作者
- year: 出版年份
- genre: 类型
- summary: 简介（50字以内）

返回 JSON 格式。
""",
    input_variables=["book_title"],
)

# 3. 构建链
chain = prompt | llm | parser

# 4. 流式解析
book_title = "三体"

print("=" * 60)
print("流式解析测试")
print("=" * 60)
print(f"\n正在解析: {book_title}\n")

try:
    # 使用 stream() 方法
    for i, chunk in enumerate(chain.stream({"book_title": book_title}), 1):
        print(f"[Chunk {i}]")
        print(f"  类型: {type(chunk)}")
        print(f"  内容: {chunk}")
        print()

    print("✓ 流式解析完成！")
except Exception as e:
    print(f"✗ 流式解析失败: {e}")

"""
预期输出：
============================================================
流式解析测试
============================================================

正在解析: 三体

[Chunk 1]
  类型: <class 'dict'>
  内容: {'title': '三体', 'author': '刘慈欣', 'year': 2008, 'genre': '科幻', 'summary': '讲述地球文明与三体文明的接触和冲突'}

✓ 流式解析完成！

注意：JsonOutputParser 在流式模式下会累积解析，
只有当完整 JSON 可用时才会输出。
如果需要逐块解析，使用 BaseCumulativeTransformOutputParser。
"""
```

**关键点**：
1. **stream() 方法**：支持流式输出
2. **累积解析**：JsonOutputParser 会等待完整 JSON
3. **partial=True**：内部使用此参数处理部分 JSON
4. **适用场景**：实时显示解析进度

---

## 示例 5：错误处理与调试

**目标**：处理常见的 JSON 解析错误

**完整代码**：

```python
"""
示例 5：错误处理与调试
演示如何处理 JsonOutputParser 的常见错误
"""

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, OutputParserException
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# 1. 初始化组件
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = JsonOutputParser()

# 2. 创建 Prompt（故意不明确格式要求）
prompt = PromptTemplate(
    template="""回答以下问题：{question}

请提供详细的答案。
""",
    input_variables=["question"],
)

# 3. 构建链
chain = prompt | llm | parser

# 4. 测试错误处理
test_cases = [
    {
        "question": "什么是 Python？",
        "expected_error": "LLM 可能返回纯文本而非 JSON",
    },
    {
        "question": "用 JSON 格式列出 Python 的 3 个特点",
        "expected_error": "可能成功（因为明确要求 JSON）",
    },
]

print("=" * 60)
print("错误处理测试")
print("=" * 60)

for i, test in enumerate(test_cases, 1):
    print(f"\n【测试 {i}】")
    print(f"问题: {test['question']}")
    print(f"预期: {test['expected_error']}")

    try:
        result = chain.invoke({"question": test["question"]})
        print(f"✓ 解析成功")
        print(f"  结果: {result}")
    except OutputParserException as e:
        print(f"✗ 解析失败（OutputParserException）")
        print(f"  错误信息: {str(e)[:100]}...")
        print(f"  原始输出: {e.llm_output[:100] if e.llm_output else 'N/A'}...")
    except Exception as e:
        print(f"✗ 其他错误: {type(e).__name__}")
        print(f"  错误信息: {str(e)[:100]}...")

# 5. 改进的错误处理策略
print("\n" + "=" * 60)
print("改进的错误处理策略")
print("=" * 60)

def safe_parse_json(chain, inputs, max_retries=3):
    """安全的 JSON 解析，带重试机制"""
    for attempt in range(max_retries):
        try:
            result = chain.invoke(inputs)
            return {"success": True, "data": result, "attempts": attempt + 1}
        except OutputParserException as e:
            print(f"  尝试 {attempt + 1}/{max_retries} 失败")
            if attempt == max_retries - 1:
                return {
                    "success": False,
                    "error": str(e),
                    "llm_output": e.llm_output,
                    "attempts": max_retries,
                }
            # 可以在这里添加 Prompt 改进逻辑
            continue

    return {"success": False, "error": "Max retries exceeded"}

# 测试改进的策略
question = "什么是 Python？"
print(f"\n测试问题: {question}")
result = safe_parse_json(chain, {"question": question})

if result["success"]:
    print(f"✓ 解析成功（尝试 {result['attempts']} 次）")
    print(f"  结果: {result['data']}")
else:
    print(f"✗ 解析失败（尝试 {result['attempts']} 次）")
    print(f"  错误: {result['error'][:100]}...")

"""
预期输出：
============================================================
错误处理测试
============================================================

【测试 1】
问题: 什么是 Python？
预期: LLM 可能返回纯文本而非 JSON
✗ 解析失败（OutputParserException）
  错误信息: Invalid json output: Python 是一种高级编程语言...
  原始输出: Python 是一种高级编程语言...

【测试 2】
问题: 用 JSON 格式列出 Python 的 3 个特点
预期: 可能成功（因为明确要求 JSON）
✓ 解析成功
  结果: {'features': ['简洁易读', '功能强大', '社区活跃']}

============================================================
改进的错误处理策略
============================================================

测试问题: 什么是 Python？
  尝试 1/3 失败
  尝试 2/3 失败
  尝试 3/3 失败
✗ 解析失败（尝试 3 次）
  错误: Invalid json output: Python 是一种高级编程语言...
"""
```

**关键点**：
1. **OutputParserException**：专门的解析异常
2. **llm_output 属性**：包含原始 LLM 输出
3. **重试机制**：可以多次尝试解析
4. **Prompt 改进**：明确要求 JSON 格式可以提高成功率

---

## 常见错误与解决方案

### 错误 1：Invalid json output

**原因**：LLM 返回的不是有效的 JSON 格式

**解决方案**：
```python
# 1. 在 Prompt 中明确要求 JSON 格式
prompt = PromptTemplate(
    template="""...

请严格按照 JSON 格式返回，不要包含任何其他文本。

示例格式：
{{"key1": "value1", "key2": "value2"}}
""",
    input_variables=[...],
)

# 2. 使用 OutputFixingParser 自动修复
from langchain.output_parsers import OutputFixingParser

fixing_parser = OutputFixingParser.from_llm(
    parser=parser,
    llm=llm,
)
```

### 错误 2：Markdown 代码块未被识别

**原因**：parse_json_markdown() 无法识别特殊格式

**解决方案**：
```python
# JsonOutputParser 已经内置 Markdown 处理
# 如果仍然失败，检查 LLM 输出格式

# 手动测试
from langchain_core.utils.json import parse_json_markdown

text = """```json
{"name": "Alice"}
```"""

result = parse_json_markdown(text)
print(result)  # {'name': 'Alice'}
```

### 错误 3：流式解析不输出中间结果

**原因**：JsonOutputParser 等待完整 JSON

**解决方案**：
```python
# 使用 BaseCumulativeTransformOutputParser
from langchain_core.output_parsers import BaseCumulativeTransformOutputParser

# 或者使用 SimpleJsonOutputParser（更宽松）
from langchain_core.output_parsers import SimpleJsonOutputParser

parser = SimpleJsonOutputParser()
```

---

## 最佳实践

### 1. Prompt 工程

```python
# ✓ 好的 Prompt
prompt = PromptTemplate(
    template="""提取信息并返回 JSON。

输入：{text}

输出格式（严格遵守）：
{{
  "field1": "value1",
  "field2": "value2"
}}

不要包含任何解释或额外文本。
""",
    input_variables=["text"],
)

# ✗ 不好的 Prompt
prompt = PromptTemplate(
    template="提取信息：{text}",
    input_variables=["text"],
)
```

### 2. 模型选择

```python
# 更强大的模型输出更稳定
llm_strong = ChatOpenAI(model="gpt-4o", temperature=0)  # 推荐
llm_weak = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # 可能不稳定
```

### 3. 错误处理

```python
# 始终使用 try-except
try:
    result = chain.invoke(inputs)
except OutputParserException as e:
    # 记录错误
    logger.error(f"解析失败: {e.llm_output}")
    # 回退策略
    result = {"error": "解析失败"}
```

### 4. 调试技巧

```python
# 打印中间结果
chain_with_debug = (
    prompt
    | llm
    | (lambda x: print(f"LLM 输出: {x.content}") or x)  # 调试
    | parser
)
```

---

## 性能优化

### 1. 使用更便宜的模型

```python
# 简单任务使用 gpt-4o-mini
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

### 2. 批量处理

```python
# 使用 batch() 方法
inputs = [{"text": text1}, {"text": text2}, {"text": text3}]
results = chain.batch(inputs)
```

### 3. 缓存

```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())
```

---

## 总结

JsonOutputParser 的核心优势：
1. **灵活性**：无需预定义 schema
2. **Markdown 友好**：自动处理代码块
3. **LCEL 集成**：无缝连接
4. **流式支持**：支持实时解析

**何时使用**：
- ✓ Schema 不固定或动态变化
- ✓ 需要快速原型开发
- ✓ 模型不支持原生结构化输出
- ✓ 需要完整的流式支持

**何时不使用**：
- ✗ 需要严格的类型验证（使用 PydanticOutputParser）
- ✗ 模型支持原生结构化输出（使用 with_structured_output()）
- ✗ 需要复杂的嵌套验证（使用 Pydantic 模型）
