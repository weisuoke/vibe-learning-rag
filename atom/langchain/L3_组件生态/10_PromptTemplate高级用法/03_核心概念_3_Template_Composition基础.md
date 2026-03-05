# 核心概念3：Template Composition 基础

> 本文档深入讲解 LangChain PromptTemplate 中 Template Composition（模板组合）的基础用法，包括 + 操作符、变量合并机制和实现原理。

---

## 什么是 Template Composition？

**Template Composition（模板组合）** 是 PromptTemplate 的一种高级特性，允许你使用 `+` 操作符将多个模板组合成一个新模板。

**核心价值**：
- 提升模板复用性
- 支持模块化提示词设计
- 自动处理变量合并
- 简化复杂提示词的构建

[来源: reference/source_prompttemplate_01.md | LangChain 源码分析]

---

## 1. 使用 + 操作符组合模板

### 1.1 基础概念

LangChain 的 PromptTemplate 重载了 `+` 操作符，使得模板组合变得非常直观。

```python
from langchain_core.prompts import PromptTemplate

# 创建两个模板
template1 = PromptTemplate.from_template("你是一个{role}。")
template2 = PromptTemplate.from_template("请回答：{question}")

# 使用 + 操作符组合
combined = template1 + template2

# 格式化
result = combined.format(role="AI助手", question="什么是RAG？")
print(result)
# 输出：你是一个AI助手。请回答：什么是RAG？
```

**关键特性**：
- 简单直观的语法
- 自动合并模板内容
- 自动处理变量

[来源: reference/search_composition_01.md | The Only LangChain Prompt Templates Guide]

### 1.2 类比理解

**前端类比**：
- 类似 JavaScript 的字符串拼接：`str1 + str2`
- 类似 React 组件组合：`<Component1 /><Component2 />`
- 类似 CSS 类名组合：`className="class1 class2"`

**日常生活类比**：
- 像是拼接乐高积木，每个模板是一个积木块
- 像是组装句子，每个模板是一个句子片段
- 像是制作三明治，每个模板是一层食材

### 1.3 基础示例

```python
from langchain_core.prompts import PromptTemplate

# ===== 示例1：系统提示 + 用户问题 =====
system_prompt = PromptTemplate.from_template(
    "你是一个专业的{domain}专家。"
)

user_prompt = PromptTemplate.from_template(
    "用户问题：{question}\n请提供详细解答。"
)

# 组合
full_prompt = system_prompt + user_prompt

# 使用
result = full_prompt.format(
    domain="机器学习",
    question="什么是梯度下降？"
)
print(result)
```

**输出**：
```
你是一个专业的机器学习专家。用户问题：什么是梯度下降？
请提供详细解答。
```

---

## 2. `__add__` 方法实现原理

### 2.1 源码分析

LangChain 通过重载 `__add__` 方法实现模板组合功能。

[来源: reference/source_prompttemplate_01.md | prompt.py:142-184]

```python
def __add__(self, other: Any) -> PromptTemplate:
    """Override the `+` operator to allow for combining prompt templates."""
    if isinstance(other, PromptTemplate):
        # 检查模板格式是否一致
        if self.template_format != other.template_format:
            msg = "Cannot add templates of different formats"
            raise ValueError(msg)

        # 合并 input_variables（取并集）
        input_variables = list(
            set(self.input_variables) | set(other.input_variables)
        )

        # 拼接模板内容
        template = self.template + other.template

        # 合并 validate_template（取与）
        validate_template = self.validate_template and other.validate_template

        # 合并 partial_variables（检查冲突）
        partial_variables = dict(self.partial_variables.items())
        for k, v in other.partial_variables.items():
            if k in partial_variables:
                msg = "Cannot have same variable partialed twice."
                raise ValueError(msg)
            partial_variables[k] = v

        # 创建新模板
        return PromptTemplate(
            template=template,
            input_variables=input_variables,
            partial_variables=partial_variables,
            template_format=self.template_format,
            validate_template=validate_template,
        )

    if isinstance(other, str):
        # 支持 PromptTemplate + str
        prompt = PromptTemplate.from_template(
            other,
            template_format=self.template_format,
        )
        return self + prompt
```

### 2.2 实现原理解析

#### 原理1：模板格式一致性检查

```python
if self.template_format != other.template_format:
    msg = "Cannot add templates of different formats"
    raise ValueError(msg)
```

**为什么需要？**
- f-string、mustache、jinja2 三种格式的语法不兼容
- 混合使用会导致格式化错误

**示例**：
```python
# ❌ 错误：不同格式无法组合
template1 = PromptTemplate(
    template="Say {foo}",
    input_variables=["foo"],
    template_format="f-string"
)

template2 = PromptTemplate(
    template="This is {{bar}}",
    input_variables=["bar"],
    template_format="mustache"
)

# 会抛出 ValueError
combined = template1 + template2
```

#### 原理2：input_variables 自动合并（并集）

```python
input_variables = list(
    set(self.input_variables) | set(other.input_variables)
)
```

**为什么使用并集？**
- 保留所有需要的变量
- 避免变量丢失
- 支持变量复用

**示例**：
```python
template1 = PromptTemplate.from_template("角色：{role}")
template2 = PromptTemplate.from_template("问题：{question}")

combined = template1 + template2

print(combined.input_variables)
# ['role', 'question']
```

#### 原理3：partial_variables 冲突检查

```python
partial_variables = dict(self.partial_variables.items())
for k, v in other.partial_variables.items():
    if k in partial_variables:
        msg = "Cannot have same variable partialed twice."
        raise ValueError(msg)
    partial_variables[k] = v
```

**为什么需要冲突检查？**
- 避免同一变量被赋予不同的值
- 确保 partial 变量的唯一性
- 防止意外覆盖

**示例**：
```python
from datetime import datetime

def get_date():
    return datetime.now().strftime("%Y-%m-%d")

def get_time():
    return datetime.now().strftime("%H:%M:%S")

# ✅ 正确：不同的 partial 变量
template1 = PromptTemplate(
    template="日期：{date}",
    input_variables=[],
    partial_variables={"date": get_date}
)

template2 = PromptTemplate(
    template="时间：{time}",
    input_variables=[],
    partial_variables={"time": get_time}
)

combined = template1 + template2  # 成功

# ❌ 错误：相同的 partial 变量
template3 = PromptTemplate(
    template="另一个日期：{date}",
    input_variables=[],
    partial_variables={"date": lambda: "2026-01-01"}
)

# 会抛出 ValueError
combined = template1 + template3
```

#### 原理4：validate_template 合并（取与）

```python
validate_template = self.validate_template and other.validate_template
```

**为什么使用与运算？**
- 只有两个模板都启用验证时，组合后的模板才验证
- 保守策略，避免验证错误

---

## 3. 自动合并 input_variables 和 partial_variables

### 3.1 变量合并规则

| 变量类型 | 合并策略 | 冲突处理 |
|----------|----------|----------|
| input_variables | 并集（Union） | 自动去重 |
| partial_variables | 字典合并 | 抛出异常 |

### 3.2 input_variables 合并示例

```python
from langchain_core.prompts import PromptTemplate

# 模板1：需要 role 和 domain
template1 = PromptTemplate.from_template(
    "你是一个{role}，专注于{domain}领域。"
)

# 模板2：需要 question
template2 = PromptTemplate.from_template(
    "请回答：{question}"
)

# 组合后：需要 role、domain、question
combined = template1 + template2

print("模板1变量：", template1.input_variables)  # ['role', 'domain']
print("模板2变量：", template2.input_variables)  # ['question']
print("组合后变量：", combined.input_variables)  # ['role', 'domain', 'question']

# 使用时必须提供所有变量
result = combined.format(
    role="AI助手",
    domain="机器学习",
    question="什么是过拟合？"
)
```

### 3.3 partial_variables 合并示例

```python
from langchain_core.prompts import PromptTemplate
from datetime import datetime

def get_date():
    return datetime.now().strftime("%Y-%m-%d")

def get_version():
    return "v1.0"

# 模板1：预填充日期
template1 = PromptTemplate(
    template="[{date}] ",
    input_variables=[],
    partial_variables={"date": get_date}
)

# 模板2：预填充版本
template2 = PromptTemplate(
    template="[{version}] {message}",
    input_variables=["message"],
    partial_variables={"version": get_version}
)

# 组合后：同时包含两个 partial 变量
combined = template1 + template2

print("组合后 partial 变量：", combined.partial_variables.keys())
# dict_keys(['date', 'version'])

# 使用时只需提供 message
result = combined.format(message="系统启动")
print(result)
# [2026-02-26] [v1.0] 系统启动
```

### 3.4 变量复用场景

```python
from langchain_core.prompts import PromptTemplate

# 两个模板使用相同的变量
template1 = PromptTemplate.from_template("用户名：{user}")
template2 = PromptTemplate.from_template("欢迎回来，{user}！")

# 组合后，user 变量只需提供一次
combined = template1 + template2

print(combined.input_variables)  # ['user']（自动去重）

result = combined.format(user="张三")
print(result)
# 用户名：张三欢迎回来，张三！
```

---

## 4. PromptTemplate + str 组合

### 4.1 基础用法

LangChain 支持 PromptTemplate 与字符串直接组合。

```python
from langchain_core.prompts import PromptTemplate

# 创建模板
template = PromptTemplate.from_template("你是一个{role}。")

# 与字符串组合
combined = template + "请保持专业和友好。"

result = combined.format(role="AI助手")
print(result)
# 你是一个AI助手。请保持专业和友好。
```

### 4.2 实现原理

当 `other` 是字符串时，会自动转换为 PromptTemplate：

```python
if isinstance(other, str):
    prompt = PromptTemplate.from_template(
        other,
        template_format=self.template_format,
    )
    return self + prompt
```

**关键点**：
- 字符串会继承左侧模板的 `template_format`
- 自动提取字符串中的变量
- 然后按照 PromptTemplate + PromptTemplate 的逻辑处理

### 4.3 实际应用场景

#### 场景1：添加固定后缀

```python
from langchain_core.prompts import PromptTemplate

# 基础模板
base_template = PromptTemplate.from_template(
    "问题：{question}\n"
)

# 添加固定指令
full_template = base_template + "请用简洁的语言回答，不超过100字。"

result = full_template.format(question="什么是机器学习？")
print(result)
```

#### 场景2：添加动态后缀

```python
from langchain_core.prompts import PromptTemplate

# 基础模板
base_template = PromptTemplate.from_template(
    "角色：{role}\n"
)

# 添加带变量的后缀
full_template = base_template + "任务：{task}"

result = full_template.format(
    role="数据分析师",
    task="分析销售数据"
)
print(result)
```

#### 场景3：构建多层次提示词

```python
from langchain_core.prompts import PromptTemplate

# 第一层：系统角色
system = PromptTemplate.from_template("你是一个{role}。")

# 第二层：固定规则
rules = "请遵循以下规则：\n1. 保持客观\n2. 提供证据\n3. 承认不确定性\n\n"

# 第三层：用户输入
user = PromptTemplate.from_template("用户问题：{question}")

# 组合
full_prompt = system + rules + user

result = full_prompt.format(
    role="科学顾问",
    question="气候变化的主要原因是什么？"
)
print(result)
```

---

## 5. 基础代码示例

### 5.1 完整示例：RAG 系统提示词构建

```python
"""
Template Composition 实战示例
演示：构建 RAG 系统的分层提示词
"""

from langchain_core.prompts import PromptTemplate
from datetime import datetime

# ===== 1. 定义各层模板 =====

# 第一层：系统角色（带 partial 变量）
def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

system_template = PromptTemplate(
    template="[系统] 日期：{date}\n你是一个专业的{domain}助手。\n\n",
    input_variables=["domain"],
    partial_variables={"date": get_current_date}
)

# 第二层：上下文信息
context_template = PromptTemplate.from_template(
    "[上下文]\n{context}\n\n"
)

# 第三层：用户问题
question_template = PromptTemplate.from_template(
    "[用户问题]\n{question}\n\n"
)

# 第四层：固定指令
instruction = """[指令]
请基于上述上下文回答用户问题。
- 如果上下文中有相关信息，请引用
- 如果上下文中没有相关信息，请明确说明
- 保持回答简洁准确
"""

# ===== 2. 组合模板 =====
rag_prompt = system_template + context_template + question_template + instruction

# ===== 3. 查看组合结果 =====
print("=== 组合后的模板信息 ===")
print(f"Input Variables: {rag_prompt.input_variables}")
print(f"Partial Variables: {list(rag_prompt.partial_variables.keys())}")
print(f"Template Format: {rag_prompt.template_format}")
print()

# ===== 4. 使用模板 =====
result = rag_prompt.format(
    domain="机器学习",
    context="向量检索是一种基于向量相似度的搜索技术，常用于 RAG 系统中。",
    question="什么是向量检索？"
)

print("=== 格式化结果 ===")
print(result)
```

**输出**：
```
=== 组合后的模板信息 ===
Input Variables: ['domain', 'context', 'question']
Partial Variables: ['date']
Template Format: f-string

=== 格式化结果 ===
[系统] 日期：2026-02-26
你是一个专业的机器学习助手。

[上下文]
向量检索是一种基于向量相似度的搜索技术，常用于 RAG 系统中。

[用户问题]
什么是向量检索？

[指令]
请基于上述上下文回答用户问题。
- 如果上下文中有相关信息，请引用
- 如果上下文中没有相关信息，请明确说明
- 保持回答简洁准确
```

### 5.2 示例：多语言支持

```python
from langchain_core.prompts import PromptTemplate

# 定义不同语言的模板片段
greeting_en = PromptTemplate.from_template("Hello, {name}!")
greeting_zh = PromptTemplate.from_template("你好，{name}！")

instruction = PromptTemplate.from_template("\nPlease answer: {question}")

# 根据语言选择组合
def create_prompt(language: str):
    if language == "en":
        return greeting_en + instruction
    elif language == "zh":
        return greeting_zh + instruction
    else:
        raise ValueError(f"Unsupported language: {language}")

# 使用
prompt_en = create_prompt("en")
print(prompt_en.format(name="Alice", question="What is AI?"))

prompt_zh = create_prompt("zh")
print(prompt_zh.format(name="小明", question="什么是AI？"))
```

### 5.3 示例：条件组合

```python
from langchain_core.prompts import PromptTemplate

# 基础模板
base = PromptTemplate.from_template("问题：{question}\n")

# 可选的上下文模板
context_template = PromptTemplate.from_template("参考信息：{context}\n")

# 可选的示例模板
example_template = PromptTemplate.from_template("示例：{example}\n")

# 根据条件组合
def build_prompt(include_context=False, include_example=False):
    prompt = base

    if include_context:
        prompt = prompt + context_template

    if include_example:
        prompt = prompt + example_template

    prompt = prompt + "请回答："

    return prompt

# 使用
prompt1 = build_prompt(include_context=True)
print(prompt1.input_variables)  # ['question', 'context']

prompt2 = build_prompt(include_context=True, include_example=True)
print(prompt2.input_variables)  # ['question', 'context', 'example']
```

---

## 6. 最佳实践

### 6.1 模板组织原则

**原则1：单一职责**
```python
# ✅ 推荐：每个模板负责一个功能
system_role = PromptTemplate.from_template("你是{role}。")
task_desc = PromptTemplate.from_template("任务：{task}")
user_input = PromptTemplate.from_template("输入：{input}")

# ❌ 不推荐：一个模板包含所有内容
monolithic = PromptTemplate.from_template(
    "你是{role}。任务：{task}输入：{input}"
)
```

**原则2：从通用到具体**
```python
# ✅ 推荐：从通用到具体的组合顺序
prompt = system_role + context + user_question + specific_instruction

# ❌ 不推荐：顺序混乱
prompt = user_question + system_role + specific_instruction + context
```

**原则3：复用常见片段**
```python
# 定义可复用的模板片段
SYSTEM_ROLE = PromptTemplate.from_template("你是{role}。")
POLITE_ENDING = "请保持礼貌和专业。"

# 在多个场景中复用
customer_service = SYSTEM_ROLE + "处理客户咨询。" + POLITE_ENDING
technical_support = SYSTEM_ROLE + "提供技术支持。" + POLITE_ENDING
```

### 6.2 性能优化

**优化1：预组合常用模板**
```python
# ❌ 不推荐：每次都重新组合
def get_prompt(question):
    return template1 + template2 + template3

# ✅ 推荐：预先组合
COMBINED_TEMPLATE = template1 + template2 + template3

def get_prompt(question):
    return COMBINED_TEMPLATE
```

**优化2：避免过度组合**
```python
# ❌ 不推荐：过度拆分导致频繁组合
t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8

# ✅ 推荐：合理分组
group1 = t1 + t2 + t3
group2 = t4 + t5 + t6
final = group1 + group2 + t7 + t8
```

### 6.3 错误处理

```python
from langchain_core.prompts import PromptTemplate

def safe_combine(template1, template2):
    """安全地组合两个模板"""
    try:
        return template1 + template2
    except ValueError as e:
        if "different formats" in str(e):
            print(f"错误：模板格式不一致")
            print(f"模板1格式：{template1.template_format}")
            print(f"模板2格式：{template2.template_format}")
        elif "partialed twice" in str(e):
            print(f"错误：partial 变量冲突")
            print(f"模板1 partial：{template1.partial_variables.keys()}")
            print(f"模板2 partial：{template2.partial_variables.keys()}")
        raise
```

---

## 7. 常见问题

### 问题1：组合后变量顺序

**问题**：组合后的 input_variables 顺序是什么？

**答案**：顺序不保证，因为使用了集合（set）操作。如果需要特定顺序，应该在使用时明确指定。

```python
# input_variables 的顺序不确定
combined = template1 + template2
print(combined.input_variables)  # 可能是 ['a', 'b'] 或 ['b', 'a']

# 如果需要特定顺序，在格式化时明确指定
result = combined.format(a="value1", b="value2")
```

### 问题2：能否组合不同格式的模板？

**问题**：f-string 模板能和 mustache 模板组合吗？

**答案**：不能。会抛出 ValueError。

```python
# ❌ 错误
template1 = PromptTemplate(
    template="Say {foo}",
    input_variables=["foo"],
    template_format="f-string"
)

template2 = PromptTemplate(
    template="This is {{bar}}",
    input_variables=["bar"],
    template_format="mustache"
)

combined = template1 + template2  # ValueError
```

### 问题3：组合后如何修改？

**问题**：组合后的模板能否再次修改？

**答案**：组合后返回的是新的 PromptTemplate 对象，可以继续组合或使用 partial 方法。

```python
# 继续组合
combined1 = template1 + template2
combined2 = combined1 + template3

# 使用 partial
combined_partial = combined1.partial(role="AI助手")
```

---

## 8. 总结

### 核心要点

1. **+ 操作符**：简单直观的模板组合语法
2. **自动合并**：input_variables 取并集，partial_variables 检查冲突
3. **格式一致性**：只能组合相同 template_format 的模板
4. **字符串支持**：支持 PromptTemplate + str 的组合
5. **模块化设计**：支持构建可复用的模板片段

### 适用场景

- 构建分层提示词（系统 + 上下文 + 问题）
- RAG 系统的提示词管理
- 多语言提示词支持
- 条件化提示词构建
- 模板片段复用

### 注意事项

- 确保模板格式一致
- 避免 partial_variables 冲突
- 合理组织模板层次
- 注意变量命名规范
- 预组合常用模板以提升性能

---

**参考资料**：
- [LangChain 源码分析](reference/source_prompttemplate_01.md)
- [The Only LangChain Prompt Templates Guide](https://medium.com/@shoaibahamedshafi/the-only-langchain-prompt-templates-guide-youll-ever-need-2219293708eb)
- [LangChain Prompt Templates Complete Guide](https://latenode.com/blog/ai-frameworks-technical-infrastructure/langchain-setup-tools-agents-memory/langchain-prompt-templates-complete-guide-with-examples)
- [Dynamic Prompts with LangChain](https://www.newline.co/@zaoyang/dynamic-prompts-with-langchain-templates--71d0c244)

**版本**：v1.0
**最后更新**：2026-02-26
**适用于**：LangChain 0.3.x+
