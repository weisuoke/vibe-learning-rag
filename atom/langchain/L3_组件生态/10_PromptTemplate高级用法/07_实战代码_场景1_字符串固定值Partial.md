# 实战代码 - 场景1：字符串固定值 Partial

> 本文档展示如何使用字符串固定值的 Partial Variables 预填充常量变量，减少重复传递

---

## 场景概述

**核心问题：** 在实际开发中，某些变量（如系统设置、公司名称、API 版本等）在多次调用中保持不变，每次都传递这些固定值会导致代码冗余。

**解决方案：** 使用 `partial_variables` 参数预填充这些固定值，让模板"记住"这些常量。

**适用场景：**
- 系统配置信息（公司名、产品名、版本号）
- 固定的日期或时间戳
- 共享的上下文信息
- RAG 系统中的固定元数据

[来源: reference/search_partial_01.md | LangChain 官方文档]

---

## 双重类比

### 前端类比：默认参数

```javascript
// 前端：函数默认参数
function createUser(name, role = "user", company = "Acme Inc") {
    return { name, role, company };
}

// 只需传递变化的参数
createUser("Alice");  // { name: "Alice", role: "user", company: "Acme Inc" }
```

**LangChain 对应：**
```python
# LangChain：Partial Variables
prompt = PromptTemplate(
    template="User: {name}, Role: {role}, Company: {company}",
    input_variables=["name"],
    partial_variables={"role": "user", "company": "Acme Inc"}
)

# 只需传递变化的参数
prompt.format(name="Alice")
```

### 日常生活类比：预填表单

想象你在填写多份相似的表单：
- **传统方式**：每次都要填写姓名、地址、电话、公司名
- **Partial 方式**：表单已经预填了地址、电话、公司名，你只需填写姓名

---

## 核心实现原理

### 1. 基础语法

```python
from langchain_core.prompts import PromptTemplate

# 方式1：构造函数中指定
prompt = PromptTemplate(
    template="Hello {name}, welcome to {company}!",
    input_variables=["name"],
    partial_variables={"company": "OpenAI"}
)

# 方式2：使用 partial() 方法
prompt = PromptTemplate.from_template("Hello {name}, welcome to {company}!")
prompt_partial = prompt.partial(company="OpenAI")
```

[来源: reference/source_prompttemplate_01.md:72-85]

### 2. 变量合并机制

**源码分析：**
```python
# 在格式化时，partial_variables 会自动合并到用户提供的变量中
# sourcecode/langchain/libs/core/langchain_core/prompts/base.py:65-70

partial_variables: Mapping[str, Any] = Field(default_factory=dict)
"""A dictionary of the partial variables the prompt template carries.

Partial variables populate the template so that you don't need to pass them in every
time you call the prompt.
"""
```

[来源: reference/source_prompttemplate_01.md:70-85]

---

## 完整实战代码

### 场景1：RAG 系统配置管理

```python
"""
场景1：RAG 系统配置管理
演示：使用 Partial Variables 管理系统级配置信息
"""

import os
from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# ===== 1. 定义系统配置（固定值） =====
print("=== 1. 定义系统配置 ===")

SYSTEM_CONFIG = {
    "company": "TechCorp AI",
    "product": "SmartDoc RAG",
    "version": "v2.1.0",
    "support_email": "support@techcorp.ai"
}

print(f"系统配置: {SYSTEM_CONFIG}")
print()

# ===== 2. 创建带 Partial Variables 的模板 =====
print("=== 2. 创建带 Partial Variables 的模板 ===")

# 传统方式：每次都要传递所有变量
traditional_template = PromptTemplate.from_template(
    """You are a helpful assistant for {company}'s {product} ({version}).

User Question: {question}

Please provide a helpful answer. If you need support, contact {support_email}.
"""
)

# Partial 方式：预填充固定配置
partial_template = PromptTemplate(
    template="""You are a helpful assistant for {company}'s {product} ({version}).

User Question: {question}

Please provide a helpful answer. If you need support, contact {support_email}.
""",
    input_variables=["question"],
    partial_variables=SYSTEM_CONFIG
)

print("✅ 模板创建成功")
print(f"需要传递的变量: {partial_template.input_variables}")
print(f"预填充的变量: {list(partial_template.partial_variables.keys())}")
print()

# ===== 3. 对比使用方式 =====
print("=== 3. 对比使用方式 ===")

question = "How do I upload documents?"

# 传统方式：需要传递所有变量
print("传统方式（需要传递所有变量）:")
traditional_prompt = traditional_template.format(
    company=SYSTEM_CONFIG["company"],
    product=SYSTEM_CONFIG["product"],
    version=SYSTEM_CONFIG["version"],
    support_email=SYSTEM_CONFIG["support_email"],
    question=question
)
print(traditional_prompt[:100] + "...")
print()

# Partial 方式：只需传递变化的变量
print("Partial 方式（只需传递变化的变量）:")
partial_prompt = partial_template.format(question=question)
print(partial_prompt[:100] + "...")
print()

# ===== 4. 实际应用：多个问题处理 =====
print("=== 4. 实际应用：多个问题处理 ===")

questions = [
    "How do I upload documents?",
    "What file formats are supported?",
    "How do I search for specific content?"
]

# 使用 Partial 模板处理多个问题
for i, q in enumerate(questions, 1):
    prompt = partial_template.format(question=q)
    print(f"问题 {i}: {q}")
    print(f"生成的 Prompt 长度: {len(prompt)} 字符")
    print()

# ===== 5. 与 LLM 集成 =====
print("=== 5. 与 LLM 集成 ===")

# 创建 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 创建链
chain = partial_template | llm

# 调用链（只需传递变化的参数）
print("调用 LLM...")
response = chain.invoke({"question": "How do I upload documents?"})
print(f"回答: {response.content[:200]}...")
print()

# ===== 6. 动态更新配置 =====
print("=== 6. 动态更新配置 ===")

# 场景：产品升级到新版本
new_version_template = partial_template.partial(version="v2.2.0")

print("原版本模板:")
print(f"  Version: {partial_template.partial_variables['version']}")
print()

print("新版本模板:")
print(f"  Version: {new_version_template.partial_variables['version']}")
print()

# ===== 7. 多语言支持 =====
print("=== 7. 多语言支持 ===")

# 英文模板
en_template = PromptTemplate(
    template="""You are a helpful assistant for {company}'s {product}.

User Question: {question}

Please provide a helpful answer in English.
""",
    input_variables=["question"],
    partial_variables={"company": SYSTEM_CONFIG["company"], "product": SYSTEM_CONFIG["product"]}
)

# 中文模板
zh_template = PromptTemplate(
    template="""你是 {company} 的 {product} 的智能助手。

用户问题：{question}

请用中文提供有帮助的回答。
""",
    input_variables=["question"],
    partial_variables={"company": SYSTEM_CONFIG["company"], "product": SYSTEM_CONFIG["product"]}
)

print("英文模板:")
print(en_template.format(question="How do I upload documents?")[:100] + "...")
print()

print("中文模板:")
print(zh_template.format(question="如何上传文档？")[:100] + "...")
print()

print("✅ 所有场景演示完成！")
```

**运行输出示例：**
```
=== 1. 定义系统配置 ===
系统配置: {'company': 'TechCorp AI', 'product': 'SmartDoc RAG', 'version': 'v2.1.0', 'support_email': 'support@techcorp.ai'}

=== 2. 创建带 Partial Variables 的模板 ===
✅ 模板创建成功
需要传递的变量: ['question']
预填充的变量: ['company', 'product', 'version', 'support_email']

=== 3. 对比使用方式 ===
传统方式（需要传递所有变量）:
You are a helpful assistant for TechCorp AI's SmartDoc RAG (v2.1.0).

User Question: How do I...

Partial 方式（只需传递变化的变量）:
You are a helpful assistant for TechCorp AI's SmartDoc RAG (v2.1.0).

User Question: How do I...

=== 4. 实际应用：多个问题处理 ===
问题 1: How do I upload documents?
生成的 Prompt 长度: 156 字符

问题 2: What file formats are supported?
生成的 Prompt 长度: 163 字符

问题 3: How do I search for specific content?
生成的 Prompt 长度: 167 字符

✅ 所有场景演示完成！
```

---

## 场景2：多租户系统

```python
"""
场景2：多租户系统
演示：为不同租户创建带有租户信息的模板
"""

from langchain_core.prompts import PromptTemplate
from typing import Dict

# ===== 1. 定义租户配置 =====
print("=== 1. 定义租户配置 ===")

TENANTS = {
    "tenant_a": {
        "name": "Acme Corporation",
        "industry": "Manufacturing",
        "tier": "Enterprise"
    },
    "tenant_b": {
        "name": "Beta Solutions",
        "industry": "Healthcare",
        "tier": "Professional"
    },
    "tenant_c": {
        "name": "Gamma Tech",
        "industry": "Finance",
        "tier": "Starter"
    }
}

print(f"配置了 {len(TENANTS)} 个租户")
print()

# ===== 2. 为每个租户创建专属模板 =====
print("=== 2. 为每个租户创建专属模板 ===")

def create_tenant_template(tenant_id: str, tenant_config: Dict) -> PromptTemplate:
    """为指定租户创建带有租户信息的模板"""
    return PromptTemplate(
        template="""You are an AI assistant for {tenant_name} ({industry} industry).
Subscription Tier: {tier}

User Query: {query}

Please provide a response tailored to the {industry} industry.
""",
        input_variables=["query"],
        partial_variables={
            "tenant_name": tenant_config["name"],
            "industry": tenant_config["industry"],
            "tier": tenant_config["tier"]
        }
    )

# 创建租户模板字典
tenant_templates = {
    tenant_id: create_tenant_template(tenant_id, config)
    for tenant_id, config in TENANTS.items()
}

print(f"✅ 为 {len(tenant_templates)} 个租户创建了专属模板")
print()

# ===== 3. 使用租户模板 =====
print("=== 3. 使用租户模板 ===")

query = "What are the best practices for data security?"

for tenant_id, template in tenant_templates.items():
    print(f"租户: {tenant_id}")
    prompt = template.format(query=query)
    print(prompt[:150] + "...")
    print()

# ===== 4. 动态切换租户 =====
print("=== 4. 动态切换租户 ===")

def process_tenant_query(tenant_id: str, query: str) -> str:
    """处理指定租户的查询"""
    if tenant_id not in tenant_templates:
        raise ValueError(f"Unknown tenant: {tenant_id}")

    template = tenant_templates[tenant_id]
    return template.format(query=query)

# 模拟不同租户的查询
queries = [
    ("tenant_a", "How do I optimize production efficiency?"),
    ("tenant_b", "What are HIPAA compliance requirements?"),
    ("tenant_c", "How do I manage financial risk?")
]

for tenant_id, query in queries:
    print(f"处理租户 {tenant_id} 的查询:")
    prompt = process_tenant_query(tenant_id, query)
    print(f"  查询: {query}")
    print(f"  Prompt 长度: {len(prompt)} 字符")
    print()

print("✅ 多租户场景演示完成！")
```

---

## 场景3：API 版本管理

```python
"""
场景3：API 版本管理
演示：使用 Partial Variables 管理不同 API 版本的提示词
"""

from langchain_core.prompts import PromptTemplate
from datetime import datetime

# ===== 1. 定义 API 版本配置 =====
print("=== 1. 定义 API 版本配置 ===")

API_VERSIONS = {
    "v1": {
        "version": "1.0",
        "deprecated": False,
        "features": "Basic search and retrieval",
        "release_date": "2024-01-01"
    },
    "v2": {
        "version": "2.0",
        "deprecated": False,
        "features": "Advanced search, filtering, and ranking",
        "release_date": "2024-06-01"
    },
    "v3": {
        "version": "3.0",
        "deprecated": False,
        "features": "AI-powered semantic search and recommendations",
        "release_date": "2025-01-01"
    }
}

print(f"配置了 {len(API_VERSIONS)} 个 API 版本")
print()

# ===== 2. 为每个版本创建模板 =====
print("=== 2. 为每个版本创建模板 ===")

def create_api_template(version_key: str, version_config: dict) -> PromptTemplate:
    """为指定 API 版本创建模板"""
    deprecation_notice = " (DEPRECATED)" if version_config["deprecated"] else ""

    return PromptTemplate(
        template="""API Version: {version}{deprecation}
Release Date: {release_date}
Features: {features}

User Request: {request}

Please process this request using API v{version} capabilities.
""",
        input_variables=["request"],
        partial_variables={
            "version": version_config["version"],
            "deprecation": deprecation_notice,
            "release_date": version_config["release_date"],
            "features": version_config["features"]
        }
    )

# 创建版本模板字典
api_templates = {
    version_key: create_api_template(version_key, config)
    for version_key, config in API_VERSIONS.items()
}

print(f"✅ 为 {len(api_templates)} 个 API 版本创建了模板")
print()

# ===== 3. 使用不同版本的模板 =====
print("=== 3. 使用不同版本的模板 ===")

request = "Search for documents about machine learning"

for version_key, template in api_templates.items():
    print(f"API 版本: {version_key}")
    prompt = template.format(request=request)
    print(prompt[:200] + "...")
    print()

# ===== 4. 版本迁移提示 =====
print("=== 4. 版本迁移提示 ===")

# 标记 v1 为已弃用
API_VERSIONS["v1"]["deprecated"] = True

# 重新创建 v1 模板
api_templates["v1"] = create_api_template("v1", API_VERSIONS["v1"])

print("v1 API 已标记为弃用:")
prompt = api_templates["v1"].format(request=request)
print(prompt[:200] + "...")
print()

print("✅ API 版本管理场景演示完成！")
```

---

## 最佳实践

### 1. 何时使用字符串固定值 Partial

**适用场景：**
- ✅ 系统配置信息（公司名、产品名、版本号）
- ✅ 固定的元数据（租户信息、API 版本）
- ✅ 不会变化的上下文信息
- ✅ 多次调用中保持不变的参数

**不适用场景：**
- ❌ 需要动态计算的值（如当前时间）
- ❌ 依赖运行时状态的值
- ❌ 需要延迟计算的值

[来源: reference/search_partial_01.md]

### 2. 性能优化

```python
# ✅ 推荐：复用模板对象
template = PromptTemplate(
    template="...",
    input_variables=["query"],
    partial_variables={"company": "Acme"}
)

# 多次调用
for query in queries:
    prompt = template.format(query=query)

# ❌ 不推荐：每次都创建新模板
for query in queries:
    template = PromptTemplate(
        template="...",
        input_variables=["query"],
        partial_variables={"company": "Acme"}
    )
    prompt = template.format(query=query)
```

### 3. 变量命名规范

```python
# ✅ 推荐：清晰的变量名
partial_variables = {
    "company_name": "Acme Corp",
    "product_version": "v2.0",
    "support_email": "support@acme.com"
}

# ❌ 不推荐：模糊的变量名
partial_variables = {
    "c": "Acme Corp",
    "v": "v2.0",
    "e": "support@acme.com"
}
```

### 4. 配置管理

```python
# ✅ 推荐：集中管理配置
from dataclasses import dataclass

@dataclass
class SystemConfig:
    company: str
    product: str
    version: str
    support_email: str

config = SystemConfig(
    company="Acme Corp",
    product="SmartDoc",
    version="v2.0",
    support_email="support@acme.com"
)

template = PromptTemplate(
    template="...",
    input_variables=["query"],
    partial_variables={
        "company": config.company,
        "product": config.product,
        "version": config.version,
        "support_email": config.support_email
    }
)
```

---

## 常见问题

### Q1: Partial Variables 会影响性能吗？

**答：** 不会。Partial Variables 只是在模板创建时存储固定值，在格式化时直接使用，不会增加额外开销。

### Q2: 可以覆盖 Partial Variables 吗？

**答：** 不可以。如果在 `format()` 中传递了与 `partial_variables` 同名的参数，会抛出错误。

```python
template = PromptTemplate(
    template="{company} - {query}",
    input_variables=["query"],
    partial_variables={"company": "Acme"}
)

# ❌ 错误：不能覆盖 partial_variables
template.format(company="Beta", query="test")  # 会报错
```

### Q3: 如何更新 Partial Variables？

**答：** 使用 `partial()` 方法创建新模板。

```python
# 原模板
template = PromptTemplate(
    template="{company} - {query}",
    input_variables=["query"],
    partial_variables={"company": "Acme"}
)

# 创建新模板（更新 company）
new_template = template.partial(company="Beta")
```

---

## 总结

**字符串固定值 Partial Variables 的核心价值：**

1. **减少代码冗余**：避免重复传递固定参数
2. **提高可维护性**：集中管理系统配置
3. **增强可读性**：调用时只需关注变化的参数
4. **支持多租户**：为不同租户创建专属模板
5. **版本管理**：轻松管理不同 API 版本

**关键要点：**
- 使用 `partial_variables` 参数预填充固定值
- 使用 `partial()` 方法动态更新配置
- 适用于系统配置、租户信息、API 版本等固定值
- 不适用于需要动态计算的值（使用函数动态值 Partial）

---

**参考资料：**
- [LangChain PromptTemplate 官方文档](https://reference.langchain.com/v0.3/python/core/prompts/langchain_core.prompts.prompt.PromptTemplate.html)
- [来源: reference/search_partial_01.md]
- [来源: reference/source_prompttemplate_01.md]
- [来源: reference/context7_langchain_02.md]

**下一步学习：**
- 07_实战代码_场景2_函数动态值Partial.md - 学习如何使用函数动态计算变量
- 07_实战代码_场景3_基础模板组合.md - 学习如何组合多个模板
