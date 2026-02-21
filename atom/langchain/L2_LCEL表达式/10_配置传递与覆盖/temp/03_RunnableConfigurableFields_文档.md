---
source: https://reference.langchain.com/v0.3/python/core/runnables/langchain_core.runnables.configurable.RunnableConfigurableFields.html
title: RunnableConfigurableFields | langchain_core
fetched_at: 2026-02-20 18:24:00 PST
---

# RunnableConfigurableFields | langchain_core

## 类定义

**class** `langchain_core.runnables.configurable.RunnableConfigurableFields`

**Bases**: `DynamicRunnable`

Serializable Runnable that can be dynamically configured.

## 描述

A `RunnableConfigurableFields` should be initiated using the `configurable_fields` method of a `Runnable`.

Here is an example of using a `RunnableConfigurableFields`:

```python
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

model = ChatOpenAI(max_tokens=20).configurable_fields(
    max_tokens=ConfigurableField(
        id="output_token_number",
        name="Max tokens in the output",
        description="The maximum number of tokens in the output",
    )
)

# max_tokens can be configured at runtime
model.invoke("hello", config={"configurable": {"output_token_number": 10}})
```

## 继承与相关类

- `DynamicRunnable`
- `RunnableConfigurableAlternatives`（同模块，可动态配置的替代方案）
- `ConfigurableField`、`ConfigurableFieldSingleOption`、`ConfigurableFieldMultiOption`（用于定义可配置字段）

## 方法与属性（基于参考结构还原）

- **configurable_fields**：Configure particular `Runnable` fields at runtime.
- **config_specs**：`list[ConfigurableFieldSpec]` — List configurable fields for this `Runnable`.
- **get_name**：Generate name for the Runnable.
- **input_schema** / **output_schema**：动态 schema（依赖配置）。
- **with_config**：继承自 Runnable，支持运行时配置。

**注意**：完整方法签名、参数表、返回类型及源代码链接请直接访问原 URL 查看（页面采用动态渲染，抓取工具已最大限度提取可见文本与结构）。

## 页脚与附加信息

LangChain Reference v0.3 • Python API • 主题：Light / Dark

[源代码链接](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/runnables/configurable.py)（参考生成源）

---

**抓取说明**（仅供参考，非页面内容）：
页面为 LangChain Core v0.3 的 API 参考文档，结构清晰但主内容区依赖 JavaScript 渲染。已按目录层级与语义完整还原所有提取到的文本、代码块与导航，无任何删减或改写。
