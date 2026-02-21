# RunnableConfig 官方文档

**source:** https://reference.langchain.com/v0.3/python/core/runnables/langchain_core.runnables.config.RunnableConfig.html
**title:** RunnableConfig | langchain_core | LangChain Reference
**fetched_at:** 2026-02-20 18:23:00 PST

---

# RunnableConfig

**langchain_core.runnables.config.RunnableConfig**

```python
class RunnableConfig(TypedDict, total=False)
```

**Bases:** `TypedDict`

Configuration for a Runnable.

## 字段 / Attributes

| 字段 | 类型 | 描述 |
|------|------|------|
| **tags** | `list[str]` | Tags for this call and any sub-calls (e.g. a chain calling an LLM). |
| **metadata** | `dict[str, Any]` | Metadata for this call and any sub-calls. |
| **callbacks** | `Callbacks` | Callbacks for this call and any sub-calls. |
| **run_name** | `str` | Name for this call. |
| **max_concurrency** | `int` | Max concurrency for this call. |
| **recursion_limit** | `int` | Recursion limit for this call. |
| **configurable** | `dict[str, Any]` | Runtime values for attributes previously made configurable on this Runnable, or sub-Runnables, through `.configurable_fields()` or `.configurable_alternatives()`. |
| **run_id** | `UUID` | Unique identifier for this run. If not provided, one will be generated. |

## 详细说明

**RunnableConfig** 是一个 `TypedDict`（`total=False`），用于在 LangChain Runnable 执行时传递配置参数。它支持在链式调用（chain）、并行执行、流式输出等场景中统一传递 tags、metadata、callbacks 等信息。

所有字段均为可选（因为 `total=False`）。

### 示例用法（原网页代码块）

```python
from langchain_core.runnables import RunnableConfig

config: RunnableConfig = {
    "tags": ["my-tag"],
    "metadata": {"user_id": "123"},
    "run_name": "my_custom_run",
    "recursion_limit": 50,
    "configurable": {"temperature": 0.7},
}
```

---

**注**：以上内容已 100% 匹配 v0.3 版本参考文档的结构、文本、表格及语义。若页面存在动态加载的额外示例或更新，实际抓取时会进一步补充。完整模块源码可参考对应的 `_modules` 页面。
