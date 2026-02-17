# 04_task_decomposition.py 修复记录

目标：让 `examples/l2_llm/04_prompt_engineer_basic/02_concept_clear_command/04_task_decomposition.py` 在多步骤调用 LLM 时稳定得到 **可解析的 JSON**，避免 `json.loads(...)` 崩溃。

## 1) 第一次报错：`JSONDecodeError: Expecting value (line 1 column 1)`

**现象**
- 报错位置：在 `json.loads(step*_result.choices[0].message.content)` 解析模型输出时抛出
- 典型错误：`json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)`

**原因**
- `message.content` 不是合法 JSON（可能为空字符串、带前缀解释文字、或被 Markdown code fence 包裹）。
- 代码假设模型“永远只输出纯 JSON”，直接 `json.loads(...)`，没有任何容错/诊断信息。

**修复方式**
- 增加 `_parse_json_object()`：
  - 先做 `strip()` 空值判断
  - 去掉常见 Markdown code fence（```json / ```python 等）
  - 尝试从文本中提取/解析第一个 JSON 对象
  - 失败时抛出带 `preview` 的异常，便于定位模型到底返回了什么
- 同时把每一步 `temperature` 设为 `0.0` 并在 system prompt 强调“只输出 JSON，不要解释/不要 Markdown”。

对应改动：`examples/l2_llm/04_prompt_engineer_basic/02_concept_clear_command/04_task_decomposition.py`

## 2) 发现的硬错误：Step 3 `response_format` 写错（set 而不是 dict）

**现象**
- Step 3 原代码：`response_format={"type", "json_object"}`

**原因**
- 这在 Python 里是一个 `set`，不是 OpenAI SDK 期待的 `dict`。
- 即便前两步侥幸能跑通，第三步在请求参数上也会出问题（或导致 SDK 行为异常）。

**修复方式**
- 修正为：`response_format={"type": "json_object"}`（后续又统一切到 `json_schema`，见第 4 节）。

对应改动：`examples/l2_llm/04_prompt_engineer_basic/02_concept_clear_command/04_task_decomposition.py`

## 3) 第二次报错：`JSONDecodeError: Extra data ...`

**现象**
- 报错示例：`json.decoder.JSONDecodeError: Extra data: line 39 column 1 (char 878)`

**原因**
- 模型返回的内容里 **不止一个 JSON** 或在 JSON 后面追加了其它文本，导致 `json.loads()` 认为“一个 JSON 结束后还有额外数据”。
- 另外，prompt 里“输出格式”最初给的是“类 JSON 示例”（值是 `步骤编号/步骤结果` 之类的未加引号占位符），容易诱导模型生成非严格 JSON。

**修复方式**
1. **把 prompt 的“输出格式”示例改成严格 JSON 示例**（可直接被解析器接受），并明确：
   - 必须输出 RFC8259 严格 JSON
   - 禁止 Markdown/代码块
2. **改进 `_parse_json_object()` 的解析策略**：
   - 使用 `json.JSONDecoder().raw_decode(...)`：可以从字符串开头解析出“第一个 JSON 值”，并忽略其后尾巴文本
   - code fence 去除逻辑支持 ```python 等语言标记

对应改动：`examples/l2_llm/04_prompt_engineer_basic/02_concept_clear_command/04_task_decomposition.py`

## 4) 关键修复：改用 `json_schema + strict`，从源头强制输出结构

**现象（从报错 preview 定位）**
- preview 显示模型直接返回了一段 Python 代码（以 ```python 或 `python\n import json ...` 开头），完全不是 JSON。

**原因**
- `response_format={"type":"json_object"}` 在某些场景/模型输出下仍可能“漂移”到解释文本或代码块。

**修复方式**
- 统一把三步的 `response_format` 切换为：
  - `{"type": "json_schema", "json_schema": {"name": "...", "schema": {...}, "strict": True}}`
- 新增并复用一个 schema（示例字段）：
  - `step`: 1/2/3（整数）
  - `result`: object
  - `next_step`: integer 或 null
- 这样模型会被强约束只能输出符合 schema 的 JSON；从根源上杜绝“返回 python/解释文本/markdown”。

对应改动：`examples/l2_llm/04_prompt_engineer_basic/02_concept_clear_command/04_task_decomposition.py`

## 5) 最终状态与验证方式

**最终状态**
- 当前脚本已能正常运行（你反馈“当前不报错了”）。
- 输出解析不再依赖“模型自觉”，而是通过 `json_schema strict` 强制结构 + 解析器容错兜底。

**验证方式**
- 直接运行：
  - `/Users/wuxiao/Documents/codeWithFelix/vibe-learning/vibe-learning-rag/.venv/bin/python examples/l2_llm/04_prompt_engineer_basic/02_concept_clear_command/04_task_decomposition.py`
- 预期：
  - 三步都能打印出 `dict`（步骤 1/2/3 的结构化结果）
  - 不再出现 `Expecting value` / `Extra data` 等 JSON 解析错误

