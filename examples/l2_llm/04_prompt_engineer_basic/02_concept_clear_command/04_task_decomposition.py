COMPLEX_TASK_PROMPT="""
任务：分析技术文档并生成学习路径

这是一个复杂任务，分解为以下步骤：

步骤1：文档分类
- 识别文档类型（教程、API文档、概念解释）
- 标注难度级别（初级、中级、高级）
- 输出：文档分类结果

步骤 2：知识点提取
- 从每个文档提取核心知识点
- 识别知识点之间的依赖关系
- 输出：知识点列表和依赖图

步骤 3：路径生成
- 根据依赖关系排序知识点
- 生成学习路径（初级→中级→高级）
- 输出：结构化学习路径

每个步骤的输出格式
{{
    "step": 1,
    "result": {{ "...": "..." }},
    "next_step": 2
}}

输出要求（非常重要）：
- 必须输出“严格 JSON”（RFC 8259），不能包含任何非 JSON 文本
- 不能使用 Markdown（例如 ```json 代码块）
- 字段说明：
  - step: 整数（1/2/3）
  - next_step: 整数（下一步编号）或 null（表示没有下一步）
  - result: 对象（该步骤的详细结果）

当前执行：步骤 {current_step}
当前的输入：{input_data}
"""

from openai import OpenAI
import json
import re
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

STEP_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "step": {"type": "integer", "enum": [1, 2, 3]},
        "result": {"type": "object", "additionalProperties": True},
        "next_step": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
    },
    "required": ["step", "result", "next_step"],
    "additionalProperties": False,
}

JSON_SCHEMA_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "complex_task_step_result",
        "schema": STEP_RESULT_SCHEMA,
        "strict": True,
    },
}

def _parse_json_object(content: str) -> dict:
    """
    Best-effort JSON parsing for LLM outputs.

    Even with `response_format={"type":"json_object"}`, some setups/models may still
    wrap JSON in markdown fences or add brief prefix/suffix text. This keeps the
    example resilient and makes failures easier to debug.
    """
    if content is None or not str(content).strip():
        raise ValueError("Empty model response; expected a JSON object string.")

    raw = str(content).strip()

    # Remove common markdown code-fence wrappers.
    if raw.startswith("```"):
        # Support ```json, ```python, etc.
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

    try:
        obj, _end = json.JSONDecoder().raw_decode(raw)
        return obj
    except json.JSONDecodeError:
        # Try extracting and decoding the first JSON object.
        start = raw.find("{")
        if start != -1:
            try:
                obj, _end = json.JSONDecoder().raw_decode(raw[start:])
                return obj
            except json.JSONDecodeError as e:
                preview = raw[:300].replace("\n", "\\n")
                raise ValueError(
                    "Model returned an invalid JSON object string "
                    f"(JSON error: {e}; preview): {preview}"
                )
        # Surface a helpful error for learners.
        preview = raw[:300].replace("\n", "\\n")
        raise ValueError(f"Model returned non-JSON content (preview): {preview}")

def execute_complex_task(documents: list):
    """执行复杂任务（分步骤）"""
    results = []

    # 步骤 1： 文档分类
    step1_result = client.chat.completions.create(
        model="gpt-4o",
        response_format=JSON_SCHEMA_RESPONSE_FORMAT,
        messages=[
            {"role": "system", "content": "你是文档分析专家。只输出 JSON 对象，不要输出任何解释或 Markdown 代码块。"},
            {"role": "user", "content": COMPLEX_TASK_PROMPT.format(
                current_step=1,
                input_data=json.dumps(documents, ensure_ascii=False)
            )}
        ],
        temperature=0.0
    )

    results.append(_parse_json_object(step1_result.choices[0].message.content))

    # 步骤 2：知识点提取（基于步骤1的结果）
    step2_result = client.chat.completions.create(
        model="gpt-4o",
        response_format=JSON_SCHEMA_RESPONSE_FORMAT,
        messages=[
            {"role": "system", "content": "你是知识图谱专家。只输出 JSON 对象，不要输出任何解释或 Markdown 代码块。"},
            {"role": "user", "content": COMPLEX_TASK_PROMPT.format(
                current_step=2,
                input_data=json.dumps(results[0], ensure_ascii=False)
            )}
        ],
        temperature=0.0
    )

    results.append(_parse_json_object(step2_result.choices[0].message.content))

    # 步骤 3： 路径生成（基于步骤2的结果）
    step3_result = client.chat.completions.create(
        model="gpt-4o",
        response_format=JSON_SCHEMA_RESPONSE_FORMAT,
        messages=[
            {"role": "system", "content": "你是学习路径设计专家。只输出 JSON 对象，不要输出任何解释或 Markdown 代码块。"},
            {"role": "user", "content": COMPLEX_TASK_PROMPT.format(
                current_step=3,
                input_data=json.dumps(results[1], ensure_ascii=False)
            )}
        ],
        temperature=0.0
    )

    results.append(_parse_json_object(step3_result.choices[0].message.content))

    return results

# 测试
docs = [
    {"title": "RAG 入门", "content": "..."},
    {"title": "向量数据库", "content": "..."},
    {"title": "LangChain 高级", "content": "..."}
]

results = execute_complex_task(docs)
for i, result in enumerate(results, 1):
    print(f"步骤 {i} 结果: {result}")
