FEW_SHOT_PROMPT = """
任务：从技术文档中提取 API 信息

示例 1：
输入:
"使用 GET /api/users 获取用户列表。需要 Authorization 头。"

输出
{{
    "method": "GET",
    "endpoint": "/api/users",
    "description": "获取用户列表",
    "auth_required": true,
    "parameters": []
}}

示例 2：
输入：
"POST /api/users 创建新用户。参数：name（必需）、email（必需）。"

输出：
{{
  "method": "POST",
  "endpoint": "/api/users",
  "description": "创建新用户",
  "auth_required": false,
  "parameters": [
    {{"name": "name", "required": true}},
    {{"name": "email", "required": true}}
  ]
}}

现在处理：
输入：{input_text}
输出：
"""
from openai import OpenAI
import json
import re
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def _parse_json_object(content: str) -> dict:
    """
    Best-effort JSON parsing for LLM outputs.

    Even with `response_format={"type": "json_object"}`, some setups/models may still
    wrap JSON in markdown fences or add brief prefix/suffix text. This keeps the
    example resilient and makes failures easier to debug.
    """
    # 如果模型输出为空/全空白，直接报错：后续 json 解析会给出更难懂的异常。
    if content is None or not str(content).strip():
        raise ValueError("Empty model response; expected a JSON object string.")
    
    # 统一把 content 转成字符串并去掉首尾空白，作为后续解析输入。
    raw = str(content).strip()

    # 如果输出被 Markdown 代码块包裹（```json / ```python 等），先把 fence 去掉再解析。
    if raw.startswith("```"):
        # 支持 ```json、```python 等：去掉开头的 ``` + 可选语言标记 + 紧随空白。
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw, flags=re.IGNORECASE)
        # 去掉结尾的 ```（允许结尾前有空白/换行）。
        raw = re.sub(r"\s*```$", "", raw)
        # 清理 fence 去除后残留的首尾空白。
        raw = raw.strip()

    try:
        # raw_decode 会从字符串开头解析“第一个 JSON 值”，并返回 (对象, 结束位置)。
        obj, _end = json.JSONDecoder().raw_decode(raw)
        # 解析成功则直接返回对象（通常是 dict）。
        return obj
    except json.JSONDecodeError:
        # 如果开头不是 JSON，尝试在文本中找到第一个 “{” 并从那里开始解析 JSON 对象。
        start = raw.find("{")
        # 找不到 “{” 基本可判定为“完全不是 JSON 对象输出”。
        if start != -1:
            try:
                # 从第一个 “{” 处 raw_decode：即便 JSON 后面还有多余文本，也能先拿到对象本体。
                obj, _end = json.JSONDecoder().raw_decode(raw[start:])
                # 解析成功则返回对象。
                return obj
            except json.JSONDecodeError as e:
                # 截取前 300 字符做预览，并把换行转义，方便报错信息单行展示。
                preview = raw[:300].replace("\n", "\\n")
                # 抛出更可读的错误：包含 JSON 解析器原因 + 原始输出预览，便于调 prompt/定位问题。
                raise ValueError(
                    "Model returned an invalid JSON object string "
                    f"(JSON error: {e}; preview): {preview}"
                )
        
        # 走到这里说明：既不能从开头解析，也找不到 “{”，因此模型输出很可能不是 JSON。
        preview = raw[:300].replace("\n", "\\n")
        raise ValueError(f"Model returned non-JSON content (preview): {preview}")

def extract_api_info(text: str) -> dict:
    """提取 API 信息（使用 Few-shot）"""
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "你是 API 文档分析专家 "},
            {"role": "use", "content": FEW_SHOT_PROMPT.format(
                input_text=text
            )}
        ],
        temperature=0.0
    )

    return _parse_json_object(response.choices[0].message.content)

# 测试
text = "DELETE /api/users/:id 删除用户。需要 admin 权限。"
result = extract_api_info(text)
print(json.dumps(result, indent=2, ensure_ascii=False))
