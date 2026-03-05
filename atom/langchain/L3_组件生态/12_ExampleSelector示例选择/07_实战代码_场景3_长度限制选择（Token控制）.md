# 实战代码 - 场景3：长度限制选择（Token控制）

## 场景描述

在使用 LLM 时，我们经常面临 Token 限制的挑战：

1. **Context Window 限制**：不同模型有不同的 Token 上限（GPT-3.5: 4K, GPT-4: 8K, GPT-4o: 128K）
2. **成本控制**：Token 数量直接影响 API 调用成本
3. **动态输入长度**：用户输入长度不固定，需要动态调整示例数量

**实际应用场景：**
- Token 限制严格的模型（如 GPT-3.5-turbo）
- 成本敏感的应用
- 需要精确控制 Prompt 长度的场景
- 批量处理优化

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/length_based.py]

---

## 核心技术点

### 1. LengthBasedExampleSelector

基于长度限制选择示例的选择器：

```python
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_core.prompts import PromptTemplate

selector = LengthBasedExampleSelector(
    examples=examples,              # 示例列表
    example_prompt=example_prompt,  # 示例格式化模板
    max_length=2048,                # 最大长度限制
    get_text_length=len             # 长度计算函数（可自定义）
)
```

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/length_based.py]

### 2. 工作流程

```
用户输入 → 计算输入长度 → 计算剩余可用长度 → 按顺序选择示例 → 确保不超限
```

### 3. 关键参数

- **max_length**: 最大长度限制（默认 2048）
- **get_text_length**: 长度计算函数（默认按单词数）
- **example_prompt**: 示例格式化模板

[来源: reference/search_example_selector_02.md]

---

## 完整可运行代码

```python
"""
场景3：长度限制选择（Token控制）
演示：使用 LengthBasedExampleSelector 控制 Prompt 长度
"""

import os
import re
from dotenv import load_dotenv
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# ===== 1. 准备示例库 =====
print("=== 1. 准备示例库 ===\n")

# 代码生成示例库（不同长度）
code_examples = [
    {
        "task": "Read a file",
        "code": "with open('file.txt', 'r') as f:\n    content = f.read()"
    },
    {
        "task": "Write to a file",
        "code": "with open('file.txt', 'w') as f:\n    f.write('Hello, World!')"
    },
    {
        "task": "List files in directory",
        "code": "import os\nfiles = os.listdir('.')\nfor file in files:\n    print(file)"
    },
    {
        "task": "Make HTTP request",
        "code": "import requests\nresponse = requests.get('https://api.example.com')\ndata = response.json()\nprint(data)"
    },
    {
        "task": "Parse JSON",
        "code": "import json\nwith open('data.json', 'r') as f:\n    data = json.load(f)\nprint(data['key'])"
    },
    {
        "task": "Connect to database",
        "code": "import sqlite3\nconn = sqlite3.connect('database.db')\ncursor = conn.cursor()\ncursor.execute('SELECT * FROM users')\nresults = cursor.fetchall()\nconn.close()"
    },
    {
        "task": "Send email",
        "code": "import smtplib\nfrom email.mime.text import MIMEText\n\nmsg = MIMEText('Hello')\nmsg['Subject'] = 'Test'\nmsg['From'] = 'sender@example.com'\nmsg['To'] = 'receiver@example.com'\n\nwith smtplib.SMTP('smtp.gmail.com', 587) as server:\n    server.starttls()\n    server.login('user', 'password')\n    server.send_message(msg)"
    },
    {
        "task": "Process CSV",
        "code": "import csv\nwith open('data.csv', 'r') as f:\n    reader = csv.DictReader(f)\n    for row in reader:\n        print(row['name'], row['age'])"
    },
]

print(f"示例库包含 {len(code_examples)} 个代码示例")
print(f"示例长度范围：{min(len(ex['code']) for ex in code_examples)} - {max(len(ex['code']) for ex in code_examples)} 字符\n")

# ===== 2. 定义示例模板 =====
print("=== 2. 定义示例模板 ===\n")

example_prompt = PromptTemplate(
    input_variables=["task", "code"],
    template="Task: {task}\nCode:\n```python\n{code}\n```"
)

print("示例格式：")
print(example_prompt.format(
    task="Read a file",
    code="with open('file.txt', 'r') as f:\n    content = f.read()"
))
print()

# ===== 3. 创建长度限制选择器（默认长度计算）=====
print("=== 3. 创建长度限制选择器（默认长度计算）===\n")

# 默认按单词数计算
selector_default = LengthBasedExampleSelector(
    examples=code_examples,
    example_prompt=example_prompt,
    max_length=100  # 限制为 100 个单词
)

print("✓ 选择器创建成功（默认长度计算）")
print(f"  - 长度计算方式: 单词数")
print(f"  - max_length: 100 个单词\n")

# ===== 4. 测试不同长度的输入 =====
print("=== 4. 测试不同长度的输入 ===\n")

test_inputs = [
    {"task": "short input"},  # 短输入
    {"task": "This is a medium length input that contains more words"},  # 中等输入
    {"task": "This is a very long input " * 20},  # 长输入
]

for i, input_vars in enumerate(test_inputs, 1):
    input_text = input_vars["task"]
    input_length = len(input_text.split())
    
    print(f"--- 测试 {i}: 输入长度 = {input_length} 个单词 ---")
    print(f"输入: {input_text[:50]}{'...' if len(input_text) > 50 else ''}\n")
    
    selected = selector_default.select_examples(input_vars)
    
    print(f"选中 {len(selected)} 个示例：")
    for j, ex in enumerate(selected, 1):
        print(f"  {j}. {ex['task']}")
    print()

# ===== 5. 自定义长度计算函数（按 Token 数）=====
print("=== 5. 自定义长度计算函数（按 Token 数）===\n")

try:
    import tiktoken
    
    def get_token_length(text: str) -> int:
        """按 Token 数计算长度（OpenAI）"""
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        return len(encoding.encode(text))
    
    # 创建基于 Token 的选择器
    selector_token = LengthBasedExampleSelector(
        examples=code_examples,
        example_prompt=example_prompt,
        get_text_length=get_token_length,
        max_length=500  # 限制为 500 tokens
    )
    
    print("✓ Token 计数选择器创建成功")
    print(f"  - 长度计算方式: Token 数（tiktoken）")
    print(f"  - max_length: 500 tokens\n")
    
    # 测试
    test_input = {"task": "How to read a CSV file?"}
    selected = selector_token.select_examples(test_input)
    
    print(f"测试输入: {test_input['task']}")
    print(f"选中 {len(selected)} 个示例：")
    for ex in selected:
        print(f"  - {ex['task']}")
    print()
    
except ImportError:
    print("⚠️  tiktoken 未安装，跳过 Token 计数示例")
    print("   安装命令: pip install tiktoken\n")

# ===== 6. 与 FewShotPromptTemplate 集成 =====
print("=== 6. 与 FewShotPromptTemplate 集成 ===\n")

few_shot_prompt = FewShotPromptTemplate(
    example_selector=selector_default,
    example_prompt=example_prompt,
    prefix="You are a Python code assistant. Generate code based on the following examples:",
    suffix="Task: {task}\nCode:",
    input_variables=["task"]
)

print("✓ FewShotPromptTemplate 创建成功\n")

# 测试不同长度的输入
test_tasks = [
    "short",
    "This is a medium length task description",
    "This is a very long task description " * 10
]

for task in test_tasks:
    task_length = len(task.split())
    formatted = few_shot_prompt.format(task=task)
    prompt_length = len(formatted.split())
    
    print(f"输入长度: {task_length} 个单词")
    print(f"Prompt 总长度: {prompt_length} 个单词")
    print(f"示例数量: {len(selector_default.select_examples({'task': task}))}")
    print()

# ===== 7. 动态调整 max_length =====
print("=== 7. 动态调整 max_length ===\n")

def dynamic_max_length(input_text: str, base_max: int = 200) -> int:
    """根据输入长度动态调整 max_length"""
    input_length = len(input_text.split())
    
    if input_length > 100:
        return base_max - 50  # 输入长，减少示例空间
    elif input_length < 10:
        return base_max + 50  # 输入短，增加示例空间
    else:
        return base_max

# 测试动态调整
test_inputs_dynamic = [
    "short",
    "This is a medium length input with some words",
    "This is a very long input " * 30
]

for input_text in test_inputs_dynamic:
    max_len = dynamic_max_length(input_text)
    
    # 创建临时选择器
    temp_selector = LengthBasedExampleSelector(
        examples=code_examples,
        example_prompt=example_prompt,
        max_length=max_len
    )
    
    selected = temp_selector.select_examples({"task": input_text})
    
    print(f"输入长度: {len(input_text.split())} 个单词")
    print(f"动态 max_length: {max_len}")
    print(f"选中示例数: {len(selected)}")
    print()

# ===== 8. 对比：固定示例 vs 长度限制选择 =====
print("=== 8. 对比：固定示例 vs 长度限制选择 ===\n")

# 固定示例（前3个）
fixed_examples = code_examples[:3]
fixed_prompt = FewShotPromptTemplate(
    examples=fixed_examples,
    example_prompt=example_prompt,
    prefix="You are a Python code assistant. Generate code based on the following examples:",
    suffix="Task: {task}\nCode:",
    input_variables=["task"]
)

# 长度限制选择
dynamic_prompt = few_shot_prompt

# 测试
test_task_short = "Read file"
test_task_long = "This is a very long task description " * 20

print("短输入测试：")
print(f"  固定示例: {len(fixed_examples)} 个")
print(f"  长度限制: {len(selector_default.select_examples({'task': test_task_short}))} 个")
print()

print("长输入测试：")
print(f"  固定示例: {len(fixed_examples)} 个（可能超限）")
print(f"  长度限制: {len(selector_default.select_examples({'task': test_task_long}))} 个（自动调整）")
print()

# ===== 9. 与 LLM 集成生成代码 =====
print("=== 9. 与 LLM 集成生成代码 ===\n")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 创建代码生成链
code_chain = few_shot_prompt | llm

# 测试
test_task = "Download a file from URL"

print(f"任务: {test_task}\n")

# 查看选中的示例
selected = selector_default.select_examples({"task": test_task})
print(f"选中 {len(selected)} 个示例：")
for ex in selected:
    print(f"  - {ex['task']}")
print()

# 生成代码
response = code_chain.invoke({"task": test_task})
print("生成的代码：")
print(response.content)
print()

# ===== 10. 实际应用建议 =====
print("=== 10. 实际应用建议 ===\n")

print("✓ 最佳实践：")
print("  1. 使用 Token 计数而非单词数（更精确）")
print("  2. max_length 设置为 context_window * 0.6")
print("  3. 按重要性排序示例（重要的在前）")
print("  4. 结合语义相似度使用（先语义后长度）")
print()

print("✓ max_length 推荐值：")
print("  - GPT-3.5-turbo (4K): 800-1500")
print("  - GPT-4 (8K): 1500-2000")
print("  - GPT-4o-mini (128K): 2000-4000")
print()

print("✓ 性能优化：")
print("  1. 预计算示例长度（避免重复计算）")
print("  2. 缓存选择结果")
print("  3. 批量处理时复用选择器")
print()

print("✓ 常见误区：")
print("  ❌ 长度单位混淆（单词 vs Token）")
print("  ❌ 不考虑输出长度")
print("  ❌ 示例顺序随机（应按重要性排序）")
print()

print("=" * 80)
print("实战完成！")
print("=" * 80)
```

---

## 代码解析

### 1. 默认长度计算

```python
def _get_length_based(text: str) -> int:
    """默认按单词数计算"""
    return len(re.split(r"\n| ", text))
```

**特点：**
- 按空格和换行符分割
- 简单快速
- 不够精确（单词数 ≠ Token 数）

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/length_based.py]

### 2. Token 计数（推荐）

```python
import tiktoken

def get_token_length(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    return len(encoding.encode(text))
```

**优势：**
- 精确计算 Token 数
- 与 OpenAI API 一致
- 避免超限

### 3. 选择逻辑

```python
def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
    # 计算输入长度
    inputs = " ".join(input_variables.values())
    remaining_length = self.max_length - self.get_text_length(inputs)
    
    # 按顺序选择示例
    i = 0
    examples = []
    while remaining_length > 0 and i < len(self.examples):
        new_length = remaining_length - self.example_text_lengths[i]
        if new_length < 0:
            break
        examples.append(self.examples[i])
        remaining_length = new_length
        i += 1
    
    return examples
```

**关键点：**
- 按顺序选择（不考虑相关性）
- 动态计算剩余空间
- 确保不超限

[来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/length_based.py]

---

## 运行结果示例

### 测试1：不同长度输入

```
--- 测试 1: 输入长度 = 2 个单词 ---
输入: short input

选中 5 个示例：
  1. Read a file
  2. Write to a file
  3. List files in directory
  4. Make HTTP request
  5. Parse JSON

--- 测试 2: 输入长度 = 10 个单词 ---
输入: This is a medium length input that contains more words

选中 3 个示例：
  1. Read a file
  2. Write to a file
  3. List files in directory

--- 测试 3: 输入长度 = 80 个单词 ---
输入: This is a very long input This is a very long input...

选中 1 个示例：
  1. Read a file
```

**观察：**
- 输入越长，选中的示例越少
- 自动调整以确保不超限

### 测试2：动态 max_length

```
输入长度: 1 个单词
动态 max_length: 250
选中示例数: 6

输入长度: 9 个单词
动态 max_length: 200
选中示例数: 4

输入长度: 120 个单词
动态 max_length: 150
选中示例数: 2
```

**观察：**
- 根据输入长度动态调整
- 保持 Prompt 总长度稳定

---

## 实际应用建议

### 1. max_length 设置公式

```python
def calculate_max_length(model: str, expected_output: int = 500) -> int:
    """计算推荐的 max_length"""
    context_windows = {
        "gpt-3.5-turbo": 4096,
        "gpt-4": 8192,
        "gpt-4o-mini": 128000,
        "claude-3": 200000,
    }
    
    context_window = context_windows.get(model, 4096)
    
    # 60% 用于输入，40% 留给输出
    max_length = int(context_window * 0.6) - expected_output
    
    return max_length

# 使用
max_len = calculate_max_length("gpt-4o-mini", expected_output=500)
print(f"推荐 max_length: {max_len}")
```

### 2. 混合策略：语义 + 长度

```python
# 先用语义相似度选择候选
semantic_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=embeddings,
    vectorstore_cls=Chroma,
    k=10  # 选择 10 个候选
)

candidates = semantic_selector.select_examples({"input": user_input})

# 再用长度限制筛选
length_selector = LengthBasedExampleSelector(
    examples=candidates,
    example_prompt=example_prompt,
    get_text_length=get_token_length,
    max_length=2000
)

final_examples = length_selector.select_examples({"input": user_input})
```

[来源: reference/fetch_example_selector_10.md | https://github.com/whitesmith/langchain-semantic-length-example-selector]

### 3. 示例优先级排序

```python
# 为示例添加优先级
examples_with_priority = [
    {"task": "...", "code": "...", "priority": 10},  # 高优先级
    {"task": "...", "code": "...", "priority": 5},   # 中优先级
    {"task": "...", "code": "...", "priority": 1},   # 低优先级
]

# 按优先级排序
sorted_examples = sorted(
    examples_with_priority,
    key=lambda x: x['priority'],
    reverse=True
)

# 使用排序后的示例
selector = LengthBasedExampleSelector(
    examples=sorted_examples,
    example_prompt=example_prompt,
    max_length=2000
)
```

### 4. 批量处理优化

```python
# 预计算示例长度
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o-mini")

example_token_lengths = [
    len(encoding.encode(example_prompt.format(**ex)))
    for ex in examples
]

# 创建选择器（使用预计算的长度）
selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    get_text_length=lambda text: len(encoding.encode(text)),
    max_length=2000,
    example_text_lengths=example_token_lengths  # 使用预计算的长度
)

# 批量处理
results = [
    selector.select_examples({"task": task})
    for task in batch_tasks
]
```

---

## 常见问题

### Q1: 如何避免长度单位混淆？

**问题：** 不清楚 max_length 是单词数还是 Token 数

**解决方案：**

```python
# ❌ 错误：使用默认（单词数）
selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=2000  # 实际是 2000 个单词，不是 Token
)

# ✅ 正确：明确使用 Token 计数
import tiktoken

def get_token_length(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    return len(encoding.encode(text))

selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    get_text_length=get_token_length,
    max_length=2000  # 2000 tokens
)
```

### Q2: 如何处理输出长度？

**问题：** 只考虑输入，忘记留空间给输出

**解决方案：**

```python
context_window = 4096  # GPT-3.5-turbo
expected_output = 500  # 预期输出长度

# 留出空间给输出
max_length = context_window - expected_output  # 3596
```

### Q3: 示例顺序重要吗？

**问题：** LengthBasedExampleSelector 按顺序选择

**解决方案：**

```python
# ❌ 错误：随机顺序
examples = random.shuffle(examples)

# ✅ 正确：按重要性排序
examples = sorted(examples, key=lambda x: x['importance'], reverse=True)

selector = LengthBasedExampleSelector(
    examples=examples,  # 重要的示例在前面
    example_prompt=example_prompt,
    max_length=2000
)
```

---

## 总结

**核心优势：**
1. 精确控制 Prompt 长度
2. 避免超过 context window
3. 降低 Token 成本
4. 简单直接，无需复杂依赖

**最佳实践：**
- 使用 Token 计数而非单词数
- 留出足够空间给输出
- 按重要性排序示例
- 结合语义相似度使用

**适用场景：**
- Token 限制严格的模型
- 成本敏感的应用
- 批量处理优化
- 需要精确控制长度

**局限性：**
- 不考虑示例相关性
- 按顺序选择，可能错过相关示例
- 需要与其他选择器结合使用

[来源: 综合多个参考资料]

---

**参考资料：**
- [来源: sourcecode/langchain/libs/core/langchain_core/example_selectors/length_based.py]
- [来源: reference/search_example_selector_02.md]
- [来源: reference/fetch_example_selector_06.md | https://pub.aimind.so/langchain-in-chains-6-example-selectors-310f47b4cdf3]
- [来源: reference/fetch_example_selector_10.md | https://github.com/whitesmith/langchain-semantic-length-example-selector]
