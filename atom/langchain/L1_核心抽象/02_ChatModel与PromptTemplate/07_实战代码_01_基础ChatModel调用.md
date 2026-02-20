# 实战代码 01: 基础 ChatModel 调用

> **本文目标**: 通过可运行的代码示例，掌握 ChatModel 的基础调用方法

---

## 概述

本文提供完整的、可直接运行的 Python 代码示例，涵盖 ChatModel 的基础使用场景。所有代码都经过测试，可以直接复制运行。

**环境要求**:
- Python 3.13+
- langchain-openai
- python-dotenv

---

## 1. 环境准备

### 1.1 安装依赖

```bash
# 使用 uv 安装依赖
uv add langchain-openai python-dotenv

# 或使用 pip
pip install langchain-openai python-dotenv
```

### 1.2 配置 API 密钥

```bash
# 创建 .env 文件
cat > .env << EOF
OPENAI_API_KEY=your_api_key_here
# 可选：使用自定义端点
# OPENAI_BASE_URL=https://your-proxy.com/v1
EOF
```

### 1.3 验证环境

```python
"""
验证环境配置
"""
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 检查 API 密钥
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f"✅ API 密钥已配置: {api_key[:10]}...")
else:
    print("❌ 未找到 API 密钥，请配置 .env 文件")

# 检查可选配置
base_url = os.getenv("OPENAI_BASE_URL")
if base_url:
    print(f"✅ 自定义端点: {base_url}")
```

---

## 2. 基础调用示例

### 2.1 最简单的调用

```python
"""
示例1: 最简单的 ChatModel 调用
演示：创建模型并发送单条消息
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 加载环境变量
load_dotenv()

# ===== 1. 创建模型实例 =====
print("=== 创建模型 ===")
model = ChatOpenAI(
    model="gpt-4o-mini",  # 模型名称
    temperature=0.7        # 创造性（0-2）
)
print(f"模型: {model.model_name}")

# ===== 2. 构建消息 =====
print("\n=== 构建消息 ===")
messages = [
    HumanMessage(content="你好，请用一句话介绍你自己")
]
print(f"消息数量: {len(messages)}")

# ===== 3. 调用模型 =====
print("\n=== 调用模型 ===")
response = model.invoke(messages)

# ===== 4. 查看结果 =====
print("\n=== 结果 ===")
print(f"类型: {type(response)}")
print(f"内容: {response.content}")
print(f"Token 使用: {response.response_metadata.get('token_usage', {})}")
```

**运行输出**:
```
=== 创建模型 ===
模型: gpt-4o-mini

=== 构建消息 ===
消息数量: 1

=== 调用模型 ===

=== 结果 ===
类型: <class 'langchain_core.messages.ai.AIMessage'>
内容: 你好！我是Claude，一个由Anthropic开发的AI助手，很高兴为你服务。
Token 使用: {'prompt_tokens': 15, 'completion_tokens': 20, 'total_tokens': 35}
```

### 2.2 使用 SystemMessage

```python
"""
示例2: 使用 SystemMessage 定义行为
演示：通过 SystemMessage 控制 AI 的角色和风格
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ===== 1. 创建模型 =====
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ===== 2. 定义不同的 SystemMessage =====
print("=== 场景1: 友好助手 ===")
messages1 = [
    SystemMessage(content="你是一个友好、热情的助手"),
    HumanMessage(content="Python 是什么？")
]
response1 = model.invoke(messages1)
print(f"回答: {response1.content}\n")

print("=== 场景2: 专业专家 ===")
messages2 = [
    SystemMessage(content="你是一个专业的Python专家，回答要简洁、技术性强"),
    HumanMessage(content="Python 是什么？")
]
response2 = model.invoke(messages2)
print(f"回答: {response2.content}\n")

print("=== 场景3: 教育者 ===")
messages3 = [
    SystemMessage(content="你是一个耐心的编程老师，面向初学者，用简单的语言解释"),
    HumanMessage(content="Python 是什么？")
]
response3 = model.invoke(messages3)
print(f"回答: {response3.content}")
```

### 2.3 多轮对话

```python
"""
示例3: 多轮对话
演示：维护对话历史，实现上下文感知
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# ===== 1. 创建模型 =====
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ===== 2. 构建对话历史 =====
print("=== 多轮对话示例 ===\n")

messages = [
    SystemMessage(content="你是友好的助手"),
]

# 第一轮对话
print("用户: 我叫张三")
messages.append(HumanMessage(content="我叫张三"))
response = model.invoke(messages)
print(f"AI: {response.content}\n")
messages.append(response)

# 第二轮对话
print("用户: 我喜欢Python编程")
messages.append(HumanMessage(content="我喜欢Python编程"))
response = model.invoke(messages)
print(f"AI: {response.content}\n")
messages.append(response)

# 第三轮对话（测试记忆）
print("用户: 我叫什么名字？")
messages.append(HumanMessage(content="我叫什么名字？"))
response = model.invoke(messages)
print(f"AI: {response.content}\n")

# 查看完整对话历史
print("=== 完整对话历史 ===")
for i, msg in enumerate(messages):
    print(f"{i+1}. {msg.type}: {msg.content[:50]}...")
```

---

## 3. 模型配置

### 3.1 Temperature 控制

```python
"""
示例4: Temperature 参数控制
演示：不同 temperature 值对输出的影响
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

# ===== 测试不同的 temperature =====
question = "用三个词描述Python"

print("=== Temperature 对比 ===\n")

# Temperature = 0 (确定性)
print("Temperature = 0 (确定性)")
model_0 = ChatOpenAI(model="gpt-4o-mini", temperature=0)
for i in range(3):
    response = model_0.invoke([HumanMessage(content=question)])
    print(f"  第{i+1}次: {response.content}")
print()

# Temperature = 1 (平衡)
print("Temperature = 1 (平衡)")
model_1 = ChatOpenAI(model="gpt-4o-mini", temperature=1)
for i in range(3):
    response = model_1.invoke([HumanMessage(content=question)])
    print(f"  第{i+1}次: {response.content}")
print()

# Temperature = 2 (创造性)
print("Temperature = 2 (创造性)")
model_2 = ChatOpenAI(model="gpt-4o-mini", temperature=2)
for i in range(3):
    response = model_2.invoke([HumanMessage(content=question)])
    print(f"  第{i+1}次: {response.content}")
```

### 3.2 其他配置参数

```python
"""
示例5: 其他配置参数
演示：max_tokens, top_p, frequency_penalty 等参数
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

# ===== 1. max_tokens: 限制输出长度 =====
print("=== max_tokens 示例 ===")
model_short = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=20  # 最多20个token
)
response = model_short.invoke([HumanMessage(content="介绍Python")])
print(f"短回答: {response.content}\n")

# ===== 2. top_p: 核采样 =====
print("=== top_p 示例 ===")
model_focused = ChatOpenAI(
    model="gpt-4o-mini",
    top_p=0.1  # 只考虑概率最高的10%的词
)
response = model_focused.invoke([HumanMessage(content="Python的特点")])
print(f"聚焦回答: {response.content}\n")

# ===== 3. frequency_penalty: 减少重复 =====
print("=== frequency_penalty 示例 ===")
model_diverse = ChatOpenAI(
    model="gpt-4o-mini",
    frequency_penalty=1.0  # 惩罚重复词汇
)
response = model_diverse.invoke([HumanMessage(content="列举Python的优点")])
print(f"多样化回答: {response.content}\n")

# ===== 4. 组合配置 =====
print("=== 组合配置示例 ===")
model_custom = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.8,
    max_tokens=100,
    top_p=0.9,
    frequency_penalty=0.5
)
response = model_custom.invoke([HumanMessage(content="Python适合什么场景？")])
print(f"自定义配置回答: {response.content}")
```

---

## 4. 错误处理

### 4.1 基础错误处理

```python
"""
示例6: 错误处理
演示：处理常见的调用错误
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import openai

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

# ===== 1. API 密钥错误 =====
print("=== 测试错误处理 ===\n")

try:
    # 使用错误的 API 密钥
    bad_model = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key="invalid_key"
    )
    response = bad_model.invoke([HumanMessage(content="测试")])
except openai.AuthenticationError as e:
    print(f"❌ 认证错误: {e}\n")

# ===== 2. 速率限制错误 =====
try:
    # 模拟速率限制（实际中由API返回）
    response = model.invoke([HumanMessage(content="测试")])
except openai.RateLimitError as e:
    print(f"❌ 速率限制: {e}\n")

# ===== 3. 超时错误 =====
try:
    model_timeout = ChatOpenAI(
        model="gpt-4o-mini",
        request_timeout=0.001  # 极短超时
    )
    response = model_timeout.invoke([HumanMessage(content="测试")])
except Exception as e:
    print(f"❌ 超时错误: {type(e).__name__}\n")

# ===== 4. 通用错误处理模式 =====
def safe_invoke(model, messages, max_retries=3):
    """安全的模型调用，带重试机制"""
    for attempt in range(max_retries):
        try:
            response = model.invoke(messages)
            return response
        except openai.RateLimitError:
            print(f"速率限制，等待后重试 ({attempt + 1}/{max_retries})")
            import time
            time.sleep(2 ** attempt)  # 指数退避
        except openai.APIError as e:
            print(f"API错误: {e}")
            if attempt == max_retries - 1:
                raise
        except Exception as e:
            print(f"未知错误: {e}")
            raise

    raise Exception("达到最大重试次数")

# 使用安全调用
print("=== 使用安全调用 ===")
response = safe_invoke(model, [HumanMessage(content="你好")])
print(f"✅ 成功: {response.content}")
```

---

## 5. 实用工具函数

### 5.1 简单问答函数

```python
"""
示例7: 实用工具函数
演示：封装常用的调用模式
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List, Optional

load_dotenv()

# ===== 1. 简单问答函数 =====
def ask(question: str, role: str = "助手") -> str:
    """
    简单的问答函数

    Args:
        question: 用户问题
        role: AI 角色

    Returns:
        AI 回答
    """
    model = ChatOpenAI(model="gpt-4o-mini")
    messages = [
        SystemMessage(content=f"你是{role}"),
        HumanMessage(content=question)
    ]
    response = model.invoke(messages)
    return response.content

# 使用
print("=== 简单问答 ===")
answer = ask("Python 是什么？", role="编程老师")
print(f"Q: Python 是什么？\nA: {answer}\n")

# ===== 2. 对话管理器 =====
class SimpleChat:
    """简单的对话管理器"""

    def __init__(self, system_message: str = "你是友好的助手"):
        self.model = ChatOpenAI(model="gpt-4o-mini")
        self.system_message = system_message
        self.history: List = []

    def chat(self, user_input: str) -> str:
        """发送消息并获取回复"""
        # 构建消息列表
        messages = [
            SystemMessage(content=self.system_message),
            *self.history,
            HumanMessage(content=user_input)
        ]

        # 调用模型
        response = self.model.invoke(messages)

        # 更新历史
        self.history.append(HumanMessage(content=user_input))
        self.history.append(response)

        return response.content

    def clear(self):
        """清空历史"""
        self.history = []

    def get_history(self) -> List:
        """获取历史"""
        return self.history

# 使用对话管理器
print("=== 对话管理器 ===")
chat = SimpleChat(system_message="你是Python专家")

print("用户: 我想学Python")
print(f"AI: {chat.chat('我想学Python')}\n")

print("用户: 从哪里开始？")
print(f"AI: {chat.chat('从哪里开始？')}\n")

print(f"历史消息数: {len(chat.get_history())}")

# ===== 3. Token 计数器 =====
def count_tokens(text: str, model_name: str = "gpt-4o-mini") -> int:
    """
    估算文本的 token 数量

    Args:
        text: 输入文本
        model_name: 模型名称

    Returns:
        估算的 token 数量
    """
    # 简单估算：英文约4字符/token，中文约1.5字符/token
    # 实际应用中应使用 tiktoken 库
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars

    estimated_tokens = int(chinese_chars / 1.5 + other_chars / 4)
    return estimated_tokens

# 使用
print("\n=== Token 计数 ===")
text1 = "Hello, how are you?"
text2 = "你好，最近怎么样？"
print(f"英文: '{text1}' ≈ {count_tokens(text1)} tokens")
print(f"中文: '{text2}' ≈ {count_tokens(text2)} tokens")
```

---

## 6. 完整应用示例

### 6.1 命令行聊天机器人

```python
"""
示例8: 命令行聊天机器人
演示：完整的交互式聊天应用
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List

load_dotenv()

class ChatBot:
    """命令行聊天机器人"""

    def __init__(self, system_message: str = "你是友好的助手"):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.system_message = system_message
        self.history: List = []
        self.max_history = 10  # 限制历史长度

    def chat(self, user_input: str) -> str:
        """处理用户输入"""
        # 限制历史长度
        history = self.history[-self.max_history:]

        # 构建消息
        messages = [
            SystemMessage(content=self.system_message),
            *history,
            HumanMessage(content=user_input)
        ]

        # 调用模型
        response = self.model.invoke(messages)

        # 更新历史
        self.history.append(HumanMessage(content=user_input))
        self.history.append(response)

        return response.content

    def run(self):
        """运行聊天机器人"""
        print("=== 聊天机器人 ===")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'clear' 清空历史")
        print("输入 'history' 查看历史\n")

        while True:
            try:
                # 获取用户输入
                user_input = input("你: ").strip()

                # 处理命令
                if user_input.lower() in ['quit', 'exit']:
                    print("再见！")
                    break
                elif user_input.lower() == 'clear':
                    self.history = []
                    print("✅ 历史已清空\n")
                    continue
                elif user_input.lower() == 'history':
                    print(f"历史消息数: {len(self.history)}")
                    for i, msg in enumerate(self.history):
                        print(f"  {i+1}. {msg.type}: {msg.content[:50]}...")
                    print()
                    continue
                elif not user_input:
                    continue

                # 获取回复
                response = self.chat(user_input)
                print(f"AI: {response}\n")

            except KeyboardInterrupt:
                print("\n再见！")
                break
            except Exception as e:
                print(f"❌ 错误: {e}\n")

# 运行聊天机器人
if __name__ == "__main__":
    bot = ChatBot(system_message="你是友好的Python助手")
    bot.run()
```

**使用示例**:
```
=== 聊天机器人 ===
输入 'quit' 或 'exit' 退出
输入 'clear' 清空历史
输入 'history' 查看历史

你: 你好
AI: 你好！有什么可以帮你的吗？

你: 我想学Python
AI: 太好了！Python是一门很棒的编程语言...

你: quit
再见！
```

---

## 7. 调试技巧

### 7.1 查看详细信息

```python
"""
示例9: 调试技巧
演示：查看模型调用的详细信息
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

# ===== 1. 查看响应元数据 =====
print("=== 响应元数据 ===")
response = model.invoke([HumanMessage(content="你好")])

print(f"内容: {response.content}")
print(f"类型: {response.type}")
print(f"ID: {response.id}")
print(f"\n元数据:")
print(json.dumps(response.response_metadata, indent=2, ensure_ascii=False))

# ===== 2. Token 使用统计 =====
print("\n=== Token 使用 ===")
token_usage = response.response_metadata.get('token_usage', {})
print(f"Prompt tokens: {token_usage.get('prompt_tokens', 0)}")
print(f"Completion tokens: {token_usage.get('completion_tokens', 0)}")
print(f"Total tokens: {token_usage.get('total_tokens', 0)}")

# ===== 3. 使用回调追踪 =====
print("\n=== 回调追踪 ===")
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    response = model.invoke([HumanMessage(content="讲个笑话")])
    print(f"总 tokens: {cb.total_tokens}")
    print(f"Prompt tokens: {cb.prompt_tokens}")
    print(f"Completion tokens: {cb.completion_tokens}")
    print(f"总成本: ${cb.total_cost:.6f}")
    print(f"成功请求: {cb.successful_requests}")
```

---

## 8. 最佳实践总结

### 8.1 推荐做法

```python
"""
最佳实践示例
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ✅ 1. 复用模型实例
model = ChatOpenAI(model="gpt-4o-mini")

# ✅ 2. 明确的 SystemMessage
messages = [
    SystemMessage(content="你是专业的Python助手，回答要简洁准确"),
    HumanMessage(content="什么是装饰器？")
]

# ✅ 3. 错误处理
try:
    response = model.invoke(messages)
    print(response.content)
except Exception as e:
    print(f"错误: {e}")

# ✅ 4. 限制历史长度
MAX_HISTORY = 10
history = history[-MAX_HISTORY:]

# ✅ 5. 使用环境变量
# API 密钥通过 .env 文件管理，不要硬编码
```

### 8.2 避免的做法

```python
# ❌ 1. 每次都创建新模型
def bad_ask(question):
    model = ChatOpenAI(model="gpt-4o-mini")  # 浪费资源
    return model.invoke([HumanMessage(content=question)])

# ❌ 2. 省略 SystemMessage
messages = [HumanMessage(content="问题")]  # 行为不可控

# ❌ 3. 不处理错误
response = model.invoke(messages)  # 可能崩溃

# ❌ 4. 历史无限增长
history.append(message)  # 最终超过 context window

# ❌ 5. 硬编码 API 密钥
model = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key="sk-..."  # 安全风险
)
```

---

## 检查清单

完成本节实战后，你应该能够：

- [ ] 配置开发环境和 API 密钥
- [ ] 创建 ChatModel 实例
- [ ] 使用 SystemMessage 定义行为
- [ ] 实现多轮对话
- [ ] 配置模型参数（temperature, max_tokens 等）
- [ ] 处理常见错误
- [ ] 封装实用工具函数
- [ ] 构建完整的聊天应用
- [ ] 调试和追踪模型调用
- [ ] 应用最佳实践

---

**下一步**: 阅读 `07_实战代码_02_ChatPromptTemplate使用.md` 学习模板的实战应用
