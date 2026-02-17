# 核心概念2：ConversationBufferWindowMemory（滑动窗口记忆）

## 一句话定义

**ConversationBufferWindowMemory 只保留最近 N 轮对话，像一个"滑动窗口"，自动丢弃旧对话以控制 Token 消耗。**

---

## 什么是 ConversationBufferWindowMemory？

ConversationBufferWindowMemory 就像一个"短期记忆"，只记住最近发生的事情，旧的记忆会自动被遗忘。

**核心特点：**
- ✅ 自动限制历史长度
- ✅ Token 消耗可控（固定）
- ✅ 实现简单
- ✅ 适合中等长度对话
- ❌ 会丢失旧对话信息
- ❌ 无法回忆很久之前的内容

---

## 工作原理

### 滑动窗口机制

```
假设 k=2（保留最近2轮对话）

第1轮对话后：
[用户: 我叫张三]
[AI: 你好，张三！]

第2轮对话后：
[用户: 我叫张三]
[AI: 你好，张三！]
[用户: 我喜欢打篮球]
[AI: 打篮球很好！]

第3轮对话后（窗口滑动，丢弃第1轮）：
[用户: 我喜欢打篮球]  ← 保留
[AI: 打篮球很好！]      ← 保留
[用户: 我叫什么名字？]  ← 新消息
[AI: 抱歉，我不记得了] ← AI 已经忘记名字了 ❌
```

**关键参数：**
- `k`：保留的对话轮数（1轮 = 1个用户消息 + 1个AI消息）

---

## 详细代码示例

### 示例1：基础使用

```python
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

# 1. 创建记忆（保留最近3轮对话）
memory = ConversationBufferWindowMemory(k=3)

# 2. 创建对话链
llm = ChatOpenAI(temperature=0)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# 3. 第1轮对话
response1 = conversation.predict(input="我叫张三")
print(response1)
# "你好，张三！"

# 4. 第2轮对话
response2 = conversation.predict(input="我今年25岁")
print(response2)
# "好的，张三，你今年25岁。"

# 5. 第3轮对话
response3 = conversation.predict(input="我喜欢打篮球")
print(response3)
# "很高兴知道你喜欢打篮球，张三！"

# 6. 第4轮对话（第1轮已被丢弃）
response4 = conversation.predict(input="我叫什么名字？")
print(response4)
# "抱歉，我不记得你的名字。" ❌
# 因为第1轮对话（"我叫张三"）已经被丢弃了

# 7. 查看当前记忆
print(memory.load_memory_variables({}))
# 只包含最近3轮对话（第2、3、4轮）
```

### 示例2：手动操作记忆

```python
from langchain.memory import ConversationBufferWindowMemory

# 创建记忆（保留最近2轮）
memory = ConversationBufferWindowMemory(k=2)

# 保存多轮对话
memory.save_context({"input": "第1轮：我叫张三"}, {"output": "你好，张三！"})
memory.save_context({"input": "第2轮：我25岁"}, {"output": "好的。"})
memory.save_context({"input": "第3轮：我喜欢篮球"}, {"output": "很好！"})

# 查看记忆（只有最近2轮）
history = memory.load_memory_variables({})
print(history["history"])
# Human: 第2轮：我25岁
# AI: 好的。
# Human: 第3轮：我喜欢篮球
# AI: 很好！

# 第1轮已被丢弃 ✓
```

### 示例3：返回消息列表

```python
from langchain.memory import ConversationBufferWindowMemory

# 返回消息对象列表而非字符串
memory = ConversationBufferWindowMemory(k=2, return_messages=True)

memory.save_context({"input": "你好"}, {"output": "你好！"})
memory.save_context({"input": "天气怎么样"}, {"output": "今天天气很好"})
memory.save_context({"input": "谢谢"}, {"output": "不客气"})

history = memory.load_memory_variables({})
print(history["history"])
# [
#     HumanMessage(content="天气怎么样"),
#     AIMessage(content="今天天气很好"),
#     HumanMessage(content="谢谢"),
#     AIMessage(content="不客气")
# ]
# 第1轮（"你好"）已被丢弃
```

### 示例4：动态调整窗口大小

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=2)

# 保存3轮对话
memory.save_context({"input": "第1轮"}, {"output": "回复1"})
memory.save_context({"input": "第2轮"}, {"output": "回复2"})
memory.save_context({"input": "第3轮"}, {"output": "回复3"})

print(f"当前消息数: {len(memory.chat_memory.messages)}")  # 4条（2轮）

# 动态调整窗口大小
memory.k = 3  # 扩大窗口

# 继续保存对话
memory.save_context({"input": "第4轮"}, {"output": "回复4"})

print(f"调整后消息数: {len(memory.chat_memory.messages)}")  # 6条（3轮）
```

---

## 在 FastAPI 中使用

### 示例：多用户对话 API（窗口记忆版）

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from typing import Dict

app = FastAPI()

# 存储每个用户的记忆
user_memories: Dict[str, ConversationBufferWindowMemory] = {}

class ChatRequest(BaseModel):
    user_id: str
    message: str
    window_size: int = 5  # 默认保留最近5轮对话

class ChatResponse(BaseModel):
    response: str
    window_size: int
    current_messages: int  # 当前记忆中的消息数

def get_or_create_memory(user_id: str, window_size: int) -> ConversationBufferWindowMemory:
    """获取或创建用户的记忆"""
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferWindowMemory(k=window_size)
    else:
        # 更新窗口大小
        user_memories[user_id].k = window_size
    return user_memories[user_id]

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """对话端点"""
    # 1. 获取用户的记忆
    memory = get_or_create_memory(request.user_id, request.window_size)

    # 2. 创建对话链
    llm = ChatOpenAI(temperature=0.7)
    conversation = ConversationChain(llm=llm, memory=memory)

    # 3. 生成回复
    response = conversation.predict(input=request.message)

    # 4. 返回结果
    return ChatResponse(
        response=response,
        window_size=memory.k,
        current_messages=len(memory.chat_memory.messages)
    )

@app.get("/memory-info/{user_id}")
async def get_memory_info(user_id: str):
    """获取用户的记忆信息"""
    if user_id not in user_memories:
        raise HTTPException(status_code=404, detail="User not found")

    memory = user_memories[user_id]
    history = memory.load_memory_variables({})

    return {
        "user_id": user_id,
        "window_size": memory.k,
        "current_messages": len(memory.chat_memory.messages),
        "max_messages": memory.k * 2,  # k轮 = k*2条消息
        "history": history["history"]
    }

@app.put("/window-size/{user_id}")
async def update_window_size(user_id: str, new_size: int):
    """动态调整用户的窗口大小"""
    if user_id not in user_memories:
        raise HTTPException(status_code=404, detail="User not found")

    if new_size < 1:
        raise HTTPException(status_code=400, detail="Window size must be >= 1")

    user_memories[user_id].k = new_size

    return {
        "user_id": user_id,
        "new_window_size": new_size,
        "message": "Window size updated"
    }
```

**测试 API：**

```bash
# 1. 第1轮对话（window_size=2）
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_123", "message": "我叫张三", "window_size": 2}'
# {"response": "你好，张三！", "window_size": 2, "current_messages": 2}

# 2. 第2轮对话
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_123", "message": "我25岁", "window_size": 2}'
# {"response": "好的，张三。", "window_size": 2, "current_messages": 4}

# 3. 第3轮对话（第1轮会被丢弃）
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_123", "message": "我叫什么名字？", "window_size": 2}'
# {"response": "抱歉，我不记得。", "window_size": 2, "current_messages": 4}

# 4. 查看记忆信息
curl "http://localhost:8000/memory-info/user_123"

# 5. 动态调整窗口大小
curl -X PUT "http://localhost:8000/window-size/user_123?new_size=5"
```

---

## 窗口大小选择指南

### 如何选择合适的 k 值？

| 场景 | 推荐 k 值 | 原因 |
|------|----------|------|
| **简单问答** | k=3-5 | 只需要记住最近几轮上下文 |
| **客服对话** | k=5-10 | 需要记住问题和解决方案 |
| **聊天机器人** | k=10-15 | 需要较长的上下文连贯性 |
| **技术支持** | k=15-20 | 需要记住详细的技术信息 |
| **长期陪伴** | 不适合 | 应该使用 Summary 或持久化 |

### Token 消耗计算

假设每轮对话平均 100 tokens：

| k 值 | 消息数 | Token 消耗 | 成本（GPT-4） |
|------|--------|-----------|--------------|
| k=3 | 6条 | 300 tokens | $0.009 |
| k=5 | 10条 | 500 tokens | $0.015 |
| k=10 | 20条 | 1000 tokens | $0.030 |
| k=20 | 40条 | 2000 tokens | $0.060 |

**结论：**
- Token 消耗是固定的（k * 100 tokens）
- 不会随对话轮次增长而增长
- 比 ConversationBufferMemory 更可控

---

## 优缺点分析

### 优点

1. **Token 消耗可控**
   - 固定的历史长度
   - 不会随对话增长而增长
   - 成本可预测

2. **实现简单**
   - 只需设置 k 值
   - 自动管理历史

3. **适合中等长度对话**
   - 10-30 轮对话
   - 平衡了上下文和成本

4. **性能稳定**
   - 内存占用固定
   - 处理速度稳定

### 缺点

1. **会丢失旧信息**
   - 超过窗口的对话会被遗忘
   - 无法回忆很久之前的内容

2. **可能丢失重要信息**
   - 用户在第1轮提到的重要信息
   - 在第10轮可能已经被遗忘

3. **窗口大小难以确定**
   - 太小：丢失重要信息
   - 太大：Token 消耗高

---

## 适用场景

### ✅ 适合使用的场景

1. **中等长度对话**
   - 客服机器人（10-20 轮）
   - 技术支持（15-30 轮）

2. **成本敏感场景**
   - 需要控制 Token 成本
   - 大量用户并发

3. **上下文要求不高**
   - 只需要记住最近的对话
   - 不需要回忆很久之前的内容

### ❌ 不适合使用的场景

1. **需要长期记忆**
   - 个人助理（需要记住用户偏好）
   - 教育辅导（需要记住学习进度）

2. **重要信息在开头**
   - 用户在第1轮提到关键信息
   - 后续对话需要引用

3. **复杂任务**
   - 需要完整上下文的任务
   - 多步骤任务

---

## 与其他记忆类型的对比

| 特性 | BufferMemory | BufferWindowMemory | SummaryMemory |
|------|-------------|-------------------|---------------|
| **存储方式** | 完整历史 | 最近 N 轮 | 总结 + 最近几轮 |
| **Token 消耗** | 线性增长 | 固定 | 低（压缩） |
| **信息丢失** | 无 | 旧对话丢失 | 细节丢失 |
| **实现复杂度** | 简单 | 简单 | 中等 |
| **适用对话长度** | < 10 轮 | 10-30 轮 | > 30 轮 |

---

## 最佳实践

### 1. 动态调整窗口大小

根据对话类型动态调整 k 值：

```python
from langchain.memory import ConversationBufferWindowMemory

def get_adaptive_memory(conversation_type: str) -> ConversationBufferWindowMemory:
    """根据对话类型返回合适的记忆"""
    window_sizes = {
        "simple_qa": 3,      # 简单问答
        "customer_service": 10,  # 客服
        "chat": 15,          # 聊天
        "technical_support": 20  # 技术支持
    }

    k = window_sizes.get(conversation_type, 5)  # 默认5
    return ConversationBufferWindowMemory(k=k)

# 使用
memory = get_adaptive_memory("customer_service")
```

### 2. 监控窗口使用情况

```python
from langchain.memory import ConversationBufferWindowMemory

def get_window_usage(memory: ConversationBufferWindowMemory) -> dict:
    """获取窗口使用情况"""
    current_messages = len(memory.chat_memory.messages)
    max_messages = memory.k * 2
    usage_percent = (current_messages / max_messages) * 100

    return {
        "current_messages": current_messages,
        "max_messages": max_messages,
        "usage_percent": usage_percent,
        "is_full": current_messages >= max_messages
    }

# 使用
memory = ConversationBufferWindowMemory(k=5)
memory.save_context({"input": "你好"}, {"output": "你好！"})

usage = get_window_usage(memory)
print(usage)
# {
#     "current_messages": 2,
#     "max_messages": 10,
#     "usage_percent": 20.0,
#     "is_full": False
# }
```

### 3. 保存重要信息到外部存储

对于重要信息，不要依赖窗口记忆：

```python
from langchain.memory import ConversationBufferWindowMemory
from typing import Dict

class HybridMemory:
    """混合记忆：窗口记忆 + 持久化存储"""
    def __init__(self, k: int = 5):
        self.window_memory = ConversationBufferWindowMemory(k=k)
        self.persistent_facts: Dict[str, str] = {}  # 持久化的事实

    def save_context(self, input_dict: dict, output_dict: dict):
        """保存对话"""
        # 保存到窗口记忆
        self.window_memory.save_context(input_dict, output_dict)

        # 提取重要信息（简化示例）
        user_input = input_dict.get("input", "")
        if "我叫" in user_input:
            name = user_input.split("我叫")[1].strip()
            self.persistent_facts["name"] = name
        elif "我喜欢" in user_input:
            hobby = user_input.split("我喜欢")[1].strip()
            self.persistent_facts["hobby"] = hobby

    def load_memory_variables(self, inputs: dict) -> dict:
        """加载记忆"""
        # 获取窗口记忆
        window_history = self.window_memory.load_memory_variables(inputs)

        # 添加持久化事实
        facts_text = "\n".join([f"{k}: {v}" for k, v in self.persistent_facts.items()])
        if facts_text:
            full_history = f"[重要信息]\n{facts_text}\n\n[最近对话]\n{window_history['history']}"
        else:
            full_history = window_history['history']

        return {"history": full_history}

# 使用
memory = HybridMemory(k=2)

memory.save_context({"input": "我叫张三"}, {"output": "你好，张三！"})
memory.save_context({"input": "我喜欢打篮球"}, {"output": "很好！"})
memory.save_context({"input": "今天天气怎么样"}, {"output": "天气很好"})
memory.save_context({"input": "我叫什么名字？"}, {"output": "你叫张三"})

# 即使第1轮对话被丢弃，重要信息仍然保留
history = memory.load_memory_variables({})
print(history["history"])
# [重要信息]
# name: 张三
# hobby: 打篮球
#
# [最近对话]
# Human: 今天天气怎么样
# AI: 天气很好
# Human: 我叫什么名字？
# AI: 你叫张三
```

### 4. 窗口大小自适应

根据 Token 使用情况自动调整窗口大小：

```python
import tiktoken
from langchain.memory import ConversationBufferWindowMemory

class AdaptiveWindowMemory:
    """自适应窗口记忆"""
    def __init__(self, max_tokens: int = 2000, initial_k: int = 5):
        self.memory = ConversationBufferWindowMemory(k=initial_k)
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model("gpt-4")

    def count_tokens(self) -> int:
        """计算当前记忆的 token 数"""
        history = self.memory.load_memory_variables({})
        return len(self.encoding.encode(history["history"]))

    def save_context(self, input_dict: dict, output_dict: dict):
        """保存对话并自动调整窗口大小"""
        self.memory.save_context(input_dict, output_dict)

        # 检查 token 使用情况
        current_tokens = self.count_tokens()

        if current_tokens > self.max_tokens:
            # Token 超限，减小窗口
            self.memory.k = max(1, self.memory.k - 1)
            print(f"Token 超限，窗口大小减小到 {self.memory.k}")
        elif current_tokens < self.max_tokens * 0.5 and self.memory.k < 20:
            # Token 使用不足，增大窗口
            self.memory.k += 1
            print(f"Token 使用不足，窗口大小增大到 {self.memory.k}")

    def load_memory_variables(self, inputs: dict) -> dict:
        return self.memory.load_memory_variables(inputs)

# 使用
memory = AdaptiveWindowMemory(max_tokens=500, initial_k=5)

for i in range(10):
    memory.save_context(
        {"input": f"第{i+1}轮：这是一个测试消息"},
        {"output": f"收到第{i+1}轮消息"}
    )
    print(f"当前窗口大小: {memory.memory.k}, Token: {memory.count_tokens()}")
```

---

## 总结

**ConversationBufferWindowMemory 的核心特点：**

1. **滑动窗口机制** - 只保留最近 N 轮对话
2. **Token 消耗可控** - 固定的历史长度
3. **适合中等长度对话** - 10-30 轮对话
4. **会丢失旧信息** - 需要注意重要信息的保存

**何时使用：**
- ✅ 中等长度对话（10-30 轮）
- ✅ 成本敏感场景
- ✅ 上下文要求不高

**何时避免：**
- ❌ 需要长期记忆
- ❌ 重要信息在开头
- ❌ 复杂任务

**最佳实践：**
- 根据场景选择合适的 k 值
- 监控窗口使用情况
- 重要信息保存到外部存储
- 考虑使用自适应窗口大小

**下一步：**
- 如果需要长期记忆，考虑使用 **ConversationSummaryMemory**（总结历史）
- 如果需要持久化，考虑使用 **数据库存储**（PostgreSQL/Redis）
