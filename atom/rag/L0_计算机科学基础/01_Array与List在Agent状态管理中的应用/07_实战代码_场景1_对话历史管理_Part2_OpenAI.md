# 实战代码 - 场景1：对话历史管理（Part 2: OpenAI SDK）

## 场景描述

使用 OpenAI SDK 直接管理对话历史，实现：
- 手动消息列表管理
- Context Window 控制
- 会话持久化
- 消息修剪策略

---

## 完整代码实现

```python
"""
OpenAI SDK 对话历史管理示例
演示：手动管理消息列表和会话持久化
"""

import os
import json
import time
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# ===== 1. 创建 OpenAI 客户端 =====
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")  # 可选：使用代理
)


# ===== 2. 对话历史管理器 =====
class ConversationManager:
    """对话历史管理器"""

    def __init__(self, model: str = "gpt-4", max_messages: int = 50):
        self.model = model
        self.max_messages = max_messages
        self.conversations: Dict[str, List[Dict]] = {}

    def add_message(self, user_id: str, role: str, content: str):
        """添加消息到对话历史"""
        if user_id not in self.conversations:
            self.conversations[user_id] = []

        message = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }

        self.conversations[user_id].append(message)

        # 自动修剪
        self._trim_messages(user_id)

    def _trim_messages(self, user_id: str):
        """修剪消息历史（保留最近 N 条）"""
        messages = self.conversations[user_id]

        if len(messages) > self.max_messages:
            # 保留系统消息 + 最近的消息
            system_msgs = [msg for msg in messages if msg["role"] == "system"]
            recent_msgs = messages[-(self.max_messages - len(system_msgs)):]

            self.conversations[user_id] = system_msgs + recent_msgs

            print(f"[修剪] 用户 {user_id}: {len(messages)} -> {len(self.conversations[user_id])} 条消息")

    def get_messages(self, user_id: str) -> List[Dict]:
        """获取对话历史（OpenAI 格式）"""
        if user_id not in self.conversations:
            return []

        # 转换为 OpenAI 格式（移除 timestamp）
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.conversations[user_id]
        ]

    def chat(self, user_id: str, user_message: str) -> str:
        """发送消息并获取回复"""
        # 添加用户消息
        self.add_message(user_id, "user", user_message)

        # 获取对话历史
        messages = self.get_messages(user_id)

        print(f"\n[Chat] 用户 {user_id}: {user_message}")
        print(f"[Chat] 当前消息数: {len(messages)}")

        # 调用 OpenAI API
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0
        )

        # 提取回复
        assistant_message = response.choices[0].message.content

        # 添加助手消息
        self.add_message(user_id, "assistant", assistant_message)

        print(f"[Chat] AI 回复: {assistant_message[:50]}...")

        return assistant_message

    def save_to_file(self, user_id: str, filepath: str):
        """保存对话历史到文件"""
        if user_id not in self.conversations:
            return

        data = {
            "user_id": user_id,
            "messages": self.conversations[user_id],
            "saved_at": time.time()
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[保存] 用户 {user_id} 的对话历史已保存到 {filepath}")

    def load_from_file(self, user_id: str, filepath: str):
        """从文件加载对话历史"""
        if not Path(filepath).exists():
            print(f"[加载] 文件不存在: {filepath}")
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.conversations[user_id] = data["messages"]

        print(f"[加载] 用户 {user_id} 的对话历史已加载，共 {len(data['messages'])} 条消息")

    def get_stats(self, user_id: str) -> Dict:
        """获取对话统计信息"""
        if user_id not in self.conversations:
            return {"total": 0, "user": 0, "assistant": 0, "system": 0}

        messages = self.conversations[user_id]

        return {
            "total": len(messages),
            "user": sum(1 for msg in messages if msg["role"] == "user"),
            "assistant": sum(1 for msg in messages if msg["role"] == "assistant"),
            "system": sum(1 for msg in messages if msg["role"] == "system"),
        }


# ===== 3. 使用示例 =====
def main():
    """主函数"""
    print("=" * 60)
    print("OpenAI SDK 对话历史管理示例")
    print("=" * 60)

    # 创建管理器
    manager = ConversationManager(model="gpt-4", max_messages=20)

    # 用户 ID
    user_id = "user_123"

    # 添加系统消息
    manager.add_message(user_id, "system", "你是一个有帮助的 AI 助手")

    # ===== 第 1 轮对话 =====
    print("\n" + "=" * 60)
    print("第 1 轮对话")
    print("=" * 60)

    response1 = manager.chat(user_id, "我叫张三，我在学习 RAG")
    print(f"\n[结果] {response1}")

    # ===== 第 2 轮对话 =====
    print("\n" + "=" * 60)
    print("第 2 轮对话")
    print("=" * 60)

    response2 = manager.chat(user_id, "我叫什么名字？")
    print(f"\n[结果] {response2}")

    # ===== 第 3 轮对话 =====
    print("\n" + "=" * 60)
    print("第 3 轮对话")
    print("=" * 60)

    response3 = manager.chat(user_id, "我在学习什么？")
    print(f"\n[结果] {response3}")

    # ===== 查看统计信息 =====
    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)

    stats = manager.get_stats(user_id)
    print(f"总消息数: {stats['total']}")
    print(f"用户消息: {stats['user']}")
    print(f"助手消息: {stats['assistant']}")
    print(f"系统消息: {stats['system']}")

    # ===== 保存对话历史 =====
    print("\n" + "=" * 60)
    print("保存对话历史")
    print("=" * 60)

    manager.save_to_file(user_id, "data/conversations/user_123.json")

    # ===== 加载对话历史 =====
    print("\n" + "=" * 60)
    print("加载对话历史")
    print("=" * 60)

    new_manager = ConversationManager()
    new_manager.load_from_file(user_id, "data/conversations/user_123.json")

    # 继续对话
    response4 = new_manager.chat(user_id, "总结一下我们的对话")
    print(f"\n[结果] {response4}")


# ===== 4. 高级示例：Context Window 管理 =====
class ContextWindowManager(ConversationManager):
    """带 Context Window 管理的对话管理器"""

    def __init__(self, model: str = "gpt-4", max_tokens: int = 8000):
        super().__init__(model)
        self.max_tokens = max_tokens

    def estimate_tokens(self, messages: List[Dict]) -> int:
        """估算 token 数量（简化版）"""
        # 简化估算：1 token ≈ 4 字符
        total_chars = sum(len(msg["content"]) for msg in messages)
        return total_chars // 4

    def _trim_messages(self, user_id: str):
        """基于 token 数量修剪消息"""
        messages = self.conversations[user_id]

        # 估算当前 token 数
        current_tokens = self.estimate_tokens(
            [{"role": msg["role"], "content": msg["content"]} for msg in messages]
        )

        if current_tokens > self.max_tokens:
            # 保留系统消息
            system_msgs = [msg for msg in messages if msg["role"] == "system"]

            # 从最新消息开始，逐步添加直到达到 token 限制
            recent_msgs = []
            token_count = self.estimate_tokens(
                [{"role": msg["role"], "content": msg["content"]} for msg in system_msgs]
            )

            for msg in reversed(messages):
                if msg["role"] == "system":
                    continue

                msg_tokens = self.estimate_tokens([{"role": msg["role"], "content": msg["content"]}])

                if token_count + msg_tokens > self.max_tokens:
                    break

                recent_msgs.insert(0, msg)
                token_count += msg_tokens

            self.conversations[user_id] = system_msgs + recent_msgs

            print(f"[修剪] 用户 {user_id}: {current_tokens} -> {token_count} tokens")


def context_window_example():
    """Context Window 管理示例"""
    print("\n" + "=" * 60)
    print("Context Window 管理示例")
    print("=" * 60)

    manager = ContextWindowManager(model="gpt-4", max_tokens=1000)

    user_id = "user_456"
    manager.add_message(user_id, "system", "你是一个有帮助的助手")

    # 模拟长对话
    for i in range(20):
        response = manager.chat(user_id, f"这是第 {i+1} 条消息，请简短回复")
        print(f"第 {i+1} 轮: {len(manager.conversations[user_id])} 条消息")


# ===== 5. 高级示例：多用户并发 =====
def multi_user_example():
    """多用户并发示例"""
    print("\n" + "=" * 60)
    print("多用户并发示例")
    print("=" * 60)

    manager = ConversationManager()

    # 用户 1
    manager.add_message("user_001", "system", "你是一个有帮助的助手")
    response1 = manager.chat("user_001", "我叫李四")
    print(f"\n[用户1] {response1}")

    # 用户 2
    manager.add_message("user_002", "system", "你是一个有帮助的助手")
    response2 = manager.chat("user_002", "我叫王五")
    print(f"[用户2] {response2}")

    # 用户 1 继续对话
    response3 = manager.chat("user_001", "我叫什么名字？")
    print(f"\n[用户1] {response3}")

    # 用户 2 继续对话
    response4 = manager.chat("user_002", "我叫什么名字？")
    print(f"[用户2] {response4}")

    # 统计信息
    print("\n统计信息:")
    print(f"用户1: {manager.get_stats('user_001')}")
    print(f"用户2: {manager.get_stats('user_002')}")


# ===== 6. 性能测试 =====
def performance_test():
    """性能测试"""
    print("\n" + "=" * 60)
    print("性能测试")
    print("=" * 60)

    manager = ConversationManager()
    user_id = "perf_test"

    manager.add_message(user_id, "system", "你是一个有帮助的助手")

    # 测试 100 轮对话
    start = time.perf_counter()

    for i in range(100):
        manager.chat(user_id, f"消息 {i}")

    elapsed = time.perf_counter() - start

    print(f"\n100 轮对话耗时: {elapsed:.2f}s")
    print(f"平均每轮: {elapsed/100*1000:.2f}ms")

    # 最终统计
    stats = manager.get_stats(user_id)
    print(f"最终消息数: {stats['total']}")


# ===== 7. 对比：OpenAI SDK vs LangGraph =====
def comparison_example():
    """对比示例"""
    print("\n" + "=" * 60)
    print("OpenAI SDK vs LangGraph 对比")
    print("=" * 60)

    print("\n【OpenAI SDK】")
    print("优点:")
    print("  - 直接控制消息列表")
    print("  - 灵活的修剪策略")
    print("  - 简单的持久化（JSON）")
    print("\n缺点:")
    print("  - 需要手动管理状态")
    print("  - 需要手动实现持久化")
    print("  - 需要手动处理并发")

    print("\n【LangGraph】")
    print("优点:")
    print("  - 自动状态管理")
    print("  - 内置持久化（checkpointer）")
    print("  - 自动消息去重")
    print("\n缺点:")
    print("  - 学习曲线较陡")
    print("  - 抽象层较多")
    print("  - 灵活性稍低")


if __name__ == "__main__":
    # 运行主示例
    main()

    # 运行 Context Window 示例
    # context_window_example()  # 注释掉，避免调用 API

    # 运行多用户示例
    multi_user_example()

    # 运行性能测试
    # performance_test()  # 注释掉，避免调用 API

    # 运行对比示例
    comparison_example()
```

---

## 运行输出示例

```
============================================================
OpenAI SDK 对话历史管理示例
============================================================

============================================================
第 1 轮对话
============================================================

[Chat] 用户 user_123: 我叫张三，我在学习 RAG
[Chat] 当前消息数: 2
[Chat] AI 回复: 你好，张三！很高兴认识你。RAG（Retrieval-Augmented...

[结果] 你好，张三！很高兴认识你。RAG（Retrieval-Augmented Generation）是一种...

============================================================
第 2 轮对话
============================================================

[Chat] 用户 user_123: 我叫什么名字？
[Chat] 当前消息数: 4
[Chat] AI 回复: 你叫张三。

[结果] 你叫张三。

============================================================
第 3 轮对话
============================================================

[Chat] 用户 user_123: 我在学习什么？
[Chat] 当前消息数: 6
[Chat] AI 回复: 你在学习 RAG（Retrieval-Augmented Generation）。

[结果] 你在学习 RAG（Retrieval-Augmented Generation）。

============================================================
统计信息
============================================================
总消息数: 7
用户消息: 3
助手消息: 3
系统消息: 1

============================================================
保存对话历史
============================================================
[保存] 用户 user_123 的对话历史已保存到 data/conversations/user_123.json

============================================================
加载对话历史
============================================================
[加载] 用户 user_123 的对话历史已加载，共 7 条消息

[Chat] 用户 user_123: 总结一下我们的对话
[Chat] 当前消息数: 8
[Chat] AI 回复: 我们的对话主要围绕你的自我介绍和学习内容展开...

[结果] 我们的对话主要围绕你的自我介绍和学习内容展开...
```

---

## 保存的 JSON 文件示例

```json
{
  "user_id": "user_123",
  "messages": [
    {
      "role": "system",
      "content": "你是一个有帮助的 AI 助手",
      "timestamp": 1707825600.123
    },
    {
      "role": "user",
      "content": "我叫张三，我在学习 RAG",
      "timestamp": 1707825601.456
    },
    {
      "role": "assistant",
      "content": "你好，张三！很高兴认识你...",
      "timestamp": 1707825602.789
    }
  ],
  "saved_at": 1707825610.123
}
```

---

## 关键要点

1. **手动状态管理**
   - 使用 Python List 存储消息
   - 手动追加用户和助手消息
   - 需要自己实现修剪逻辑

2. **持久化策略**
   - JSON 文件存储（简单）
   - 包含时间戳（可追溯）
   - 支持加载和恢复

3. **Context Window 管理**
   - 估算 token 数量
   - 基于 token 限制修剪
   - 保留系统消息

4. **多用户隔离**
   - 使用字典管理多个会话
   - 每个用户独立的消息列表
   - 支持并发访问

5. **性能特性**
   - List append：O(1) 摊销
   - 索引访问：O(1)
   - 修剪操作：O(n)

---

## 参考来源（2025-2026）

### OpenAI 官方文档
- **OpenAI Chat Completions API** (2026)
  - URL: https://platform.openai.com/docs/api-reference/chat
  - 描述：OpenAI Chat API 官方文档

- **OpenAI Python SDK** (2026)
  - URL: https://github.com/openai/openai-python
  - 描述：OpenAI Python SDK 官方仓库

### 最佳实践
- **Managing Conversation History** (2026)
  - URL: https://platform.openai.com/docs/guides/conversation-history
  - 描述：OpenAI 官方对话历史管理指南
