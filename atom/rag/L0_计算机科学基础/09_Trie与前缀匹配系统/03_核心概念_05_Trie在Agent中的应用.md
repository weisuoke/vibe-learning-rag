# Trie 在 AI Agent 中的应用

## 核心概念

**Trie 在 AI Agent 中的核心作用 = 路由匹配 + 实体识别 + 知识图谱索引**

AI Agent 系统需要快速匹配用户意图、识别实体、检索知识，Trie 提供了高效的解决方案。

---

## 1. Agent 路由匹配

### 1.1 什么是 Agent 路由？

**定义：** 根据用户输入，将请求路由到合适的 Agent 处理。

**场景：** 多 Agent 系统
```python
# 系统中有多个专业 Agent
agents = {
    "weather_agent": "处理天气查询",
    "time_agent": "处理时间查询",
    "search_agent": "处理搜索请求",
    "calculator_agent": "处理计算任务",
}

# 用户输入："明天天气怎么样？"
# 需要路由到：weather_agent
```

**传统方案问题：**
- 关键词匹配：不准确，容易误判
- 规则引擎：维护成本高
- LLM 分类：延迟高，成本高

**Trie 方案：**
- 构建意图关键词 Trie
- O(m) 时间复杂度匹配
- 支持前缀匹配和模糊匹配

---

### 1.2 基于 Trie 的路由系统

**核心思想：** 为每个 Agent 定义关键词模式，使用 Trie 快速匹配。

```python
"""
AI Agent 路由系统
演示：基于 Trie 的意图识别和路由
"""

class AgentRouter:
    """AI Agent 路由器"""

    def __init__(self):
        self.trie = Trie()
        self.agents = {}

    def register_agent(self, agent_name: str, keywords: list):
        """注册 Agent 及其关键词"""
        self.agents[agent_name] = keywords
        for keyword in keywords:
            # 存储关键词 -> Agent 映射
            self.trie.insert(keyword.lower(), value=agent_name)

    def route(self, user_input: str) -> str:
        """路由用户请求"""
        # 1. 分词（简化：按空格分词）
        words = user_input.lower().split()

        # 2. 匹配关键词
        matched_agents = {}
        for word in words:
            # 精确匹配
            if self.trie.search(word):
                node = self.trie._get_node(word)
                agent = node.value
                matched_agents[agent] = matched_agents.get(agent, 0) + 1

            # 前缀匹配
            matches = self.trie.get_words_with_prefix(word)
            for match in matches:
                node = self.trie._get_node(match)
                agent = node.value
                matched_agents[agent] = matched_agents.get(agent, 0) + 0.5

        # 3. 返回得分最高的 Agent
        if matched_agents:
            return max(matched_agents.items(), key=lambda x: x[1])[0]

        return "default_agent"


# ===== 使用示例 =====
router = AgentRouter()

# 注册 Agent
router.register_agent("weather_agent", ["天气", "气温", "温度", "下雨", "晴天"])
router.register_agent("time_agent", ["时间", "几点", "现在", "日期"])
router.register_agent("search_agent", ["搜索", "查询", "找", "查找"])
router.register_agent("calculator_agent", ["计算", "加", "减", "乘", "除"])

# 测试路由
print(router.route("明天天气怎么样？"))  # weather_agent
print(router.route("现在几点了？"))      # time_agent
print(router.route("帮我搜索一下"))      # search_agent
print(router.route("1+1等于多少"))      # calculator_agent
```

**来源：** "Real-World Applications of Trie Data Structure" (Medium 2025)
https://medium.com/@kvaibhaw300/real-world-applications-of-trie-data-structure-187a68417cbb

---

### 1.3 多级路由

**场景：** 层级化的 Agent 系统

```python
# 一级路由：领域分类
# - 信息查询 → 二级路由
# - 任务执行 → 二级路由
# - 对话聊天 → 直接处理

# 二级路由：具体 Agent
# 信息查询 → weather_agent / time_agent / search_agent
# 任务执行 → calculator_agent / file_agent / email_agent
```

**实现：**

```python
class HierarchicalRouter:
    """层级化路由器"""

    def __init__(self):
        # 一级路由 Trie
        self.level1_trie = Trie()
        # 二级路由 Trie（每个领域一个）
        self.level2_tries = {}

    def register_domain(self, domain: str, keywords: list):
        """注册领域"""
        for keyword in keywords:
            self.level1_trie.insert(keyword, value=domain)
        self.level2_tries[domain] = Trie()

    def register_agent(self, domain: str, agent: str, keywords: list):
        """注册 Agent 到特定领域"""
        trie = self.level2_tries[domain]
        for keyword in keywords:
            trie.insert(keyword, value=agent)

    def route(self, user_input: str) -> str:
        """两级路由"""
        words = user_input.lower().split()

        # 一级路由：匹配领域
        domain_scores = {}
        for word in words:
            matches = self.level1_trie.get_words_with_prefix(word)
            for match in matches:
                node = self.level1_trie._get_node(match)
                domain = node.value
                domain_scores[domain] = domain_scores.get(domain, 0) + 1

        if not domain_scores:
            return "default_agent"

        # 选择得分最高的领域
        domain = max(domain_scores.items(), key=lambda x: x[1])[0]

        # 二级路由：匹配 Agent
        trie = self.level2_tries[domain]
        agent_scores = {}
        for word in words:
            matches = trie.get_words_with_prefix(word)
            for match in matches:
                node = trie._get_node(match)
                agent = node.value
                agent_scores[agent] = agent_scores.get(agent, 0) + 1

        if agent_scores:
            return max(agent_scores.items(), key=lambda x: x[1])[0]

        return f"{domain}_default_agent"


# ===== 使用示例 =====
router = HierarchicalRouter()

# 注册领域
router.register_domain("information", ["查询", "搜索", "找", "什么"])
router.register_domain("task", ["计算", "发送", "创建", "执行"])

# 注册 Agent
router.register_agent("information", "weather_agent", ["天气", "气温"])
router.register_agent("information", "time_agent", ["时间", "几点"])
router.register_agent("task", "calculator_agent", ["计算", "加", "减"])
router.register_agent("task", "email_agent", ["邮件", "发送"])

# 测试
print(router.route("查询明天的天气"))  # weather_agent
print(router.route("计算1+1"))        # calculator_agent
```

---

## 2. 实体识别（NER）

### 2.1 基于 Trie 的实体识别

**定义：** 从文本中识别出预定义的实体（人名、地名、机构名等）。

**传统 NER 方案：**
- 基于规则：维护成本高
- 基于模型：计算成本高
- 基于词典：遍历效率低

**Trie 方案：**
- 构建实体词典 Trie
- 最长前缀匹配
- O(m) 时间复杂度

---

### 2.2 最长匹配算法

**核心思想：** 从文本的每个位置开始，尝试匹配最长的实体。

```python
"""
基于 Trie 的实体识别
演示：最长匹配算法
"""

class EntityRecognizer:
    """实体识别器"""

    def __init__(self):
        self.trie = Trie()

    def add_entity(self, entity: str, entity_type: str):
        """添加实体到词典"""
        self.trie.insert(entity, value=entity_type)

    def recognize(self, text: str) -> list:
        """识别文本中的实体"""
        entities = []
        i = 0

        while i < len(text):
            # 尝试最长匹配
            max_len = 0
            max_entity = None
            max_type = None

            for j in range(i + 1, len(text) + 1):
                substring = text[i:j]
                if self.trie.search(substring):
                    node = self.trie._get_node(substring)
                    max_len = j - i
                    max_entity = substring
                    max_type = node.value

            if max_entity:
                entities.append({
                    "text": max_entity,
                    "type": max_type,
                    "start": i,
                    "end": i + max_len
                })
                i += max_len
            else:
                i += 1

        return entities


# ===== 使用示例 =====
recognizer = EntityRecognizer()

# 添加实体
recognizer.add_entity("北京大学", "ORG")
recognizer.add_entity("北京", "LOC")
recognizer.add_entity("清华大学", "ORG")
recognizer.add_entity("张三", "PER")
recognizer.add_entity("李四", "PER")

# 识别实体
text = "张三在北京大学读书，李四在清华大学工作。"
entities = recognizer.recognize(text)

for entity in entities:
    print(f"{entity['text']} ({entity['type']}): {entity['start']}-{entity['end']}")

# 输出：
# 张三 (PER): 0-2
# 北京大学 (ORG): 3-7
# 李四 (PER): 11-13
# 清华大学 (ORG): 14-18
```

---

### 2.3 在 RAG 中的应用

**场景：** 文档问答中的实体识别

```python
"""
RAG 中的实体识别
演示：识别查询中的实体，增强检索
"""

class RAGEntityRecognizer:
    """RAG 实体识别器"""

    def __init__(self, entity_trie):
        self.entity_trie = entity_trie

    def enhance_query(self, query: str) -> dict:
        """增强查询（识别实体）"""
        # 识别实体
        recognizer = EntityRecognizer()
        recognizer.trie = self.entity_trie
        entities = recognizer.recognize(query)

        # 提取实体类型
        entity_types = {}
        for entity in entities:
            entity_type = entity['type']
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append(entity['text'])

        return {
            "original_query": query,
            "entities": entities,
            "entity_types": entity_types
        }


# ===== 使用示例 =====
# 构建实体 Trie
entity_trie = Trie()
entity_trie.insert("Python", value="TECH")
entity_trie.insert("FastAPI", value="TECH")
entity_trie.insert("机器学习", value="TECH")
entity_trie.insert("北京", value="LOC")

# 增强查询
rag_recognizer = RAGEntityRecognizer(entity_trie)
enhanced = rag_recognizer.enhance_query("如何在 Python 中使用 FastAPI？")

print(enhanced)
# 输出：
# {
#   "original_query": "如何在 Python 中使用 FastAPI？",
#   "entities": [
#     {"text": "Python", "type": "TECH", "start": 4, "end": 10},
#     {"text": "FastAPI", "type": "TECH", "start": 14, "end": 21}
#   ],
#   "entity_types": {
#     "TECH": ["Python", "FastAPI"]
#   }
# }
```

---

## 3. 知识图谱前缀索引

### 3.1 知识图谱中的 Trie

**场景：** 快速查询知识图谱中的实体和关系

**知识图谱结构：**
```python
# 三元组：(主体, 关系, 客体)
triples = [
    ("北京大学", "位于", "北京"),
    ("北京大学", "类型", "大学"),
    ("清华大学", "位于", "北京"),
    ("张三", "就读于", "北京大学"),
]
```

**Trie 索引：**
- 主体索引：快速查找以某前缀开头的主体
- 关系索引：快速查找特定关系
- 客体索引：快速查找以某前缀开头的客体

---

### 3.2 实现示例

```python
"""
知识图谱 Trie 索引
演示：快速查询实体和关系
"""

class KnowledgeGraphIndex:
    """知识图谱索引"""

    def __init__(self):
        self.subject_trie = Trie()  # 主体索引
        self.relation_trie = Trie()  # 关系索引
        self.object_trie = Trie()   # 客体索引

    def add_triple(self, subject: str, relation: str, obj: str):
        """添加三元组"""
        # 索引主体
        if not self.subject_trie.search(subject):
            self.subject_trie.insert(subject, value=[])
        node = self.subject_trie._get_node(subject)
        node.value.append((relation, obj))

        # 索引关系
        if not self.relation_trie.search(relation):
            self.relation_trie.insert(relation, value=[])
        node = self.relation_trie._get_node(relation)
        node.value.append((subject, obj))

        # 索引客体
        if not self.object_trie.search(obj):
            self.object_trie.insert(obj, value=[])
        node = self.object_trie._get_node(obj)
        node.value.append((subject, relation))

    def query_by_subject(self, subject_prefix: str) -> list:
        """查询主体"""
        matches = self.subject_trie.get_words_with_prefix(subject_prefix)
        results = []
        for subject in matches:
            node = self.subject_trie._get_node(subject)
            for relation, obj in node.value:
                results.append((subject, relation, obj))
        return results

    def query_by_relation(self, relation: str) -> list:
        """查询关系"""
        if self.relation_trie.search(relation):
            node = self.relation_trie._get_node(relation)
            return [(s, relation, o) for s, o in node.value]
        return []


# ===== 使用示例 =====
kg = KnowledgeGraphIndex()

# 添加三元组
kg.add_triple("北京大学", "位于", "北京")
kg.add_triple("北京大学", "类型", "大学")
kg.add_triple("北京师范大学", "位于", "北京")
kg.add_triple("清华大学", "位于", "北京")

# 查询
print("=== 查询主体（前缀：北京）===")
results = kg.query_by_subject("北京")
for triple in results:
    print(triple)
# 输出：
# ('北京大学', '位于', '北京')
# ('北京大学', '类型', '大学')
# ('北京师范大学', '位于', '北京')

print("\n=== 查询关系（位于）===")
results = kg.query_by_relation("位于")
for triple in results:
    print(triple)
# 输出：
# ('北京大学', '位于', '北京')
# ('北京师范大学', '位于', '北京')
# ('清华大学', '位于', '北京')
```

---

## 4. 对话上下文管理

### 4.1 对话历史索引

**场景：** 快速检索对话历史中的关键信息

```python
"""
对话上下文管理
演示：使用 Trie 索引对话历史
"""

class ConversationIndex:
    """对话索引"""

    def __init__(self):
        self.trie = Trie()
        self.messages = []

    def add_message(self, message: str, metadata: dict):
        """添加消息"""
        msg_id = len(self.messages)
        self.messages.append({"text": message, "metadata": metadata})

        # 索引关键词
        words = message.lower().split()
        for word in words:
            if not self.trie.search(word):
                self.trie.insert(word, value=[])
            node = self.trie._get_node(word)
            node.value.append(msg_id)

    def search_history(self, keyword: str) -> list:
        """搜索对话历史"""
        matches = self.trie.get_words_with_prefix(keyword.lower())
        msg_ids = set()

        for match in matches:
            node = self.trie._get_node(match)
            msg_ids.update(node.value)

        return [self.messages[msg_id] for msg_id in sorted(msg_ids)]


# ===== 使用示例 =====
conv = ConversationIndex()

# 添加对话
conv.add_message("今天天气怎么样？", {"role": "user", "timestamp": "10:00"})
conv.add_message("今天天气晴朗，温度 25 度。", {"role": "assistant", "timestamp": "10:01"})
conv.add_message("明天天气呢？", {"role": "user", "timestamp": "10:02"})

# 搜索历史
results = conv.search_history("天气")
for msg in results:
    print(f"[{msg['metadata']['role']}] {msg['text']}")
# 输出：
# [user] 今天天气怎么样？
# [assistant] 今天天气晴朗，温度 25 度。
# [user] 明天天气呢？
```

---

## 5. 命令补全

### 5.1 Agent 命令系统

**场景：** Agent 支持命令行式交互

```python
"""
Agent 命令补全
演示：使用 Trie 实现命令自动补全
"""

class AgentCommandSystem:
    """Agent 命令系统"""

    def __init__(self):
        self.trie = Trie()
        self.commands = {}

    def register_command(self, command: str, handler, description: str):
        """注册命令"""
        self.commands[command] = {"handler": handler, "description": description}
        self.trie.insert(command, value=command)

    def autocomplete(self, prefix: str) -> list:
        """自动补全"""
        matches = self.trie.get_words_with_prefix(prefix)
        return [(cmd, self.commands[cmd]["description"]) for cmd in matches]

    def execute(self, command: str, *args):
        """执行命令"""
        if command in self.commands:
            return self.commands[command]["handler"](*args)
        return "Unknown command"


# ===== 使用示例 =====
agent = AgentCommandSystem()

# 注册命令
agent.register_command("search", lambda q: f"Searching for: {q}", "搜索")
agent.register_command("calculate", lambda expr: f"Calculating: {expr}", "计算")
agent.register_command("weather", lambda loc: f"Weather in: {loc}", "天气查询")

# 自动补全
print("=== 自动补全（输入：se）===")
suggestions = agent.autocomplete("se")
for cmd, desc in suggestions:
    print(f"{cmd}: {desc}")
# 输出：
# search: 搜索

print("\n=== 自动补全（输入：ca）===")
suggestions = agent.autocomplete("ca")
for cmd, desc in suggestions:
    print(f"{cmd}: {desc}")
# 输出：
# calculate: 计算

# 执行命令
print("\n=== 执行命令 ===")
print(agent.execute("search", "Python"))
# 输出：Searching for: Python
```

---

## 6. 性能优化

### 6.1 缓存热点路由

**问题：** 高频查询重复计算

**优化：** 缓存热点路由结果

```python
from functools import lru_cache

class CachedRouter(AgentRouter):
    """带缓存的路由器"""

    @lru_cache(maxsize=1000)
    def route(self, user_input: str) -> str:
        """缓存路由结果"""
        return super().route(user_input)
```

---

### 6.2 并行匹配

**问题：** 多个 Trie 串行查询慢

**优化：** 并行查询多个 Trie

```python
import concurrent.futures

class ParallelRouter:
    """并行路由器"""

    def __init__(self):
        self.tries = {}  # domain -> trie

    def route_parallel(self, user_input: str) -> str:
        """并行路由"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._match_trie, domain, trie, user_input): domain
                for domain, trie in self.tries.items()
            }

            results = {}
            for future in concurrent.futures.as_completed(futures):
                domain = futures[future]
                score = future.result()
                results[domain] = score

        return max(results.items(), key=lambda x: x[1])[0]

    def _match_trie(self, domain, trie, user_input):
        """匹配单个 Trie"""
        # 匹配逻辑
        pass
```

---

## 7. 完整示例

```python
"""
AI Agent 路由系统完整实现
演示：多 Agent 系统的路由、实体识别、命令补全
"""

class CompleteAgentSystem:
    """完整的 Agent 系统"""

    def __init__(self):
        self.router = AgentRouter()
        self.entity_recognizer = EntityRecognizer()
        self.command_system = AgentCommandSystem()

    def setup(self):
        """初始化系统"""
        # 注册 Agent
        self.router.register_agent("weather_agent", ["天气", "气温", "温度"])
        self.router.register_agent("time_agent", ["时间", "几点", "现在"])
        self.router.register_agent("search_agent", ["搜索", "查询", "找"])

        # 注册实体
        self.entity_recognizer.add_entity("北京", "LOC")
        self.entity_recognizer.add_entity("上海", "LOC")
        self.entity_recognizer.add_entity("Python", "TECH")

        # 注册命令
        self.command_system.register_command("help", self.show_help, "显示帮助")
        self.command_system.register_command("list", self.list_agents, "列出 Agent")

    def process(self, user_input: str):
        """处理用户输入"""
        # 1. 识别实体
        entities = self.entity_recognizer.recognize(user_input)
        print(f"识别实体: {entities}")

        # 2. 路由到 Agent
        agent = self.router.route(user_input)
        print(f"路由到: {agent}")

        # 3. 执行
        return f"[{agent}] 处理: {user_input}"

    def show_help(self):
        return "可用命令: help, list"

    def list_agents(self):
        return f"可用 Agent: {list(self.router.agents.keys())}"


# ===== 测试 =====
if __name__ == "__main__":
    system = CompleteAgentSystem()
    system.setup()

    # 测试
    print("=== 测试 1 ===")
    result = system.process("北京明天天气怎么样？")
    print(result)

    print("\n=== 测试 2 ===")
    result = system.process("搜索 Python 教程")
    print(result)

    print("\n=== 命令补全 ===")
    suggestions = system.command_system.autocomplete("he")
    print(suggestions)
```

**预期输出：**
```
=== 测试 1 ===
识别实体: [{'text': '北京', 'type': 'LOC', 'start': 0, 'end': 2}]
路由到: weather_agent
[weather_agent] 处理: 北京明天天气怎么样？

=== 测试 2 ===
识别实体: [{'text': 'Python', 'type': 'TECH', 'start': 2, 'end': 8}]
路由到: search_agent
[search_agent] 处理: 搜索 Python 教程

=== 命令补全 ===
[('help', '显示帮助')]
```

---

## 8. 总结

### 8.1 Trie 在 AI Agent 中的核心价值

1. **路由匹配**：O(m) 时间复杂度，快速识别意图
2. **实体识别**：最长匹配，准确识别实体
3. **知识图谱**：前缀索引，快速查询
4. **对话管理**：历史索引，快速检索
5. **命令补全**：自动补全，提升体验

### 8.2 适用场景

- ✅ 多 Agent 系统的路由
- ✅ 实体识别和信息抽取
- ✅ 知识图谱查询
- ✅ 对话历史检索
- ✅ 命令行式 Agent

### 8.3 注意事项

- 定期更新 Trie（新增关键词、实体）
- 处理同义词和变体
- 结合语义理解（LLM）提升准确性
- 监控路由准确率

---

**版本**: v1.0
**最后更新**: 2026-02-14
**参考文献**:
- Medium 2025: "Real-World Applications of Trie Data Structure"

**下一步**: 学习最小可用知识（`04_最小可用.md`）
