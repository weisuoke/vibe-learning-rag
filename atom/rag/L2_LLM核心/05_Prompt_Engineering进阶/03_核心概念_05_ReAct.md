# 核心概念 5: ReAct (Reasoning + Acting)

## 一句话定义

**通过交替进行推理(Reasoning)和行动(Acting),让大模型能够动态调用工具、获取信息并完成需要多步交互的复杂任务。**

**RAG应用:** 在RAG系统中,ReAct让模型根据推理结果动态决定是否需要更多检索,实现自适应的迭代检索优化。

---

## 为什么重要?

### 问题场景

```python
# 场景:需要多次信息查询的任务
from openai import OpenAI

client = OpenAI()

# ❌ 静态方法:一次性检索
query = "Python和JavaScript哪个更适合Web开发?"
docs = retriever.search(query, top_k=5)
answer = llm.generate(f"文档:{docs}\n问题:{query}")

# 问题:
# 1. 如果检索结果不足,无法补充
# 2. 无法根据中间结果调整策略
# 3. 缺乏灵活性
```

### 解决方案

```python
# ✅ ReAct:推理与行动交替
def react_rag(query, max_steps=5):
    context = ""
    
    for step in range(max_steps):
        # Thought:推理当前状态
        thought = llm.generate(f"""
        任务:{query}
        当前信息:{context}
        
        思考:我现在知道什么?还需要什么信息?
        """)
        
        if "足够" in thought or "完成" in thought:
            break
        
        # Action:执行检索
        search_query = extract_query(thought)
        new_docs = retriever.search(search_query)
        
        # Observation:观察结果
        context += f"\n新信息:{new_docs}"
    
    # 最终答案
    answer = llm.generate(f"完整信息:{context}\n问题:{query}")
    return answer
```

**性能提升:**

| 任务类型 | 静态方法 | ReAct | 提升 |
|---------|---------|-------|------|
| 多跳问答 | 45% | 62% | +38% |
| 事实验证 | 56% | 71% | +27% |
| 复杂推理 | 38% | 58% | +53% |

**来源:** [ReAct: Synergizing Reasoning and Acting (2022)](https://arxiv.org/abs/2210.03629)

---

## 核心原理

### 原理1:推理-行动循环

**定义:** 交替进行思考(Thought)、行动(Action)、观察(Observation)。

**循环结构:**

```
Thought → Action → Observation → Thought → Action → ...
  ↓         ↓          ↓
 推理     工具调用    结果反馈
```

**与传统方法的区别:**

```
传统方法:
输入 → 处理 → 输出
     (黑盒,无中间调整)

ReAct:
输入 → 思考1 → 行动1 → 观察1 → 思考2 → 行动2 → 观察2 → 输出
      ↓        ↓        ↓
    可调整   可验证   可迭代
```

**来源:** [ReAct Paper (2022)](https://arxiv.org/abs/2210.03629)

---

### 原理2:工具使用能力

**定义:** 模型能够识别何时需要外部工具,并正确调用。

**工具类型:**

```python
# 1. 搜索工具
def search(query: str) -> List[str]:
    """搜索相关文档"""
    return retriever.search(query)

# 2. 计算工具
def calculate(expression: str) -> float:
    """执行数学计算"""
    return eval(expression)

# 3. API调用
def call_api(endpoint: str, params: dict) -> dict:
    """调用外部API"""
    return requests.get(endpoint, params=params).json()

# 4. 数据库查询
def query_db(sql: str) -> List[dict]:
    """查询数据库"""
    return db.execute(sql).fetchall()
```

**工具选择机制:**

```python
def select_tool(thought: str, available_tools: List[str]) -> str:
    """根据推理选择工具"""
    prompt = f"""
    当前思考:{thought}
    可用工具:{', '.join(available_tools)}
    
    应该使用哪个工具?
    """
    response = llm.generate(prompt)
    return extract_tool_name(response)
```

---

### 原理3:自我纠错能力

**定义:** 通过观察行动结果,模型可以发现错误并调整策略。

**纠错示例:**

```python
# 步骤1:初始尝试
Thought: "需要查询Python的创建时间"
Action: search("Python")
Observation: "返回了Python编程语言、Python蛇等多个结果"

# 步骤2:发现问题并调整
Thought: "查询太宽泛,需要更精确"
Action: search("Python编程语言 创建时间")
Observation: "Python由Guido van Rossum于1991年创建"

# 步骤3:验证答案
Thought: "信息已足够,可以回答"
Answer: "1991年"
```

---

## 手写实现

### 从零实现 ReAct

```python
"""
ReAct Implementation
功能:推理与行动交替的任务执行
"""

from typing import List, Dict, Callable, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
import re

@dataclass
class Step:
    """执行步骤"""
    step_number: int
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None

class ReActAgent:
    """ReAct智能体"""
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.tools = {}
        self.history = []
    
    def register_tool(self, name: str, func: Callable, description: str):
        """注册工具"""
        self.tools[name] = {
            "function": func,
            "description": description
        }
    
    def think(
        self,
        task: str,
        context: str,
        model: str = "gpt-4o-mini"
    ) -> str:
        """推理步骤"""
        prompt = f"""
任务:{task}

当前上下文:
{context}

可用工具:
{self._format_tools()}

请思考:
1. 我现在知道什么?
2. 还需要什么信息?
3. 应该采取什么行动?

如果信息已足够,回答"任务完成"。
否则,说明需要什么信息和使用什么工具。

思考:
"""
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    def _format_tools(self) -> str:
        """格式化工具列表"""
        lines = []
        for name, info in self.tools.items():
            lines.append(f"- {name}: {info['description']}")
        return "\n".join(lines)
    
    def parse_action(self, thought: str) -> Optional[Tuple[str, str]]:
        """从思考中解析行动"""
        # 查找工具调用模式
        for tool_name in self.tools.keys():
            if tool_name in thought.lower():
                # 提取参数
                param = self._extract_param(thought, tool_name)
                return tool_name, param
        return None
    
    def _extract_param(self, text: str, tool_name: str) -> str:
        """提取工具参数"""
        # 简单实现:查找引号内的内容
        patterns = [
            r'"([^"]+)"',
            r"'([^']+)'",
            r'查询[：:]\s*(.+)',
            r'搜索[：:]\s*(.+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return text
    
    def execute_action(
        self,
        tool_name: str,
        param: str
    ) -> str:
        """执行行动"""
        if tool_name not in self.tools:
            return f"错误:工具'{tool_name}'不存在"
        
        try:
            tool_func = self.tools[tool_name]["function"]
            result = tool_func(param)
            return str(result)
        except Exception as e:
            return f"错误:{str(e)}"
    
    def run(
        self,
        task: str,
        max_steps: int = 10,
        model: str = "gpt-4o-mini"
    ) -> Dict:
        """运行ReAct循环"""
        context = ""
        self.history = []
        
        for step_num in range(1, max_steps + 1):
            # Thought:推理
            thought = self.think(task, context, model)
            
            step = Step(
                step_number=step_num,
                thought=thought
            )
            
            # 检查是否完成
            if "任务完成" in thought or "足够" in thought:
                self.history.append(step)
                break
            
            # Action:解析并执行行动
            action_info = self.parse_action(thought)
            
            if action_info:
                tool_name, param = action_info
                step.action = f"{tool_name}({param})"
                
                # Observation:观察结果
                observation = self.execute_action(tool_name, param)
                step.observation = observation
                
                # 更新上下文
                context += f"\n\n步骤{step_num}:\n"
                context += f"行动:{step.action}\n"
                context += f"结果:{observation}"
            
            self.history.append(step)
        
        # 生成最终答案
        final_answer = self._generate_final_answer(task, context, model)
        
        return {
            "task": task,
            "steps": len(self.history),
            "history": [
                {
                    "step": s.step_number,
                    "thought": s.thought,
                    "action": s.action,
                    "observation": s.observation
                }
                for s in self.history
            ],
            "final_answer": final_answer
        }
    
    def _generate_final_answer(
        self,
        task: str,
        context: str,
        model: str
    ) -> str:
        """生成最终答案"""
        prompt = f"""
任务:{task}

执行过程:
{context}

基于以上信息,给出最终答案:
"""
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content


# 使用示例
if __name__ == "__main__":
    from dotenv import load_dotenv
    import chromadb
    
    load_dotenv()
    
    client = OpenAI()
    agent = ReActAgent(client)
    
    # 设置向量数据库
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("docs")
    
    docs = [
        "Python由Guido van Rossum于1991年创建",
        "JavaScript由Brendan Eich于1995年创建",
        "Java由James Gosling于1995年创建"
    ]
    collection.add(documents=docs, ids=[f"doc{i}" for i in range(len(docs))])
    
    # 注册工具
    def search_docs(query: str) -> str:
        """搜索文档"""
        results = collection.query(query_texts=[query], n_results=2)
        return " | ".join(results['documents'][0])
    
    def calculate(expression: str) -> str:
        """计算"""
        try:
            result = eval(expression)
            return str(result)
        except:
            return "计算错误"
    
    agent.register_tool(
        "search",
        search_docs,
        "搜索相关文档"
    )
    
    agent.register_tool(
        "calculate",
        calculate,
        "执行数学计算"
    )
    
    # 测试
    result = agent.run(
        "Python和JavaScript哪个创建得更早?相差几年?",
        max_steps=5
    )
    
    print(f"任务:{result['task']}")
    print(f"步骤数:{result['steps']}")
    print("\n执行历史:")
    for step in result['history']:
        print(f"\n步骤{step['step']}:")
        print(f"思考:{step['thought'][:100]}...")
        if step['action']:
            print(f"行动:{step['action']}")
            print(f"观察:{step['observation'][:100]}...")
    print(f"\n最终答案:{result['final_answer']}")
```

---

## RAG 应用场景

### 场景1:自适应检索

**问题:** 初始检索结果可能不足

**解决方案:** ReAct动态补充检索

```python
def adaptive_rag(query: str) -> Dict:
    """自适应RAG"""
    agent = ReActAgent(client)
    
    # 注册检索工具
    def search_semantic(q: str) -> str:
        return " | ".join(collection.query(query_texts=[q], n_results=2)['documents'][0])
    
    def search_keyword(q: str) -> str:
        # 关键词检索实现
        keywords = extract_keywords(q)
        return " | ".join(collection.query(query_texts=[keywords], n_results=2)['documents'][0])
    
    agent.register_tool("semantic_search", search_semantic, "语义检索")
    agent.register_tool("keyword_search", search_keyword, "关键词检索")
    
    result = agent.run(query, max_steps=5)
    return result
```

---

## 最佳实践

### 1. 工具设计

```python
# ✅ 好:工具功能单一明确
def search(query: str) -> List[str]:
    """搜索文档"""
    pass

def calculate(expr: str) -> float:
    """计算"""
    pass

# ❌ 坏:工具功能模糊
def do_something(input: str) -> str:
    """做某事"""
    pass
```

### 2. 步数限制

```python
# 根据任务复杂度设置
simple_task = 3      # 简单查询
medium_task = 5      # 中等复杂
complex_task = 10    # 复杂任务
```

### 3. 错误处理

```python
def execute_action_safe(tool_name, param):
    try:
        return tool_func(param)
    except Exception as e:
        return f"错误:{e},请尝试其他方法"
```

---

## 参考资源

- [ReAct: Synergizing Reasoning and Acting (2022)](https://arxiv.org/abs/2210.03629)
- [Prompt Engineering Guide - ReAct](https://www.promptingguide.ai/techniques/react)
- [Oracle RAG Enhancement](https://blogs.oracle.com/ai-and-datascience/enhancing-rag-with-advanced-prompting)
