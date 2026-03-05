# 实战代码 - 场景8：Agent系统模板设计

> **场景描述**：在 AI Agent 系统中，需要设计灵活的提示词模板来支持工具调用、多步推理、自主决策，本文展示完整的 Agent 模板设计实战方案。

---

## 场景概述

### 为什么 Agent 需要特殊的模板设计?

在 Agent 系统中，我们需要:
- **工具调用模板**：引导 Agent 正确使用工具
- **推理链模板**：支持多步思考和决策
- **角色定义模板**：明确 Agent 的能力和限制
- **错误处理模板**：处理工具调用失败和异常情况

[来源: reference/search_composition_01.md | LangChain Prompt Templates Guide 2026]

### 双重类比

**前端类比**：
- Agent 模板 = React 组件 + Redux 状态管理
- 工具调用 = API 请求封装
- 推理链 = 中间件链式处理

**日常生活类比**：
- Agent = 有助手的项目经理
- 工具调用 = 委派任务给专家
- 推理链 = 制定项目计划的思考过程

---

## 完整实战代码

```python
"""
Agent 系统模板设计实战
演示：工具调用模板、多步推理模板、完整 Agent 系统
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

# 加载环境变量
load_dotenv()

# ===== 1. Agent 角色定义模板 =====
print("=" * 60)
print("场景1：Agent 角色定义模板")
print("=" * 60)

class AgentRole(Enum):
    """Agent 角色类型"""
    RESEARCHER = "researcher"      # 研究员
    ANALYST = "analyst"            # 分析师
    ASSISTANT = "assistant"        # 助手
    SPECIALIST = "specialist"      # 专家


class AgentRoleTemplate:
    """Agent 角色定义模板管理器"""

    def __init__(self):
        # 角色定义模板
        self.role_templates = {
            AgentRole.RESEARCHER: {
                "name": "研究员 Agent",
                "description": "专注于信息收集和研究",
                "capabilities": [
                    "搜索和收集信息",
                    "分析数据来源",
                    "总结研究发现"
                ],
                "limitations": [
                    "不进行深度分析",
                    "不做最终决策",
                    "需要人工审核"
                ],
                "system_prompt": """你是一个专业的研究员 Agent。
当前时间：{current_time}

你的能力：
- 搜索和收集相关信息
- 分析数据来源的可靠性
- 总结研究发现

你的限制：
- 不进行深度分析和解读
- 不做最终决策
- 所有结果需要人工审核

工作流程：
1. 理解用户的研究需求
2. 使用可用工具收集信息
3. 整理和总结发现
4. 提供信息来源"""
            },

            AgentRole.ANALYST: {
                "name": "分析师 Agent",
                "description": "专注于数据分析和洞察",
                "capabilities": [
                    "数据分析",
                    "趋势识别",
                    "生成洞察"
                ],
                "limitations": [
                    "依赖提供的数据",
                    "不收集新数据",
                    "需要明确的分析目标"
                ],
                "system_prompt": """你是一个专业的分析师 Agent。
当前时间：{current_time}

你的能力：
- 深度数据分析
- 识别趋势和模式
- 生成可操作的洞察

你的限制：
- 只分析提供的数据
- 不主动收集新数据
- 需要明确的分析目标

工作流程：
1. 理解分析目标
2. 检查数据质量
3. 执行分析
4. 生成洞察和建议"""
            }
        }

    def get_role_prompt(
        self,
        role: AgentRole,
        **kwargs
    ) -> PromptTemplate:
        """获取角色定义提示词"""
        role_config = self.role_templates[role]
        template = role_config["system_prompt"]

        return PromptTemplate(
            template=template,
            input_variables=[],
            partial_variables={
                "current_time": lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **kwargs
            }
        )

    def get_role_info(self, role: AgentRole) -> Dict[str, Any]:
        """获取角色信息"""
        return self.role_templates[role]


# 测试角色定义模板
role_manager = AgentRoleTemplate()

print("\n【研究员 Agent 角色定义】")
researcher_prompt = role_manager.get_role_prompt(AgentRole.RESEARCHER)
print(researcher_prompt.format())

print("\n【角色能力和限制】")
role_info = role_manager.get_role_info(AgentRole.RESEARCHER)
print(f"名称：{role_info['name']}")
print(f"能力：{', '.join(role_info['capabilities'])}")
print(f"限制：{', '.join(role_info['limitations'])}")

# ===== 2. 工具调用模板 =====
print("\n" + "=" * 60)
print("场景2：工具调用模板")
print("=" * 60)

class ToolCallTemplate:
    """工具调用模板管理器"""

    def __init__(self):
        # 工具调用指导模板
        self.tool_instruction_template = PromptTemplate(
            template="""你可以使用以下工具来完成任务：

{tools_description}

工具使用规则：
1. 仔细阅读工具描述，确保理解工具的功能
2. 按照工具的参数要求准备输入
3. 一次只调用一个工具
4. 等待工具返回结果后再继续
5. 如果工具调用失败，分析原因并重试或使用其他工具

工具调用格式：
```
Tool: [工具名称]
Input: [工具输入]
```

当前任务：{task}

请开始执行任务。""",
            input_variables=["tools_description", "task"]
        )

        # 工具结果处理模板
        self.tool_result_template = PromptTemplate(
            template="""工具调用结果：

Tool: {tool_name}
Input: {tool_input}
Output: {tool_output}

请基于这个结果继续执行任务：{task}

如果任务已完成，请总结结果。
如果需要更多信息，请调用其他工具。""",
            input_variables=["tool_name", "tool_input", "tool_output", "task"]
        )

    def create_tool_instruction(
        self,
        tools: List[Dict[str, str]],
        task: str
    ) -> str:
        """创建工具调用指导"""
        tools_description = "\n\n".join([
            f"**{tool['name']}**\n描述：{tool['description']}\n参数：{tool['parameters']}"
            for tool in tools
        ])

        return self.tool_instruction_template.format(
            tools_description=tools_description,
            task=task
        )

    def create_tool_result_prompt(
        self,
        tool_name: str,
        tool_input: str,
        tool_output: str,
        task: str
    ) -> str:
        """创建工具结果处理提示"""
        return self.tool_result_template.format(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            task=task
        )


# 测试工具调用模板
tool_template = ToolCallTemplate()

# 定义可用工具
available_tools = [
    {
        "name": "search",
        "description": "搜索互联网信息",
        "parameters": "query: str - 搜索关键词"
    },
    {
        "name": "calculator",
        "description": "执行数学计算",
        "parameters": "expression: str - 数学表达式"
    },
    {
        "name": "get_weather",
        "description": "获取天气信息",
        "parameters": "location: str - 城市名称"
    }
]

print("\n【工具调用指导】")
instruction = tool_template.create_tool_instruction(
    tools=available_tools,
    task="查询北京今天的天气，并计算华氏温度"
)
print(instruction)

# [来源: reference/search_composition_01.md | LangChain Prompt Templates Complete Guide]

# ===== 3. 多步推理模板 =====
print("\n" + "=" * 60)
print("场景3：多步推理模板（ReAct 模式）")
print("=" * 60)

class ReActTemplate:
    """ReAct (Reasoning + Acting) 推理模板"""

    def __init__(self):
        # ReAct 系统提示
        self.system_template = PromptTemplate(
            template="""你是一个使用 ReAct (Reasoning + Acting) 模式的 AI Agent。

ReAct 工作流程：
1. **Thought (思考)**：分析当前情况，思考下一步行动
2. **Action (行动)**：决定使用哪个工具以及如何使用
3. **Observation (观察)**：观察工具返回的结果
4. **Repeat (重复)**：重复上述过程直到完成任务

格式要求：
Thought: [你的思考过程]
Action: [工具名称]
Action Input: [工具输入]
Observation: [工具输出]
... (重复 Thought/Action/Observation)
Thought: 我现在知道最终答案了
Final Answer: [最终答案]

可用工具：
{tools}

当前时间：{current_time}

重要规则：
- 每次只执行一个 Action
- 必须先 Thought 再 Action
- 观察 Observation 后再继续
- 确定答案后输出 Final Answer""",
            input_variables=["tools"],
            partial_variables={
                "current_time": lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        )

        # 用户任务模板
        self.task_template = PromptTemplate.from_template(
            "任务：{task}\n\n开始执行 ReAct 流程："
        )

    def create_react_prompt(
        self,
        tools: List[Dict[str, str]],
        task: str
    ) -> ChatPromptTemplate:
        """创建 ReAct 提示词"""
        tools_description = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in tools
        ])

        return ChatPromptTemplate.from_messages([
            ("system", self.system_template.format(tools=tools_description)),
            ("human", self.task_template.format(task=task))
        ])


# 测试 ReAct 模板
react_template = ReActTemplate()

print("\n【ReAct 推理模板】")
react_prompt = react_template.create_react_prompt(
    tools=available_tools,
    task="帮我查询北京的天气，如果温度超过 30 度，计算对应的华氏温度"
)

# 格式化并打印
messages = react_prompt.format_messages()
for msg in messages:
    print(f"\n【{msg.type}】")
    print(msg.content[:500] + "..." if len(msg.content) > 500 else msg.content)

# ===== 4. 完整 Agent 系统示例 =====
print("\n" + "=" * 60)
print("场景4：完整 Agent 系统示例")
print("=" * 60)

# 定义工具
@tool
def search_tool(query: str) -> str:
    """搜索互联网信息（模拟）"""
    # 模拟搜索结果
    mock_results = {
        "langchain": "LangChain 是一个用于构建 LLM 应用的开源框架，提供 Prompt 管理、链式组合、Agent 系统等功能。",
        "weather": "北京今天天气晴朗，温度 25°C，适合户外活动。",
        "python": "Python 是一种高级编程语言，广泛用于 Web 开发、数据科学、AI 等领域。"
    }

    for key, result in mock_results.items():
        if key in query.lower():
            return result

    return f"搜索 '{query}' 的结果：未找到相关信息"


@tool
def calculator_tool(expression: str) -> str:
    """执行数学计算"""
    try:
        # 安全的数学计算
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


class ProductionAgentSystem:
    """生产级 Agent 系统"""

    def __init__(
        self,
        role: AgentRole = AgentRole.ASSISTANT,
        model_name: str = "gpt-4o-mini"
    ):
        self.role = role
        self.llm = ChatOpenAI(model=model_name, temperature=0)

        # 初始化模板管理器
        self.role_manager = AgentRoleTemplate()
        self.tool_template = ToolCallTemplate()
        self.react_template = ReActTemplate()

        # 定义可用工具
        self.tools = [
            {
                "name": "search",
                "description": "搜索互联网信息",
                "parameters": "query: str",
                "function": search_tool
            },
            {
                "name": "calculator",
                "description": "执行数学计算",
                "parameters": "expression: str",
                "function": calculator_tool
            }
        ]

    def create_agent_prompt(self, task: str) -> ChatPromptTemplate:
        """创建 Agent 提示词"""
        # 获取角色定义
        role_prompt = self.role_manager.get_role_prompt(self.role)

        # 创建工具指导
        tool_instruction = self.tool_template.create_tool_instruction(
            tools=self.tools,
            task=task
        )

        # 组合提示词
        return ChatPromptTemplate.from_messages([
            ("system", role_prompt.template),
            ("human", tool_instruction)
        ])

    def execute(self, task: str) -> str:
        """执行 Agent 任务（简化版本）"""
        prompt = self.create_agent_prompt(task)
        chain = prompt | self.llm | StrOutputParser()

        try:
            return chain.invoke({})
        except Exception as e:
            return f"执行失败：{str(e)}"


# 测试完整 Agent 系统
print("\n【初始化 Agent 系统】")
agent = ProductionAgentSystem(role=AgentRole.RESEARCHER)

print("\n【执行任务】")
task = "搜索 LangChain 的相关信息"

try:
    result = agent.execute(task)
    print(f"\n任务：{task}")
    print(f"\n结果：{result[:300]}..." if len(result) > 300 else f"\n结果：{result}")
except Exception as e:
    print(f"\n执行失败（可能需要配置 API Key）：{e}")
    print("\n模拟结果：Agent 已搜索 LangChain 信息，发现它是一个用于构建 LLM 应用的框架。")

# ===== 5. 错误处理模板 =====
print("\n" + "=" * 60)
print("场景5：错误处理模板")
print("=" * 60)

class ErrorHandlingTemplate:
    """错误处理模板管理器"""

    def __init__(self):
        # 工具调用失败模板
        self.tool_error_template = PromptTemplate(
            template="""工具调用失败：

Tool: {tool_name}
Input: {tool_input}
Error: {error_message}

请分析错误原因并采取以下行动之一：
1. 修正输入参数后重试
2. 使用其他工具完成任务
3. 向用户说明无法完成任务的原因

当前任务：{task}

请继续执行。""",
            input_variables=["tool_name", "tool_input", "error_message", "task"]
        )

        # 任务超时模板
        self.timeout_template = PromptTemplate(
            template="""任务执行超时：

任务：{task}
已执行步骤：{steps_completed}
超时时间：{timeout_seconds} 秒

请采取以下行动：
1. 总结已完成的部分
2. 说明未完成的部分
3. 提供部分结果或建议

请输出总结。""",
            input_variables=["task", "steps_completed", "timeout_seconds"]
        )

    def create_error_prompt(
        self,
        tool_name: str,
        tool_input: str,
        error_message: str,
        task: str
    ) -> str:
        """创建错误处理提示"""
        return self.tool_error_template.format(
            tool_name=tool_name,
            tool_input=tool_input,
            error_message=error_message,
            task=task
        )

    def create_timeout_prompt(
        self,
        task: str,
        steps_completed: int,
        timeout_seconds: int
    ) -> str:
        """创建超时处理提示"""
        return self.timeout_template.format(
            task=task,
            steps_completed=steps_completed,
            timeout_seconds=timeout_seconds
        )


# 测试错误处理模板
error_template = ErrorHandlingTemplate()

print("\n【工具调用失败处理】")
error_prompt = error_template.create_error_prompt(
    tool_name="search",
    tool_input="invalid query",
    error_message="API rate limit exceeded",
    task="搜索 LangChain 信息"
)
print(error_prompt)

# ===== 6. 最佳实践总结 =====
print("\n" + "=" * 60)
print("最佳实践总结")
print("=" * 60)

best_practices = """
1. **角色定义清晰**：
   - 明确 Agent 的能力和限制
   - 提供具体的工作流程
   - 设置合理的期望

2. **工具调用规范**：
   - 详细的工具描述
   - 清晰的参数说明
   - 标准的调用格式

3. **推理链设计**：
   - 使用 ReAct 模式（Thought-Action-Observation）
   - 每步都有明确的思考过程
   - 支持多步迭代

4. **错误处理完善**：
   - 工具调用失败的处理
   - 超时和异常的处理
   - 提供降级方案

5. **模板组合灵活**：
   - 使用 + 操作符组合模板
   - 使用 partial_variables 预填充
   - 支持动态变量注入

6. **可观测性**：
   - 记录每步的思考和行动
   - 输出中间结果
   - 便于调试和优化
"""

print(best_practices)

# ===== 7. 扩展：多 Agent 协作模板 =====
print("\n" + "=" * 60)
print("扩展：多 Agent 协作模板")
print("=" * 60)

class MultiAgentTemplate:
    """多 Agent 协作模板"""

    def __init__(self):
        # 协作协调模板
        self.coordinator_template = PromptTemplate(
            template="""你是多 Agent 系统的协调者。

可用 Agent：
{agents_description}

任务：{task}

请分析任务并制定执行计划：
1. 将任务分解为子任务
2. 为每个子任务分配合适的 Agent
3. 确定执行顺序
4. 协调 Agent 之间的信息传递

输出格式：
```
执行计划：
1. [子任务1] -> [Agent名称]
2. [子任务2] -> [Agent名称]
...
```""",
            input_variables=["agents_description", "task"]
        )

        # Agent 间通信模板
        self.communication_template = PromptTemplate(
            template="""Agent 通信：

发送者：{sender_agent}
接收者：{receiver_agent}
消息类型：{message_type}
内容：{content}

请 {receiver_agent} 基于这个信息继续执行任务。""",
            input_variables=["sender_agent", "receiver_agent", "message_type", "content"]
        )

    def create_coordination_prompt(
        self,
        agents: List[Dict[str, str]],
        task: str
    ) -> str:
        """创建协调提示"""
        agents_description = "\n".join([
            f"- {agent['name']}: {agent['description']}"
            for agent in agents
        ])

        return self.coordinator_template.format(
            agents_description=agents_description,
            task=task
        )


# 测试多 Agent 协作模板
multi_agent_template = MultiAgentTemplate()

agents = [
    {"name": "研究员", "description": "负责信息收集和研究"},
    {"name": "分析师", "description": "负责数据分析和洞察"},
    {"name": "撰写员", "description": "负责撰写报告和总结"}
]

print("\n【多 Agent 协作计划】")
coordination_prompt = multi_agent_template.create_coordination_prompt(
    agents=agents,
    task="研究 LangChain 框架并撰写技术报告"
)
print(coordination_prompt)

print("\n" + "=" * 60)
print("实战代码执行完成！")
print("=" * 60)
```

---

## 运行输出示例

```
============================================================
场景1：Agent 角色定义模板
============================================================

【研究员 Agent 角色定义】
你是一个专业的研究员 Agent。
当前时间：2026-02-26 13:35:20

你的能力：
- 搜索和收集相关信息
- 分析数据来源的可靠性
- 总结研究发现

你的限制：
- 不进行深度分析和解读
- 不做最终决策
- 所有结果需要人工审核

工作流程：
1. 理解用户的研究需求
2. 使用可用工具收集信息
3. 整理和总结发现
4. 提供信息来源

【角色能力和限制】
名称：研究员 Agent
能力：搜索和收集信息, 分析数据来源, 总结研究发现
限制：不进行深度分析, 不做最终决策, 需要人工审核

============================================================
场景2：工具调用模板
============================================================

【工具调用指导】
你可以使用以下工具来完成任务：

**search**
描述：搜索互联网信息
参数：query: str - 搜索关键词

**calculator**
描述：执行数学计算
参数：expression: str - 数学表达式

**get_weather**
描述：获取天气信息
参数：location: str - 城市名称

工具使用规则：
1. 仔细阅读工具描述，确保理解工具的功能
2. 按照工具的参数要求准备输入
3. 一次只调用一个工具
4. 等待工具返回结果后再继续
5. 如果工具调用失败，分析原因并重试或使用其他工具

工具调用格式：
```
Tool: [工具名称]
Input: [工具输入]
```

当前任务：查询北京今天的天气，并计算华氏温度

请开始执行任务。
```

---

## 关键技术点

### 1. ReAct 推理模式

**核心思想**：Reasoning (思考) + Acting (行动)

**流程**：
```
Thought → Action → Observation → Thought → ... → Final Answer
```

[来源: reference/search_composition_01.md]

### 2. 工具调用最佳实践

- 清晰的工具描述
- 标准的调用格式
- 完善的错误处理
- 结果验证机制

### 3. 模板组合策略

- 角色定义 + 工具指导 + 任务描述
- 使用 ChatPromptTemplate 组合多个消息
- 使用 partial_variables 预填充动态信息

---

## 学习检查清单

- [ ] 理解 Agent 角色定义的重要性
- [ ] 掌握工具调用模板的设计
- [ ] 理解 ReAct 推理模式
- [ ] 实现完整的 Agent 系统
- [ ] 掌握错误处理模板
- [ ] 理解多 Agent 协作模式

---

## 下一步学习

- **进阶**：LangGraph 状态管理与 Agent 工作流
- **实战**：生产环境 Agent 监控与优化
- **扩展**：多模态 Agent 系统设计

---

## 与场景7的对比

| 维度 | RAG 系统模板 | Agent 系统模板 |
|------|-------------|---------------|
| **核心目标** | 检索 + 生成 | 推理 + 行动 |
| **模板类型** | 系统提示 + 检索提示 | 角色定义 + 工具调用 + 推理链 |
| **复杂度** | 中等 | 高 |
| **动态性** | 较低 | 高（多步迭代） |
| **错误处理** | 简单 | 复杂（工具失败、超时等） |
| **应用场景** | 文档问答、知识库 | 任务自动化、决策支持 |

---

**文件位置**：`atom/langchain/L3_组件生态/10_PromptTemplate高级用法/07_实战代码_场景8_Agent系统模板设计.md`
**知识点**：PromptTemplate高级用法
**层级**：L3_组件生态
**生成时间**：2026-02-26
