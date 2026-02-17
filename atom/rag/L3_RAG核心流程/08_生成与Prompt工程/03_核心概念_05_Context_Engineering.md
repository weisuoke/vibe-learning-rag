# 核心概念5：Context Engineering

> 从Prompt Engineering到Context Engineering的范式转变 - 2025-2026生产标准

---

## 概述

Context Engineering是2025-2026年RAG领域的重大范式转变，将上下文从"背景信息"提升为"基础设施"，需要像管理数据库和API一样系统化地管理上下文。

**核心观点：** "Context is not just background information—it's infrastructure."

**来源：** Redis "Context engineering: Best practices for an emerging discipline" (2025年9月)
https://redis.io/blog/context-engineering-best-practices-for-an-emerging-discipline

---

## 1. 范式转变

### 1.1 传统Prompt Engineering vs Context Engineering

```python
# ===== 传统Prompt Engineering（2023-2024）=====

# 关注点：如何写好提示词
prompt = f"""
你是助手。

参考资料：{docs}

问题：{query}

回答：
"""

# 局限性：
# - 依赖人工经验
# - 难以规模化
# - 缺乏系统性
# - 不可复现

# ===== Context Engineering（2025-2026）=====

class ContextInfrastructure:
    """
    上下文基础设施
    """
    
    def __init__(self):
        # 1. System Layer（系统层）
        self.system_prompts = SystemPromptManager()
        
        # 2. RAG Layer（检索层）
        self.retrieval = RetrievalEngine()
        
        # 3. Tools Layer（工具层）
        self.tools = ToolRegistry()
        
        # 4. Memory Layer（记忆层）
        self.memory = MemoryStore()
    
    def build_context(self, query: str) -> dict:
        """
        系统化构建上下文
        """
        # 组装完整上下文
        context = {
            "system": self.system_prompts.get_for_task(query),
            "retrieved": self.retrieval.search(query),
            "tools": self.tools.get_available(),
            "memory": self.memory.get_relevant(query)
        }
        
        return context

# 优势：
# - 系统化管理
# - 可复现
# - 可扩展
# - 工程化
```

### 1.2 四层上下文架构

```python
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ContextLayers:
    """
    四层上下文架构
    """
    system: str          # 系统层：角色和约束
    rag: List[str]       # RAG层：检索内容
    tools: List[dict]    # 工具层：可用工具
    memory: List[dict]   # 记忆层：历史对话

class ContextEngineer:
    """
    上下文工程师
    """
    
    def design_context_architecture(
        self,
        task_type: str,
        domain: str
    ) -> ContextLayers:
        """
        设计上下文架构
        """
        # 1. System Layer设计
        system = self._design_system_layer(task_type, domain)
        
        # 2. RAG Layer配置
        rag_config = self._configure_rag_layer(task_type)
        
        # 3. Tools Layer选择
        tools = self._select_tools(task_type)
        
        # 4. Memory Layer策略
        memory_strategy = self._design_memory_strategy(task_type)
        
        return ContextLayers(
            system=system,
            rag=[],  # 运行时填充
            tools=tools,
            memory=[]  # 运行时填充
        )
    
    def _design_system_layer(
        self,
        task_type: str,
        domain: str
    ) -> str:
        """
        设计系统层
        """
        base_role = {
            "qa": "知识助手",
            "summarization": "总结专家",
            "analysis": "分析师"
        }.get(task_type, "助手")
        
        domain_expertise = {
            "medical": "医疗领域专家",
            "legal": "法律领域专家",
            "technical": "技术领域专家"
        }.get(domain, "")
        
        return f"""
你是一个{base_role}{f'，专注于{domain_expertise}' if domain_expertise else ''}。

核心能力：
- 基于参考资料提供准确答案
- 识别信息不足的情况
- 提供可追溯的引用

行为约束：
- 只使用参考资料中的信息
- 不推测或添加额外内容
- 如果资料不足，明确说明
"""
    
    def _configure_rag_layer(self, task_type: str) -> dict:
        """
        配置RAG层
        """
        return {
            "top_k": 5 if task_type == "qa" else 10,
            "min_score": 0.7,
            "rerank": True,
            "context_window": 2000
        }
    
    def _select_tools(self, task_type: str) -> List[dict]:
        """
        选择工具层
        """
        base_tools = [
            {"name": "search", "description": "搜索知识库"},
            {"name": "calculate", "description": "数学计算"}
        ]
        
        if task_type == "analysis":
            base_tools.append({
                "name": "visualize",
                "description": "数据可视化"
            })
        
        return base_tools
    
    def _design_memory_strategy(self, task_type: str) -> dict:
        """
        设计记忆层策略
        """
        return {
            "window_size": 5 if task_type == "qa" else 10,
            "summarization": task_type == "analysis",
            "persistence": True
        }
```

---

## 2. Context as Infrastructure

### 2.1 基础设施特征

```python
class ContextInfrastructure:
    """
    上下文基础设施
    
    特征：
    1. 持久化：像数据库一样持久存储
    2. 可查询：像API一样可查询
    3. 可扩展：像微服务一样可扩展
    4. 可监控：像系统一样可监控
    """
    
    def __init__(self):
        # 持久化存储
        self.storage = ContextStorage()
        
        # 查询接口
        self.query_engine = ContextQueryEngine()
        
        # 扩展机制
        self.plugins = PluginRegistry()
        
        # 监控系统
        self.monitor = ContextMonitor()
    
    def store_context(
        self,
        session_id: str,
        context: dict
    ):
        """
        持久化存储上下文
        """
        self.storage.save(session_id, context)
        self.monitor.log_storage(session_id, len(str(context)))
    
    def query_context(
        self,
        session_id: str,
        query: str
    ) -> dict:
        """
        查询上下文
        """
        context = self.storage.load(session_id)
        relevant = self.query_engine.search(context, query)
        self.monitor.log_query(session_id, query)
        return relevant
    
    def extend_context(
        self,
        plugin_name: str,
        plugin_func: callable
    ):
        """
        扩展上下文能力
        """
        self.plugins.register(plugin_name, plugin_func)
    
    def get_metrics(self) -> dict:
        """
        获取监控指标
        """
        return self.monitor.get_metrics()
```

### 2.2 上下文生命周期管理

```python
from enum import Enum
from datetime import datetime

class ContextState(Enum):
    """上下文状态"""
    CREATED = "created"
    ACTIVE = "active"
    CACHED = "cached"
    EXPIRED = "expired"

class ContextLifecycleManager:
    """
    上下文生命周期管理
    """
    
    def __init__(self):
        self.contexts = {}
        self.ttl = 3600  # 1小时过期
    
    def create_context(
        self,
        session_id: str,
        initial_context: dict
    ) -> dict:
        """
        创建上下文
        """
        context = {
            "id": session_id,
            "state": ContextState.CREATED,
            "data": initial_context,
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "access_count": 0
        }
        
        self.contexts[session_id] = context
        return context
    
    def activate_context(self, session_id: str):
        """
        激活上下文
        """
        if session_id in self.contexts:
            self.contexts[session_id]["state"] = ContextState.ACTIVE
            self.contexts[session_id]["last_accessed"] = datetime.now()
    
    def cache_context(self, session_id: str):
        """
        缓存上下文
        """
        if session_id in self.contexts:
            self.contexts[session_id]["state"] = ContextState.CACHED
    
    def expire_context(self, session_id: str):
        """
        过期上下文
        """
        if session_id in self.contexts:
            self.contexts[session_id]["state"] = ContextState.EXPIRED
    
    def cleanup_expired(self):
        """
        清理过期上下文
        """
        now = datetime.now()
        expired = [
            sid for sid, ctx in self.contexts.items()
            if (now - ctx["last_accessed"]).seconds > self.ttl
        ]
        
        for sid in expired:
            self.expire_context(sid)
            del self.contexts[sid]
```

---

## 3. Context Selection（上下文选择）

### 3.1 智能上下文选择

```python
class ContextSelector:
    """
    智能上下文选择器
    """
    
    def select_optimal_context(
        self,
        query: str,
        available_contexts: List[dict],
        max_tokens: int = 2000
    ) -> List[dict]:
        """
        选择最优上下文
        """
        # 1. 相关性评分
        scored = self._score_relevance(query, available_contexts)
        
        # 2. 多样性过滤
        diverse = self._ensure_diversity(scored)
        
        # 3. Token预算管理
        selected = self._fit_token_budget(diverse, max_tokens)
        
        return selected
    
    def _score_relevance(
        self,
        query: str,
        contexts: List[dict]
    ) -> List[dict]:
        """
        相关性评分
        """
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        query_emb = model.encode(query)
        
        for ctx in contexts:
            ctx_emb = model.encode(ctx['content'])
            ctx['relevance_score'] = np.dot(query_emb, ctx_emb)
        
        return sorted(contexts, key=lambda x: x['relevance_score'], reverse=True)
    
    def _ensure_diversity(
        self,
        contexts: List[dict],
        threshold: float = 0.8
    ) -> List[dict]:
        """
        确保多样性
        """
        selected = []
        
        for ctx in contexts:
            # 检查与已选择上下文的相似度
            is_diverse = True
            for sel in selected:
                if self._similarity(ctx, sel) > threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(ctx)
        
        return selected
    
    def _similarity(self, ctx1: dict, ctx2: dict) -> float:
        """
        计算相似度
        """
        # 简化实现
        return 0.5
    
    def _fit_token_budget(
        self,
        contexts: List[dict],
        max_tokens: int
    ) -> List[dict]:
        """
        适配Token预算
        """
        selected = []
        total_tokens = 0
        
        for ctx in contexts:
            ctx_tokens = len(ctx['content'].split()) * 1.3
            if total_tokens + ctx_tokens <= max_tokens:
                selected.append(ctx)
                total_tokens += ctx_tokens
            else:
                break
        
        return selected
```

---

## 4. Context Structuring（上下文结构化）

### 4.1 层次化结构

```python
class HierarchicalContext:
    """
    层次化上下文
    """
    
    def structure_context(
        self,
        raw_contexts: List[dict]
    ) -> dict:
        """
        结构化上下文
        """
        structured = {
            "high_priority": [],
            "medium_priority": [],
            "low_priority": [],
            "metadata": {}
        }
        
        for ctx in raw_contexts:
            priority = self._determine_priority(ctx)
            structured[f"{priority}_priority"].append(ctx)
        
        return structured
    
    def _determine_priority(self, ctx: dict) -> str:
        """
        确定优先级
        """
        score = ctx.get('relevance_score', 0)
        
        if score > 0.8:
            return "high"
        elif score > 0.6:
            return "medium"
        else:
            return "low"
    
    def format_for_llm(self, structured: dict) -> str:
        """
        格式化为LLM输入
        """
        output = ""
        
        if structured["high_priority"]:
            output += "## 高相关性内容\n\n"
            for ctx in structured["high_priority"]:
                output += f"- {ctx['content']}\n"
        
        if structured["medium_priority"]:
            output += "\n## 中等相关性内容\n\n"
            for ctx in structured["medium_priority"]:
                output += f"- {ctx['content']}\n"
        
        return output
```

---

## 5. Context Delivery（上下文传递）

### 5.1 优化传递策略

```python
class ContextDeliveryOptimizer:
    """
    上下文传递优化器
    """
    
    def optimize_delivery(
        self,
        context: dict,
        model_config: dict
    ) -> dict:
        """
        优化上下文传递
        """
        # 1. 压缩策略
        compressed = self._compress_if_needed(
            context,
            model_config['max_context_length']
        )
        
        # 2. 分块策略
        chunked = self._chunk_if_needed(
            compressed,
            model_config['chunk_size']
        )
        
        # 3. 缓存策略
        cached = self._cache_common_contexts(chunked)
        
        return cached
    
    def _compress_if_needed(
        self,
        context: dict,
        max_length: int
    ) -> dict:
        """
        压缩上下文
        """
        current_length = len(str(context))
        
        if current_length > max_length:
            # 使用LLM总结
            context['content'] = self._summarize(
                context['content'],
                target_length=max_length
            )
        
        return context
    
    def _summarize(self, content: str, target_length: int) -> str:
        """
        总结内容
        """
        # 简化实现
        return content[:target_length]
    
    def _chunk_if_needed(
        self,
        context: dict,
        chunk_size: int
    ) -> List[dict]:
        """
        分块处理
        """
        content = context['content']
        
        if len(content) <= chunk_size:
            return [context]
        
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunk = context.copy()
            chunk['content'] = content[i:i+chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    def _cache_common_contexts(self, contexts: List[dict]) -> List[dict]:
        """
        缓存常见上下文
        """
        # 实现缓存逻辑
        return contexts
```

---

## 6. 生产环境实践

### 6.1 完整Context Engineering系统

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ProductionContextEngine:
    """
    生产级上下文引擎
    """
    
    def __init__(self):
        self.infrastructure = ContextInfrastructure()
        self.selector = ContextSelector()
        self.structurer = HierarchicalContext()
        self.optimizer = ContextDeliveryOptimizer()
        self.lifecycle = ContextLifecycleManager()
    
    def process_query(
        self,
        session_id: str,
        query: str,
        domain: str = "general"
    ) -> dict:
        """
        处理查询（完整流程）
        """
        # 1. 创建/激活上下文
        if session_id not in self.lifecycle.contexts:
            self.lifecycle.create_context(session_id, {})
        self.lifecycle.activate_context(session_id)
        
        # 2. 检索相关上下文
        retrieved = self.infrastructure.query_context(session_id, query)
        
        # 3. 选择最优上下文
        selected = self.selector.select_optimal_context(
            query,
            retrieved,
            max_tokens=2000
        )
        
        # 4. 结构化上下文
        structured = self.structurer.structure_context(selected)
        
        # 5. 优化传递
        optimized = self.optimizer.optimize_delivery(
            structured,
            {"max_context_length": 2000, "chunk_size": 500}
        )
        
        # 6. 生成答案
        answer = self._generate_answer(query, optimized, domain)
        
        # 7. 存储上下文
        self.infrastructure.store_context(session_id, {
            "query": query,
            "context": optimized,
            "answer": answer
        })
        
        return {
            "answer": answer,
            "context_used": len(selected),
            "session_id": session_id
        }
    
    def _generate_answer(
        self,
        query: str,
        context: dict,
        domain: str
    ) -> str:
        """
        生成答案
        """
        # 构建System Prompt
        engineer = ContextEngineer()
        layers = engineer.design_context_architecture("qa", domain)
        
        # 格式化上下文
        context_str = self.structurer.format_for_llm(context)
        
        # 生成
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": layers.system},
                {"role": "user", "content": f"{context_str}\n\n问题：{query}"}
            ],
            temperature=0.1
        )
        
        return response.choices[0].message.content

# 使用示例
engine = ProductionContextEngine()

result = engine.process_query(
    session_id="session_123",
    query="什么是RAG？",
    domain="technical"
)

print("答案：", result["answer"])
print("使用上下文数：", result["context_used"])
```

---

## 7. 监控与优化

### 7.1 上下文质量监控

```python
class ContextQualityMonitor:
    """
    上下文质量监控
    """
    
    def __init__(self):
        self.metrics = {
            "context_relevance": [],
            "context_utilization": [],
            "context_efficiency": []
        }
    
    def monitor_context_quality(
        self,
        query: str,
        context: dict,
        answer: str
    ) -> dict:
        """
        监控上下文质量
        """
        # 1. 相关性
        relevance = self._measure_relevance(query, context)
        
        # 2. 利用率
        utilization = self._measure_utilization(context, answer)
        
        # 3. 效率
        efficiency = self._measure_efficiency(context, answer)
        
        self.metrics["context_relevance"].append(relevance)
        self.metrics["context_utilization"].append(utilization)
        self.metrics["context_efficiency"].append(efficiency)
        
        return {
            "relevance": relevance,
            "utilization": utilization,
            "efficiency": efficiency
        }
    
    def _measure_relevance(self, query: str, context: dict) -> float:
        """
        测量相关性
        """
        # 简化实现
        return 0.85
    
    def _measure_utilization(self, context: dict, answer: str) -> float:
        """
        测量利用率（答案中使用了多少上下文）
        """
        # 简化实现
        return 0.75
    
    def _measure_efficiency(self, context: dict, answer: str) -> float:
        """
        测量效率（Token使用效率）
        """
        context_tokens = len(str(context).split())
        answer_tokens = len(answer.split())
        return answer_tokens / context_tokens if context_tokens > 0 else 0
    
    def get_summary(self) -> dict:
        """
        获取监控摘要
        """
        return {
            "avg_relevance": sum(self.metrics["context_relevance"]) / len(self.metrics["context_relevance"]),
            "avg_utilization": sum(self.metrics["context_utilization"]) / len(self.metrics["context_utilization"]),
            "avg_efficiency": sum(self.metrics["context_efficiency"]) / len(self.metrics["context_efficiency"])
        }
```

---

## 总结

### 核心原则

1. **Infrastructure Mindset**：上下文是基础设施，不是背景信息
2. **Systematic Management**：系统化管理，不是临时拼接
3. **Lifecycle Awareness**：全生命周期管理
4. **Quality Monitoring**：持续监控和优化
5. **Scalability**：可扩展的架构设计

### 2025-2026标准配置

```python
# Context Engineering生产配置
CONTEXT_ENGINEERING_CONFIG_2026 = {
    "architecture": "four_layer",  # System + RAG + Tools + Memory
    "selection": "intelligent",    # 智能选择
    "structuring": "hierarchical", # 层次化结构
    "delivery": "optimized",       # 优化传递
    "monitoring": True,            # 启用监控
    "caching": True,              # 启用缓存
    "lifecycle_management": True   # 生命周期管理
}
```

---

**版本：** v1.0 (2025-2026最新标准)
**最后更新：** 2026-02-16
**参考来源：**
- Redis "Context Engineering Best Practices" (2025-09)
- https://redis.io/blog/context-engineering-best-practices-for-an-emerging-discipline
