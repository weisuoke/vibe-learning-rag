# 实战代码08：AI Agent日志监控

## 学习目标

实现完整的AI Agent日志监控系统，追踪LLM调用、RAG检索、Token消耗和性能指标。

---

## 第一步：LLM调用日志

### 示例1：标准化LLM调用日志

```python
# utils/llm_logger.py
"""
LLM调用日志工具
"""

import time
import hashlib
import structlog
from typing import Optional, Dict, Any
from openai import OpenAI

logger = structlog.get_logger()

class LLMLogger:
    """LLM调用日志记录器"""

    def __init__(self, client: OpenAI):
        self.client = client

    async def call_with_logging(
        self,
        model: str,
        messages: list,
        **kwargs
    ):
        """带日志的LLM调用"""
        # 计算prompt哈希（用于去重和追踪）
        prompt_text = str(messages)
        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:16]

        # 记录调用开始
        start_time = time.time()

        logger.info("llm_call_start",
            model=model,
            prompt_length=len(prompt_text),
            prompt_hash=prompt_hash,
            temperature=kwargs.get("temperature", 1.0),
            max_tokens=kwargs.get("max_tokens")
        )

        try:
            # 调用LLM
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )

            # 计算耗时
            duration_ms = (time.time() - start_time) * 1000

            # 计算成本（示例价格）
            cost_per_1k_tokens = {
                "gpt-4": 0.03,
                "gpt-3.5-turbo": 0.002
            }
            price = cost_per_1k_tokens.get(model, 0.03)
            cost_usd = (response.usage.total_tokens / 1000) * price

            # 记录调用成功
            logger.info("llm_call_success",
                model=model,
                prompt_hash=prompt_hash,
                tokens_prompt=response.usage.prompt_tokens,
                tokens_completion=response.usage.completion_tokens,
                tokens_total=response.usage.total_tokens,
                duration_ms=duration_ms,
                cost_usd=cost_usd,
                finish_reason=response.choices[0].finish_reason
            )

            return response

        except Exception as e:
            # 计算耗时
            duration_ms = (time.time() - start_time) * 1000

            # 记录调用失败
            logger.error("llm_call_failed",
                model=model,
                prompt_hash=prompt_hash,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=duration_ms
            )

            raise

# 使用
client = OpenAI()
llm_logger = LLMLogger(client)

response = await llm_logger.call_with_logging(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is Python?"}],
    temperature=0.7,
    max_tokens=100
)
```

---

## 第二步：RAG检索日志

### 示例2：RAG检索日志记录

```python
# utils/rag_logger.py
"""
RAG检索日志工具
"""

import time
import structlog
from typing import List, Dict, Any

logger = structlog.get_logger()

class RAGLogger:
    """RAG检索日志记录器"""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    async def search_with_logging(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """带日志的RAG检索"""
        start_time = time.time()

        # 记录检索开始
        logger.info("rag_search_start",
            query=query,
            query_length=len(query),
            k=k,
            threshold=threshold
        )

        try:
            # 执行检索
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k
            )

            # 过滤低分结果
            filtered_results = [
                (doc, score) for doc, score in results
                if score >= threshold
            ]

            # 计算耗时
            duration_ms = (time.time() - start_time) * 1000

            # 提取分数统计
            scores = [score for _, score in filtered_results]
            top_score = max(scores) if scores else 0
            avg_score = sum(scores) / len(scores) if scores else 0

            # 记录检索成功
            logger.info("rag_search_success",
                query=query,
                results_count=len(filtered_results),
                results_before_filter=len(results),
                top_score=top_score,
                avg_score=avg_score,
                duration_ms=duration_ms
            )

            # 如果结果太少，记录警告
            if len(filtered_results) < k // 2:
                logger.warning("rag_low_results",
                    query=query,
                    results_count=len(filtered_results),
                    expected_min=k // 2
                )

            return filtered_results

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            logger.error("rag_search_failed",
                query=query,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=duration_ms
            )

            raise

# 使用
rag_logger = RAGLogger(vector_store)

results = await rag_logger.search_with_logging(
    query="Python programming",
    k=5,
    threshold=0.7
)
```

---

## 第三步：完整的AI Agent请求链路

### 示例3：端到端AI Agent日志

```python
# services/ai_agent.py
"""
AI Agent服务（带完整日志）
"""

import time
import structlog
from typing import List, Dict, Any
from openai import OpenAI

logger = structlog.get_logger()

class AIAgentService:
    """AI Agent服务"""

    def __init__(self, llm_client: OpenAI, vector_store):
        self.llm_client = llm_client
        self.vector_store = vector_store

    async def chat(self, message: str, user_id: str) -> str:
        """
        处理聊天请求

        完整链路：
        1. 接收用户消息
        2. RAG检索相关文档
        3. 构建prompt
        4. 调用LLM
        5. 返回响应
        """
        # 记录请求开始
        total_start = time.time()

        logger.info("chat_request",
            message_length=len(message),
            user_id=user_id
        )

        try:
            # 步骤1：RAG检索
            rag_start = time.time()
            docs = await self._rag_search(message)
            rag_duration = (time.time() - rag_start) * 1000

            # 步骤2：构建prompt
            prompt_start = time.time()
            prompt = self._build_prompt(message, docs)
            prompt_duration = (time.time() - prompt_start) * 1000

            logger.info("prompt_built",
                prompt_length=len(prompt),
                docs_count=len(docs),
                duration_ms=prompt_duration
            )

            # 步骤3：调用LLM
            llm_start = time.time()
            response = await self._call_llm(prompt)
            llm_duration = (time.time() - llm_start) * 1000

            # 计算总耗时
            total_duration = (time.time() - total_start) * 1000

            # 记录完整链路
            logger.info("chat_complete",
                total_duration_ms=total_duration,
                rag_duration_ms=rag_duration,
                prompt_duration_ms=prompt_duration,
                llm_duration_ms=llm_duration,
                rag_percentage=(rag_duration / total_duration) * 100,
                llm_percentage=(llm_duration / total_duration) * 100
            )

            return response

        except Exception as e:
            total_duration = (time.time() - total_start) * 1000

            logger.error("chat_failed",
                error=str(e),
                error_type=type(e).__name__,
                total_duration_ms=total_duration
            )

            raise

    async def _rag_search(self, query: str) -> List[Dict]:
        """RAG检索"""
        start = time.time()

        logger.info("rag_search_start", query=query)

        results = self.vector_store.similarity_search_with_score(query, k=5)

        duration_ms = (time.time() - start) * 1000
        scores = [score for _, score in results]

        logger.info("rag_search_complete",
            results_count=len(results),
            top_score=max(scores) if scores else 0,
            avg_score=sum(scores) / len(scores) if scores else 0,
            duration_ms=duration_ms
        )

        return [doc for doc, _ in results]

    def _build_prompt(self, message: str, docs: List[Dict]) -> str:
        """构建prompt"""
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {message}

Answer:"""

        return prompt

    async def _call_llm(self, prompt: str) -> str:
        """调用LLM"""
        start = time.time()

        logger.info("llm_call_start",
            prompt_length=len(prompt),
            model="gpt-4"
        )

        response = self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )

        duration_ms = (time.time() - start) * 1000

        logger.info("llm_call_complete",
            tokens_used=response.usage.total_tokens,
            duration_ms=duration_ms,
            cost_usd=(response.usage.total_tokens / 1000) * 0.03
        )

        return response.choices[0].message.content
```

---

## 第四步：Token消耗监控

### 示例4：Token使用统计

```python
# monitoring/token_tracker.py
"""
Token消耗追踪
"""

import structlog
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict

logger = structlog.get_logger()

class TokenTracker:
    """Token消耗追踪器"""

    def __init__(self):
        self.usage = defaultdict(lambda: {
            "total_tokens": 0,
            "total_cost": 0,
            "call_count": 0
        })

    def track(
        self,
        model: str,
        tokens: int,
        cost: float,
        user_id: str = None
    ):
        """记录Token使用"""
        # 按模型统计
        self.usage[f"model:{model}"]["total_tokens"] += tokens
        self.usage[f"model:{model}"]["total_cost"] += cost
        self.usage[f"model:{model}"]["call_count"] += 1

        # 按用户统计
        if user_id:
            self.usage[f"user:{user_id}"]["total_tokens"] += tokens
            self.usage[f"user:{user_id}"]["total_cost"] += cost
            self.usage[f"user:{user_id}"]["call_count"] += 1

        # 记录日志
        logger.info("token_usage",
            model=model,
            user_id=user_id,
            tokens=tokens,
            cost_usd=cost
        )

        # 检查是否超过预算
        if user_id:
            user_cost = self.usage[f"user:{user_id}"]["total_cost"]
            if user_cost > 10:  # $10预算
                logger.warning("user_budget_exceeded",
                    user_id=user_id,
                    total_cost=user_cost,
                    budget=10
                )

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return dict(self.usage)

    def reset(self):
        """重置统计"""
        self.usage.clear()

# 全局tracker
token_tracker = TokenTracker()

# 在LLM调用后使用
def log_llm_usage(model: str, response, user_id: str = None):
    """记录LLM使用"""
    tokens = response.usage.total_tokens

    # 计算成本
    cost_per_1k = {"gpt-4": 0.03, "gpt-3.5-turbo": 0.002}
    cost = (tokens / 1000) * cost_per_1k.get(model, 0.03)

    # 追踪
    token_tracker.track(model, tokens, cost, user_id)
```

---

## 第五步：性能监控仪表板

### 示例5：实时性能指标

```python
# monitoring/performance_dashboard.py
"""
性能监控仪表板
"""

import structlog
from collections import deque
from datetime import datetime
import statistics

logger = structlog.get_logger()

class PerformanceDashboard:
    """性能监控仪表板"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

        # 滑动窗口
        self.llm_durations = deque(maxlen=window_size)
        self.rag_durations = deque(maxlen=window_size)
        self.total_durations = deque(maxlen=window_size)

        # 错误计数
        self.error_count = 0
        self.total_count = 0

    def record_request(
        self,
        total_duration: float,
        llm_duration: float,
        rag_duration: float,
        success: bool
    ):
        """记录请求"""
        self.total_count += 1

        if success:
            self.llm_durations.append(llm_duration)
            self.rag_durations.append(rag_duration)
            self.total_durations.append(total_duration)
        else:
            self.error_count += 1

        # 定期记录统计
        if self.total_count % 10 == 0:
            self._log_stats()

    def _log_stats(self):
        """记录统计信息"""
        if not self.total_durations:
            return

        stats = {
            "total_requests": self.total_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.total_count if self.total_count > 0 else 0,

            # 总耗时统计
            "total_duration_avg": statistics.mean(self.total_durations),
            "total_duration_p50": statistics.median(self.total_durations),
            "total_duration_p95": self._percentile(self.total_durations, 0.95),
            "total_duration_p99": self._percentile(self.total_durations, 0.99),

            # LLM耗时统计
            "llm_duration_avg": statistics.mean(self.llm_durations),
            "llm_duration_p95": self._percentile(self.llm_durations, 0.95),

            # RAG耗时统计
            "rag_duration_avg": statistics.mean(self.rag_durations),
            "rag_duration_p95": self._percentile(self.rag_durations, 0.95),
        }

        logger.info("performance_stats", **stats)

        # 检查异常
        if stats["error_rate"] > 0.05:  # 错误率超过5%
            logger.warning("high_error_rate", error_rate=stats["error_rate"])

        if stats["total_duration_p95"] > 5000:  # P95超过5秒
            logger.warning("high_latency", p95_ms=stats["total_duration_p95"])

    def _percentile(self, data, percentile):
        """计算百分位数"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def get_current_stats(self) -> dict:
        """获取当前统计"""
        if not self.total_durations:
            return {}

        return {
            "total_requests": self.total_count,
            "error_rate": self.error_count / self.total_count if self.total_count > 0 else 0,
            "avg_duration_ms": statistics.mean(self.total_durations),
            "p95_duration_ms": self._percentile(self.total_durations, 0.95)
        }

# 全局dashboard
dashboard = PerformanceDashboard()
```

---

## 第六步：完整的FastAPI集成

### 示例6：AI Agent API with完整监控

```python
# main.py
"""
AI Agent API with完整日志监控
"""

import uuid
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import structlog

# 导入服务和监控
from services.ai_agent import AIAgentService
from monitoring.token_tracker import token_tracker
from monitoring.performance_dashboard import dashboard
from config.production_logging import setup_production_logging

# 配置日志
setup_production_logging()

app = FastAPI()
logger = structlog.get_logger()

# 初始化服务
ai_agent = AIAgentService(llm_client, vector_store)

@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    """监控中间件"""
    # 跳过健康检查
    if request.url.path == "/health":
        return await call_next(request)

    # 生成请求ID
    request_id = str(uuid.uuid4())

    # 绑定上下文
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        user_id=request.headers.get("X-User-ID")
    )

    # 记录请求开始
    start_time = time.time()
    logger.info("request_start")

    try:
        # 处理请求
        response = await call_next(request)

        # 计算耗时
        duration_ms = (time.time() - start_time) * 1000

        # 记录请求结束
        logger.info("request_end",
            status_code=response.status_code,
            duration_ms=duration_ms
        )

        # 添加响应头
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        return response

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000

        logger.error("request_failed",
            error=str(e),
            error_type=type(e).__name__,
            duration_ms=duration_ms
        )

        # 记录到dashboard
        dashboard.record_request(
            total_duration=duration_ms,
            llm_duration=0,
            rag_duration=0,
            success=False
        )

        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
            headers={"X-Request-ID": request_id}
        )

    finally:
        structlog.contextvars.clear_contextvars()

@app.post("/api/chat")
async def chat(message: str, user_id: str = None):
    """
    AI Agent聊天接口

    完整监控：
    - 请求链路追踪
    - LLM调用日志
    - RAG检索日志
    - Token消耗追踪
    - 性能指标记录
    """
    logger.info("chat_start", message_length=len(message))

    # 记录时间
    total_start = time.time()
    rag_duration = 0
    llm_duration = 0

    try:
        # 调用AI Agent
        response = await ai_agent.chat(message, user_id)

        # 计算耗时（从日志中提取）
        total_duration = (time.time() - total_start) * 1000

        # 记录到dashboard
        dashboard.record_request(
            total_duration=total_duration,
            llm_duration=llm_duration,
            rag_duration=rag_duration,
            success=True
        )

        logger.info("chat_success",
            response_length=len(response),
            total_duration_ms=total_duration
        )

        return {
            "response": response,
            "metadata": {
                "duration_ms": total_duration
            }
        }

    except Exception as e:
        logger.error("chat_error", error=str(e))

        # 记录失败
        dashboard.record_request(
            total_duration=(time.time() - total_start) * 1000,
            llm_duration=0,
            rag_duration=0,
            success=False
        )

        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """获取统计信息"""
    return {
        "performance": dashboard.get_current_stats(),
        "token_usage": token_tracker.get_stats()
    }

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}
```

---

## 第七步：日志分析工具

### 示例7：日志分析脚本

```python
# tools/analyze_logs.py
"""
AI Agent日志分析工具

使用方法：
    python tools/analyze_logs.py --log-file logs/all.log
"""

import json
import argparse
from collections import defaultdict
from datetime import datetime
import statistics

def analyze_logs(log_file: str):
    """分析日志文件"""
    # 统计数据
    stats = {
        "total_requests": 0,
        "llm_calls": 0,
        "rag_searches": 0,
        "errors": 0,
        "llm_durations": [],
        "rag_durations": [],
        "total_durations": [],
        "token_usage": defaultdict(int),
        "total_cost": 0
    }

    # 读取日志
    with open(log_file, 'r') as f:
        for line in f:
            try:
                log = json.loads(line)

                # 统计请求
                if log.get("event") == "request_start":
                    stats["total_requests"] += 1

                # 统计LLM调用
                elif log.get("event") == "llm_call_success":
                    stats["llm_calls"] += 1
                    stats["llm_durations"].append(log.get("duration_ms", 0))
                    stats["token_usage"][log.get("model")] += log.get("tokens_total", 0)
                    stats["total_cost"] += log.get("cost_usd", 0)

                # 统计RAG检索
                elif log.get("event") == "rag_search_success":
                    stats["rag_searches"] += 1
                    stats["rag_durations"].append(log.get("duration_ms", 0))

                # 统计完整请求
                elif log.get("event") == "chat_complete":
                    stats["total_durations"].append(log.get("total_duration_ms", 0))

                # 统计错误
                elif log.get("level") == "error":
                    stats["errors"] += 1

            except json.JSONDecodeError:
                continue

    # 打印报告
    print("=" * 60)
    print("AI Agent日志分析报告")
    print("=" * 60)

    print(f"\n总请求数: {stats['total_requests']}")
    print(f"LLM调用数: {stats['llm_calls']}")
    print(f"RAG检索数: {stats['rag_searches']}")
    print(f"错误数: {stats['errors']}")
    print(f"错误率: {stats['errors'] / stats['total_requests'] * 100:.2f}%" if stats['total_requests'] > 0 else "N/A")

    if stats["llm_durations"]:
        print(f"\nLLM调用耗时:")
        print(f"  平均: {statistics.mean(stats['llm_durations']):.2f}ms")
        print(f"  中位数: {statistics.median(stats['llm_durations']):.2f}ms")
        print(f"  P95: {sorted(stats['llm_durations'])[int(len(stats['llm_durations']) * 0.95)]:.2f}ms")

    if stats["rag_durations"]:
        print(f"\nRAG检索耗时:")
        print(f"  平均: {statistics.mean(stats['rag_durations']):.2f}ms")
        print(f"  中位数: {statistics.median(stats['rag_durations']):.2f}ms")

    if stats["total_durations"]:
        print(f"\n总请求耗时:")
        print(f"  平均: {statistics.mean(stats['total_durations']):.2f}ms")
        print(f"  P95: {sorted(stats['total_durations'])[int(len(stats['total_durations']) * 0.95)]:.2f}ms")

    print(f"\nToken使用:")
    for model, tokens in stats["token_usage"].items():
        print(f"  {model}: {tokens:,} tokens")

    print(f"\n总成本: ${stats['total_cost']:.4f}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析AI Agent日志")
    parser.add_argument("--log-file", required=True, help="日志文件路径")

    args = parser.parse_args()

    analyze_logs(args.log_file)
```

**使用：**
```bash
python tools/analyze_logs.py --log-file logs/all.log
```

---

## 总结

### 核心要点

1. **LLM调用日志**
   - 记录model、tokens、duration、cost
   - 使用prompt哈希去重
   - 追踪finish_reason

2. **RAG检索日志**
   - 记录query、results_count、scores
   - 监控检索质量
   - 低结果告警

3. **完整链路追踪**
   - 端到端耗时
   - 各步骤占比
   - 性能瓶颈分析

4. **Token消耗监控**
   - 按模型统计
   - 按用户统计
   - 预算告警

5. **性能监控**
   - 实时指标
   - 百分位数统计
   - 异常告警

### 最佳实践

1. 标准化日志格式
2. 记录关键性能指标
3. 实时监控和告警
4. 定期分析日志
5. 优化成本和性能

### 完成

恭喜！你已经完成了结构化日志的完整学习路径，包括：
- 10个维度的理论知识
- 8个实战代码示例
- 完整的AI Agent日志监控系统

现在你可以在生产环境中实现专业的日志系统了！
