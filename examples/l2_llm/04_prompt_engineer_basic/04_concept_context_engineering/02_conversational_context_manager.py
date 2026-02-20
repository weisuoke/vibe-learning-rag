from sentence_transformers import CrossEncoder
from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

class ContextOptimizer:
    """上下文优化器"""

    def __init__(self, max_tokens: int = 6000):
        self.max_tokens = max_tokens
        # 使用 CrossEncoder 进行 ReRank
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def estimate_tokens(self, text: str) -> int:
        """估算 token 数量（粗略估计：1 token ≈ 4 字符）"""
        return len(text) // 4
    
    def optimize(self, query: str, documents: list, max_tokens: int | None = None) -> str:
        """优化上下文注入"""

        # Allow callers (e.g. a context manager) to provide a dynamic budget.
        token_budget = self.max_tokens if max_tokens is None else max_tokens
        if token_budget <= 0 or not documents:
            return "\n".join(
                [
                    "# 检索结果",
                    f"检索到 {len(documents)} 篇文档，选择了最相关的 0 篇",
                    f"总 tokens：0/{token_budget}\n",
                ]
            )

        # 步骤1： ReRank 重排序
        print("=== 步骤 1： ReRank 重排序 ===")
        pairs = [(query, doc['content']) for doc in documents]
        scores = self.reranker.predict(pairs)

        # 按相关性排序
        ranked_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        print(f"重排序后的相关度分数:")
        for i, (doc, score) in enumerate(ranked_docs):
            print(f"  文档 {i+1}: {score:.3f}")

        # 步骤 2：动态选择文档（基于 Token 预算）
        print(f"\n===步骤 2： 动态选择（预算：{token_budget}tokens）===")
        selected_docs=[]
        current_tokens = 0

        for doc, score in ranked_docs:
            doc_tokens = self.estimate_tokens(doc['content'])

            if current_tokens + doc_tokens <= token_budget:
                selected_docs.append((doc, score))
                current_tokens += doc_tokens
                print(f"  ✓ 选择文档（{doc_tokens} tokens，累计：{current_tokens}）")
            else:
                print(f"  ✗ 跳过文档（{doc_tokens} tokens，超出预算）")
                break

        # 步骤 3：构建结构化上下文
        print(f"\n=== 步骤 3：构建结构化上下文 ===")
        context_parts = []

        # 添加元信息
        context_parts.append(f"# 检索结果")
        context_parts.append(f"检索到 {len(documents)} 篇文档，选择了最相关的 {len(selected_docs)} 篇")
        context_parts.append(f"总 tokens：{current_tokens}/{token_budget}\n")

        # 添加文档内容
        for i, (doc, score) in enumerate(selected_docs, 1):
            context_parts.append(f"## 文档 {i} (相关度: {score:.2f})")
            context_parts.append(f"来源：{doc.get('source', '未知')}")
            context_parts.append(f"内容：{doc['content']}\n")

        return "\n".join(context_parts)

class ConversationalContextManager:
    """对话式上下文管理器"""

    def __init__(self, context_window: int=8192):
        self.context_window = context_window
        self.conversation_history = []
        self.optimizer = ContextOptimizer()

    def estimate_tokens(self, text: str) -> int:
        """估算 tokens"""
        return len(text) // 4
    
    def manage_context(
        self,
        user_query: str,
        retrieved_docs: list,
        system_prompt: str
    ) -> dict:
        """管理上下文分配"""
        
        # 固定开销
        system_tokens = self.estimate_tokens(system_prompt)
        query_tokens = self.estimate_tokens(user_query)
        output_reserve = 500

        # 计算对话历史的 tokens
        history_tokens = sum([
            self.estimate_tokens(msg['content'])
            for msg in self.conversation_history
        ])

        # 可用空间
        available = self.context_window - system_tokens - query_tokens - output_reserve

        print(f"=== 上下文预算分配 ===")
        print(f"Context Window: {self.context_window} tokens")
        print(f"  - System Prompt: {system_tokens} tokens")
        print(f"  - User Query: {query_tokens} tokens")
        print(f"  - Output Reserve: {output_reserve} tokens")
        print(f"  - Conversation History: {history_tokens} tokens")
        print(f"  - Available: {available} tokens")

        # 策略：如果对话历史太长， 压缩它
        max_history_tokens = available // 3 # 对话历史最多占 1/3
        if history_tokens > max_history_tokens:
            print(f"\n⚠️  对话历史过长，压缩到 {max_history_tokens} tokens")
            self.conversation_history = self._compress_history(max_history_tokens)
            history_tokens = max_history_tokens

        # 剩余空间给检索内容
        retrieval_budget = max(0, available - history_tokens)
        print(f"\n检索内容预算: {retrieval_budget} tokens")

        # 优化检索内容
        optimized_context = self.optimizer.optimize(
            user_query,
            retrieved_docs,
            max_tokens=retrieval_budget
        )

        return {
            "system_prompt": system_prompt,
            "conversation_history": self.conversation_history,
            "retrieved_context": optimized_context,
            "user_query": user_query,
            "budget": {
                "total": self.context_window,
                "system": system_tokens,
                "history": history_tokens,
                "retrieval": self.estimate_tokens(optimized_context),
                "query": query_tokens,
                "output_reserve": output_reserve
            }
        }
    
    def _compress_history(self, max_tokens: int) -> list:
        """压缩对话历史"""
        # 策略：保留最近的对话
        compressed = []
        current_tokens = 0

        for msg in reversed(self.conversation_history):
            msg_tokens = self.estimate_tokens(msg['content'])
            if current_tokens + msg_tokens <= max_tokens:
                compressed.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break

        return compressed
    
    def add_to_history(self, role: str, content: str):
        """添加到对话历史"""
        self.conversation_history.append({"role": role, "content": content})

# 测试
manager = ConversationalContextManager(context_window=8192)

# 模拟检索到的文档
documents = [
    {
        "content": "RAG（检索增强生成）是一种结合检索和生成的技术，通过在生成前检索相关文档来提升答案准确性。",
        "source": "doc1.pdf"
    },
    {
        "content": "向量数据库用于存储和检索 embeddings，支持语义相似度搜索。",
        "source": "doc2.pdf"
    },
    {
        "content": "LangChain 是一个流行的 RAG 框架，提供了完整的工具链。",
        "source": "doc3.pdf"
    },
    {
        "content": "Python 是一种高级编程语言，广泛用于数据科学和机器学习。",
        "source": "doc4.pdf"
    },
    {
        "content": "Transformer 是现代 LLM 的基础架构，使用自注意力机制。",
        "source": "doc5.pdf"
    }
]

# 第一轮对话
context1 = manager.manage_context(
    user_query="什么是 RAG?",
    retrieved_docs=documents,
    system_prompt="你是专业的 RAG 助手"
)

print(f"\n第1轮对话预算:")
for key, value in context1['budget'].items():
    print(f" {key}: {value}")

# 模拟添加对话历史
manager.add_to_history("user", "什么是 RAG？")
manager.add_to_history("assistant", "RAG 是检索增强生成技术...")


# 第二轮对话
context2 = manager.manage_context(
    user_query="它有什么优势？",
    retrieved_docs=documents,
    system_prompt="你是专业的 RAG 助手"
)

print(f"\n第二轮对话预算:")
for key, value in context2['budget'].items():
    print(f" {key}: {value}")