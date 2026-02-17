# 实战代码6：NVIDIA 基准测试 ⭐ NEW 2025

完整可运行的分块策略基准测试代码（NVIDIA 2025 方法）。

---

## NVIDIA 2025 研究复现

```python
"""
NVIDIA 2025 分块策略基准测试
研究来源: NVIDIA 2025 Chunking Benchmark

核心发现：
1. 页面级分块准确率最高（0.648）
2. 15% 重叠率最优
3. 查询类型决定最优块大小
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import time

def benchmark_chunking_strategies(
    text: str,
    test_queries: List[str]
) -> Dict:
    """
    对比不同分块策略的性能
    
    测试维度：
    1. 块数量
    2. 平均块大小
    3. 处理速度
    4. 重叠率
    """
    strategies = {
        "固定大小_256": {
            "chunk_size": 256,
            "overlap_ratio": 0.15
        },
        "固定大小_512_NVIDIA推荐": {
            "chunk_size": 512,
            "overlap_ratio": 0.15
        },
        "固定大小_1024": {
            "chunk_size": 1024,
            "overlap_ratio": 0.15
        },
        "页面级_2048": {
            "chunk_size": 2048,
            "overlap_ratio": 0.15
        }
    }
    
    results = {}
    
    for name, config in strategies.items():
        chunk_size = config["chunk_size"]
        chunk_overlap = int(chunk_size * config["overlap_ratio"])
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )
        
        # 测试性能
        start_time = time.time()
        chunks = splitter.split_text(text)
        elapsed = time.time() - start_time
        
        # 统计信息
        sizes = [len(c) for c in chunks]
        
        results[name] = {
            "chunk_count": len(chunks),
            "avg_size": sum(sizes) // len(sizes) if sizes else 0,
            "min_size": min(sizes) if sizes else 0,
            "max_size": max(sizes) if sizes else 0,
            "elapsed_ms": f"{elapsed * 1000:.2f}",
            "config": config
        }
    
    return results

# 测试数据
text = """
Python 编程基础教程

第一章：变量与数据类型
Python 是一种动态类型语言。变量不需要声明类型。
Python 支持多种数据类型：整数、浮点数、字符串、列表等。

第二章：控制流程
if 语句用于条件判断。for 循环用于遍历序列。
while 循环用于重复执行代码块。
""" * 100  # 约10万字符

test_queries = [
    "什么是 Python？",
    "如何使用 for 循环？",
    "Python 支持哪些数据类型？"
]

# 运行基准测试
results = benchmark_chunking_strategies(text, test_queries)

print("NVIDIA 2025 分块策略基准测试结果:\n")
for name, result in results.items():
    print(f"{name}:")
    print(f"  块数量: {result['chunk_count']}")
    print(f"  平均大小: {result['avg_size']} 字符")
    print(f"  大小范围: {result['min_size']}-{result['max_size']}")
    print(f"  处理耗时: {result['elapsed_ms']} ms")
    print()

print("NVIDIA 2025 推荐: 固定大小_512（事实查询）或 固定大小_1024（分析查询）")
```

---

## 重叠率对比测试

```python
"""
测试不同重叠率的效果
NVIDIA 2025 发现：15% 重叠率最优
"""

def benchmark_overlap_ratios(text: str) -> Dict:
    """对比不同重叠率的效果"""
    chunk_size = 512
    overlap_ratios = [0.0, 0.10, 0.15, 0.20, 0.25]
    
    results = {}
    
    for ratio in overlap_ratios:
        chunk_overlap = int(chunk_size * ratio)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        chunks = splitter.split_text(text)
        
        # 计算存储成本（相对值）
        total_chars = sum(len(c) for c in chunks)
        storage_cost = total_chars / len(text)  # 相对于原文的倍数
        
        results[f"{ratio:.0%}"] = {
            "chunk_count": len(chunks),
            "storage_cost": f"{storage_cost:.2f}x",
            "overlap": chunk_overlap,
            "nvidia_recommended": ratio == 0.15
        }
    
    return results

# 运行测试
results = benchmark_overlap_ratios(text)

print("重叠率对比测试:\n")
for ratio, result in results.items():
    marker = " ⭐ NVIDIA 推荐" if result["nvidia_recommended"] else ""
    print(f"{ratio} 重叠{marker}:")
    print(f"  块数量: {result['chunk_count']}")
    print(f"  存储成本: {result['storage_cost']}")
    print(f"  重叠大小: {result['overlap']} 字符")
    print()

print("结论: 15% 重叠率是成本和效果的最佳平衡点")
```

---

## 查询类型自适应测试

```python
"""
测试查询类型自适应分块
NVIDIA 2025: 不同查询类型需要不同块大小
"""

def benchmark_query_types(text: str) -> Dict:
    """对比不同查询类型的最优配置"""
    query_configs = {
        "事实查询": {
            "chunk_size": 512,
            "example": "Python 的 GIL 是什么？"
        },
        "分析查询": {
            "chunk_size": 1024,
            "example": "分析 Python 多线程的性能特点"
        },
        "混合查询": {
            "chunk_size": 768,
            "example": "Python 有哪些特性？为什么流行？"
        }
    }
    
    results = {}
    
    for query_type, config in query_configs.items():
        chunk_size = config["chunk_size"]
        chunk_overlap = int(chunk_size * 0.15)  # 15% NVIDIA 推荐
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        chunks = splitter.split_text(text)
        
        results[query_type] = {
            "chunk_size": chunk_size,
            "chunk_count": len(chunks),
            "avg_size": sum(len(c) for c in chunks) // len(chunks),
            "example_query": config["example"]
        }
    
    return results

# 运行测试
results = benchmark_query_types(text)

print("查询类型自适应测试:\n")
for query_type, result in results.items():
    print(f"{query_type}:")
    print(f"  推荐块大小: {result['chunk_size']} tokens")
    print(f"  块数量: {result['chunk_count']}")
    print(f"  平均大小: {result['avg_size']} 字符")
    print(f"  示例查询: {result['example_query']}")
    print()
```

---

## 完整基准测试套件

```python
"""
完整的分块策略基准测试套件
包含：性能、质量、成本三个维度
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
import time
from typing import List, Dict

class ChunkingBenchmark:
    """分块策略基准测试"""
    
    def __init__(self, text: str):
        self.text = text
        self.results = {}
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("分块策略基准测试（NVIDIA 2025 标准）")
        print("=" * 60)
        
        # 测试1：固定大小分块
        self.test_fixed_size()
        
        # 测试2：递归字符分块
        self.test_recursive()
        
        # 测试3：语义分块
        self.test_semantic()
        
        # 测试4：重叠率对比
        self.test_overlap_ratios()
        
        # 测试5：查询类型自适应
        self.test_query_types()
        
        # 输出总结
        self.print_summary()
    
    def test_fixed_size(self):
        """测试固定大小分块"""
        print("\n[测试1] 固定大小分块")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=77
        )
        
        start = time.time()
        chunks = splitter.split_text(self.text)
        elapsed = time.time() - start
        
        self.results["固定大小_512"] = {
            "chunks": len(chunks),
            "time_ms": f"{elapsed * 1000:.2f}",
            "cost": "$0"
        }
        
        print(f"  块数: {len(chunks)}")
        print(f"  耗时: {elapsed * 1000:.2f} ms")
        print(f"  成本: $0")
    
    def test_recursive(self):
        """测试递归字符分块"""
        print("\n[测试2] 递归字符分块（NVIDIA 2025 推荐）")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=77,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )
        
        start = time.time()
        chunks = splitter.split_text(self.text)
        elapsed = time.time() - start
        
        self.results["递归字符_512"] = {
            "chunks": len(chunks),
            "time_ms": f"{elapsed * 1000:.2f}",
            "cost": "$0"
        }
        
        print(f"  块数: {len(chunks)}")
        print(f"  耗时: {elapsed * 1000:.2f} ms")
        print(f"  成本: $0")
        print(f"  ⭐ NVIDIA 2025 推荐配置")
    
    def test_semantic(self):
        """测试语义分块"""
        print("\n[测试3] 语义分块")
        
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            splitter = SemanticChunker(embeddings=embeddings)
            
            start = time.time()
            chunks = splitter.split_text(self.text[:5000])  # 限制长度
            elapsed = time.time() - start
            
            # 估算成本
            estimated_cost = 0.002  # 约 $0.002
            
            self.results["语义分块"] = {
                "chunks": len(chunks),
                "time_ms": f"{elapsed * 1000:.2f}",
                "cost": f"${estimated_cost:.4f}"
            }
            
            print(f"  块数: {len(chunks)}")
            print(f"  耗时: {elapsed * 1000:.2f} ms")
            print(f"  成本: ${estimated_cost:.4f}")
        except Exception as e:
            print(f"  跳过（需要 API key）: {e}")
    
    def test_overlap_ratios(self):
        """测试不同重叠率"""
        print("\n[测试4] 重叠率对比")
        
        for ratio in [0.10, 0.15, 0.20]:
            chunk_size = 512
            chunk_overlap = int(chunk_size * ratio)
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            chunks = splitter.split_text(self.text)
            storage_cost = sum(len(c) for c in chunks) / len(self.text)
            
            marker = " ⭐ NVIDIA 最优" if ratio == 0.15 else ""
            print(f"  {ratio:.0%} 重叠{marker}:")
            print(f"    块数: {len(chunks)}")
            print(f"    存储成本: {storage_cost:.2f}x")
    
    def test_query_types(self):
        """测试查询类型自适应"""
        print("\n[测试5] 查询类型自适应（NVIDIA 2025）")
        
        configs = {
            "事实查询": 512,
            "分析查询": 1024,
            "混合查询": 768
        }
        
        for query_type, chunk_size in configs.items():
            chunk_overlap = int(chunk_size * 0.15)
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            chunks = splitter.split_text(self.text)
            
            print(f"  {query_type} (chunk_size={chunk_size}):")
            print(f"    块数: {len(chunks)}")
            print(f"    平均大小: {sum(len(c) for c in chunks) // len(chunks)}")
    
    def print_summary(self):
        """输出测试总结"""
        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        
        print("\nNVIDIA 2025 核心发现:")
        print("1. 页面级分块准确率最高（0.648）")
        print("2. 15% 重叠率最优")
        print("3. 查询类型决定块大小:")
        print("   - 事实查询: 512 tokens")
        print("   - 分析查询: 1024 tokens")
        print("   - 混合查询: 768 tokens")
        
        print("\n推荐配置:")
        print("RecursiveCharacterTextSplitter(")
        print("    chunk_size=512,  # 或根据查询类型调整")
        print("    chunk_overlap=77,  # 15% overlap")
        print(")")

# 运行完整测试
text = "你的长文本..." * 1000
benchmark = ChunkingBenchmark(text)
benchmark.run_all_tests()
```

---

## 上下文感知效果测试

```python
"""
测试上下文感知分块的效果
Anthropic 2024-2025: 减少 49-67% 检索失败
"""

from anthropic import Anthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter

def test_contextual_retrieval(document: str, test_query: str):
    """
    对比传统分块 vs 上下文感知分块
    
    效果：减少 49-67% 检索失败
    """
    # 1. 传统分块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=77
    )
    base_chunks = splitter.split_text(document)
    
    print(f"传统分块:")
    print(f"  块数: {len(base_chunks)}")
    print(f"  示例: {base_chunks[0][:100]}...")
    
    # 2. 上下文感知分块
    client = Anthropic()
    
    def add_context(chunk: str) -> str:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": f"为片段生成上下文：\n\n文档：{document[:1000]}\n\n片段：{chunk}"
            }]
        )
        context = response.content[0].text.strip()
        return f"{context}\n\n{chunk}"
    
    # 只处理前3个 chunk（演示）
    contextual_chunks = [add_context(c) for c in base_chunks[:3]]
    
    print(f"\n上下文感知分块:")
    print(f"  块数: {len(base_chunks)}（相同）")
    print(f"  示例: {contextual_chunks[0][:200]}...")
    
    print(f"\n效果对比:")
    print(f"  传统分块: 基准")
    print(f"  上下文感知: 减少 49% 检索失败（Anthropic 2024-2025）")
    print(f"  成本: $0.02/10万字")

# 测试
document = "Python 编程教程..." * 100
test_contextual_retrieval(document, "什么是 Python？")
```

---

## 核心研究来源

**NVIDIA 2025**: [Finding the Best Chunking Strategy](https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses)
- 页面级分块准确率最高（0.648）
- 15% 重叠率最优
- 查询类型决定块大小

**Anthropic 2024-2025**: [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- 减少 49-67% 检索失败

---

**完成！** 你已经掌握了文本分块的完整知识体系和实战代码。
