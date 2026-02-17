# 核心概念4：代理式分块（Agentic Chunking）⭐ NEW 2025-2026

**使用 LLM 像人类编辑一样理解文档结构，智能确定分块边界和元数据，是 2025-2026 年最高质量的分块策略。**

---

## 一句话定义

**代理式分块是让 LLM 作为智能代理理解文档的语义结构、主题边界和内容重要性，动态确定最佳分块边界并自动生成元数据标签的方法，比传统方法提升 15-20% 检索准确率，是 2025-2026 年质量最高但成本也最高的分块策略。**

**研究来源**: [IBM Agentic Chunking](https://www.ibm.com/think/topics/agentic-chunking)

---

## 核心原理

### 基本思想

```
传统分块：基于固定规则
→ 规则无法理解语义

代理式分块：LLM 智能理解
→ 像人类编辑一样分块

示例：
文档："Python 基础教程

第一章：变量与数据类型
Python 支持多种数据类型...

第二章：控制流程
if 语句用于条件判断..."

传统分块：
- 固定大小：可能在章节中间切断
- 递归字符：在 \n\n 处切分，但不理解章节含义

代理式分块：
- LLM 识别：这是教程文档，有章节结构
- 智能边界：在章节边界切分
- 自动元数据：{"chapter": "第一章", "topic": "变量与数据类型"}
```

### 4步工作流程

```
1. 文本准备（Text Preparation）
   - 清洗文档
   - 格式化内容
   ↓
2. 智能分块（Splitting）
   - LLM 理解文档结构
   - 识别语义边界
   - 动态确定分块点
   ↓
3. 元数据标注（Labeling）
   - LLM 为每个 chunk 生成元数据
   - 包括：主题、关键词、章节信息、重要性
   ↓
4. 向量化（Embedding）
   - 将 chunk + 元数据一起向量化
   - 增强检索效果
```

**研究来源**: [IBM Agentic Chunking](https://www.ibm.com/think/topics/agentic-chunking)

---

## Python 实现

### 基础实现

```python
from openai import OpenAI
from typing import List, Dict
import json

client = OpenAI()

def agentic_chunking(document: str, model: str = "gpt-4") -> List[Dict]:
    """
    代理式分块（IBM 2025-2026 方法）

    Args:
        document: 原始文档
        model: LLM 模型（推荐 gpt-4 或 claude-3-opus）

    Returns:
        分块结果，包含 chunk 和 metadata
    """
    prompt = f"""
你是文档分块专家。分析以下文档，确定最佳分块边界。

要求：
1. 识别文档结构（章节、段落、主题）
2. 在语义边界处切分（章节转换、主题变化）
3. 为每个 chunk 生成元数据（主题、关键词、章节信息）
4. 每个 chunk 大小控制在 300-800 tokens

文档：
{document}

返回 JSON 格式：
[
  {{
    "chunk": "chunk 内容",
    "metadata": {{
      "topic": "主题",
      "keywords": ["关键词1", "关键词2"],
      "section": "章节名称",
      "importance": "high/medium/low"
    }}
  }}
]
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)
    return result.get("chunks", [])

# 使用示例
document = """
Python 编程基础教程

第一章：变量与数据类型
Python 是一种动态类型语言。变量不需要声明类型。
Python 支持多种数据类型：整数、浮点数、字符串、列表等。

第二章：控制流程
if 语句用于条件判断。for 循环用于遍历序列。
while 循环用于重复执行代码块。
"""

chunks = agentic_chunking(document)
for i, chunk_data in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(f"内容: {chunk_data['chunk'][:100]}...")
    print(f"元数据: {chunk_data['metadata']}")
```

### 生产级实现

```python
from typing import List, Dict, Optional
import json
from openai import OpenAI

class AgenticChunker:
    """代理式分块器（生产级）"""

    def __init__(
        self,
        model: str = "gpt-4",
        target_chunk_size: int = 512,
        max_chunk_size: int = 800
    ):
        self.client = OpenAI()
        self.model = model
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk(self, document: str) -> List[Dict]:
        """执行代理式分块"""
        # 1. 分析文档结构
        structure = self._analyze_structure(document)

        # 2. 确定分块边界
        boundaries = self._determine_boundaries(document, structure)

        # 3. 生成 chunks 和元数据
        chunks = self._create_chunks(document, boundaries)

        # 4. 验证和优化
        chunks = self._validate_chunks(chunks)

        return chunks

    def _analyze_structure(self, document: str) -> Dict:
        """分析文档结构"""
        prompt = f"""
分析文档结构，识别：
1. 文档类型（教程、API文档、论文等）
2. 章节层级
3. 主题分布

文档：
{document[:2000]}

返回 JSON：
{{
  "doc_type": "文档类型",
  "sections": ["章节1", "章节2"],
  "topics": ["主题1", "主题2"]
}}
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def _determine_boundaries(
        self,
        document: str,
        structure: Dict
    ) -> List[int]:
        """确定分块边界"""
        prompt = f"""
根据文档结构确定最佳分块边界。

文档结构：{json.dumps(structure, ensure_ascii=False)}
目标块大小：{self.target_chunk_size} tokens

文档：
{document}

返回边界位置（字符索引）：
{{"boundaries": [0, 150, 300, ...]}}
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("boundaries", [])

    def _create_chunks(
        self,
        document: str,
        boundaries: List[int]
    ) -> List[Dict]:
        """根据边界创建 chunks 并生成元数据"""
        chunks = []

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            chunk_text = document[start:end].strip()

            # 为每个 chunk 生成元数据
            metadata = self._generate_metadata(chunk_text, document)

            chunks.append({
                "chunk": chunk_text,
                "metadata": metadata,
                "start": start,
                "end": end
            })

        return chunks

    def _generate_metadata(
        self,
        chunk: str,
        full_document: str
    ) -> Dict:
        """为 chunk 生成元数据"""
        prompt = f"""
为文档片段生成元数据。

完整文档开头：{full_document[:1000]}
片段：{chunk}

返回 JSON：
{{
  "topic": "主题",
  "keywords": ["关键词1", "关键词2"],
  "section": "章节名称",
  "importance": "high/medium/low",
  "summary": "一句话概括"
}}
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # 使用便宜的模型
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=200
        )
        return json.loads(response.choices[0].message.content)

    def _validate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """验证和优化 chunks"""
        validated = []

        for chunk_data in chunks:
            chunk = chunk_data["chunk"]

            # 检查大小
            if len(chunk) < 100:
                # 太小，合并到前一个
                if validated:
                    validated[-1]["chunk"] += "\n\n" + chunk
                continue

            if len(chunk) > self.max_chunk_size:
                # 太大，需要进一步切分
                # 这里简化处理，实际应该递归调用
                continue

            validated.append(chunk_data)

        return validated

# 使用示例
chunker = AgenticChunker(model="gpt-4", target_chunk_size=512)
chunks = chunker.chunk(document)

for i, chunk_data in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(f"内容: {chunk_data['chunk'][:100]}...")
    print(f"主题: {chunk_data['metadata']['topic']}")
    print(f"关键词: {chunk_data['metadata']['keywords']}")
    print(f"重要性: {chunk_data['metadata']['importance']}")
```

---

## 2025-2026 优化方案

### 优化1：多模型协作

```python
def multi_model_agentic_chunking(document: str) -> List[Dict]:
    """
    多模型协作分块
    - GPT-4: 结构分析
    - Claude: 边界确定
    - GPT-4o-mini: 元数据生成
    """
    # 1. GPT-4 分析结构（最强理解能力）
    structure = analyze_with_gpt4(document)

    # 2. Claude 确定边界（最佳推理能力）
    boundaries = determine_with_claude(document, structure)

    # 3. GPT-4o-mini 生成元数据（成本优化）
    chunks = create_chunks_with_mini(document, boundaries)

    return chunks
```

### 优化2：增量分块（大文档优化）

```python
def incremental_agentic_chunking(
    document: str,
    window_size: int = 5000
) -> List[Dict]:
    """
    增量代理式分块（处理超长文档）

    原理：
    1. 将文档分成多个窗口
    2. 每个窗口独立进行代理式分块
    3. 合并相邻窗口的边界
    """
    chunks = []
    for i in range(0, len(document), window_size):
        window = document[i:i+window_size*2]  # 重叠窗口
        window_chunks = agentic_chunking(window)
        chunks.extend(window_chunks)

    # 合并重叠部分
    chunks = merge_overlapping_chunks(chunks)
    return chunks
```

### 优化3：缓存优化（成本控制）

```python
import hashlib
from pathlib import Path

class CachedAgenticChunker:
    """带缓存的代理式分块器"""

    def __init__(self, cache_dir: str = ".agentic_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.chunker = AgenticChunker()

    def chunk(self, document: str) -> List[Dict]:
        """带缓存的分块"""
        # 计算文档哈希
        doc_hash = hashlib.md5(document.encode()).hexdigest()
        cache_file = self.cache_dir / f"{doc_hash}.json"

        # 检查缓存
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)

        # 执行分块
        chunks = self.chunker.chunk(document)

        # 保存缓存
        with open(cache_file, 'w') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        return chunks
```

---

## 效果与成本分析

### 效果提升

**IBM 2025-2026 研究数据：**

| 指标 | 传统分块 | 代理式分块 | 提升 |
|------|----------|-----------|------|
| **检索准确率** | 0.65 | 0.78 | +15-20% |
| **召回率** | 0.70 | 0.85 | +21% |
| **语义完整性** | 中 | 高 | +40% |
| **元数据质量** | 无 | 高 | N/A |

**研究来源**: [IBM Agentic Chunking](https://www.ibm.com/think/topics/agentic-chunking)

### 成本分析

**假设：10万字文档（约 25,000 tokens）**

| 步骤 | 模型 | Tokens | 成本 |
|------|------|--------|------|
| 结构分析 | GPT-4 | 2,000 | $0.06 |
| 边界确定 | GPT-4 | 25,000 | $0.75 |
| 元数据生成（10个chunk） | GPT-4o-mini | 5,000 | $0.01 |
| **总计** | - | 32,000 | **$0.82** |

**成本对比：**
- 递归分块：$0（无 API 调用）
- 语义分块：$0.002（embedding）
- **代理式分块：$0.82**（LLM 调用）
- 上下文感知：$0.10（轻量 LLM）

**结论**: 代理式分块成本最高，但质量也最高。适合高价值文档。

---

## 优缺点分析

### 优点

| 优点 | 说明 | 适用场景 |
|------|------|---------|
| ✅ **质量最高** | 提升 15-20% 准确率 | 关键业务场景 |
| ✅ **智能理解** | LLM 理解文档结构和语义 | 复杂文档 |
| ✅ **自动元数据** | 无需手动标注 | 大规模文档库 |
| ✅ **适应性强** | 自动适应不同文档类型 | 多种格式 |
| ✅ **边界最优** | 在最佳语义边界切分 | 高质量要求 |

### 缺点

| 缺点 | 说明 | 影响 |
|------|------|------|
| ❌ **成本最高** | LLM 调用成本高 | 预算限制 |
| ❌ **速度最慢** | API 调用耗时 | 不适合实时 |
| ❌ **依赖 LLM** | 需要高质量 LLM（GPT-4/Claude） | 技术门槛 |
| ❌ **不可控性** | LLM 输出可能不稳定 | 需要验证 |

---

## 在 RAG 开发中的应用

### 适用场景（✅ 强烈推荐）

1. **高价值文档**
   - 医疗、法律、金融文档
   - 企业核心知识库
   - 示例：医疗诊断指南、法律合同库

2. **复杂结构文档**
   - 多层级技术文档
   - 学术论文、研究报告
   - 示例：API 文档、技术规范

3. **需要元数据的场景**
   - 需要按主题、章节检索
   - 需要重要性排序
   - 示例：企业知识管理系统

4. **预算充足的项目**
   - 可以承担 LLM 成本
   - 追求最佳效果
   - 示例：企业级 RAG 系统

### 不适用场景（❌ 不推荐）

1. **大规模文档处理**
   - 成本过高
   - 处理时间长
   - 示例：TB 级文档库

2. **实时处理场景**
   - API 延迟高
   - 无法满足实时要求
   - 示例：实时聊天机器人

3. **预算有限的项目**
   - LLM 成本不可接受
   - 示例：个人项目、MVP

4. **简单文档**
   - 结构简单，递归分块已足够
   - 示例：纯文本文章

---

## 最佳实践

### 实践1：混合策略（成本优化）

```python
def hybrid_chunking_strategy(document: str, doc_type: str) -> List[Dict]:
    """
    混合分块策略：
    - 重要文档：代理式分块
    - 普通文档：递归分块 + 上下文感知
    """
    if doc_type in ["medical", "legal", "financial"]:
        # 高价值：代理式分块
        return agentic_chunking(document)
    elif doc_type in ["technical", "academic"]:
        # 中等价值：递归 + 上下文
        return recursive_with_context(document)
    else:
        # 普通文档：递归分块
        return recursive_chunking(document)
```

### 实践2：批量处理优化

```python
async def batch_agentic_chunking(
    documents: List[str],
    batch_size: int = 5
) -> List[List[Dict]]:
    """批量代理式分块（并发优化）"""
    import asyncio

    async def chunk_one(doc: str) -> List[Dict]:
        return agentic_chunking(doc)

    results = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_results = await asyncio.gather(*[chunk_one(doc) for doc in batch])
        results.extend(batch_results)

    return results
```

### 实践3：质量验证

```python
def validate_agentic_chunks(chunks: List[Dict]) -> bool:
    """验证代理式分块质量"""
    # 1. 检查元数据完整性
    for chunk_data in chunks:
        if not chunk_data.get("metadata"):
            return False
        metadata = chunk_data["metadata"]
        if not all(k in metadata for k in ["topic", "keywords"]):
            return False

    # 2. 检查块大小分布
    sizes = [len(c["chunk"]) for c in chunks]
    avg_size = sum(sizes) / len(sizes)
    if avg_size < 200 or avg_size > 1000:
        return False

    # 3. 检查语义完整性
    for chunk_data in chunks:
        chunk = chunk_data["chunk"]
        if not chunk.strip().endswith(('.', '。', '!', '！', '?', '？')):
            # 可能切断句子
            pass  # 警告但不失败

    return True
```

---

## 与其他策略对比

### 代理式 vs 递归字符

| 特性 | 代理式分块 | 递归字符 |
|------|-----------|----------|
| **质量** | ✅✅✅ 最高 | ✅ 好 |
| **成本** | ❌ 高（$0.82/10万字） | ✅ 无 |
| **速度** | ❌ 慢 | ✅ 快 |
| **元数据** | ✅ 自动生成 | ❌ 无 |
| **推荐场景** | 高价值文档 | **通用场景** |

### 代理式 vs 语义分块

| 特性 | 代理式分块 | 语义分块 |
|------|-----------|----------|
| **质量** | ✅✅✅ 最高 | ✅✅ 很好 |
| **成本** | ❌ 高（LLM） | ❌ 中（embedding） |
| **智能程度** | ✅ 理解结构 | ❌ 仅相似度 |
| **元数据** | ✅ 自动生成 | ❌ 无 |
| **推荐场景** | 复杂文档 | 主题多变文档 |

### 代理式 vs 上下文感知

| 特性 | 代理式分块 | 上下文感知 |
|------|-----------|-----------|
| **质量** | ✅✅✅ 最高 | ✅✅ 很好 |
| **成本** | ❌ 高 | ❌ 中 |
| **分块边界** | ✅ 智能确定 | ❌ 依赖基础分块 |
| **元数据** | ✅ 丰富 | ✅ 上下文 |
| **推荐场景** | 复杂文档 | 生产环境优化 |

---

## 实战示例

### 示例1：处理技术文档

```python
# 技术文档代理式分块
tech_doc = """
API 文档

## 认证
使用 JWT token 进行认证...

## 端点列表
### GET /users
获取用户列表...

### POST /users
创建新用户...
"""

chunker = AgenticChunker(model="gpt-4")
chunks = chunker.chunk(tech_doc)

# 结果示例
# Chunk 1: {"chunk": "## 认证\n使用 JWT...", "metadata": {"section": "认证", "topic": "JWT认证"}}
# Chunk 2: {"chunk": "### GET /users...", "metadata": {"section": "端点列表", "topic": "用户查询"}}
```

### 示例2：医疗文档分块

```python
# 医疗文档（高价值场景）
medical_doc = """
糖尿病诊疗指南

第一章：诊断标准
空腹血糖 ≥7.0 mmol/L...

第二章：治疗方案
一线药物：二甲双胍...
"""

chunks = agentic_chunking(medical_doc, model="gpt-4")

# 自动生成的元数据
# {"importance": "high", "keywords": ["糖尿病", "诊断标准", "血糖"]}
```

---

## 常见问题

### Q1: 代理式分块适合生产环境吗？

**A**: ✅ **适合高价值场景**。

- 医疗、法律、金融等关键领域
- 企业核心知识库
- 需要最高质量的场景

**不适合**：成本敏感、大规模处理、实时场景

### Q2: 如何降低成本？

**A**: 3个策略：

1. **混合策略**：重要文档用代理式，普通文档用递归
2. **缓存**：相同文档不重复处理
3. **模型选择**：结构分析用 GPT-4，元数据用 GPT-4o-mini

### Q3: 如何保证质量稳定性？

**A**: 4个方法：

1. **Prompt 工程**：精心设计 prompt
2. **结构化输出**：使用 JSON mode
3. **验证机制**：检查输出质量
4. **多次采样**：temperature=0 提高稳定性

### Q4: 与上下文感知分块如何选择？

**A**: 根据需求选择：

- **代理式**：需要智能边界 + 元数据
- **上下文感知**：已有好的基础分块，只需增强上下文
- **组合使用**：代理式分块 + 上下文感知（最佳效果）

---

## 核心研究来源

1. **IBM 2025-2026**: [Agentic Chunking](https://www.ibm.com/think/topics/agentic-chunking)
   - 提升 15-20% 检索准确率
   - 4步工作流程
   - 自动元数据生成

2. **NVIDIA 2025**: 验证智能分块的有效性

3. **Anthropic 2024-2025**: 可与上下文感知结合使用

---

## 下一步学习

**同级概念：**
- ← [03_核心概念_03_语义分块](./03_核心概念_03_语义分块.md) - 基于相似度
- → [03_核心概念_05_上下文感知分块Contextual](./03_核心概念_05_上下文感知分块Contextual.md) - Anthropic 方法

**实战代码：**
- → [07_实战代码_04_代理式分块实现](./07_实战代码_04_代理式分块实现.md) - 完整实现代码

**对比分析：**
- → [06_反直觉点](./06_反直觉点.md) - 代理式分块的常见误区
