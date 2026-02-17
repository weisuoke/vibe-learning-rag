# 核心概念4：引用溯源（Attribution）

> **为每个生成声明提供源文档引用，提升RAG系统的可信度和透明度**

---

## 概念定义

**引用溯源（Attribution）**：为生成内容中的每个声明标注其来源文档的位置，使用户可以追溯到原始信息源，验证声明的真实性。

**核心目标：**
- **可追溯性**：用户可以点击引用查看原始文档
- **可验证性**：用户可以验证声明是否被源文档支持
- **透明度**：明确展示信息来源，增强信任

---

## 为什么需要引用溯源？

### 问题：缺乏引用的风险

```
生成内容："Python由Guido van Rossum于1991年创建。"

用户疑问：
- 这个信息从哪里来的？
- 我如何验证这是真的？
- 如果有错误，责任在谁？

解决方案：添加引用
"Python由Guido van Rossum于1991年创建[1]。"

[1] Python官方文档，https://docs.python.org/3/faq/general.html
```

### 引用溯源的价值

| 维度 | 无引用 | 有引用 |
|------|--------|--------|
| **用户信任度** | 低（无法验证） | 高（可追溯） |
| **法律合规** | 风险高 | 符合要求 |
| **调试效率** | 难以定位错误来源 | 快速定位问题 |
| **用户体验** | 被动接受 | 主动验证 |

---

## 引用溯源的实现方法

### 方法1：语义相似度匹配（推荐）

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def find_supporting_context(claim: str, contexts: list[str], threshold: float = 0.7) -> dict:
    """为声明找到最相似的上下文"""

    # 编码声明和上下文
    claim_emb = model.encode(claim, convert_to_tensor=True)
    context_embs = model.encode(contexts, convert_to_tensor=True)

    # 计算相似度
    similarities = util.cos_sim(claim_emb, context_embs)[0]

    # 找到最相似的上下文
    best_idx = similarities.argmax().item()
    best_score = similarities[best_idx].item()

    if best_score >= threshold:
        return {
            "supported": True,
            "source_index": best_idx,
            "source_text": contexts[best_idx],
            "similarity": best_score
        }
    else:
        return {
            "supported": False,
            "similarity": best_score
        }

# 使用示例
claim = "Python由Guido van Rossum创建"
contexts = [
    "Python是一种高级编程语言。",
    "Python由Guido van Rossum于1991年创建。",
    "Python广泛用于数据科学。"
]

result = find_supporting_context(claim, contexts)
if result["supported"]:
    print(f"✅ 声明被支持")
    print(f"   来源: {result['source_text']}")
    print(f"   相似度: {result['similarity']:.2f}")
```

### 方法2：精确匹配

```python
def exact_match_attribution(claim: str, contexts: list[str]) -> dict:
    """精确匹配：查找包含声明的上下文"""

    claim_lower = claim.lower()

    for idx, context in enumerate(contexts):
        if claim_lower in context.lower():
            return {
                "supported": True,
                "source_index": idx,
                "source_text": context,
                "match_type": "exact"
            }

    return {"supported": False}

# 使用示例
claim = "Python由Guido创建"
contexts = [
    "Python是一种高级编程语言。",
    "Python由Guido van Rossum创建。",
    "Python广泛用于数据科学。"
]

result = exact_match_attribution(claim, contexts)
if result["supported"]:
    print(f"✅ 精确匹配")
    print(f"   来源: {result['source_text']}")
```

### 方法3：NLI验证 + 溯源

```python
from transformers import pipeline

nli_model = pipeline("text-classification", model="microsoft/deberta-v3-base-mnli-fever-anli")

def nli_based_attribution(claim: str, contexts: list[str]) -> dict:
    """基于NLI的引用溯源"""

    best_result = {
        "supported": False,
        "best_score": 0.0,
        "source_index": -1
    }

    for idx, context in enumerate(contexts):
        # NLI验证
        input_text = f"{context} [SEP] {claim}"
        result = nli_model(input_text)[0]

        if result["label"] == "entailment" and result["score"] > best_result["best_score"]:
            best_result = {
                "supported": True,
                "best_score": result["score"],
                "source_index": idx,
                "source_text": context
            }

    return best_result

# 使用示例
result = nli_based_attribution(claim, contexts)
if result["supported"]:
    print(f"✅ NLI验证通过")
    print(f"   来源: {result['source_text']}")
    print(f"   置信度: {result['best_score']:.2f}")
```

---

## 引用格式

### 格式1：行内引用（推荐）

```python
def format_inline_citation(claims: list[dict]) -> str:
    """生成行内引用格式"""

    cited_text = ""
    citations = []

    for i, claim_info in enumerate(claims):
        if claim_info["supported"]:
            citation_num = claim_info["source_index"] + 1
            cited_text += f"{claim_info['claim']}[{citation_num}]"
            citations.append({
                "num": citation_num,
                "text": claim_info["source_text"]
            })
        else:
            # 未被支持的声明不添加引用
            cited_text += f"[未验证：{claim_info['claim']}]"

    # 生成引用列表
    citation_list = "\n\n引用来源：\n"
    for cite in citations:
        citation_list += f"[{cite['num']}] {cite['text']}\n"

    return cited_text + citation_list

# 使用示例
claims = [
    {"claim": "Python是高级语言", "supported": True, "source_index": 0, "source_text": "Python是一种高级编程语言。"},
    {"claim": "Python由Guido创建", "supported": True, "source_index": 1, "source_text": "Python由Guido van Rossum创建。"},
    {"claim": "Python用于AI开发", "supported": False}
]

result = format_inline_citation(claims)
print(result)
```

**输出：**
```
Python是高级语言[1]Python由Guido创建[2][未验证：Python用于AI开发]

引用来源：
[1] Python是一种高级编程语言。
[2] Python由Guido van Rossum创建。
```

### 格式2：脚注引用

```python
def format_footnote_citation(claims: list[dict]) -> str:
    """生成脚注引用格式"""

    cited_text = ""
    footnotes = []

    for i, claim_info in enumerate(claims):
        if claim_info["supported"]:
            footnote_num = len(footnotes) + 1
            cited_text += f"{claim_info['claim']}^{footnote_num}"
            footnotes.append({
                "num": footnote_num,
                "text": claim_info["source_text"],
                "url": claim_info.get("source_url", "")
            })

    # 生成脚注
    footnote_list = "\n\n---\n"
    for fn in footnotes:
        if fn["url"]:
            footnote_list += f"^{fn['num']} {fn['text']} ({fn['url']})\n"
        else:
            footnote_list += f"^{fn['num']} {fn['text']}\n"

    return cited_text + footnote_list
```

### 格式3：Span-level引用（高级）

```python
def format_span_citation(text: str, attributions: list[dict]) -> str:
    """生成Span-level引用（HTML格式）"""

    html = "<div>"

    for attr in attributions:
        span_text = attr["span_text"]
        source_idx = attr["source_index"]

        # 替换为带引用的span
        cited_span = f'<span data-source="{source_idx}" class="cited">{span_text}</span>'
        html = html.replace(span_text, cited_span)

    html += "</div>"

    # 添加引用列表
    html += "\n<div class='citations'>"
    for i, source in enumerate(attr["sources"]):
        html += f"<div id='source-{i}'>[{i+1}] {source}</div>"
    html += "</div>"

    return html
```

---

## 在RAG中的集成

### 完整引用溯源流程

```python
def rag_with_attribution(question: str) -> dict:
    """集成引用溯源的RAG系统"""

    # 1. 检索
    contexts = retrieve_contexts(question)

    # 2. 生成
    answer = generate_answer(question, contexts)

    # 3. 声明分解
    claims = decompose_claims(answer)

    # 4. 为每个声明找到引用
    attributed_claims = []
    for claim in claims:
        attribution = find_supporting_context(claim, contexts)

        attributed_claims.append({
            "claim": claim,
            "supported": attribution["supported"],
            "source_index": attribution.get("source_index", -1),
            "source_text": attribution.get("source_text", ""),
            "similarity": attribution.get("similarity", 0.0)
        })

    # 5. 生成带引用的回答
    cited_answer = format_inline_citation(attributed_claims)

    # 6. 计算支持率
    support_ratio = sum(1 for c in attributed_claims if c["supported"]) / len(attributed_claims)

    return {
        "answer": cited_answer,
        "original_answer": answer,
        "support_ratio": support_ratio,
        "attributed_claims": attributed_claims,
        "contexts": contexts
    }
```

---

## 2026年最新技术

### 1. 统一归因管道（arXiv 2601.19927, 2026）

```python
from attribution_pipeline import UnifiedAttributionPipeline

pipeline = UnifiedAttributionPipeline()

def unified_attribution(response: str, contexts: list[str]) -> dict:
    """使用统一归因管道"""

    result = pipeline.attribute(
        generated_text=response,
        source_documents=contexts,
        granularity="claim",  # 声明级归因
        method="hybrid"  # 混合方法（语义相似度 + NLI）
    )

    return {
        "attributed_text": result["attributed_text"],
        "attributions": result["attributions"],
        "confidence_scores": result["confidence_scores"]
    }
```

### 2. 多源引用

```python
def multi_source_attribution(claim: str, contexts: list[str], top_k: int = 3) -> dict:
    """为一个声明找到多个支持来源"""

    model = SentenceTransformer('all-MiniLM-L6-v2')

    claim_emb = model.encode(claim, convert_to_tensor=True)
    context_embs = model.encode(contexts, convert_to_tensor=True)

    similarities = util.cos_sim(claim_emb, context_embs)[0]

    # 找到Top-K最相似的上下文
    top_k_indices = similarities.argsort(descending=True)[:top_k]

    sources = []
    for idx in top_k_indices:
        if similarities[idx] > 0.7:  # 阈值
            sources.append({
                "source_index": idx.item(),
                "source_text": contexts[idx.item()],
                "similarity": similarities[idx].item()
            })

    return {
        "claim": claim,
        "sources": sources,
        "num_sources": len(sources)
    }

# 使用示例
claim = "Python是一种高级编程语言"
contexts = [
    "Python是一种高级编程语言，易于学习。",
    "Python是高级语言，广泛用于数据科学。",
    "Python语法简洁，是高级编程语言。"
]

result = multi_source_attribution(claim, contexts, top_k=3)
print(f"声明：{result['claim']}")
print(f"找到{result['num_sources']}个支持来源：")
for i, source in enumerate(result['sources'], 1):
    print(f"  [{i}] {source['source_text']} (相似度: {source['similarity']:.2f})")
```

### 3. 交互式引用

```python
def interactive_citation(attributed_claims: list[dict]) -> str:
    """生成交互式引用（Markdown格式）"""

    markdown = ""

    for claim_info in attributed_claims:
        if claim_info["supported"]:
            source_idx = claim_info["source_index"]
            claim = claim_info["claim"]

            # 生成可点击的引用链接
            markdown += f"{claim} [[{source_idx+1}]](#source-{source_idx})\n\n"

    # 添加引用列表（带锚点）
    markdown += "\n---\n\n## 引用来源\n\n"
    for i, claim_info in enumerate(attributed_claims):
        if claim_info["supported"]:
            markdown += f"<a id='source-{i}'></a>\n"
            markdown += f"**[{i+1}]** {claim_info['source_text']}\n\n"

    return markdown
```

---

## 用户体验优化

### 1. 悬浮预览

```javascript
// 前端实现：鼠标悬停显示源文档片段
document.querySelectorAll('.citation').forEach(cite => {
    cite.addEventListener('mouseover', (e) => {
        const sourceId = e.target.dataset.source;
        const sourceText = document.getElementById(`source-${sourceId}`).textContent;

        // 显示悬浮窗
        showTooltip(e.target, sourceText);
    });
});
```

### 2. 高亮匹配

```python
def highlight_supporting_text(claim: str, source_text: str) -> str:
    """在源文档中高亮支持该声明的部分"""

    # 使用模糊匹配找到最相关的片段
    from difflib import SequenceMatcher

    claim_words = claim.lower().split()
    source_words = source_text.lower().split()

    # 找到最长公共子序列
    matcher = SequenceMatcher(None, claim_words, source_words)
    match = matcher.find_longest_match(0, len(claim_words), 0, len(source_words))

    if match.size > 0:
        # 高亮匹配部分
        highlighted = source_text
        start_idx = sum(len(w) + 1 for w in source_words[:match.b])
        end_idx = start_idx + sum(len(w) + 1 for w in source_words[match.b:match.b + match.size])

        highlighted = (
            source_text[:start_idx] +
            f"**{source_text[start_idx:end_idx]}**" +
            source_text[end_idx:]
        )

        return highlighted

    return source_text
```

### 3. 置信度显示

```python
def format_citation_with_confidence(claim_info: dict) -> str:
    """显示引用的置信度"""

    claim = claim_info["claim"]
    source_idx = claim_info["source_index"]
    confidence = claim_info["similarity"]

    # 根据置信度选择图标
    if confidence > 0.9:
        icon = "🟢"  # 高置信度
    elif confidence > 0.7:
        icon = "🟡"  # 中置信度
    else:
        icon = "🔴"  # 低置信度

    return f"{claim} [{source_idx+1}] {icon} {confidence:.0%}"

# 使用示例
claim_info = {
    "claim": "Python由Guido创建",
    "source_index": 1,
    "similarity": 0.95
}

print(format_citation_with_confidence(claim_info))
# 输出: Python由Guido创建 [2] 🟢 95%
```

---

## 常见问题

### Q1: 如何处理一个声明需要多个来源支持的情况？

**A:** 使用多源引用

```python
def multi_source_citation(claim: str, contexts: list[str]) -> str:
    """为一个声明添加多个引用"""

    sources = find_all_supporting_contexts(claim, contexts, threshold=0.7)

    if len(sources) == 0:
        return f"[未验证：{claim}]"
    elif len(sources) == 1:
        return f"{claim}[{sources[0]['index']+1}]"
    else:
        # 多个来源
        indices = ",".join(str(s['index']+1) for s in sources)
        return f"{claim}[{indices}]"

# 示例
claim = "Python是高级语言"
contexts = [
    "Python是一种高级编程语言。",
    "Python是高级语言，易于学习。",
    "Python属于高级编程语言类别。"
]

result = multi_source_citation(claim, contexts)
print(result)
# 输出: Python是高级语言[1,2,3]
```

### Q2: 如何处理部分支持的情况？

**A:** 标注支持程度

```python
def partial_support_citation(claim: str, context: str, similarity: float) -> str:
    """标注部分支持的声明"""

    if similarity > 0.9:
        support_level = "完全支持"
    elif similarity > 0.7:
        support_level = "部分支持"
    else:
        support_level = "弱支持"

    return f"{claim} [{support_level}, 相似度: {similarity:.0%}]"
```

### Q3: 引用溯源的性能开销如何优化？

**A:** 3种优化策略

1. **批量Embedding**
   ```python
   # 一次性编码所有声明和上下文
   claim_embs = model.encode(claims)
   context_embs = model.encode(contexts)

   # 批量计算相似度
   similarities = util.cos_sim(claim_embs, context_embs)
   ```

2. **缓存Embedding**
   ```python
   # 缓存上下文的Embedding
   context_emb_cache = {}

   def get_context_embedding(context: str):
       if context not in context_emb_cache:
           context_emb_cache[context] = model.encode(context)
       return context_emb_cache[context]
   ```

3. **异步处理**
   ```python
   import asyncio

   async def async_attribution(claims: list[str], contexts: list[str]):
       tasks = [
           find_supporting_context_async(claim, contexts)
           for claim in claims
       ]
       return await asyncio.gather(*tasks)
   ```

---

## 实际案例

### 案例1：学术论文引用

```python
# 场景：学术问答系统
question = "什么是深度学习？"
contexts = [
    "深度学习是机器学习的一个分支，使用多层神经网络。（LeCun et al., Nature 2015）",
    "深度学习在图像识别中取得了突破性进展。（Krizhevsky et al., NIPS 2012）"
]

answer = "深度学习是机器学习的一个分支，使用多层神经网络，在图像识别中取得了突破性进展。"

# 引用溯源
claims = [
    "深度学习是机器学习的一个分支",
    "深度学习使用多层神经网络",
    "深度学习在图像识别中取得了突破性进展"
]

attributed_answer = """
深度学习是机器学习的一个分支[1]，使用多层神经网络[1]，在图像识别中取得了突破性进展[2]。

引用来源：
[1] LeCun et al., "Deep learning", Nature 2015
[2] Krizhevsky et al., "ImageNet Classification with Deep CNNs", NIPS 2012
"""
```

### 案例2：医疗咨询引用

```python
# 场景：医疗问答系统
question = "感冒应该如何治疗？"
contexts = [
    "感冒的常见治疗方法包括多喝水、休息和服用退烧药。（《感冒诊疗指南2025版》第5页）",
    "感冒患者应避免剧烈运动。（《家庭医疗手册》第120页）"
]

answer = "感冒的治疗方法包括多喝水、休息、服用退烧药，并避免剧烈运动。"

# 引用溯源（医疗场景要求更严格）
attributed_answer = """
感冒的治疗方法包括：
- 多喝水[1]
- 休息[1]
- 服用退烧药[1]
- 避免剧烈运动[2]

引用来源：
[1] 《感冒诊疗指南2025版》第5页
[2] 《家庭医疗手册》第120页

⚠️ 以上信息仅供参考，请咨询专业医生。
"""
```

---

## 学习资源

### 论文

- **Attribution Techniques for RAG Systems** (arXiv 2601.19927, 2026)
- **Faithful Attribution in RAG** (2025)
- **Span-level Attribution for LLMs** (2024)

### 工具

- **LangChain Attribution**: https://python.langchain.com/docs/modules/chains/additional/attribution
- **LlamaIndex Citation**: https://docs.llamaindex.ai/en/stable/examples/query_engine/citation_query_engine.html

### 最佳实践

- **Datadog LLM Observability** (2025)
- **AWS RAG Attribution Guide** (2025)

---

**记住：引用溯源不仅是技术要求，更是提升用户信任和系统透明度的关键。在医疗、法律等高风险场景中，引用溯源是必需的。**
