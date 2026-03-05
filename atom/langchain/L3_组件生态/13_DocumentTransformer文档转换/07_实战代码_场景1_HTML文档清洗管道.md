# 实战代码 - 场景1：HTML 文档清洗管道

**生成时间：** 2026-02-27
**知识点层级：** L3_组件生态
**知识点编号：** 13
**场景编号：** 1

---

## 场景描述

### 业务背景

构建一个企业知识库系统，需要从公司内部网站、技术博客、产品文档等 HTML 页面中提取有效内容。
原始 HTML 包含大量导航栏、广告、脚本标签等噪声，直接用于 RAG 检索会严重影响质量。

### 技术挑战

1. **HTML 噪声多**：导航、侧边栏、页脚、脚本等占内容 60%+
2. **格式不统一**：不同页面结构差异大
3. **需要保留语义**：标题层级、列表结构、代码块等需要保留
4. **管道化处理**：清洗 → 转换 → 分块需要串联

### 学习目标

- 使用 `BeautifulSoupTransformer` 提取特定 HTML 标签
- 使用 `Html2TextTransformer` 将 HTML 转为纯文本
- 将清洗后的文档与 `TextSplitter` 串联
- 对比清洗前后的文档质量

[来源: reference/source_document_transformer_01.md | 源码分析]
[来源: reference/context7_langchain_01.md | LangChain 官方文档]

---

## 环境准备

### 依赖安装

```bash
# 安装依赖
pip install langchain langchain-community langchain-text-splitters beautifulsoup4 html2text lxml
```

### 导入模块

```python
"""
HTML 文档清洗管道实战
演示：从网页加载 HTML → BeautifulSoup 清洗 → Html2Text 转换 → 文本分块
"""

from langchain_core.documents import Document
from langchain_community.document_transformers import (
    BeautifulSoupTransformer,
    Html2TextTransformer,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

---

## 第一步：准备模拟 HTML 数据

> 使用内嵌 HTML 字符串模拟真实网页，无需网络访问即可运行。

```python
# ============================================================
# 模拟 HTML 数据 —— 一个典型的技术博客页面
# ============================================================

MOCK_HTML_BLOG = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>RAG 架构入门指南 - TechBlog</title>
    <script src="analytics.js"></script>
    <style>
        body { font-family: Arial; }
        .nav { background: #333; }
        .sidebar { width: 300px; }
    </style>
</head>
<body>
    <!-- 导航栏 —— 噪声 -->
    <nav class="nav">
        <ul>
            <li><a href="/">首页</a></li>
            <li><a href="/blog">博客</a></li>
            <li><a href="/about">关于</a></li>
            <li><a href="/contact">联系我们</a></li>
        </ul>
    </nav>

    <!-- 主内容区 —— 有效内容 -->
    <main>
        <article>
            <h1>RAG 架构入门指南</h1>
            <p class="meta">作者：张三 | 发布时间：2026-01-15</p>

            <h2>什么是 RAG？</h2>
            <p>RAG（Retrieval-Augmented Generation）是一种将<strong>检索</strong>与<strong>生成</strong>
            结合的技术架构。它通过从外部知识库中检索相关文档，然后将这些文档作为上下文
            提供给大语言模型，从而生成更准确、更有依据的回答。</p>

            <h2>RAG 的核心组件</h2>
            <p>一个完整的 RAG 系统包含以下核心组件：</p>
            <ul>
                <li><strong>文档加载器</strong>：从各种数据源加载原始文档</li>
                <li><strong>文本分块器</strong>：将长文档切分为适合检索的小块</li>
                <li><strong>嵌入模型</strong>：将文本转换为向量表示</li>
                <li><strong>向量存储</strong>：存储和检索向量化的文档</li>
                <li><strong>检索器</strong>：根据查询找到最相关的文档</li>
                <li><strong>生成器</strong>：基于检索结果生成最终回答</li>
            </ul>

            <h2>代码示例</h2>
            <p>下面是一个最简单的 RAG 实现：</p>
            <pre><code class="python">
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template(
    "根据以下上下文回答问题：\\n{context}\\n\\n问题：{question}"
)
            </code></pre>

            <h2>总结</h2>
            <p>RAG 是当前最实用的 LLM 应用架构之一。通过合理的文档处理管道，
            可以显著提升生成质量，减少幻觉问题。</p>
        </article>
    </main>

    <!-- 侧边栏 —— 噪声 -->
    <aside class="sidebar">
        <h3>热门文章</h3>
        <ul>
            <li><a href="/post/1">LangChain 入门</a></li>
            <li><a href="/post/2">向量数据库对比</a></li>
            <li><a href="/post/3">Prompt 工程技巧</a></li>
        </ul>
        <h3>广告</h3>
        <div class="ad">购买我们的 AI 课程，限时优惠！</div>
    </aside>

    <!-- 页脚 —— 噪声 -->
    <footer>
        <p>© 2026 TechBlog. All rights reserved.</p>
        <p>京ICP备12345678号</p>
    </footer>

    <script>
        // Google Analytics 跟踪代码
        (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;})
    </script>
</body>
</html>
"""

# 模拟第二个页面 —— 产品文档
MOCK_HTML_DOCS = """
<!DOCTYPE html>
<html>
<head><title>API 文档 - 向量检索接口</title></head>
<body>
    <nav><a href="/">Home</a> | <a href="/docs">Docs</a></nav>

    <div class="content">
        <h1>向量检索 API</h1>

        <h2>接口说明</h2>
        <p>本接口用于在向量数据库中执行相似度检索。支持余弦相似度、
        欧氏距离和内积三种度量方式。</p>

        <h2>请求参数</h2>
        <table>
            <tr><th>参数名</th><th>类型</th><th>说明</th></tr>
            <tr><td>query</td><td>string</td><td>查询文本</td></tr>
            <tr><td>top_k</td><td>int</td><td>返回结果数量，默认 5</td></tr>
            <tr><td>threshold</td><td>float</td><td>相似度阈值，默认 0.7</td></tr>
        </table>

        <h2>响应示例</h2>
        <pre><code>
{
    "results": [
        {"id": "doc_001", "score": 0.95, "content": "..."},
        {"id": "doc_002", "score": 0.87, "content": "..."}
    ]
}
        </code></pre>

        <h2>注意事项</h2>
        <p>1. 查询文本长度不超过 512 个 token</p>
        <p>2. top_k 最大值为 100</p>
        <p>3. 建议在生产环境中设置合理的 threshold 以过滤低质量结果</p>
    </div>

    <footer><p>© 2026 API Docs</p></footer>
</body>
</html>
"""

# 将 HTML 包装为 LangChain Document 对象
html_documents = [
    Document(
        page_content=MOCK_HTML_BLOG,
        metadata={"source": "https://techblog.example.com/rag-guide", "type": "blog"}
    ),
    Document(
        page_content=MOCK_HTML_DOCS,
        metadata={"source": "https://api.example.com/docs/vector-search", "type": "api_doc"}
    ),
]

print(f"原始文档数量: {len(html_documents)}")
print(f"文档 1 长度: {len(html_documents[0].page_content)} 字符")
print(f"文档 2 长度: {len(html_documents[1].page_content)} 字符")
```

**预期输出：**
```
原始文档数量: 2
文档 1 长度: 2247 字符
文档 2 长度: 1098 字符
```

---

## 第二步：BeautifulSoupTransformer 提取特定标签

> BeautifulSoupTransformer 的核心能力：从 HTML 中只提取你关心的标签内容。

### 2.1 基础用法 —— 提取所有文本

```python
# ============================================================
# BeautifulSoupTransformer 基础用法
# ============================================================

bs_transformer = BeautifulSoupTransformer()

# 默认提取所有文本内容（去除 HTML 标签）
docs_basic = bs_transformer.transform_documents(html_documents)

print("=" * 60)
print("BeautifulSoup 默认提取（所有文本）")
print("=" * 60)
for i, doc in enumerate(docs_basic):
    print(f"\n--- 文档 {i+1} ---")
    print(f"来源: {doc.metadata['source']}")
    print(f"长度: {len(doc.page_content)} 字符")
    print(f"内容预览:\n{doc.page_content[:300]}...")
```

**问题分析：** 默认提取会包含导航栏、侧边栏、页脚等噪声内容。

### 2.2 精准提取 —— 只要正文标签

```python
# ============================================================
# 精准提取：只提取正文相关的标签
# ============================================================

# tags_to_extract 参数：只提取指定标签的内容
bs_transformer_precise = BeautifulSoupTransformer()

docs_precise = bs_transformer_precise.transform_documents(
    html_documents,
    tags_to_extract=["h1", "h2", "h3", "p", "li", "pre", "code", "td", "th"],
)

print("=" * 60)
print("BeautifulSoup 精准提取（仅正文标签）")
print("=" * 60)
for i, doc in enumerate(docs_precise):
    print(f"\n--- 文档 {i+1} ---")
    print(f"来源: {doc.metadata['source']}")
    print(f"长度: {len(doc.page_content)} 字符")
    print(f"内容预览:\n{doc.page_content[:500]}")
```

### 2.3 对比提取效果

```python
# ============================================================
# 对比：默认提取 vs 精准提取
# ============================================================

print("\n" + "=" * 60)
print("提取效果对比")
print("=" * 60)

for i in range(len(html_documents)):
    original_len = len(html_documents[i].page_content)
    basic_len = len(docs_basic[i].page_content)
    precise_len = len(docs_precise[i].page_content)

    print(f"\n文档 {i+1} ({html_documents[i].metadata['type']}):")
    print(f"  原始 HTML:     {original_len:>6} 字符 (100%)")
    print(f"  默认提取:      {basic_len:>6} 字符 ({basic_len/original_len*100:.1f}%)")
    print(f"  精准提取:      {precise_len:>6} 字符 ({precise_len/original_len*100:.1f}%)")
    print(f"  噪声去除率:    {(1 - precise_len/original_len)*100:.1f}%")
```

**预期输出（示意）：**
```
文档 1 (blog):
  原始 HTML:       2247 字符 (100%)
  默认提取:        1180 字符 (52.5%)
  精准提取:         920 字符 (40.9%)
  噪声去除率:      59.1%
```

> **关键洞察：** 精准提取可以去除 50-70% 的噪声内容，显著提升后续 RAG 检索质量。

---

## 第三步：Html2TextTransformer 转为纯文本

> Html2TextTransformer 使用 `html2text` 库，将 HTML 转为 Markdown 风格的纯文本，
> 保留标题层级和列表结构。

### 3.1 基础转换

```python
# ============================================================
# Html2TextTransformer 基础用法
# ============================================================

html2text_transformer = Html2TextTransformer()

docs_text = html2text_transformer.transform_documents(html_documents)

print("=" * 60)
print("Html2TextTransformer 转换结果")
print("=" * 60)
for i, doc in enumerate(docs_text):
    print(f"\n--- 文档 {i+1} ---")
    print(f"来源: {doc.metadata['source']}")
    print(f"长度: {len(doc.page_content)} 字符")
    print(f"内容:\n{doc.page_content[:600]}")
```

**预期输出（示意）：**
```
--- 文档 1 ---
来源: https://techblog.example.com/rag-guide
长度: 680 字符
内容:
# RAG 架构入门指南

作者：张三 | 发布时间：2026-01-15

## 什么是 RAG？

RAG（Retrieval-Augmented Generation）是一种将**检索**与**生成**结合的技术架构...

## RAG 的核心组件

一个完整的 RAG 系统包含以下核心组件：

  * **文档加载器**：从各种数据源加载原始文档
  * **文本分块器**：将长文档切分为适合检索的小块
  ...
```

> **关键区别：** Html2TextTransformer 保留了 Markdown 格式（`#` 标题、`*` 列表），
> 而 BeautifulSoupTransformer 只提取纯文本。

### 3.2 对比两种转换器

```python
# ============================================================
# 对比：BeautifulSoup vs Html2Text
# ============================================================

print("\n" + "=" * 60)
print("两种转换器对比")
print("=" * 60)

# 取第一个文档做对比
doc_bs = docs_precise[0].page_content[:200]
doc_h2t = docs_text[0].page_content[:200]

print("\n【BeautifulSoupTransformer 输出】")
print(f"特点: 纯文本，无格式")
print(f"预览: {doc_bs}")

print("\n【Html2TextTransformer 输出】")
print(f"特点: Markdown 格式，保留结构")
print(f"预览: {doc_h2t}")

print("\n【选择建议】")
print("├── 需要纯文本 → BeautifulSoupTransformer")
print("├── 需要保留结构 → Html2TextTransformer")
print("├── 需要精确控制提取标签 → BeautifulSoupTransformer")
print("└── 需要 Markdown 输出 → Html2TextTransformer")
```

---

## 第四步：组合管道 —— 清洗 + 分块

> 真实 RAG 场景中，清洗和分块通常需要串联使用。

### 4.1 管道式处理

```python
# ============================================================
# 完整管道：HTML → 清洗 → 纯文本 → 分块
# ============================================================

def html_cleaning_pipeline(
    documents: list[Document],
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[Document]:
    """
    HTML 文档清洗管道

    流程：
    1. Html2TextTransformer: HTML → Markdown 纯文本
    2. RecursiveCharacterTextSplitter: 长文本 → 小块

    Args:
        documents: 原始 HTML 文档列表
        chunk_size: 分块大小（字符数）
        chunk_overlap: 分块重叠（字符数）

    Returns:
        清洗并分块后的文档列表
    """
    # 第一步：HTML → 纯文本
    html2text = Html2TextTransformer()
    clean_docs = html2text.transform_documents(documents)

    # 第二步：纯文本 → 分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],  # Markdown 友好的分隔符
    )
    chunks = text_splitter.split_documents(clean_docs)

    return chunks


# 执行管道
chunks = html_cleaning_pipeline(html_documents)

print("=" * 60)
print("管道处理结果")
print("=" * 60)
print(f"输入: {len(html_documents)} 个 HTML 文档")
print(f"输出: {len(chunks)} 个文本块")

for i, chunk in enumerate(chunks):
    print(f"\n--- 块 {i+1} ---")
    print(f"来源: {chunk.metadata['source']}")
    print(f"长度: {len(chunk.page_content)} 字符")
    print(f"内容: {chunk.page_content[:150]}...")
```

### 4.2 带元数据增强的管道

```python
# ============================================================
# 增强版管道：清洗 + 分块 + 元数据增强
# ============================================================

def enhanced_html_pipeline(
    documents: list[Document],
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[Document]:
    """
    增强版 HTML 清洗管道

    额外功能：
    - 记录原始文档长度
    - 记录清洗后长度
    - 记录分块索引
    - 计算噪声去除率
    """
    results = []

    for doc in documents:
        original_len = len(doc.page_content)

        # 第一步：HTML → 纯文本
        html2text = Html2TextTransformer()
        clean_docs = html2text.transform_documents([doc])
        clean_doc = clean_docs[0]
        clean_len = len(clean_doc.page_content)

        # 第二步：分块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        )
        chunks = text_splitter.split_documents([clean_doc])

        # 第三步：增强元数据
        for idx, chunk in enumerate(chunks):
            chunk.metadata.update({
                "original_html_length": original_len,
                "clean_text_length": clean_len,
                "noise_removal_rate": f"{(1 - clean_len/original_len)*100:.1f}%",
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "pipeline": "html_cleaning_v1",
            })
            results.append(chunk)

    return results


# 执行增强管道
enhanced_chunks = enhanced_html_pipeline(html_documents)

print("=" * 60)
print("增强管道处理结果")
print("=" * 60)
print(f"总块数: {len(enhanced_chunks)}")

for i, chunk in enumerate(enhanced_chunks[:3]):  # 只展示前 3 个
    print(f"\n--- 块 {i+1} ---")
    print(f"内容: {chunk.page_content[:100]}...")
    print(f"元数据: {chunk.metadata}")
```

**预期输出（示意）：**
```
--- 块 1 ---
内容: # RAG 架构入门指南

作者：张三 | 发布时间：2026-01-15

## 什么是 RAG？...
元数据: {
    'source': 'https://techblog.example.com/rag-guide',
    'type': 'blog',
    'original_html_length': 2247,
    'clean_text_length': 680,
    'noise_removal_rate': '69.7%',
    'chunk_index': 0,
    'total_chunks': 3,
    'pipeline': 'html_cleaning_v1'
}
```

---

## 第五步：质量对比实验

> 对比直接分块 vs 清洗后分块的效果差异。

```python
# ============================================================
# 质量对比：直接分块 vs 清洗后分块
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)

# 方案 A：直接对 HTML 分块（不清洗）
chunks_dirty = text_splitter.split_documents(html_documents)

# 方案 B：清洗后分块
chunks_clean = html_cleaning_pipeline(html_documents, chunk_size=300, chunk_overlap=50)

print("=" * 60)
print("质量对比实验")
print("=" * 60)

print(f"\n方案 A（直接分块）: {len(chunks_dirty)} 个块")
print(f"方案 B（清洗后分块）: {len(chunks_clean)} 个块")

# 分析噪声块
def contains_noise(text: str) -> bool:
    """检查文本块是否包含 HTML 噪声"""
    noise_indicators = [
        "<script", "<style", "<nav", "<footer",
        "analytics", "GoogleAnalytics", "<!DOCTYPE",
        "charset=", "font-family",
    ]
    return any(indicator in text for indicator in noise_indicators)

dirty_noise_count = sum(1 for c in chunks_dirty if contains_noise(c.page_content))
clean_noise_count = sum(1 for c in chunks_clean if contains_noise(c.page_content))

print(f"\n方案 A 噪声块数: {dirty_noise_count}/{len(chunks_dirty)} "
      f"({dirty_noise_count/len(chunks_dirty)*100:.1f}%)")
print(f"方案 B 噪声块数: {clean_noise_count}/{len(chunks_clean)} "
      f"({clean_noise_count/len(chunks_clean)*100:.1f}%)")

# 展示噪声块示例
print("\n--- 方案 A 的噪声块示例 ---")
for chunk in chunks_dirty:
    if contains_noise(chunk.page_content):
        print(f"  [{len(chunk.page_content)}字符] {chunk.page_content[:120]}...")
        break

print("\n--- 方案 B 的干净块示例 ---")
if chunks_clean:
    print(f"  [{len(chunks_clean[0].page_content)}字符] {chunks_clean[0].page_content[:120]}...")
```

---

## 总结与最佳实践

### 管道选择决策树

```
HTML 文档清洗决策树
│
├── 需要精确控制提取哪些标签？
│   ├── 是 → BeautifulSoupTransformer (tags_to_extract)
│   └── 否 → Html2TextTransformer（自动转 Markdown）
│
├── 输出格式要求？
│   ├── 纯文本 → BeautifulSoupTransformer
│   └── Markdown → Html2TextTransformer
│
└── 后续处理？
    └── 一定要接 TextSplitter 分块！
```

### 关键参数速查

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `tags_to_extract` | `["h1","h2","h3","p","li","pre","code"]` | BS4 提取的标签 |
| `chunk_size` | 300-500 | 分块大小（字符） |
| `chunk_overlap` | 50-100 | 分块重叠 |
| `separators` | `["\n## ", "\n\n", "\n"]` | Markdown 友好分隔符 |

### RAG 场景建议

1. **技术博客/文档**：Html2TextTransformer → 保留 Markdown 结构
2. **产品页面**：BeautifulSoupTransformer + `tags_to_extract` → 精确提取
3. **混合来源**：先 BeautifulSoup 提取 `<main>` 区域，再 Html2Text 转换
4. **始终增强元数据**：记录来源 URL、清洗率、分块索引等信息

> **一句话总结：** HTML 清洗是 RAG 管道的第一道防线，清洗质量直接决定检索质量。
> 用 BeautifulSoup 精准提取 + Html2Text 格式转换 + TextSplitter 分块，三步搞定。
