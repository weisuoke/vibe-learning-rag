---
type: source_code_analysis
source: sourcecode/langchain/libs/text-splitters/langchain_text_splitters/html.py
analyzed_files:
  - sourcecode/langchain/libs/text-splitters/langchain_text_splitters/html.py
analyzed_at: 2026-02-25
knowledge_point: 04_TextSplitter文本分块
---

# 源码分析：HTML 文本分块器

## 分析的文件
- `sourcecode/langchain/libs/text-splitters/langchain_text_splitters/html.py` - HTML 专用分块器

## 关键发现

### 1. HTMLHeaderTextSplitter 类

**特点**：
- 基于 HTML 标题标签（h1, h2, h3等）分块
- 保留 HTML 文档的层级结构
- 不继承自 TextSplitter（独立实现）

**核心参数**：
```python
def __init__(
    self,
    headers_to_split_on: list[tuple[str, str]],
    return_each_element: bool = False,
) -> None:
```

- `headers_to_split_on`: 标题标签列表，如 `[("h1", "Header 1"), ("h2", "Header 2")]`
- `return_each_element`: 是否将每个 HTML 元素作为单独的 Document 返回

### 2. 核心方法

#### split_text()
- 解析 HTML 内容
- 识别指定的标题标签
- 创建带有层级元数据的 Document 对象

#### split_text_from_url()
- 从 URL 获取 HTML 内容
- 调用 split_text() 处理

### 3. 依赖库

- `BeautifulSoup` (bs4): HTML 解析
- `lxml`: XML/HTML 处理
- `requests`: HTTP 请求

### 4. 使用场景

**适用场景**：
- 网页内容提取
- 技术文档爬取
- 保留 HTML 结构的分块

**示例**：
```python
headers_to_split_on = [("h1", "Main Topic"), ("h2", "Sub Topic")]
splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text(html_content)
```

### 5. 与 MarkdownHeaderTextSplitter 的对比

| 特性 | HTMLHeaderTextSplitter | MarkdownHeaderTextSplitter |
|------|------------------------|----------------------------|
| 输入格式 | HTML | Markdown |
| 解析库 | BeautifulSoup | 正则表达式 |
| 标题识别 | HTML 标签 | Markdown 语法 |
| 代码块处理 | 自动 | 手动检测 |
