---
type: context7_documentation
library: langgraph
version: main (2026-02-17)
fetched_at: 2026-02-26
knowledge_point: Reducer函数与状态更新
context7_query: state update partial return overwrite
library_id: /langchain-ai/langgraph
---

# Context7 文档：LangGraph 部分状态更新与返回策略

## 文档来源
- 库名称: LangGraph
- Library ID: `/langchain-ai/langgraph`
- 版本: main branch
- 最后更新: 2026-02-17
- 查询关键词: state update partial return overwrite

## 关键信息提取

### 1. 部分状态更新的核心概念

**节点签名**: `State -> Partial<State>`

节点函数不需要返回完整的状态对象,只需要返回需要更新的字段即可。

### 2. 实际应用示例

#### 示例 1: Transform Query Node (查询转换)

**来源**: https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag.ipynb

```python
def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}
```

**关键点**:
- 只返回需要更新的字段: `question` 和 `documents`
- 不需要返回完整的状态对象
- `question` 字段被更新为新值
- `documents` 字段保持不变 (但仍需返回)

#### 示例 2: Generate Node (生成答案)

**来源**: https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag.ipynb

```python
def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
```

**关键点**:
- 添加新字段: `generation`
- 保持现有字段: `documents` 和 `question`
- 展示了如何向状态添加新字段

#### 示例 3: Grade Documents Node (文档评分)

**来源**: https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag.ipynb

```python
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}
```

**关键点**:
- 过滤 `documents` 列表
- 只返回相关文档
- `question` 字段保持不变

#### 示例 4: Web Search Node (网络搜索)

**来源**: https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag.ipynb

```python
def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents, "question": question}
```

**关键点**:
- 追加到现有列表: `documents.append(web_results)`
- 如果 `documents` 字段有 Reducer (如 `operator.add`),会自动合并
- 如果没有 Reducer,会直接替换

#### 示例 5: Grade Documents with Web Search Flag

**来源**: https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag.ipynb

```python
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}
```

**关键点**:
- 添加控制流标志: `web_search`
- 用于条件路由决策
- 展示了如何在状态中添加元数据

### 3. 部分状态更新的规则

#### 规则 1: 只返回需要更新的字段
```python
# ✓ 正确: 只返回需要更新的字段
def node(state):
    return {"field1": new_value}

# ✗ 错误: 不需要返回完整状态
def node(state):
    return {
        "field1": new_value,
        "field2": state["field2"],  # 不需要
        "field3": state["field3"],  # 不需要
    }
```

#### 规则 2: 有 Reducer 的字段会合并
```python
# 状态定义
class State(TypedDict):
    messages: Annotated[list, operator.add]  # 有 Reducer
    counter: int  # 无 Reducer

# 节点 A
def node_a(state):
    return {"messages": ["A"], "counter": 1}

# 节点 B
def node_b(state):
    return {"messages": ["B"], "counter": 2}

# 结果:
# messages: ["A", "B"]  # 使用 operator.add 合并
# counter: 2  # 直接替换
```

#### 规则 3: 无 Reducer 的字段会替换
```python
# 状态定义
class State(TypedDict):
    value: int  # 无 Reducer

# 节点 A
def node_a(state):
    return {"value": 1}

# 节点 B
def node_b(state):
    return {"value": 2}

# 结果:
# value: 2  # 直接替换,不合并
```

### 4. 实际应用场景

#### 场景 1: RAG 查询改写
```python
def rewrite_query(state):
    """只更新 query 字段"""
    original_query = state["query"]
    rewritten_query = rewriter.invoke(original_query)
    return {"query": rewritten_query}
```

#### 场景 2: 文档过滤
```python
def filter_documents(state):
    """只更新 documents 字段"""
    documents = state["documents"]
    filtered = [d for d in documents if is_relevant(d)]
    return {"documents": filtered}
```

#### 场景 3: 添加元数据
```python
def add_metadata(state):
    """添加新字段"""
    return {
        "timestamp": datetime.now(),
        "source": "web_search"
    }
```

#### 场景 4: 控制流标志
```python
def check_quality(state):
    """添加控制流标志"""
    quality_score = evaluate(state["generation"])
    return {
        "quality_score": quality_score,
        "needs_retry": quality_score < 0.7
    }
```

## 技术要点总结

### 1. 部分状态更新的优势
- **简洁**: 只返回需要更新的字段
- **灵活**: 可以选择性地更新状态
- **高效**: 减少不必要的数据传递

### 2. 与 Reducer 的交互
- **有 Reducer**: 新值与旧值合并
- **无 Reducer**: 新值直接替换旧值

### 3. 最佳实践
- 只返回需要更新的字段
- 使用 Reducer 处理累积型数据 (如列表、消息)
- 使用直接替换处理单值数据 (如计数器、标志)
- 添加控制流标志用于条件路由

### 4. 常见模式
- **查询改写**: 更新 `query` 字段
- **文档过滤**: 更新 `documents` 字段
- **生成答案**: 添加 `generation` 字段
- **添加元数据**: 添加 `timestamp`, `source` 等字段
- **控制流**: 添加 `needs_retry`, `web_search` 等标志

## 官方文档链接

1. **Self-RAG Example**: https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag.ipynb
2. **CRAG Example**: https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag.ipynb

## 版本信息

- **最后更新**: 2026-02-17
- **分支**: main
- **总代码片段**: 234 个
- **Trust Score**: 9.2/10
- **Benchmark Score**: 77.5/100
