---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/langgraph/langgraph/errors.py
analyzed_at: 2026-03-07
knowledge_point: 08_错误处理与重试
---

# 源码分析：LangGraph 错误体系

## 分析的文件
- `libs/langgraph/langgraph/errors.py` - LangGraph 核心错误定义

## 关键发现

### ErrorCode 枚举
LangGraph 定义了以下错误代码：
- `GRAPH_RECURSION_LIMIT` - 图递归深度超限
- `INVALID_CONCURRENT_GRAPH_UPDATE` - 无效的并发图更新
- `INVALID_GRAPH_NODE_RETURN_VALUE` - 无效的图节点返回值
- `MULTIPLE_SUBGRAPHS` - 多个子图冲突
- `INVALID_CHAT_HISTORY` - 无效的聊天历史

### 错误层次结构

```
Exception
├── RecursionError
│   └── GraphRecursionError          # 图执行步数超限
├── InvalidUpdateError               # 无效的 Channel 更新
├── GraphBubbleUp                    # 图中断基类（不会被重试）
│   ├── GraphInterrupt               # 子图中断（被根图抑制）
│   │   └── NodeInterrupt (deprecated) # 节点中断（已弃用）
│   └── ParentCommand                # 命令冒泡到父图
├── EmptyInputError                  # 空输入错误
└── TaskNotFound                     # 分布式模式中任务未找到
```

### 关键设计决策

1. **GraphBubbleUp 不会被重试**：`run_with_retry()` 中对 `GraphBubbleUp` 异常直接 `raise`，不进入重试逻辑
2. **ParentCommand 特殊处理**：会根据命名空间判断是当前图还是父图的命令
3. **NodeInterrupt 已弃用**：推荐使用 `interrupt()` 函数代替
4. **错误代码追踪**：每个错误都有对应的在线文档链接

## 代码片段

```python
class ErrorCode(Enum):
    GRAPH_RECURSION_LIMIT = "GRAPH_RECURSION_LIMIT"
    INVALID_CONCURRENT_GRAPH_UPDATE = "INVALID_CONCURRENT_GRAPH_UPDATE"
    INVALID_GRAPH_NODE_RETURN_VALUE = "INVALID_GRAPH_NODE_RETURN_VALUE"
    MULTIPLE_SUBGRAPHS = "MULTIPLE_SUBGRAPHS"
    INVALID_CHAT_HISTORY = "INVALID_CHAT_HISTORY"

class GraphRecursionError(RecursionError):
    """图执行步数超限，防止无限循环"""
    pass

class GraphBubbleUp(Exception):
    """图中断基类，不会被重试机制捕获"""
    pass

class GraphInterrupt(GraphBubbleUp):
    """子图中断信号"""
    def __init__(self, interrupts: Sequence[Interrupt] = ()) -> None:
        super().__init__(interrupts)
```
