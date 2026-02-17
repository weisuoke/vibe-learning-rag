# 核心概念07：状态持久化Checkpointing

> **定义**：Checkpointing是保存Agent状态快照的机制，支持故障恢复、长运行任务和人机协作

---

## 一、为什么需要状态持久化？

### 1.1 传统Agent的问题

**问题1：状态易失**
```python
# 传统方式：状态在内存中
agent_state = {"messages": [], "context": ""}

# 进程崩溃 → 状态丢失 ❌
# 无法恢复 ❌
```

**问题2：长运行任务**
```python
# 长时间运行的任务
for i in range(1000):
    result = process_item(i)
    # 如果在第500步崩溃，需要从头开始 ❌
```

**问题3：人机协作**
```python
# 需要人类确认
answer = agent.generate()
# 如何暂停等待人类确认？❌
# 如何恢复执行？❌
```

---

### 1.2 Checkpointing的解决方案

**解决方案**：
```python
from langgraph.checkpoint.memory import MemorySaver

# 1. 创建checkpointer
checkpointer = MemorySaver()

# 2. 编译图时绑定
app = graph.compile(checkpointer=checkpointer)

# 3. 运行时自动保存
config = {"configurable": {"thread_id": "user_123"}}
result = app.invoke(input, config=config)
# 每个节点执行后自动保存checkpoint ✅

# 4. 故障恢复
state = app.get_state(config)  # 获取最新状态
app.invoke(None, config=config)  # 从checkpoint恢复 ✅
```

---

## 二、Checkpointing核心概念

### 2.1 Thread（线程）

**定义**：Thread是一个独立的会话标识符

```python
# 不同用户使用不同thread_id
config_user1 = {"configurable": {"thread_id": "user_1"}}
config_user2 = {"configurable": {"thread_id": "user_2"}}

# 每个thread有独立的状态
app.invoke(input1, config=config_user1)  # user_1的状态
app.invoke(input2, config=config_user2)  # user_2的状态
```

**类比**：
- **前端**：Session ID（每个用户一个会话）
- **生活**：游戏存档槽位（每个槽位独立）

---

### 2.2 Checkpoint（检查点）

**定义**：Checkpoint是某个时刻的状态快照

```python
# Checkpoint包含：
{
    "v": 1,                    # 版本号
    "id": "checkpoint_123",    # 检查点ID
    "ts": "2026-02-14T...",    # 时间戳
    "channel_values": {        # 状态值
        "messages": [...],
        "context": "..."
    },
    "channel_versions": {...}, # 版本信息
    "versions_seen": {...}     # 已见版本
}
```

---

### 2.3 Checkpointer（检查点保存器）

**定义**：Checkpointer是保存和加载checkpoint的接口

**内置实现**：
1. **MemorySaver**：内存存储（开发/测试）
2. **PostgresSaver**：PostgreSQL存储（生产）
3. **DynamoDBSaver**：DynamoDB存储（AWS生产）
4. **RedisSaver**：Redis存储（高性能）

---

## 三、Checkpointer实现

### 3.1 MemorySaver（内存存储）

**适用场景**：开发、测试、演示

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

# 1. 创建MemorySaver
checkpointer = MemorySaver()

# 2. 编译图
app = graph.compile(checkpointer=checkpointer)

# 3. 使用
config = {"configurable": {"thread_id": "demo"}}
result = app.invoke({"query": "test"}, config=config)

# 4. 获取状态
state = app.get_state(config)
print(state.values)  # 当前状态
print(state.next)    # 下一个节点

# 5. 获取历史
history = app.get_state_history(config)
for checkpoint in history:
    print(f"Checkpoint {checkpoint.config['configurable']['checkpoint_id']}")
```

**优势**：
- ✅ 简单易用
- ✅ 无需配置
- ✅ 适合开发测试

**劣势**：
- ❌ 进程重启后丢失
- ❌ 不支持分布式
- ❌ 不适合生产环境

---

### 3.2 PostgresSaver（生产级）

**适用场景**：生产环境、需要持久化

```python
from langgraph.checkpoint.postgres import PostgresSaver

# 1. 创建PostgresSaver
DB_URI = "postgresql://user:pass@localhost:5432/langgraph"
checkpointer = PostgresSaver.from_conn_string(DB_URI)

# 2. 初始化数据库表
checkpointer.setup()

# 3. 编译图
app = graph.compile(checkpointer=checkpointer)

# 4. 使用（与MemorySaver相同）
config = {"configurable": {"thread_id": "user_123"}}
result = app.invoke(input, config=config)
```

**数据库表结构**：
```sql
CREATE TABLE checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_id)
);
```

**优势**：
- ✅ 持久化存储
- ✅ 支持分布式
- ✅ 事务性保证
- ✅ 适合生产环境

**劣势**：
- ❌ 需要配置数据库
- ❌ 性能略低于内存

---

### 3.3 DynamoDBSaver（AWS生产级）

**适用场景**：AWS环境、无服务器架构

```python
from langgraph.checkpoint.dynamodb import DynamoDBSaver
import boto3

# 1. 创建DynamoDB客户端
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

# 2. 创建DynamoDBSaver
checkpointer = DynamoDBSaver(
    table_name='langgraph_checkpoints',
    dynamodb_resource=dynamodb
)

# 3. 创建表（首次使用）
checkpointer.setup()

# 4. 编译图
app = graph.compile(checkpointer=checkpointer)
```

**DynamoDB表结构**：
```python
{
    "TableName": "langgraph_checkpoints",
    "KeySchema": [
        {"AttributeName": "thread_id", "KeyType": "HASH"},
        {"AttributeName": "checkpoint_id", "KeyType": "RANGE"}
    ],
    "AttributeDefinitions": [
        {"AttributeName": "thread_id", "AttributeType": "S"},
        {"AttributeName": "checkpoint_id", "AttributeType": "S"}
    ]
}
```

**优势**：
- ✅ 无服务器
- ✅ 自动扩展
- ✅ 高可用性
- ✅ 与AWS生态集成

**劣势**：
- ❌ AWS专用
- ❌ 成本较高

---

## 四、完整实战示例

### 4.1 长运行任务的Checkpointing

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
import time

# 1. 定义状态
class ProcessState(TypedDict):
    items: List[str]
    processed: List[str]
    current_index: int
    total: int

# 2. 定义处理节点
def process_batch(state: ProcessState) -> ProcessState:
    """批量处理节点"""
    items = state["items"]
    current_index = state["current_index"]
    processed = state["processed"]

    # 每次处理10个
    batch_size = 10
    end_index = min(current_index + batch_size, len(items))

    print(f"处理 {current_index} 到 {end_index}...")

    for i in range(current_index, end_index):
        # 模拟处理
        result = f"processed_{items[i]}"
        processed.append(result)
        time.sleep(0.1)  # 模拟耗时操作

    return {
        "processed": processed,
        "current_index": end_index
    }

def should_continue(state: ProcessState) -> str:
    """判断是否继续"""
    if state["current_index"] >= state["total"]:
        return "end"
    return "process"

# 3. 构建图
def create_long_running_graph():
    graph = StateGraph(ProcessState)

    graph.add_node("process", process_batch)

    graph.add_conditional_edges(
        "process",
        should_continue,
        {
            "end": END,
            "process": "process"
        }
    )

    graph.set_entry_point("process")

    return graph

# 4. 使用Checkpointing
if __name__ == "__main__":
    # 创建checkpointer
    checkpointer = PostgresSaver.from_conn_string(
        "postgresql://localhost/langgraph"
    )
    checkpointer.setup()

    # 编译图
    graph = create_long_running_graph()
    app = graph.compile(checkpointer=checkpointer)

    # 初始状态
    items = [f"item_{i}" for i in range(100)]
    initial_state = {
        "items": items,
        "processed": [],
        "current_index": 0,
        "total": len(items)
    }

    config = {"configurable": {"thread_id": "long_task_1"}}

    try:
        # 运行任务
        result = app.invoke(initial_state, config=config)
        print(f"✅ 完成！处理了 {len(result['processed'])} 个项目")

    except KeyboardInterrupt:
        print("\n⚠️ 任务中断！")

        # 获取当前状态
        state = app.get_state(config)
        print(f"已处理: {state.values['current_index']}/{state.values['total']}")

        # 恢复执行
        print("恢复执行...")
        result = app.invoke(None, config=config)
        print(f"✅ 完成！处理了 {len(result['processed'])} 个项目")
```

---

### 4.2 人机协作的Checkpointing

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 1. 定义状态
class ApprovalState(TypedDict):
    query: str
    draft_answer: str
    approved: bool
    final_answer: str

# 2. 定义节点
def generate_draft(state: ApprovalState) -> ApprovalState:
    """生成草稿"""
    query = state["query"]
    draft = f"草稿答案：关于'{query}'的回答..."
    print(f"📝 生成草稿: {draft}")
    return {"draft_answer": draft}

def human_approval(state: ApprovalState) -> ApprovalState:
    """人类批准节点（会在此中断）"""
    draft = state["draft_answer"]
    print(f"\n等待人类批准...")
    print(f"草稿: {draft}")
    # LangGraph会在这里中断
    return state

def finalize(state: ApprovalState) -> ApprovalState:
    """最终化"""
    if state["approved"]:
        final = state["draft_answer"]
    else:
        final = "已拒绝"
    print(f"✅ 最终答案: {final}")
    return {"final_answer": final}

# 3. 构建图
def create_approval_graph():
    graph = StateGraph(ApprovalState)

    graph.add_node("generate", generate_draft)
    graph.add_node("approval", human_approval)
    graph.add_node("finalize", finalize)

    graph.add_edge("generate", "approval")
    graph.add_edge("approval", "finalize")
    graph.add_edge("finalize", END)

    graph.set_entry_point("generate")

    return graph

# 4. 使用中断机制
if __name__ == "__main__":
    checkpointer = MemorySaver()
    graph = create_approval_graph()

    # 编译时指定中断点
    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["approval"]  # 在approval节点前中断
    )

    config = {"configurable": {"thread_id": "approval_1"}}

    # 第一步：运行到中断点
    print("=== 第一步：生成草稿 ===")
    initial_state = {
        "query": "什么是LangGraph？",
        "draft_answer": "",
        "approved": False,
        "final_answer": ""
    }

    result = app.invoke(initial_state, config=config)
    print(f"\n当前状态: {result}")

    # 第二步：人类批准
    print("\n=== 第二步：人类批准 ===")
    approval = input("批准草稿？(y/n): ")

    # 更新状态
    app.update_state(
        config,
        {"approved": approval.lower() == 'y'}
    )

    # 第三步：继续执行
    print("\n=== 第三步：继续执行 ===")
    result = app.invoke(None, config=config)
    print(f"\n最终结果: {result['final_answer']}")
```

---

## 五、状态管理操作

### 5.1 获取状态

```python
# 获取当前状态
state = app.get_state(config)

print(state.values)      # 状态值
print(state.next)        # 下一个节点
print(state.config)      # 配置信息
print(state.metadata)    # 元数据
```

---

### 5.2 更新状态

```python
# 更新状态（不执行节点）
app.update_state(
    config,
    {"key": "new_value"}
)

# 更新状态并指定下一个节点
app.update_state(
    config,
    {"key": "new_value"},
    as_node="node_name"
)
```

---

### 5.3 获取历史

```python
# 获取所有checkpoint历史
history = app.get_state_history(config)

for checkpoint in history:
    print(f"Checkpoint ID: {checkpoint.config['configurable']['checkpoint_id']}")
    print(f"Values: {checkpoint.values}")
    print(f"Next: {checkpoint.next}")
    print("---")
```

---

### 5.4 回溯到历史状态

```python
# 获取历史
history = list(app.get_state_history(config))

# 回溯到第3个checkpoint
old_checkpoint = history[2]
old_config = old_checkpoint.config

# 从旧checkpoint继续执行
result = app.invoke(None, config=old_config)
```

---

## 六、性能优化

### 6.1 增量更新

**问题**：每次保存完整状态开销大

**解决方案**：只保存变化部分

```python
# LangGraph自动实现增量更新
# 只保存changed的channel
class State(TypedDict):
    messages: Annotated[List[str], operator.add]  # 增量添加
    context: str  # 覆盖更新

def node(state: State) -> State:
    # 只返回变化的部分
    return {"messages": ["new_message"]}
    # LangGraph只保存新增的message
```

---

### 6.2 异步保存

**问题**：同步保存阻塞执行

**解决方案**：使用异步checkpointer

```python
from langgraph.checkpoint.postgres import AsyncPostgresSaver

# 异步checkpointer
checkpointer = AsyncPostgresSaver.from_conn_string(DB_URI)

# 异步执行
async def run_agent():
    result = await app.ainvoke(input, config=config)
    return result
```

---

### 6.3 批量写入

**问题**：频繁写入数据库

**解决方案**：批量提交

```python
# PostgresSaver自动批量写入
# 在事务中批量提交多个checkpoint
```

---

## 七、生产级最佳实践

### 7.1 选择合适的Checkpointer

| 场景 | 推荐 | 原因 |
|------|------|------|
| **开发/测试** | MemorySaver | 简单快速 |
| **生产环境** | PostgresSaver | 可靠持久 |
| **AWS环境** | DynamoDBSaver | 无服务器 |
| **高性能** | RedisSaver | 低延迟 |

---

### 7.2 Thread ID设计

**原则**：
- 用户级：`user_{user_id}`
- 会话级：`session_{session_id}`
- 任务级：`task_{task_id}`

```python
# 用户级（跨会话）
config = {"configurable": {"thread_id": f"user_{user_id}"}}

# 会话级（单次对话）
config = {"configurable": {"thread_id": f"session_{session_id}"}}

# 任务级（单个任务）
config = {"configurable": {"thread_id": f"task_{task_id}"}}
```

---

### 7.3 清理策略

**问题**：checkpoint累积占用存储

**解决方案**：定期清理

```python
# 清理30天前的checkpoint
DELETE FROM checkpoints
WHERE created_at < NOW() - INTERVAL '30 days';

# 只保留最近N个checkpoint
DELETE FROM checkpoints
WHERE checkpoint_id NOT IN (
    SELECT checkpoint_id
    FROM checkpoints
    WHERE thread_id = 'xxx'
    ORDER BY created_at DESC
    LIMIT 10
);
```

---

### 7.4 监控与告警

**监控指标**：
- Checkpoint保存频率
- Checkpoint大小
- 保存延迟
- 存储使用量

```python
import time

class MonitoredCheckpointer:
    def __init__(self, base_checkpointer):
        self.base = base_checkpointer

    def put(self, config, checkpoint, metadata):
        start = time.time()
        result = self.base.put(config, checkpoint, metadata)
        duration = time.time() - start

        # 记录指标
        print(f"Checkpoint saved in {duration:.3f}s")
        print(f"Size: {len(str(checkpoint))} bytes")

        return result
```

---

## 八、常见问题

### 8.1 Checkpoint过大

**问题**：状态包含大量数据

**解决方案**：
1. 只保存必要数据
2. 使用引用而非完整数据
3. 压缩大对象

```python
# ❌ 不好：保存完整文档
class BadState(TypedDict):
    documents: List[str]  # 可能很大

# ✅ 好：只保存文档ID
class GoodState(TypedDict):
    document_ids: List[str]  # 小

def retrieve_documents(state):
    # 从数据库加载文档
    docs = db.get_documents(state["document_ids"])
    return {"documents": docs}
```

---

### 8.2 并发冲突

**问题**：多个进程同时更新同一thread

**解决方案**：使用乐观锁

```python
# PostgresSaver自动处理并发
# 使用checkpoint_id作为版本号
# 更新时检查版本是否匹配
```

---

### 8.3 状态迁移

**问题**：状态schema变化

**解决方案**：版本化状态

```python
class StateV1(TypedDict):
    query: str

class StateV2(TypedDict):
    query: str
    version: int  # 新增字段

def migrate_state(old_state):
    """迁移旧状态"""
    if "version" not in old_state:
        old_state["version"] = 1
    return old_state
```

---

## 九、总结

### 核心要点

1. **Checkpointing**：保存状态快照
2. **Thread**：会话标识符
3. **Checkpointer**：存储后端（Memory、Postgres、DynamoDB）
4. **应用场景**：故障恢复、长运行任务、人机协作
5. **性能优化**：增量更新、异步保存、批量写入

### Checkpointer选择

| Checkpointer | 适用场景 | 优势 | 劣势 |
|--------------|---------|------|------|
| **MemorySaver** | 开发/测试 | 简单快速 | 不持久 |
| **PostgresSaver** | 生产环境 | 可靠持久 | 需配置 |
| **DynamoDBSaver** | AWS环境 | 无服务器 | AWS专用 |
| **RedisSaver** | 高性能 | 低延迟 | 需Redis |

### 最佳实践

1. **开发用MemorySaver，生产用PostgresSaver**
2. **合理设计Thread ID**（用户级/会话级/任务级）
3. **定期清理旧checkpoint**
4. **监控checkpoint大小和频率**
5. **使用增量更新减少开销**

### 学习建议

1. **理解Thread概念**：会话标识符
2. **掌握基本操作**：get_state、update_state、get_state_history
3. **实践人机协作**：interrupt_before机制
4. **学习生产配置**：PostgresSaver、DynamoDBSaver
5. **优化性能**：增量更新、异步保存

---

## 参考资料

1. **官方文档**：
   - LangGraph Persistence (2025)
   - LangGraph Checkpointing Reference
   - langgraph-checkpoint-postgres - PyPI

2. **教程**：
   - Sparkco.ai - Mastering LangGraph Checkpointing (2025)
   - AWS - Build durable AI agents with LangGraph and DynamoDB (2026)
   - Medium - Mastering Persistence in LangGraph

3. **最佳实践**：
   - LinkedIn - Why Persistence is the Secret to Reliable LangGraph Agents
   - LangGraph Patterns & Best Practices Guide (2025)

---

**版本**: v1.0
**最后更新**: 2026-02-14
**代码行数**: ~450行
