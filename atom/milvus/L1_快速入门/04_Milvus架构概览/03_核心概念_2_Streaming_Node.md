# Milvus架构概览 - 核心概念2：Streaming Node

> 理解 Milvus 2.6 的核心创新：Streaming Node 统一流式数据处理

---

## 原理讲解

### 1. Streaming Node 的诞生背景

**传统架构的问题（Milvus 2.5 及之前）：**

```
数据流转路径：
Client → Proxy → Pulsar/Kafka → Data Node → Object Storage
                              ↓
                         Query Node

问题：
1. 依赖外部消息队列（Pulsar/Kafka）
   - 需要独立部署和运维
   - 增加系统复杂度
   - 额外的成本开销

2. 数据流转延迟高
   - 多次网络跳转
   - 序列化/反序列化开销
   - 消息队列的延迟

3. 资源消耗大
   - Pulsar/Kafka 需要独立资源
   - 数据需要在多个组件间复制
   - 存储和计算资源浪费
```

**Milvus 2.6 的解决方案：Streaming Node**

```
新的数据流转路径：
Client → Proxy → Streaming Node → Data Node → Object Storage
                               ↓
                          Query Node

优势：
1. 内置流式处理
   - 无需外部消息队列
   - 简化部署和运维
   - 降低成本（节省 30-40%）

2. 降低延迟
   - 减少网络跳转
   - 优化数据传输
   - 延迟降低 50%

3. 统一管理
   - 与 Milvus 深度集成
   - 统一监控和调优
   - 简化故障排查
```

### 2. Streaming Node 的核心功能

#### 2.1 统一数据流管理

**Streaming Node 管理所有数据流：**

```
1. 写入流（Insert/Upsert/Delete）
   Client → Streaming Node → Data Node

2. 同步流（Data Node → Query Node）
   Data Node → Streaming Node → Query Node

3. 订阅流（实时更新）
   Streaming Node → 订阅者（如监控系统）
```

**类比：** Streaming Node 就像快递分拣中心，所有包裹（数据）都经过这里统一分拣和分发。

#### 2.2 与 Woodpecker WAL 深度集成

**零磁盘架构的基础：**

```
传统架构：
  数据 → 本地磁盘 WAL → 消息队列 → 处理

Streaming Node + Woodpecker WAL：
  数据 → Streaming Node → Woodpecker WAL（对象存储）→ 处理

优势：
- 不需要本地磁盘
- WAL 直接写入对象存储
- 数据可靠性更高（11 个 9）
```

#### 2.3 流式数据优化

**针对向量数据的特殊优化：**

```python
# 1. 批量处理
"""
Streaming Node 会自动批量处理数据：
- 小批量写入 → 合并为大批量
- 减少网络开销
- 提高吞吐量
"""

# 2. 数据压缩
"""
向量数据通常可以压缩：
- Streaming Node 自动压缩
- 减少网络传输量
- 降低存储成本
"""

# 3. 优先级调度
"""
不同类型的数据流有不同优先级：
- 实时查询：高优先级
- 批量写入：中优先级
- 后台任务：低优先级
"""
```

### 3. Streaming Node 的工作原理

#### 3.1 数据接收

```
1. Proxy 发送数据到 Streaming Node
2. Streaming Node 验证数据格式
3. 分配 Timestamp（时间戳）
4. 写入 Woodpecker WAL（持久化保障）
5. 缓存到内存（等待批量处理）
```

#### 3.2 数据分发

```
1. 根据 Collection 和 Partition 分组
2. 批量发送到 Data Node
3. 同时通知 Query Node（数据同步）
4. 更新元数据（通过 Root Coord）
```

#### 3.3 故障恢复

```
1. Streaming Node 崩溃
   → 从 Woodpecker WAL 恢复未完成的操作
   → 重新分发数据

2. Data Node 崩溃
   → Streaming Node 重试发送
   → 或分发到其他 Data Node

3. 网络故障
   → Streaming Node 缓存数据
   → 网络恢复后继续发送
```

---

## 手写实现

### 简化版 Streaming Node 实现

```python
"""
Streaming Node 的简化实现
展示核心的流式处理逻辑
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from queue import Queue
import threading
import time

@dataclass
class StreamMessage:
    """流式消息"""
    collection: str
    partition: str
    data: Dict[str, Any]
    timestamp: float
    msg_id: str

class WoodpeckerWAL:
    """Woodpecker WAL（简化版）"""
    def __init__(self):
        self.log = []
    
    def append(self, message: StreamMessage):
        """追加日志"""
        self.log.append({
            "msg_id": message.msg_id,
            "collection": message.collection,
            "data": message.data,
            "timestamp": message.timestamp
        })
        print(f"[WAL] Logged message {message.msg_id}")
    
    def recover(self, from_timestamp: float) -> List[Dict]:
        """恢复日志"""
        return [msg for msg in self.log if msg["timestamp"] >= from_timestamp]

class StreamingNode:
    """Streaming Node 核心实现"""
    
    def __init__(self, wal: WoodpeckerWAL):
        self.wal = wal
        self.buffer = {}  # collection -> messages
        self.subscribers = {}  # collection -> subscribers
        self.running = False
        self.msg_counter = 0
        
        # 批量处理配置
        self.batch_size = 100
        self.batch_timeout = 1.0  # 秒
        
        # 统计信息
        self.stats = {
            "received": 0,
            "processed": 0,
            "batches": 0
        }
    
    def start(self):
        """启动 Streaming Node"""
        self.running = True
        # 启动后台批量处理线程
        self.batch_thread = threading.Thread(target=self._batch_processor)
        self.batch_thread.daemon = True
        self.batch_thread.start()
        print("[StreamingNode] Started")
    
    def stop(self):
        """停止 Streaming Node"""
        self.running = False
        self.batch_thread.join()
        print("[StreamingNode] Stopped")
    
    def receive(self, collection: str, partition: str, data: Dict) -> str:
        """接收数据"""
        # 生成消息 ID
        self.msg_counter += 1
        msg_id = f"msg_{self.msg_counter}"
        
        # 创建流式消息
        message = StreamMessage(
            collection=collection,
            partition=partition,
            data=data,
            timestamp=time.time(),
            msg_id=msg_id
        )
        
        # 写入 WAL（持久化保障）
        self.wal.append(message)
        
        # 缓存到内存
        key = f"{collection}:{partition}"
        if key not in self.buffer:
            self.buffer[key] = []
        self.buffer[key].append(message)
        
        # 更新统计
        self.stats["received"] += 1
        
        print(f"[StreamingNode] Received message {msg_id} for {collection}/{partition}")
        
        # 通知订阅者
        self._notify_subscribers(collection, message)
        
        return msg_id
    
    def subscribe(self, collection: str, callback):
        """订阅数据流"""
        if collection not in self.subscribers:
            self.subscribers[collection] = []
        self.subscribers[collection].append(callback)
        print(f"[StreamingNode] New subscriber for {collection}")
    
    def _notify_subscribers(self, collection: str, message: StreamMessage):
        """通知订阅者"""
        if collection in self.subscribers:
            for callback in self.subscribers[collection]:
                try:
                    callback(message)
                except Exception as e:
                    print(f"[StreamingNode] Subscriber error: {e}")
    
    def _batch_processor(self):
        """后台批量处理线程"""
        print("[StreamingNode] Batch processor started")
        
        while self.running:
            time.sleep(self.batch_timeout)
            
            # 处理所有缓存的数据
            for key, messages in list(self.buffer.items()):
                if not messages:
                    continue
                
                # 批量处理
                if len(messages) >= self.batch_size:
                    batch = messages[:self.batch_size]
                    self.buffer[key] = messages[self.batch_size:]
                else:
                    # 超时处理（即使未达到 batch_size）
                    batch = messages
                    self.buffer[key] = []
                
                if batch:
                    self._process_batch(key, batch)
    
    def _process_batch(self, key: str, batch: List[StreamMessage]):
        """处理批量数据"""
        collection, partition = key.split(":")
        
        print(f"[StreamingNode] Processing batch of {len(batch)} messages for {collection}/{partition}")
        
        # 模拟发送到 Data Node
        # 实际实现中会调用 Data Node 的 gRPC 接口
        
        # 更新统计
        self.stats["processed"] += len(batch)
        self.stats["batches"] += 1
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()

class DataNode:
    """Data Node（简化版）"""
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.received_batches = []
    
    def process_batch(self, collection: str, partition: str, messages: List[StreamMessage]):
        """处理批量数据"""
        print(f"[DataNode-{self.node_id}] Received batch of {len(messages)} messages")
        self.received_batches.append({
            "collection": collection,
            "partition": partition,
            "count": len(messages),
            "timestamp": time.time()
        })

# ===== 完整演示 =====

def demo_streaming_node():
    """演示 Streaming Node 的工作流程"""
    
    print("=" * 60)
    print("Streaming Node 演示")
    print("=" * 60)
    
    # 1. 初始化组件
    print("\n[1] 初始化组件")
    wal = WoodpeckerWAL()
    streaming_node = StreamingNode(wal)
    data_node = DataNode("node-1")
    
    # 2. 启动 Streaming Node
    print("\n[2] 启动 Streaming Node")
    streaming_node.start()
    
    # 3. 订阅数据流
    print("\n[3] 订阅数据流")
    def on_message(message: StreamMessage):
        print(f"[Subscriber] Received: {message.msg_id}")
    
    streaming_node.subscribe("documents", on_message)
    
    # 4. 发送数据
    print("\n[4] 发送数据")
    for i in range(10):
        streaming_node.receive(
            collection="documents",
            partition="default",
            data={"id": i, "vector": [0.1 * i] * 128}
        )
        time.sleep(0.1)
    
    # 5. 等待批量处理
    print("\n[5] 等待批量处理")
    time.sleep(2)
    
    # 6. 查看统计信息
    print("\n[6] 统计信息")
    stats = streaming_node.get_stats()
    print(f"  接收消息数: {stats['received']}")
    print(f"  处理消息数: {stats['processed']}")
    print(f"  批次数: {stats['batches']}")
    
    # 7. 停止 Streaming Node
    print("\n[7] 停止 Streaming Node")
    streaming_node.stop()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)

if __name__ == "__main__":
    demo_streaming_node()
```

**运行输出示例：**
```
============================================================
Streaming Node 演示
============================================================

[1] 初始化组件

[2] 启动 Streaming Node
[StreamingNode] Started
[StreamingNode] Batch processor started

[3] 订阅数据流
[StreamingNode] New subscriber for documents

[4] 发送数据
[WAL] Logged message msg_1
[StreamingNode] Received message msg_1 for documents/default
[Subscriber] Received: msg_1
...

[5] 等待批量处理
[StreamingNode] Processing batch of 10 messages for documents:default

[6] 统计信息
  接收消息数: 10
  处理消息数: 10
  批次数: 1

[7] 停止 Streaming Node
[StreamingNode] Stopped

============================================================
演示完成！
============================================================
```

---

## RAG相关应用场景

### 场景1：实时文档入库

```python
"""
利用 Streaming Node 实现实时文档入库
"""

from pymilvus import Collection, connections

# 连接到 Milvus 2.6
connections.connect("default", host="localhost", port="19530")

collection = Collection("documents")

# 实时入库流程
"""
1. 用户上传文档
   ↓
2. 文档解析和 Embedding
   ↓
3. 发送到 Milvus
   ↓
4. Streaming Node 接收
   ↓
5. 写入 Woodpecker WAL（持久化）
   ↓
6. 批量发送到 Data Node
   ↓
7. 数据立即可搜索（< 1秒）

优势：
- 实时性：文档上传后 < 1秒可搜索
- 可靠性：Woodpecker WAL 保证不丢失
- 高吞吐：批量处理提高效率
"""

# 实际代码
def upload_document(doc_text: str, doc_id: int):
    """上传文档到 Milvus"""
    # 生成 Embedding（假设已有函数）
    embedding = generate_embedding(doc_text)
    
    # 插入到 Milvus
    # Streaming Node 会自动处理
    collection.insert([{
        "id": doc_id,
        "text": doc_text,
        "vector": embedding
    }])
    
    print(f"Document {doc_id} uploaded, will be searchable in < 1s")

# 批量上传
for i in range(1000):
    upload_document(f"Document {i}", i)
    
# Streaming Node 会自动批量处理，无需手动管理
```

### 场景2：用户反馈实时更新

```python
"""
用户反馈实时更新向量库
"""

# 场景：用户对搜索结果进行反馈
"""
1. 用户点击"有用"或"无用"
   ↓
2. 更新文档的相关性分数
   ↓
3. 通过 Streaming Node 实时更新
   ↓
4. 下次搜索立即生效
"""

def update_document_score(doc_id: int, feedback: str):
    """根据用户反馈更新文档"""
    # 获取当前文档
    result = collection.query(expr=f"id == {doc_id}", output_fields=["*"])
    
    if result:
        doc = result[0]
        # 更新相关性分数
        new_score = doc.get("score", 0.5)
        if feedback == "useful":
            new_score += 0.1
        else:
            new_score -= 0.1
        
        # 删除旧文档
        collection.delete(expr=f"id == {doc_id}")
        
        # 插入更新后的文档
        # Streaming Node 会实时处理
        collection.insert([{
            "id": doc_id,
            "text": doc["text"],
            "vector": doc["vector"],
            "score": new_score
        }])
        
        print(f"Document {doc_id} updated, new score: {new_score}")

# 用户反馈
update_document_score(123, "useful")
# 下次搜索立即生效（得益于 Streaming Node 的实时处理）
```

### 场景3：多租户数据隔离

```python
"""
利用 Streaming Node 实现多租户数据隔离
"""

# 每个租户一个 Collection
"""
Streaming Node 会为每个 Collection 维护独立的数据流：
- tenant_1 → Streaming Node → Data Node (tenant_1)
- tenant_2 → Streaming Node → Data Node (tenant_2)
- tenant_3 → Streaming Node → Data Node (tenant_3)

优势：
- 数据隔离：租户间数据不会混淆
- 性能隔离：一个租户的高负载不影响其他租户
- 灵活扩展：可以为不同租户分配不同的资源
"""

def create_tenant_collection(tenant_id: str):
    """为租户创建 Collection"""
    from pymilvus import CollectionSchema, FieldSchema, DataType
    
    # 定义 Schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
    ]
    schema = CollectionSchema(fields, description=f"Tenant {tenant_id} collection")
    
    # 创建 Collection
    collection = Collection(name=f"tenant_{tenant_id}", schema=schema)
    print(f"Collection created for tenant {tenant_id}")
    
    return collection

# 为 1000 个租户创建 Collection
# Milvus 2.6 支持 100K collections
for i in range(1000):
    create_tenant_collection(f"tenant_{i}")
```

---

## Streaming Node vs 传统消息队列

### 对比表

| 特性 | Kafka/Pulsar | Streaming Node |
|------|-------------|----------------|
| **部署** | 独立部署 | 内置组件 |
| **运维** | 需要独立运维 | 统一运维 |
| **延迟** | 50-100ms | 20-50ms |
| **成本** | 额外成本 | 无额外成本 |
| **集成** | 需要适配 | 深度集成 |
| **优化** | 通用优化 | 向量数据优化 |
| **监控** | 独立监控 | 统一监控 |
| **故障排查** | 复杂 | 简单 |

### 成本对比

```
传统架构（Kafka/Pulsar）：
  Milvus 集群：$10,000/月
  Kafka 集群：$5,000/月
  总成本：$15,000/月

Milvus 2.6（Streaming Node）：
  Milvus 集群（含 Streaming Node）：$10,000/月
  节省：$5,000/月（33%）
```

---

## 学习检查清单

- [ ] 理解 Streaming Node 的诞生背景
- [ ] 理解 Streaming Node 的核心功能
- [ ] 理解 Streaming Node 与 Woodpecker WAL 的集成
- [ ] 理解 Streaming Node 的工作原理
- [ ] 能够运行手写实现的 Streaming Node 模拟
- [ ] 理解 Streaming Node 在 RAG 系统中的应用
- [ ] 理解 Streaming Node vs 传统消息队列的区别
- [ ] 理解 Streaming Node 的成本优势

---

> **来源**: 
> - [Milvus 2.6 Streaming Node Design](https://github.com/milvus-io/milvus/blob/master/docs/design_docs/20240618-streaming_node.md)
> - [Milvus 2.6 Release Notes](https://milvus.io/docs/release_notes.md)
> - 获取时间: 2026-02-21

**记住：** Streaming Node 是 Milvus 2.6 的核心创新，理解它的工作原理对于优化 RAG 系统至关重要。
