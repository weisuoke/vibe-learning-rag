# 实战代码场景 2：Streaming Node 源码解析

> **来源**: [Milvus Repository](https://github.com/milvus-io/milvus) | [Milvus 2.6 Release Notes](https://milvus.io/docs/release_notes.md) | [Woodpecker Configuration](https://milvus.io/docs/use-woodpecker.md) | [Architecture Overview](https://milvus.io/docs/architecture_overview.md) | 获取时间: 2026-02-21

---

## 一、源码概览

### 1.1 Streaming Node 在 Milvus 2.6 中的位置

Streaming Node 是 Milvus 2.6 的核心新组件，负责处理实时数据流和 WAL 管理。

**源码目录结构：**
```
sourcecode/milvus/
├── cmd/components/
│   └── streaming_node.go          # Streaming Node 入口
├── internal/streamingnode/
│   ├── server/
│   │   ├── server.go              # 服务器主逻辑
│   │   ├── walmanager/            # WAL 管理器
│   │   │   ├── manager.go         # WAL 管理接口
│   │   │   ├── manager_impl.go    # WAL 管理实现
│   │   │   ├── wal_lifetime.go    # WAL 生命周期
│   │   │   └── wal_state.go       # WAL 状态管理
│   │   ├── resource/              # 资源管理
│   │   └── service/               # gRPC 服务
│   └── client/                    # 客户端接口
└── configs/milvus.yaml            # 配置文件
```

### 1.2 核心组件

**Streaming Node 的三大核心组件：**

1. **Server**：服务器主逻辑，负责初始化和生命周期管理
2. **WAL Manager**：WAL 管理器，负责 WAL 的创建、读写、状态管理
3. **Service**：gRPC 服务，提供外部接口

---

## 二、核心实现解析

### 2.1 Server 初始化（server.go）

**源码片段（简化）：**

```go
// Server is the streamingnode server.
type Server struct {
    session    *sessionutil.Session
    grpcServer *grpc.Server

    // service level instances
    handlerService service.HandlerService
    managerService service.ManagerService

    // basic component instances
    walManager walmanager.Manager
}

// Init initializes the streamingnode server.
func (s *Server) init() {
    log.Info("init streamingnode server...")

    // 1. 初始化基础组件
    s.initBasicComponent()

    // 2. 初始化服务
    s.initService()

    // 3. 初始化文件资源管理器
    fileresource.InitManager(
        resource.Resource().ChunkManager(),
        fileresource.ParseMode(paramtable.Get().CommonCfg.QNFileResourceMode.GetValue()),
    )

    // 4. 初始化查询引擎
    if err := initcore.InitQueryNode(context.TODO()); err != nil {
        panic(fmt.Sprintf("init query node segcore failed, %+v", err))
    }

    log.Info("streamingnode server initialized")
}

// initBasicComponent initialize all underlying dependency
func (s *Server) initBasicComponent() {
    var err error

    // 打开 WAL 管理器
    s.walManager, err = walmanager.OpenManager()
    if err != nil {
        panic(fmt.Sprintf("open wal manager failed, %+v", err))
    }

    // 注册 WAL 管理器到本地注册表
    registry.RegisterLocalWALManager(s.walManager)
}
```

**关键点解析：**

1. **Server 结构体**：包含 session、gRPC 服务器、服务实例和 WAL 管理器
2. **初始化流程**：
   - 初始化基础组件（WAL 管理器）
   - 初始化 gRPC 服务
   - 初始化文件资源管理器
   - 初始化查询引擎
3. **WAL 管理器**：通过 `walmanager.OpenManager()` 打开，并注册到本地注册表

### 2.2 WAL Manager 实现（walmanager/manager_impl.go）

**WAL Manager 接口定义：**

```go
// Manager is the interface for wal manager.
type Manager interface {
    // Open opens a wal instance.
    Open(ctx context.Context, opt *OpenOption) (WAL, error)

    // GetAvailableWAL returns all available wal info.
    GetAvailableWAL() []WALInfo

    // Close closes the manager.
    Close()
}
```

**WAL Manager 实现（简化）：**

```go
type managerImpl struct {
    // WAL 实例映射
    wals sync.Map // map[string]*walLifetime

    // 配置
    config *config.WalConfig
}

// Open opens a wal instance.
func (m *managerImpl) Open(ctx context.Context, opt *OpenOption) (WAL, error) {
    // 1. 生成 WAL 名称
    walName := opt.Channel.Name

    // 2. 检查是否已存在
    if lifetime, ok := m.wals.Load(walName); ok {
        return lifetime.(*walLifetime).WAL(), nil
    }

    // 3. 创建新的 WAL 实例
    wal, err := m.openWAL(ctx, opt)
    if err != nil {
        return nil, err
    }

    // 4. 创建 WAL 生命周期管理器
    lifetime := newWALLifetime(wal, opt)
    m.wals.Store(walName, lifetime)

    return wal, nil
}

// openWAL creates a new wal instance.
func (m *managerImpl) openWAL(ctx context.Context, opt *OpenOption) (WAL, error) {
    // 根据配置选择 WAL 实现（Woodpecker/Kafka/Pulsar）
    builder := walimpls.GetBuilder(m.config.WALType)

    // 构建 WAL 实例
    wal, err := builder.Build(ctx, &walimpls.BuildOption{
        Channel: opt.Channel,
        Config:  m.config,
    })

    return wal, err
}
```

**关键点解析：**

1. **WAL 实例管理**：使用 `sync.Map` 存储 WAL 实例，支持并发访问
2. **Open 流程**：
   - 生成 WAL 名称（基于 Channel）
   - 检查是否已存在
   - 创建新的 WAL 实例
   - 创建生命周期管理器
3. **WAL 实现选择**：根据配置选择 Woodpecker/Kafka/Pulsar

### 2.3 WAL 生命周期管理（walmanager/wal_lifetime.go）

**WAL Lifetime 结构：**

```go
type walLifetime struct {
    wal    WAL
    option *OpenOption

    // 状态管理
    state    atomic.Value // *walState
    stateMu  sync.Mutex

    // 引用计数
    refCount atomic.Int32
}

// WAL returns the wal instance.
func (w *walLifetime) WAL() WAL {
    w.refCount.Add(1)
    return w.wal
}

// Close closes the wal lifetime.
func (w *walLifetime) Close() {
    if w.refCount.Add(-1) == 0 {
        // 引用计数为 0，关闭 WAL
        w.wal.Close()
    }
}
```

**关键点解析：**

1. **生命周期管理**：使用引用计数管理 WAL 实例的生命周期
2. **状态管理**：使用 `atomic.Value` 存储 WAL 状态，支持并发访问
3. **自动关闭**：当引用计数为 0 时，自动关闭 WAL 实例

---

## 三、关键代码片段

### 3.1 Streaming Node 启动流程

**完整启动流程（cmd/components/streaming_node.go）：**

```go
// NewStreamingNode creates a new StreamingNode
func NewStreamingNode(ctx context.Context, factory dependency.Factory) (*StreamingNode, error) {
    // 1. 创建服务器实例
    svr, err := streamingnode.NewServer(ctx, factory)
    if err != nil {
        return nil, err
    }

    // 2. 返回 Streaming Node
    return &StreamingNode{
        Server: svr,
    }, nil
}

// Run starts the StreamingNode
func (s *StreamingNode) Run() error {
    // 1. 初始化服务器
    s.init()

    // 2. 启动 gRPC 服务
    if err := s.startGRPC(); err != nil {
        return err
    }

    // 3. 注册到 etcd
    if err := s.register(); err != nil {
        return err
    }

    log.Info("StreamingNode started successfully")
    return nil
}
```

### 3.2 WAL 写入流程

**简化的 WAL 写入实现：**

```go
// Append appends a message to the wal.
func (w *walImpl) Append(ctx context.Context, msg *message.ImmutableMessage) (*AppendResult, error) {
    // 1. 序列化消息
    data, err := msg.Marshal()
    if err != nil {
        return nil, err
    }

    // 2. 写入对象存储（零磁盘！）
    offset, err := w.writer.Write(ctx, data)
    if err != nil {
        return nil, err
    }

    // 3. 返回结果
    return &AppendResult{
        MessageID: msg.MessageID(),
        Offset:    offset,
    }, nil
}
```

### 3.3 WAL 读取流程

**简化的 WAL 读取实现：**

```go
// Read reads messages from the wal.
func (w *walImpl) Read(ctx context.Context, opt *ReadOption) (Scanner, error) {
    // 1. 创建读取器
    reader, err := w.createReader(ctx, opt)
    if err != nil {
        return nil, err
    }

    // 2. 返回扫描器
    return &scannerImpl{
        reader: reader,
        option: opt,
    }, nil
}

// Next returns the next message.
func (s *scannerImpl) Next(ctx context.Context) (*message.ImmutableMessage, error) {
    // 1. 从对象存储读取数据（零磁盘！）
    data, err := s.reader.Read(ctx)
    if err != nil {
        return nil, err
    }

    // 2. 反序列化消息
    msg, err := message.Unmarshal(data)
    if err != nil {
        return nil, err
    }

    return msg, nil
}
```

---

## 四、RAG 应用场景

### 4.1 场景 1：监控 Streaming Node 状态

**Python 代码示例：**

```python
from pymilvus import connections, utility
import time

def monitor_streaming_node():
    """监控 Streaming Node 状态"""

    connections.connect(host="localhost", port="19530")

    while True:
        # 获取 Streaming Node 信息
        # 注意：Milvus Python SDK 可能没有直接的 Streaming Node API
        # 这里展示概念性代码

        try:
            # 检查集群健康状态
            if utility.get_server_version():
                print(f"✅ Streaming Node is healthy")

            # 获取 WAL 信息（概念性）
            # wal_info = utility.get_wal_info()
            # print(f"   Active WALs: {len(wal_info)}")

        except Exception as e:
            print(f"❌ Streaming Node error: {e}")

        time.sleep(5)

# 运行监控
monitor_streaming_node()
```

### 4.2 场景 2：RAG 系统的实时数据流处理

**问题：** RAG 系统需要实时处理大量文档插入

**Streaming Node 的作用：**
- 接收实时数据流
- 管理 WAL，保证数据不丢失
- 提供实时查询能力

**Python 代码示例：**

```python
from pymilvus import connections, Collection
import numpy as np
import time

connections.connect(host="localhost", port="19530")

def realtime_rag_ingestion(collection_name: str):
    """RAG 系统的实时数据摄取"""

    collection = Collection(collection_name)

    # 模拟实时数据流
    batch_size = 100
    total_inserted = 0

    while True:
        # 生成批量数据
        documents = [
            {
                "text": f"Real-time document {total_inserted + i}",
                "embedding": np.random.rand(768).tolist(),
                "timestamp": int(time.time())
            }
            for i in range(batch_size)
        ]

        # 插入数据（Streaming Node 处理）
        texts = [doc["text"] for doc in documents]
        embeddings = [doc["embedding"] for doc in documents]
        timestamps = [doc["timestamp"] for doc in documents]

        start_time = time.time()
        collection.insert([texts, embeddings, timestamps])
        elapsed = time.time() - start_time

        total_inserted += batch_size
        throughput = batch_size / elapsed

        print(f"✅ Inserted {batch_size} docs in {elapsed:.2f}s")
        print(f"   Throughput: {throughput:.2f} docs/sec")
        print(f"   Total: {total_inserted} docs")
        print(f"   Streaming Node: Handling real-time WAL")

        time.sleep(1)

# 运行实时摄取
realtime_rag_ingestion("rag_knowledge_base")
```

**Streaming Node 的优势：**
- 零磁盘架构，直接写入对象存储
- 高吞吐量，支持大规模实时数据流
- 自动 WAL 管理，保证数据不丢失
- 实时查询能力，无需等待 Flush

---

## 五、总结

### 5.1 Streaming Node 源码核心要点

1. **模块化设计**：Server、WAL Manager、Service 三层架构
2. **生命周期管理**：使用引用计数管理 WAL 实例
3. **零磁盘实现**：WAL 直接写入对象存储
4. **并发安全**：使用 `sync.Map` 和 `atomic` 保证并发安全

### 5.2 对 RAG 开发的启示

- **实时性**：Streaming Node 提供实时数据流处理能力
- **可靠性**：WAL 机制保证数据不丢失
- **扩展性**：零磁盘架构支持无限扩展
- **简化运维**：无需管理本地磁盘

### 5.3 源码学习建议

1. **从入口开始**：`cmd/components/streaming_node.go`
2. **理解核心逻辑**：`internal/streamingnode/server/server.go`
3. **深入 WAL 管理**：`internal/streamingnode/server/walmanager/`
4. **查看配置**：`configs/milvus.yaml`

---

> **来源**: [Milvus Repository](https://github.com/milvus-io/milvus) | [Milvus 2.6 Release Notes](https://milvus.io/docs/release_notes.md) | [Woodpecker Configuration](https://milvus.io/docs/use-woodpecker.md) | [Architecture Overview](https://milvus.io/docs/architecture_overview.md) | 获取时间: 2026-02-21