---
type: source_code_analysis
source: sourcecode/langgraph
analyzed_files:
  - libs/checkpoint/langgraph/checkpoint/serde/base.py
  - libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py
  - libs/checkpoint/langgraph/checkpoint/serde/encrypted.py
  - libs/checkpoint/langgraph/checkpoint/serde/types.py
  - libs/checkpoint/langgraph/checkpoint/base/__init__.py
  - libs/langgraph/langgraph/channels/base.py
analyzed_at: 2026-02-27
knowledge_point: 10_状态持久化准备
---

# 源码分析：LangGraph 状态持久化核心架构

## 分析的文件

### 1. serde/base.py - 序列化协议定义

**关键类：**

- `UntypedSerializerProtocol` - 旧版序列化接口（dumps/loads）
- `SerializerProtocol` - 现代类型化序列化接口（dumps_typed/loads_typed）
- `SerializerCompat` - 适配器，将旧接口包装为新接口
- `CipherProtocol` - 加密接口（encrypt/decrypt）

**设计模式：**
- Protocol 模式（结构化子类型）：所有接口用 Protocol 定义，不需要继承
- 适配器模式：SerializerCompat 适配旧序列化器
- `maybe_add_typed_methods` 工厂函数：自动包装旧序列化器

**核心接口：**
```python
class SerializerProtocol(Protocol):
    def dumps_typed(self, obj: Any) -> tuple[str, bytes]: ...
    def loads_typed(self, data: tuple[str, bytes]) -> Any: ...
```

类型标签（type tag）是关键：告诉反序列化器如何解码字节（如 "msgpack"、"json"、"bytes"、"pickle"）。

### 2. serde/jsonplus.py - 核心序列化器实现

**关键类：JsonPlusSerializer**

这是所有 checkpoint saver 使用的默认序列化器。尽管名字叫 JsonPlus，实际主要使用 msgpack（通过 ormsgpack）。

**构造参数：**
- `pickle_fallback`: 默认 False，开启后无法 msgpack 编码的对象回退到 pickle
- `allowed_json_modules`: JSON 构造器反序列化的模块白名单
- `__unpack_ext_hook__`: 自定义 msgpack 扩展类型钩子

**序列化流程（dumps_typed）：**
```
obj → None? → ("null", b"")
obj → bytes? → ("bytes", obj)
obj → bytearray? → ("bytearray", obj)
obj → 其他 → try msgpack → ("msgpack", encoded)
                → 失败且 pickle_fallback → ("pickle", pickled)
```

**Msgpack 扩展类型系统（7种）：**

| 代码 | 用途 | 重建策略 |
|------|------|----------|
| 0 | UUID, Decimal, set, frozenset, deque, Enum, IP | Class(single_arg) |
| 1 | Path, regex, timedelta, date, timezone, Send | Class(*args) |
| 2 | NamedTuple, time, dataclass, Item | Class(**kwargs) |
| 3 | datetime | Class.method(arg) |
| 4 | Pydantic v1 | Class(**dict) |
| 5 | Pydantic v2 | Class(**dict) |
| 6 | NumPy arrays | np.frombuffer + reshape |

**状态大小管理：**
- msgpack 比 JSON 产生更小的二进制输出
- NumPy 数组使用 memoryview 零拷贝（C-contiguous）
- 没有内置压缩

**安全考虑：**
- Pydantic SecretStr/SecretBytes 通过 get_secret_value() 提取 → 明文存储（除非启用加密）
- allowed_json_modules 白名单控制反序列化时的模块导入
- pickle_fallback 默认关闭（pickle 反序列化不安全）
- 文档明确警告：不应用于不受信任的 Python 对象

### 3. serde/encrypted.py - 加密层

**关键类：EncryptedSerializer**

装饰器模式 - 包装任何 SerializerProtocol 并添加加密。

**加密流程：**
```
dumps_typed: obj → serde.dumps_typed → (type, bytes) → cipher.encrypt(bytes) → (type+ciphername, ciphertext)
loads_typed: (type+ciphername, ciphertext) → cipher.decrypt → (type, bytes) → serde.loads_typed → obj
```

**类型标签格式：** `"msgpack"` → `"msgpack+aes"`（用 `+` 分隔）
- 向后兼容：没有 `+` 的标签视为未加密
- 自描述：密文携带自己的解密方法标识

**AES 工厂方法：**
```python
EncryptedSerializer.from_pycryptodome_aes()
```
- 密钥来源：参数传入或 `LANGGRAPH_AES_KEY` 环境变量
- 密钥长度：16/24/32 字节（AES-128/192/256）
- 加密模式：AES EAX（认证加密，提供机密性+完整性）
- 线格式：nonce(16B) + tag(16B) + ciphertext

### 4. serde/types.py - 通道和发送协议类型

**常量：**
- ERROR = "__error__"
- SCHEDULED = "__scheduled__"
- INTERRUPT = "__interrupt__"
- RESUME = "__resume__"
- TASKS = "__pregel_tasks"

**镜像协议：**
- `ChannelProtocol` - 镜像 BaseChannel 接口
- `SendProtocol` - 镜像 Send 接口

设计目的：打破循环依赖（checkpoint 库不依赖 langgraph 核心）

### 5. base/__init__.py - Checkpoint Saver 基类

**Checkpoint 数据结构：**
```python
class Checkpoint(TypedDict):
    v: int                              # 格式版本（当前为 2）
    id: str                             # 唯一 ID（uuid6，单调递增）
    ts: str                             # ISO 8601 时间戳
    channel_values: dict[str, Any]      # 序列化的通道快照
    channel_versions: ChannelVersions   # 每个通道的版本跟踪
    versions_seen: dict[str, ChannelVersions]  # 每个节点看到的版本
    updated_channels: list[str] | None  # 哪些通道发生了变化
```

**BaseCheckpointSaver 关键设计：**
- 默认序列化器：JsonPlusSerializer()
- 双同步/异步 API
- 版本管理：get_next_version() 生成单调递增版本 ID
- CRUD 操作：get/put/list/delete_thread/copy_thread/prune

**状态大小管理：**
- create_checkpoint 只包含有版本的通道（已更新过的）
- 空通道排除在快照之外
- prune() 方法支持 "keep_latest" 和 "delete" 策略
- 元数据清理：过滤 null 字节、排除大字段（writes）

### 6. channels/base.py - 通道基类

**BaseChannel 持久化相关方法：**
- `checkpoint()` → 返回可序列化快照（调用 get()，空则返回 MISSING）
- `from_checkpoint(checkpoint)` → 从快照恢复通道状态
- `copy()` → 通过序列化/反序列化往返创建深拷贝

## 完整持久化管道

```
Channel.checkpoint()          -- 产生可序列化值
    ↓
SerializerProtocol.dumps_typed()  -- 序列化为 (type_tag, bytes)
    ↓
[可选] CipherProtocol.encrypt()  -- 加密字节
    ↓
CheckpointSaver.put()        -- 存储到数据库/文件
```

反向：
```
CheckpointSaver.get_tuple()   -- 从存储加载
    ↓
[可选] CipherProtocol.decrypt()  -- 解密字节
    ↓
SerializerProtocol.loads_typed()  -- 反序列化
    ↓
Channel.from_checkpoint()     -- 恢复通道状态
```

## 三层关注点分离

1. **数据层**（BaseChannel）：知道如何产生和消费可序列化的状态快照
2. **序列化层**（JsonPlusSerializer, EncryptedSerializer）：将 Python 对象转换为字节并返回，可选加密
3. **存储层**（BaseCheckpointSaver）：将序列化字节持久化到后端（Postgres、SQLite、内存等）
