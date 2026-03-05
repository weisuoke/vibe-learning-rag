# 核心概念:Function Executor架构

> Function Executor是Milvus Embedding Functions的核心执行引擎,负责协调多个embedding函数的并行执行

---

## 架构概述

**Function Executor**是Milvus 2.6中embedding functions的执行引擎,负责管理多个embedding函数的生命周期、并行执行和错误处理。

### 核心特点

| 特性 | 值 | 说明 |
|------|-----|------|
| **并行执行** | ✅ 支持 | 使用goroutines并行处理多个函数 |
| **批量验证** | ✅ 支持 | 插入前验证批量大小 |
| **错误聚合** | ✅ 支持 | 收集所有函数的错误信息 |
| **模式感知** | ✅ 支持 | InsertMode vs SearchMode |
| **多函数支持** | ✅ 支持 | 单个collection可配置多个函数 |

**来源**: `reference/source_architecture.md:29-57`, `reference/source_architecture.md:168-286`

---

## 架构设计

### 1. 核心组件

```go
// FunctionExecutor管理多个embedding函数
type FunctionExecutor struct {
    runners []Runner  // 函数运行器列表
}

// Runner接口定义函数执行器
type Runner interface {
    ProcessInsert(ctx context.Context, msg *msgstream.InsertMsg) error
    ProcessSearch(ctx context.Context, req *internalpb.SearchRequest) error
    ProcessBulkInsert(ctx context.Context, req *internalpb.BulkInsertRequest) error
}
```

**来源**: `reference/source_architecture.md:29-57`

### 2. 执行流程

```
用户请求
    ↓
FunctionExecutor
    ↓
并行执行多个Runner
    ├─→ Runner1 (OpenAI)
    ├─→ Runner2 (VoyageAI)
    └─→ Runner3 (Cohere)
    ↓
聚合结果/错误
    ↓
返回给用户
```

### 3. 关键方法

**ProcessInsert()**:
- 处理插入操作
- 并行执行所有embedding函数
- 验证批量大小
- 聚合错误信息

**ProcessSearch()**:
- 处理搜索操作
- 支持常规搜索和高级搜索
- 生成查询embedding
- 替换占位符

**ProcessBulkInsert()**:
- 处理批量插入操作
- 串行执行(非并行)
- 适用于大规模数据导入

**来源**: `reference/source_architecture.md:168-286`

---

## 并行执行机制

### 1. 并行执行实现

```go
// 并行执行多个embedding函数
func (executor *FunctionExecutor) ProcessInsert(ctx context.Context, msg *msgstream.InsertMsg) error {
    var wg sync.WaitGroup
    errChan := make(chan error, len(executor.runners))

    // 为每个函数启动goroutine
    for _, runner := range executor.runners {
        wg.Add(1)
        go func(r Runner) {
            defer wg.Done()
            if err := r.ProcessInsert(ctx, msg); err != nil {
                errChan <- err
            }
        }(runner)
    }

    // 等待所有goroutine完成
    wg.Wait()
    close(errChan)

    // 聚合错误
    var errors []error
    for err := range errChan {
        errors = append(errors, err)
    }

    if len(errors) > 0 {
        return fmt.Errorf("embedding functions failed: %v", errors)
    }

    return nil
}
```

**性能优势**:
- 多个函数并行执行,总延迟 = max(各函数延迟)
- 单函数延迟:300ms,3个函数并行仍为300ms
- 串行执行需要900ms,并行提升3倍

**来源**: `reference/source_architecture.md:168-220`

### 2. 错误隔离

```go
// 错误隔离机制
// 单个函数失败不影响其他函数
for _, runner := range executor.runners {
    go func(r Runner) {
        defer func() {
            if err := recover(); err != nil {
                // 捕获panic,防止影响其他函数
                errChan <- fmt.Errorf("function panic: %v", err)
            }
        }()

        if err := r.ProcessInsert(ctx, msg); err != nil {
            errChan <- err
        }
    }(runner)
}
```

**隔离效果**:
- 单个函数崩溃不影响其他函数
- 所有错误都会被收集和报告
- 用户可以看到哪些函数成功,哪些失败

---

## 批量验证机制

### 1. 批量大小验证

```go
// 插入前验证批量大小
func (executor *FunctionExecutor) ProcessInsert(ctx context.Context, msg *msgstream.InsertMsg) error {
    numRows := msg.GetNumRows()

    // 验证每个函数的批量大小
    for _, runner := range executor.runners {
        maxBatch := runner.MaxBatch()
        if numRows > maxBatch {
            return fmt.Errorf("batch size %d exceeds maximum %d for function %s",
                numRows, maxBatch, runner.Name())
        }
    }

    // 批量大小验证通过,执行插入
    return executor.parallelExecute(ctx, msg)
}
```

**验证规则**:
- 插入前检查批量大小
- 任何函数超限都会拒绝整个批次
- 避免部分成功部分失败的情况

**来源**: `reference/source_architecture.md:168-220`

### 2. MaxBatch对比

| Provider | MaxBatch | 验证策略 |
|----------|----------|---------|
| OpenAI | 128 | 超过128拒绝 |
| VoyageAI | 128 | 超过128拒绝 |
| Cohere | 96 | 超过96拒绝 |
| Zilliz | 64 | 超过64拒绝 |
| SiliconFlow | 32 | 超过32拒绝 |
| TEI | 32 | 超过32拒绝 |
| DashScope | 25/6 | 超过25/6拒绝 |
| Bedrock | 1 | 超过1拒绝 |

**最小公倍数原则**:
- 如果配置多个函数,批量大小受最小MaxBatch限制
- 例如:OpenAI(128) + DashScope(6) → 实际MaxBatch=6

---

## 模式感知机制

### 1. InsertMode vs SearchMode

```go
// 模式定义
type TextEmbeddingMode int

const (
    InsertMode TextEmbeddingMode = iota  // 文档入库模式
    SearchMode                           // 查询检索模式
)
```

**模式差异**:

| 模式 | 用途 | Provider行为 |
|------|------|-------------|
| **InsertMode** | 文档入库 | input_type="document" |
| **SearchMode** | 查询检索 | input_type="query" |

**Provider适配**:
- VoyageAI: document vs query
- Cohere: search_document vs search_query
- VertexAI: RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY
- TEI: ingestion_prompt vs search_prompt

**来源**: `reference/source_architecture.md:377-388`

### 2. 模式切换实现

```go
// 插入操作使用InsertMode
func (executor *FunctionExecutor) ProcessInsert(ctx context.Context, msg *msgstream.InsertMsg) error {
    for _, runner := range executor.runners {
        runner.SetMode(InsertMode)  // 设置为InsertMode
        runner.ProcessInsert(ctx, msg)
    }
}

// 搜索操作使用SearchMode
func (executor *FunctionExecutor) ProcessSearch(ctx context.Context, req *internalpb.SearchRequest) error {
    for _, runner := range executor.runners {
        runner.SetMode(SearchMode)  // 设置为SearchMode
        runner.ProcessSearch(ctx, req)
    }
}
```

---

## 多函数配置

### 1. 配置多个Embedding函数

```python
from pymilvus import Function, FunctionType, CollectionSchema, FieldSchema, DataType

# 定义Schema with多个embedding函数
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="openai_vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema(name="voyageai_vector", dtype=DataType.FLOAT_VECTOR, dim=1024)
]

# 定义多个embedding函数
openai_ef = Function(
    name="openai_ef",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["openai_vector"],
    params={
        "provider": "openai",
        "model_name": "text-embedding-3-small",
        "api_key": "sk-xxx"
    }
)

voyageai_ef = Function(
    name="voyageai_ef",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["text"],
    output_field_names=["voyageai_vector"],
    params={
        "provider": "voyageai",
        "model_name": "voyage-3-large",
        "api_key": "pa-xxx"
    }
)

# 创建Schema with多个函数
schema = CollectionSchema(
    fields=fields,
    functions=[openai_ef, voyageai_ef]  # 多个函数
)
```

**并行执行效果**:
- OpenAI和VoyageAI并行生成embedding
- 总延迟 = max(OpenAI延迟, VoyageAI延迟)
- 单次插入生成2个向量字段

**来源**: `reference/source_architecture.md:168-220`

### 2. 多函数使用场景

**场景1:多模型对比**
```python
# 同时使用多个模型,对比效果
functions = [
    openai_ef,      # OpenAI embedding
    voyageai_ef,    # VoyageAI embedding
    cohere_ef       # Cohere embedding
]

# 插入时自动生成3个向量
# 检索时可以选择使用哪个向量
```

**场景2:主备模型**
```python
# 主模型 + 备用模型
functions = [
    primary_ef,     # 主模型(高质量)
    backup_ef       # 备用模型(低成本)
]

# 主模型失败时使用备用模型
```

**场景3:多语言支持**
```python
# 不同语言使用不同模型
functions = [
    english_ef,     # 英文专用模型
    chinese_ef      # 中文专用模型
]
```

---

## 错误处理机制

### 1. 错误聚合

```go
// 收集所有函数的错误
var errors []error
for err := range errChan {
    errors = append(errors, err)
}

// 返回聚合错误
if len(errors) > 0 {
    return fmt.Errorf("embedding functions failed: %v", errors)
}
```

**错误信息示例**:
```
embedding functions failed: [
    openai_ef: rate limit exceeded,
    voyageai_ef: API key invalid
]
```

### 2. 部分失败处理

```python
# 用户代码中的错误处理
from pymilvus import MilvusClient, MilvusException

client = MilvusClient(uri="http://localhost:19530")

try:
    client.insert(
        collection_name="multi_func_collection",
        data=[{"text": "test"}]
    )
except MilvusException as e:
    # 检查错误信息
    if "openai_ef" in str(e):
        print("OpenAI function failed")
    if "voyageai_ef" in str(e):
        print("VoyageAI function failed")
```

---

## 性能优化

### 1. 并行执行性能

**串行执行**:
```
总延迟 = OpenAI(300ms) + VoyageAI(150ms) + Cohere(250ms) = 700ms
```

**并行执行**:
```
总延迟 = max(OpenAI(300ms), VoyageAI(150ms), Cohere(250ms)) = 300ms
```

**性能提升**: 2.3倍

**来源**: `reference/source_architecture.md:409-427`

### 2. 批量处理优化

```python
# 批量大小受最小MaxBatch限制
# OpenAI(128) + DashScope(6) → 实际MaxBatch=6

# 优化策略:分批处理
batch_size = 6  # 最小MaxBatch

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    client.insert(
        collection_name="collection",
        data=[{"text": t} for t in batch]
    )
```

### 3. 资源管理

```go
// 使用context控制超时
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

// 执行embedding函数
err := executor.ProcessInsert(ctx, msg)
if err != nil {
    // 超时或其他错误
}
```

---

## 最佳实践

### 1. 单函数配置(推荐)

```python
# 大多数场景使用单个函数即可
schema = CollectionSchema(
    fields=fields,
    functions=[openai_ef]  # 单个函数
)
```

**优势**:
- 配置简单
- 性能稳定
- 易于调试

### 2. 多函数配置(高级)

```python
# 需要多模型对比或主备模型时使用
schema = CollectionSchema(
    fields=fields,
    functions=[primary_ef, backup_ef]  # 多个函数
)
```

**注意事项**:
- 批量大小受最小MaxBatch限制
- 所有函数必须成功才算成功
- 增加API调用成本

### 3. 错误处理策略

```python
import time
from pymilvus import MilvusClient, MilvusException

def insert_with_retry(client, collection_name, data, max_retries=3):
    """带重试的插入操作"""
    for attempt in range(max_retries):
        try:
            result = client.insert(
                collection_name=collection_name,
                data=data
            )
            return result
        except MilvusException as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg:
                # 速率限制错误,等待后重试
                wait_time = 2 ** attempt
                print(f"Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif "batch size" in error_msg:
                # 批量大小错误,减小批量
                print("Batch size exceeded. Reduce batch size")
                raise
            else:
                # 其他错误,直接抛出
                raise
    raise Exception(f"Failed after {max_retries} retries")
```

---

## 生产环境建议

### 1. 监控与告警

```python
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_function_executor(client, collection_name, data):
    """监控Function Executor性能"""
    start_time = time.time()
    batch_size = len(data)

    try:
        result = client.insert(collection_name=collection_name, data=data)
        duration = time.time() - start_time

        # 计算性能指标
        throughput = batch_size / duration
        avg_latency = duration / batch_size * 1000  # ms

        logger.info(f"Function Executor performance:")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Duration: {duration:.2f}s")
        logger.info(f"  Throughput: {throughput:.2f} texts/s")
        logger.info(f"  Avg latency: {avg_latency:.2f}ms")

        # 告警:性能异常
        if avg_latency > 500:
            logger.warning(f"High latency detected: {avg_latency:.2f}ms")

        return result
    except Exception as e:
        logger.error(f"Function Executor failed: {e}")
        raise
```

### 2. 批量大小优化

```python
def get_optimal_batch_size(functions):
    """获取最优批量大小"""
    # 获取所有函数的MaxBatch
    max_batches = [f.max_batch for f in functions]

    # 返回最小值
    return min(max_batches)

# 使用示例
optimal_batch = get_optimal_batch_size([openai_ef, dashscope_ef])
print(f"Optimal batch size: {optimal_batch}")  # 输出:6
```

### 3. 成本优化

```python
# 策略1:使用单个函数(降低API调用次数)
functions = [openai_ef]  # 单个函数

# 策略2:选择低成本Provider
functions = [voyageai_lite_ef]  # 低成本模型

# 策略3:批量处理(最大化批量大小)
batch_size = get_optimal_batch_size(functions)
```

---

## 参考资料

1. **源码分析**: `reference/source_architecture.md:29-57`, `reference/source_architecture.md:168-286`
   - Function Executor实现细节
   - 并行执行机制
   - 错误处理策略

2. **社区实践**: `reference/search_github.md:1-140`
   - 多函数配置案例
   - 生产环境部署经验
   - 性能优化建议

---

## 总结

**Function Executor核心优势**:
1. **并行执行**:多个函数并行处理,性能提升2-3倍
2. **错误隔离**:单个函数失败不影响其他函数
3. **批量验证**:插入前验证批量大小,避免部分失败
4. **模式感知**:自动适配InsertMode和SearchMode

**适用场景**:
- 单函数配置:大多数场景(推荐)
- 多函数配置:多模型对比、主备模型、多语言支持

**关键注意事项**:
- 多函数配置时,批量大小受最小MaxBatch限制
- 所有函数必须成功才算成功
- 并行执行可显著提升性能

**性能指标**:
- 并行执行:总延迟 = max(各函数延迟)
- 串行执行:总延迟 = sum(各函数延迟)
- 性能提升:2-3倍(取决于函数数量)

---

**文档版本**: v1.0
**最后更新**: 2026-02-24
**维护者**: Claude Code
