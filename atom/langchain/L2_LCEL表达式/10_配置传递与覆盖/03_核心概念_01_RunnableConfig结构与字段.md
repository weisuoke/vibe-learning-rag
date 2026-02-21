# 核心概念01：RunnableConfig结构与字段

> **本节目标**: 深入理解RunnableConfig的TypedDict设计和8个配置字段的详细用法

---

## 一、RunnableConfig定义

### 1.1 源码定义

```python
# langchain_core/runnables/config.py:51-123
from typing import TypedDict, Optional, List, Dict, Any
from langchain_core.callbacks import BaseCallbackHandler

class RunnableConfig(TypedDict, total=False):
    """
    运行时配置字典

    total=False表示所有字段都是可选的
    """
    tags: Optional[List[str]]
    metadata: Optional[Dict[str, Any]]
    callbacks: Optional[List[BaseCallbackHandler]]
    run_name: Optional[str]
    max_concurrency: Optional[int]
    recursion_limit: Optional[int]
    configurable: Optional[Dict[str, Any]]
    run_id: Optional[str]
```

### 1.2 TypedDict设计原理

**为什么使用TypedDict？**

1. **轻量级**: 就是普通字典，无需实例化
2. **类型安全**: 提供类型提示，IDE自动补全
3. **易于合并**: 使用字典操作
4. **易于序列化**: 直接json.dumps

**total=False的作用**：

```python
# total=False: 所有字段可选
config: RunnableConfig = {"tags": ["test"]}  # ✓ 有效

# 如果total=True（默认）
config: RunnableConfig = {"tags": ["test"]}  # ✗ 缺少其他7个字段
```

---

## 二、8个配置字段详解

### 2.1 tags（标签列表）

**类型**: `Optional[List[str]]`

**用途**: 为执行添加标签，用于分类、过滤和追踪

**示例**:
```python
config = {
    "tags": ["production", "user-facing", "high-priority"]
}
result = chain.invoke(input, config=config)
```

**常见用法**:
- 环境标签: `["production"]`, `["development"]`, `["staging"]`
- 功能标签: `["chat"]`, `["search"]`, `["summarization"]`
- 优先级标签: `["high-priority"]`, `["low-priority"]`
- 用户类型: `["premium-user"]`, `["free-user"]`

**合并规则**: 列表连接
```python
base = {"tags": ["base"]}
custom = {"tags": ["custom"]}
merged = merge_configs(base, custom)
# 结果: {"tags": ["base", "custom"]}
```

**最佳实践**:
- 使用小写和连字符
- 保持标签简短（<20字符）
- 使用一致的命名约定
- 避免过多标签（<10个）

---

### 2.2 metadata（元数据字典）

**类型**: `Optional[Dict[str, Any]]`

**用途**: 存储任意键值对元数据，用于追踪上下文信息

**示例**:
```python
config = {
    "metadata": {
        "user_id": "user-123",
        "session_id": "session-456",
        "environment": "production",
        "version": "1.0.0",
        "request_id": "req-789"
    }
}
```

**常见用法**:
- 用户信息: `user_id`, `user_email`, `user_role`
- 会话信息: `session_id`, `conversation_id`
- 请求信息: `request_id`, `timestamp`, `ip_address`
- 业务信息: `tenant_id`, `organization_id`, `project_id`

**合并规则**: 字典浅合并
```python
base = {"metadata": {"env": "prod", "version": "1.0"}}
custom = {"metadata": {"user": "alice", "version": "2.0"}}
merged = merge_configs(base, custom)
# 结果: {"metadata": {"env": "prod", "user": "alice", "version": "2.0"}}
```

**安全注意**:
```python
# ❌ 不要存储敏感信息
config = {
    "metadata": {
        "api_key": "sk-1234",  # 危险！
        "password": "secret"   # 危险！
    }
}

# ✓ 只存储非敏感信息
config = {
    "metadata": {
        "user_id": "user-123",
        "environment": "production"
    }
}
```

---

### 2.3 callbacks（回调列表）

**类型**: `Optional[List[BaseCallbackHandler]]`

**用途**: 注册回调处理器，用于追踪、日志、监控

**示例**:
```python
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.tracers import LangChainTracer

config = {
    "callbacks": [
        StdOutCallbackHandler(),
        LangChainTracer(project_name="my-project")
    ]
}
```

**常见回调类型**:
- `StdOutCallbackHandler`: 输出到标准输出
- `LangChainTracer`: LangSmith追踪
- 自定义回调: 继承`BaseCallbackHandler`

**合并规则**: 列表连接
```python
base = {"callbacks": [callback1]}
custom = {"callbacks": [callback2]}
merged = merge_configs(base, custom)
# 结果: {"callbacks": [callback1, callback2]}
# 两个回调都会执行
```

**自定义回调示例**:
```python
from langchain_core.callbacks import BaseCallbackHandler

class MyCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM开始: {prompts}")

    def on_llm_end(self, response, **kwargs):
        print(f"LLM结束: {response}")

config = {"callbacks": [MyCallback()]}
```

---

### 2.4 run_name（运行名称）

**类型**: `Optional[str]`

**用途**: 为这次执行指定一个名称，便于在追踪中识别

**示例**:
```python
config = {
    "run_name": "customer_support_query"
}
result = chain.invoke(input, config=config)
```

**常见用法**:
- 功能名称: `"user_authentication"`, `"document_search"`
- 场景名称: `"onboarding_flow"`, `"checkout_process"`
- 测试名称: `"test_case_1"`, `"integration_test"`

**合并规则**: 后来覆盖
```python
base = {"run_name": "base_run"}
custom = {"run_name": "custom_run"}
merged = merge_configs(base, custom)
# 结果: {"run_name": "custom_run"}
```

**最佳实践**:
- 使用描述性名称
- 使用下划线分隔单词
- 保持简短（<50字符）

---

### 2.5 max_concurrency（最大并发数）

**类型**: `Optional[int]`

**用途**: 限制并行执行的最大并发数，控制资源使用

**示例**:
```python
config = {
    "max_concurrency": 5
}
results = chain.batch(inputs, config=config)
# 最多同时执行5个请求
```

**使用场景**:
- 批处理: 限制同时处理的请求数
- API限流: 避免超过API速率限制
- 资源控制: 防止内存或CPU耗尽

**合并规则**: 后来覆盖
```python
base = {"max_concurrency": 10}
custom = {"max_concurrency": 5}
merged = merge_configs(base, custom)
# 结果: {"max_concurrency": 5}
```

**性能考虑**:
```python
# 无限制（默认）
results = chain.batch(inputs)  # 可能耗尽资源

# 限制并发
config = {"max_concurrency": 5}
results = chain.batch(inputs, config=config)  # 安全

# 根据环境调整
if env == "production":
    max_conc = 3  # 保守
else:
    max_conc = 10  # 激进
```

---

### 2.6 recursion_limit（递归限制）

**类型**: `Optional[int]`

**用途**: 限制递归调用的最大深度，防止无限循环

**默认值**: 25

**示例**:
```python
config = {
    "recursion_limit": 10
}
result = agent.invoke(input, config=config)
# 最多递归10层
```

**使用场景**:
- Agent循环: 防止Agent无限循环
- 递归链: 限制递归深度
- 调试: 快速失败，便于发现问题

**合并规则**: 后来覆盖
```python
base = {"recursion_limit": 25}
custom = {"recursion_limit": 10}
merged = merge_configs(base, custom)
# 结果: {"recursion_limit": 10}
```

**错误处理**:
```python
from langchain_core.runnables import RecursionError

try:
    result = agent.invoke(input, config={"recursion_limit": 5})
except RecursionError as e:
    print(f"超过递归限制: {e}")
```

---

### 2.7 configurable（可配置字段）

**类型**: `Optional[Dict[str, Any]]`

**用途**: 运行时覆盖可配置字段的值

**示例**:
```python
# 定义可配置字段
llm = ChatOpenAI().configurable_fields(
    temperature=ConfigurableField(id="temp")
)

# 运行时配置
config = {
    "configurable": {
        "temp": 0.9
    }
}
result = llm.invoke(input, config=config)
```

**常见用法**:
- LLM参数: `temperature`, `max_tokens`, `model_name`
- Retriever参数: `k`, `score_threshold`
- 自定义参数: 任意可配置字段

**合并规则**: 字典浅合并
```python
base = {"configurable": {"temp": 0.5, "k": 5}}
custom = {"configurable": {"temp": 0.9}}
merged = merge_configs(base, custom)
# 结果: {"configurable": {"temp": 0.9, "k": 5}}
```

**详细用法**: 参见 [核心概念05 - 可配置字段系统](./03_核心概念_05_可配置字段系统.md)

---

### 2.8 run_id（运行ID）

**类型**: `Optional[str]`

**用途**: 唯一标识这次执行，通常自动生成

**示例**:
```python
import uuid

config = {
    "run_id": str(uuid.uuid4())
}
result = chain.invoke(input, config=config)
```

**使用场景**:
- 追踪: 关联多个相关的执行
- 调试: 定位特定的执行
- 幂等性: 避免重复执行

**合并规则**: 后来覆盖
```python
base = {"run_id": "id-1"}
custom = {"run_id": "id-2"}
merged = merge_configs(base, custom)
# 结果: {"run_id": "id-2"}
```

**注意**: 通常不需要手动设置，LangChain会自动生成

---

## 三、字段交互与优先级

### 3.1 字段独立性

大多数字段是独立的，互不影响：

```python
config = {
    "tags": ["test"],
    "max_concurrency": 5,
    "callbacks": [MyCallback()]
}
# tags不影响max_concurrency
# max_concurrency不影响callbacks
```

### 3.2 字段组合

某些字段经常一起使用：

```python
# 可观测性组合
config = {
    "tags": ["production"],
    "metadata": {"user_id": "123"},
    "callbacks": [LangChainTracer()],
    "run_name": "user_query"
}

# 性能控制组合
config = {
    "max_concurrency": 5,
    "recursion_limit": 10
}

# 动态配置组合
config = {
    "configurable": {"model": "gpt-4", "temperature": 0.7}
}
```

### 3.3 优先级规则

在配置合并时：

1. **简单字段**: 后来覆盖（run_name, max_concurrency, recursion_limit, run_id）
2. **列表字段**: 连接（callbacks, tags）
3. **字典字段**: 浅合并（metadata, configurable）

---

## 四、2025-2026更新

### 4.1 API稳定性

根据研究材料（temp/04_RunnableConfig_2025_2026_updates.md）：

- RunnableConfig核心结构在2025-2026保持稳定
- 主要变化在LangGraph生态系统的使用模式
- 未来可能引入context API替代configurable嵌套

### 4.2 兼容性考虑

- MCP适配器集成中的配置传递问题已解决
- 中间件函数现在可以访问RunnableConfig
- 导入路径保持兼容

---

## 五、实践建议

### 5.1 最小配置

```python
# 只提供需要的字段
config = {"tags": ["test"]}
result = chain.invoke(input, config=config)
```

### 5.2 完整配置

```python
# 生产环境完整配置
config = {
    "tags": ["production", "user-facing"],
    "metadata": {
        "user_id": "user-123",
        "session_id": "session-456",
        "environment": "production"
    },
    "callbacks": [LangChainTracer(project_name="prod")],
    "run_name": "customer_support_query",
    "max_concurrency": 3,
    "recursion_limit": 10
}
```

### 5.3 配置验证

```python
def validate_config(config: RunnableConfig) -> RunnableConfig:
    """验证配置"""
    if "max_concurrency" in config:
        assert 1 <= config["max_concurrency"] <= 100

    if "recursion_limit" in config:
        assert 1 <= config["recursion_limit"] <= 100

    if "metadata" in config:
        # 检查敏感键
        sensitive = ["api_key", "password", "token"]
        for key in config["metadata"]:
            assert not any(s in key.lower() for s in sensitive)

    return config
```

---

## 六、常见陷阱

1. **在metadata中存储秘密**: 使用环境变量或SecretStr
2. **过多的tags**: 保持在10个以内
3. **忘记设置max_concurrency**: 批处理时可能耗尽资源
4. **手动设置run_id**: 通常不需要，让LangChain自动生成
5. **混淆configurable和metadata**: configurable用于动态字段，metadata用于追踪信息

---

## 七、下一步

- 深入学习配置合并: [核心概念02 - 配置合并与覆盖策略](./03_核心概念_02_配置合并与覆盖策略.md)
- 理解配置继承: [核心概念03 - 配置继承机制](./03_核心概念_03_配置继承机制.md)
- 实战练习: [实战代码01 - 基础配置传递](./07_实战代码_01_基础配置传递.md)
