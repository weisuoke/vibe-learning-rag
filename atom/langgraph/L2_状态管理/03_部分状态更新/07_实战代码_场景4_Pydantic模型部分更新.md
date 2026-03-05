# 实战代码：场景4 - Pydantic模型部分更新

> 使用Pydantic模型实现类型安全的部分状态更新，展示实际业务场景中的应用

---

## 概述

本文档提供3个完整的实战场景，展示如何在LangGraph中使用Pydantic模型进行部分状态更新。每个场景都是完整可运行的代码，展示了不同的业务应用。

**核心价值：**
- 类型安全：编译时类型检查，运行时验证
- 智能更新：自动识别被设置的字段
- 代码清晰：明确的字段定义和验证规则

**[来源: reference/source_部分状态更新_01.md:98-131, reference/context7_langgraph_01.md:69-86]**

---

## 场景1：用户资料管理系统

这个场景展示了一个用户资料管理系统，多个节点分别更新用户的不同字段。

```python
"""
场景1：用户资料管理系统
演示：使用Pydantic模型进行部分状态更新
"""

from typing import TypedDict, Optional
from pydantic import BaseModel, Field, validator
from langgraph.graph import StateGraph, START, END

# ===== 1. 定义状态 =====
class UserState(TypedDict):
    """用户状态"""
    user_id: str
    name: str
    email: str
    age: int
    phone: str
    is_verified: bool
    profile_complete: bool

# ===== 2. 定义Pydantic更新模型 =====
class UserUpdate(BaseModel):
    """用户信息更新模型 - 所有字段都是可选的"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    email: Optional[str] = Field(None, pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    age: Optional[int] = Field(None, ge=18, le=150)
    phone: Optional[str] = Field(None, pattern=r'^\d{10,15}$')
    is_verified: Optional[bool] = None
    profile_complete: Optional[bool] = None

    @validator('email')
    def validate_email(cls, v):
        if v and '@' not in v:
            raise ValueError('Invalid email format')
        return v

# ===== 3. 节点函数 =====
def update_basic_info(state: UserState) -> UserUpdate:
    """更新基本信息"""
    print(f"\n=== 节点1: 更新用户 {state['user_id']} 的基本信息 ===")
    
    # 只更新name和age
    update = UserUpdate(name="Alice Smith", age=30)
    print(f"更新字段: {update.model_fields_set}")  # {'name', 'age'}
    
    return update

def update_contact(state: UserState) -> UserUpdate:
    """更新联系方式"""
    print(f"\n=== 节点2: 更新联系方式 ===")
    
    # 只更新email和phone
    update = UserUpdate(
        email="alice.smith@example.com",
        phone="1234567890"
    )
    print(f"更新字段: {update.model_fields_set}")  # {'email', 'phone'}
    
    return update

def verify_user(state: UserState) -> UserUpdate:
    """验证用户"""
    print(f"\n=== 节点3: 验证用户 ===")
    
    # 只更新is_verified
    update = UserUpdate(is_verified=True)
    print(f"更新字段: {update.model_fields_set}")  # {'is_verified'}
    
    return update

def check_profile_complete(state: UserState) -> UserUpdate:
    """检查资料完整性"""
    print(f"\n=== 节点4: 检查资料完整性 ===")
    
    # 检查所有必填字段
    is_complete = all([
        state.get('name'),
        state.get('email'),
        state.get('age', 0) >= 18,
        state.get('phone'),
    ])
    
    # 只更新profile_complete
    update = UserUpdate(profile_complete=is_complete)
    print(f"资料完整: {is_complete}")
    print(f"更新字段: {update.model_fields_set}")
    
    return update

# ===== 4. 构建图 =====
def build_user_management_graph():
    """构建用户管理图"""
    graph = StateGraph(UserState)
    
    # 添加节点
    graph.add_node("update_basic", update_basic_info)
    graph.add_node("update_contact", update_contact)
    graph.add_node("verify", verify_user)
    graph.add_node("check_complete", check_profile_complete)
    
    # 添加边
    graph.add_edge(START, "update_basic")
    graph.add_edge("update_basic", "update_contact")
    graph.add_edge("update_contact", "verify")
    graph.add_edge("verify", "check_complete")
    graph.add_edge("check_complete", END)
    
    return graph.compile()

# ===== 5. 运行示例 =====
if __name__ == "__main__":
    # 初始状态
    initial_state = {
        "user_id": "user_001",
        "name": "",
        "email": "",
        "age": 0,
        "phone": "",
        "is_verified": False,
        "profile_complete": False,
    }
    
    print("=" * 60)
    print("初始状态:")
    print(initial_state)
    print("=" * 60)
    
    # 运行图
    app = build_user_management_graph()
    result = app.invoke(initial_state)
    
    print("\n" + "=" * 60)
    print("最终状态:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("=" * 60)
```

**运行输出：**
```
============================================================
初始状态:
{'user_id': 'user_001', 'name': '', 'email': '', 'age': 0, 'phone': '', 'is_verified': False, 'profile_complete': False}
============================================================

=== 节点1: 更新用户 user_001 的基本信息 ===
更新字段: {'name', 'age'}

=== 节点2: 更新联系方式 ===
更新字段: {'email', 'phone'}

=== 节点3: 验证用户 ===
更新字段: {'is_verified'}

=== 节点4: 检查资料完整性 ===
资料完整: True
更新字段: {'profile_complete'}

============================================================
最终状态:
  user_id: user_001
  name: Alice Smith
  email: alice.smith@example.com
  age: 30
  phone: 1234567890
  is_verified: True
  profile_complete: True
============================================================
```

**关键特性：**
1. **智能部分更新**：每个节点只更新需要的字段
2. **类型验证**：Pydantic自动验证字段类型和格式
3. **字段追踪**：`model_fields_set`清晰显示哪些字段被更新
4. **业务逻辑清晰**：每个节点职责单一

**[来源: reference/source_部分状态更新_01.md:98-131]**

---

## 场景2：订单状态更新系统

这个场景展示了一个订单处理系统，根据不同的业务逻辑更新订单状态。

```python
"""
场景2：订单状态更新系统
演示：使用Pydantic模型处理复杂的订单状态流转
"""

from typing import TypedDict, Optional, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from datetime import datetime

# ===== 1. 定义状态 =====
class OrderState(TypedDict):
    """订单状态"""
    order_id: str
    status: Literal["pending", "paid", "processing", "shipped", "delivered", "cancelled"]
    amount: float
    payment_method: str
    shipping_address: str
    tracking_number: str
    notes: str
    updated_at: str

# ===== 2. 定义Pydantic更新模型 =====
class OrderUpdate(BaseModel):
    """订单更新模型"""
    status: Optional[Literal["pending", "paid", "processing", "shipped", "delivered", "cancelled"]] = None
    amount: Optional[float] = Field(None, gt=0)
    payment_method: Optional[str] = None
    shipping_address: Optional[str] = None
    tracking_number: Optional[str] = None
    notes: Optional[str] = None
    updated_at: Optional[str] = None

# ===== 3. 节点函数 =====
def process_payment(state: OrderState) -> OrderUpdate:
    """处理支付"""
    print(f"\n=== 处理订单 {state['order_id']} 的支付 ===")
    
    # 只更新支付相关字段
    update = OrderUpdate(
        status="paid",
        payment_method="credit_card",
        updated_at=datetime.now().isoformat()
    )
    print(f"更新字段: {update.model_fields_set}")
    
    return update

def prepare_shipment(state: OrderState) -> OrderUpdate:
    """准备发货"""
    print(f"\n=== 准备发货 ===")
    
    # 只更新发货相关字段
    update = OrderUpdate(
        status="processing",
        shipping_address="123 Main St, City, Country",
        updated_at=datetime.now().isoformat()
    )
    print(f"更新字段: {update.model_fields_set}")
    
    return update

def ship_order(state: OrderState) -> OrderUpdate:
    """发货"""
    print(f"\n=== 订单发货 ===")
    
    # 只更新物流相关字段
    update = OrderUpdate(
        status="shipped",
        tracking_number="TRACK123456789",
        notes="Order shipped via express delivery",
        updated_at=datetime.now().isoformat()
    )
    print(f"更新字段: {update.model_fields_set}")
    
    return update

def deliver_order(state: OrderState) -> OrderUpdate:
    """确认送达"""
    print(f"\n=== 确认送达 ===")
    
    # 只更新状态和时间
    update = OrderUpdate(
        status="delivered",
        updated_at=datetime.now().isoformat()
    )
    print(f"更新字段: {update.model_fields_set}")
    
    return update

# ===== 4. 构建图 =====
def build_order_processing_graph():
    """构建订单处理图"""
    graph = StateGraph(OrderState)
    
    # 添加节点
    graph.add_node("payment", process_payment)
    graph.add_node("prepare", prepare_shipment)
    graph.add_node("ship", ship_order)
    graph.add_node("deliver", deliver_order)
    
    # 添加边
    graph.add_edge(START, "payment")
    graph.add_edge("payment", "prepare")
    graph.add_edge("prepare", "ship")
    graph.add_edge("ship", "deliver")
    graph.add_edge("deliver", END)
    
    return graph.compile()

# ===== 5. 运行示例 =====
if __name__ == "__main__":
    # 初始状态
    initial_state = {
        "order_id": "ORD-2026-001",
        "status": "pending",
        "amount": 99.99,
        "payment_method": "",
        "shipping_address": "",
        "tracking_number": "",
        "notes": "",
        "updated_at": datetime.now().isoformat(),
    }
    
    print("=" * 60)
    print("初始订单状态:")
    for key, value in initial_state.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # 运行图
    app = build_order_processing_graph()
    result = app.invoke(initial_state)
    
    print("\n" + "=" * 60)
    print("最终订单状态:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("=" * 60)
```

**关键特性：**
1. **状态流转清晰**：每个节点负责一个状态转换
2. **字段隔离**：不同节点更新不同的字段组
3. **时间戳追踪**：每次更新都记录时间
4. **业务逻辑封装**：每个节点封装特定的业务逻辑

**[来源: reference/search_部分状态更新_01.md:98-125]**

---

## 场景3：配置管理系统

这个场景展示了一个应用配置管理系统，支持动态配置更新和验证。

```python
"""
场景3：配置管理系统
演示：使用Pydantic模型管理应用配置的部分更新
"""

from typing import TypedDict, Optional, Dict
from pydantic import BaseModel, Field, validator
from langgraph.graph import StateGraph, START, END

# ===== 1. 定义状态 =====
class ConfigState(TypedDict):
    """配置状态"""
    app_name: str
    debug_mode: bool
    max_connections: int
    timeout_seconds: int
    api_key: str
    feature_flags: Dict[str, bool]
    cache_enabled: bool
    log_level: str

# ===== 2. 定义Pydantic更新模型 =====
class ConfigUpdate(BaseModel):
    """配置更新模型"""
    app_name: Optional[str] = Field(None, min_length=1)
    debug_mode: Optional[bool] = None
    max_connections: Optional[int] = Field(None, gt=0, le=10000)
    timeout_seconds: Optional[int] = Field(None, gt=0, le=300)
    api_key: Optional[str] = Field(None, min_length=10)
    feature_flags: Optional[Dict[str, bool]] = None
    cache_enabled: Optional[bool] = None
    log_level: Optional[str] = None

    @validator('log_level')
    def validate_log_level(cls, v):
        if v and v not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            raise ValueError('Invalid log level')
        return v

# ===== 3. 节点函数 =====
def enable_debug_mode(state: ConfigState) -> ConfigUpdate:
    """启用调试模式"""
    print("\n=== 启用调试模式 ===")
    
    # 只更新debug相关配置
    update = ConfigUpdate(
        debug_mode=True,
        log_level="DEBUG"
    )
    print(f"更新字段: {update.model_fields_set}")
    
    return update

def optimize_performance(state: ConfigState) -> ConfigUpdate:
    """优化性能配置"""
    print("\n=== 优化性能配置 ===")
    
    # 只更新性能相关配置
    update = ConfigUpdate(
        max_connections=5000,
        timeout_seconds=30,
        cache_enabled=True
    )
    print(f"更新字段: {update.model_fields_set}")
    
    return update

def update_feature_flags(state: ConfigState) -> ConfigUpdate:
    """更新功能开关"""
    print("\n=== 更新功能开关 ===")
    
    # 只更新feature_flags
    new_flags = state.get('feature_flags', {}).copy()
    new_flags.update({
        'new_feature_a': True,
        'new_feature_b': False,
        'experimental_mode': True
    })
    
    update = ConfigUpdate(feature_flags=new_flags)
    print(f"更新字段: {update.model_fields_set}")
    print(f"新增功能开关: {list(new_flags.keys())}")
    
    return update

def finalize_config(state: ConfigState) -> ConfigUpdate:
    """最终配置检查"""
    print("\n=== 最终配置检查 ===")
    
    # 根据debug模式调整日志级别
    if state.get('debug_mode'):
        log_level = "DEBUG"
    else:
        log_level = "INFO"
    
    update = ConfigUpdate(log_level=log_level)
    print(f"更新字段: {update.model_fields_set}")
    
    return update

# ===== 4. 构建图 =====
def build_config_management_graph():
    """构建配置管理图"""
    graph = StateGraph(ConfigState)
    
    # 添加节点
    graph.add_node("debug", enable_debug_mode)
    graph.add_node("optimize", optimize_performance)
    graph.add_node("features", update_feature_flags)
    graph.add_node("finalize", finalize_config)
    
    # 添加边
    graph.add_edge(START, "debug")
    graph.add_edge("debug", "optimize")
    graph.add_edge("optimize", "features")
    graph.add_edge("features", "finalize")
    graph.add_edge("finalize", END)
    
    return graph.compile()

# ===== 5. 运行示例 =====
if __name__ == "__main__":
    # 初始状态
    initial_config = {
        "app_name": "MyApp",
        "debug_mode": False,
        "max_connections": 100,
        "timeout_seconds": 60,
        "api_key": "secret_key_123456",
        "feature_flags": {"old_feature": True},
        "cache_enabled": False,
        "log_level": "INFO",
    }
    
    print("=" * 60)
    print("初始配置:")
    for key, value in initial_config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # 运行图
    app = build_config_management_graph()
    result = app.invoke(initial_config)
    
    print("\n" + "=" * 60)
    print("最终配置:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("=" * 60)
```

**关键特性：**
1. **配置验证**：Pydantic自动验证配置值的合法性
2. **增量更新**：只更新需要改变的配置项
3. **字典合并**：feature_flags字段展示了字典的合并更新
4. **条件更新**：根据当前状态决定更新内容

**[来源: reference/search_部分状态更新_01.md:98-125]**

---

## 最佳实践总结

### 1. 定义清晰的更新模型

```python
# ✅ 好的实践：所有字段都是Optional
class UserUpdate(BaseModel):
    """用户信息更新模型"""
    name: Optional[str] = Field(None, description="用户名")
    email: Optional[str] = Field(None, description="邮箱")
    age: Optional[int] = Field(None, ge=0, description="年龄")

# ❌ 不好的实践：必填字段失去灵活性
class UserUpdate(BaseModel):
    name: str  # 必填
    email: str  # 必填
```

### 2. 使用字段验证

```python
from pydantic import validator

class UserUpdate(BaseModel):
    email: Optional[str] = None
    age: Optional[int] = Field(None, ge=18, le=150)
    
    @validator('email')
    def validate_email(cls, v):
        if v and '@' not in v:
            raise ValueError('Invalid email format')
        return v
```

### 3. 区分None值和未设置

```python
# 场景1：清空字段（设置为None）
update = UserUpdate(bio=None)  # bio会被设置为None

# 场景2：不更新字段
update = UserUpdate()  # bio保持不变
```

### 4. 复用更新模型

```python
# 定义通用的更新模型
class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None

# 在多个节点中复用
def node_a(state: State) -> UserUpdate:
    return UserUpdate(name="Alice")

def node_b(state: State) -> UserUpdate:
    return UserUpdate(email="alice@example.com")
```

**[来源: reference/search_部分状态更新_01.md:98-125]**

---

## 常见陷阱

### 陷阱1：忘记设置Optional

```python
# ❌ 错误：所有字段都是必填的
class UserUpdate(BaseModel):
    name: str
    email: str

# ✅ 正确：所有字段都是可选的
class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
```

### 陷阱2：混淆None值和未设置

```python
# 显式设置为None
update1 = DataUpdate(value=None)
print(update1.model_fields_set)  # {'value'}

# 不设置字段
update2 = DataUpdate()
print(update2.model_fields_set)  # set()
```

### 陷阱3：忽略字段验证错误

```python
from pydantic import ValidationError

try:
    update = UserUpdate(age=200)  # 超出范围
except ValidationError as e:
    print(f"验证错误: {e}")
    update = UserUpdate()  # 返回空更新
```

**[来源: reference/source_部分状态更新_01.md:98-131]**

---

## 总结

### 核心要点

1. **智能识别**：通过`model_fields_set`自动识别被设置的字段
2. **类型安全**：编译时类型检查，运行时验证
3. **默认值跳过**：未设置的字段不会触发更新
4. **None值处理**：显式设置None会触发更新

### 使用场景

- 需要类型安全的状态更新
- 复杂的字段验证逻辑
- 多个节点共享更新模型
- 需要区分None值和未设置

### 与其他机制的对比

| 特性 | 字典返回 | Pydantic模型 | Command对象 |
|------|---------|------------|-------------|
| 类型安全 | ❌ | ✅ | ✅ |
| 字段验证 | ❌ | ✅ | ❌ |
| 智能部分更新 | ❌ | ✅ | ✅ |
| 控制流程 | ❌ | ❌ | ✅ |
| 复杂度 | 低 | 中 | 高 |

**[来源: reference/source_部分状态更新_01.md:98-131, reference/search_部分状态更新_01.md:98-125]**

---

## 参考资料

- [reference/source_部分状态更新_01.md:98-131] - get_update_as_tuples函数实现
- [reference/context7_langgraph_01.md:69-86] - 节点函数返回部分状态
- [reference/search_部分状态更新_01.md:98-125] - 最佳实践和社区经验

---

**版本：** v1.0
**创建时间：** 2026-02-26
**维护者：** Claude Code
