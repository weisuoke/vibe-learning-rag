# 自定义Tool - 核心概念02：参数验证与Schema

> 深入理解Pydantic参数验证，确保Tool输入的正确性和安全性

---

## 概述

**为什么需要参数验证？**

Tool的调用者是LLM，不是人类开发者。LLM可能会：
- 提取错误的参数值
- 传递格式不正确的参数
- 遗漏必需的参数
- 传递超出范围的参数

**参数验证的价值：**
- ✅ 在执行前拦截无效参数（节省80%的无效调用）
- ✅ 提供清晰的错误提示（让LLM理解错误）
- ✅ 保护系统安全（防止SQL注入、路径遍历等）
- ✅ 降低成本（避免无效的API调用和数据库查询）

---

## Pydantic基础

### 什么是Pydantic？

Pydantic是Python的数据验证库，使用类型注解定义数据结构和验证规则。

**核心特性：**
- 基于Python类型注解
- 自动数据验证
- 清晰的错误信息
- 高性能（使用Rust实现）

### 基本用法

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: str

# 创建实例（自动验证）
user = User(name="Alice", age=25, email="alice@example.com")
print(user.name)  # Alice

# 验证失败会抛出异常
try:
    invalid_user = User(name="Bob", age="not a number", email="bob@example.com")
except ValidationError as e:
    print(e)
```

---

## Tool参数验证

### 1. 基本类型验证

```python
from langchain.tools import tool
from pydantic import BaseModel, Field

class BasicInput(BaseModel):
    """基本类型验证"""
    text: str = Field(description="文本内容")
    count: int = Field(description="数量")
    price: float = Field(description="价格")
    is_active: bool = Field(description="是否激活")

@tool(args_schema=BasicInput)
def process_data(text: str, count: int, price: float, is_active: bool) -> str:
    """处理数据"""
    return f"文本:{text}, 数量:{count}, 价格:{price}, 激活:{is_active}"
```

**验证效果：**
```python
# ✅ 正确的参数
process_data.invoke({
    "text": "hello",
    "count": 10,
    "price": 99.99,
    "is_active": True
})

# ❌ 错误的参数（count不是整数）
process_data.invoke({
    "text": "hello",
    "count": "not a number",  # 验证失败
    "price": 99.99,
    "is_active": True
})
# 抛出ValidationError
```

### 2. 字符串验证

```python
from pydantic import BaseModel, Field

class StringInput(BaseModel):
    """字符串验证"""
    # 长度限制
    keyword: str = Field(
        description="搜索关键词",
        min_length=1,      # 最小长度
        max_length=100     # 最大长度
    )

    # 正则表达式验证
    order_id: str = Field(
        description="订单ID",
        pattern=r'^ORD-\d{6}$'  # 格式：ORD-123456
    )

    # 枚举值
    status: str = Field(
        description="订单状态",
        pattern=r'^(pending|paid|shipped|completed)$'
    )

@tool(args_schema=StringInput)
def query_order(keyword: str, order_id: str, status: str) -> str:
    """查询订单"""
    return f"关键词:{keyword}, 订单:{order_id}, 状态:{status}"
```

**验证效果：**
```python
# ✅ 正确
query_order.invoke({
    "keyword": "iPhone",
    "order_id": "ORD-123456",
    "status": "shipped"
})

# ❌ keyword太短
query_order.invoke({
    "keyword": "",  # 长度为0，验证失败
    "order_id": "ORD-123456",
    "status": "shipped"
})

# ❌ order_id格式错误
query_order.invoke({
    "keyword": "iPhone",
    "order_id": "123456",  # 缺少ORD-前缀，验证失败
    "status": "shipped"
})

# ❌ status不在枚举值中
query_order.invoke({
    "keyword": "iPhone",
    "order_id": "ORD-123456",
    "status": "invalid"  # 不是有效状态，验证失败
})
```

### 3. 数值验证

```python
from pydantic import BaseModel, Field

class NumberInput(BaseModel):
    """数值验证"""
    # 范围限制
    age: int = Field(
        description="年龄",
        ge=0,    # 大于等于0 (greater than or equal)
        le=150   # 小于等于150 (less than or equal)
    )

    # 正数
    price: float = Field(
        description="价格",
        gt=0     # 大于0 (greater than)
    )

    # 限制范围
    discount: float = Field(
        description="折扣",
        ge=0.0,  # 最小0%
        le=1.0   # 最大100%
    )

    # 倍数
    quantity: int = Field(
        description="数量",
        ge=1,
        multiple_of=10  # 必须是10的倍数
    )

@tool(args_schema=NumberInput)
def create_order(age: int, price: float, discount: float, quantity: int) -> str:
    """创建订单"""
    final_price = price * (1 - discount) * quantity
    return f"年龄:{age}, 价格:{price}, 折扣:{discount}, 数量:{quantity}, 总价:{final_price}"
```

**验证效果：**
```python
# ✅ 正确
create_order.invoke({
    "age": 25,
    "price": 99.99,
    "discount": 0.2,
    "quantity": 10
})

# ❌ age为负数
create_order.invoke({
    "age": -5,  # 验证失败
    "price": 99.99,
    "discount": 0.2,
    "quantity": 10
})

# ❌ price为0
create_order.invoke({
    "age": 25,
    "price": 0,  # 必须大于0，验证失败
    "discount": 0.2,
    "quantity": 10
})

# ❌ discount超出范围
create_order.invoke({
    "age": 25,
    "price": 99.99,
    "discount": 1.5,  # 超过1.0，验证失败
    "quantity": 10
})

# ❌ quantity不是10的倍数
create_order.invoke({
    "age": 25,
    "price": 99.99,
    "discount": 0.2,
    "quantity": 15  # 不是10的倍数，验证失败
})
```

### 4. 可选参数

```python
from typing import Optional
from pydantic import BaseModel, Field

class OptionalInput(BaseModel):
    """可选参数"""
    # 必需参数
    user_id: int = Field(description="用户ID")

    # 可选参数（有默认值）
    limit: int = Field(
        default=10,
        description="返回数量",
        ge=1,
        le=100
    )

    # 可选参数（可以为None）
    status: Optional[str] = Field(
        default=None,
        description="订单状态筛选"
    )

    start_date: Optional[str] = Field(
        default=None,
        description="开始日期，格式YYYY-MM-DD"
    )

@tool(args_schema=OptionalInput)
def query_orders(
    user_id: int,
    limit: int = 10,
    status: Optional[str] = None,
    start_date: Optional[str] = None
) -> str:
    """查询用户订单"""
    filters = [f"用户ID={user_id}"]
    if status:
        filters.append(f"状态={status}")
    if start_date:
        filters.append(f"日期>={start_date}")

    return f"查询条件: {', '.join(filters)}, 限制{limit}条"
```

**验证效果：**
```python
# ✅ 只提供必需参数
query_orders.invoke({"user_id": 123})
# 输出: "查询条件: 用户ID=123, 限制10条"

# ✅ 提供部分可选参数
query_orders.invoke({
    "user_id": 123,
    "status": "shipped"
})
# 输出: "查询条件: 用户ID=123, 状态=shipped, 限制10条"

# ✅ 提供所有参数
query_orders.invoke({
    "user_id": 123,
    "limit": 20,
    "status": "shipped",
    "start_date": "2024-01-01"
})
# 输出: "查询条件: 用户ID=123, 状态=shipped, 日期>=2024-01-01, 限制20条"

# ❌ 缺少必需参数
query_orders.invoke({"status": "shipped"})
# 验证失败：缺少user_id
```

### 5. 列表和字典

```python
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class CollectionInput(BaseModel):
    """集合类型验证"""
    # 字符串列表
    keywords: List[str] = Field(
        description="搜索关键词列表",
        min_items=1,     # 至少1个元素
        max_items=10     # 最多10个元素
    )

    # 整数列表
    product_ids: List[int] = Field(
        description="产品ID列表"
    )

    # 字典
    filters: Dict[str, str] = Field(
        default={},
        description="筛选条件"
    )

    # 可选列表
    tags: Optional[List[str]] = Field(
        default=None,
        description="标签列表"
    )

@tool(args_schema=CollectionInput)
def search_products(
    keywords: List[str],
    product_ids: List[int],
    filters: Dict[str, str] = {},
    tags: Optional[List[str]] = None
) -> str:
    """搜索产品"""
    return f"关键词:{keywords}, ID:{product_ids}, 筛选:{filters}, 标签:{tags}"
```

**验证效果：**
```python
# ✅ 正确
search_products.invoke({
    "keywords": ["iPhone", "手机"],
    "product_ids": [1, 2, 3],
    "filters": {"category": "电子产品", "brand": "Apple"},
    "tags": ["热门", "新品"]
})

# ❌ keywords为空列表
search_products.invoke({
    "keywords": [],  # 至少需要1个元素，验证失败
    "product_ids": [1, 2, 3]
})

# ❌ keywords超过10个
search_products.invoke({
    "keywords": ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "k10", "k11"],  # 超过10个，验证失败
    "product_ids": [1, 2, 3]
})

# ❌ product_ids包含非整数
search_products.invoke({
    "keywords": ["iPhone"],
    "product_ids": [1, "not a number", 3]  # 验证失败
})
```

### 6. 嵌套Schema

```python
from pydantic import BaseModel, Field
from typing import List

class Address(BaseModel):
    """地址"""
    street: str = Field(description="街道")
    city: str = Field(description="城市")
    zipcode: str = Field(
        description="邮编",
        pattern=r'^\d{6}$'  # 6位数字
    )

class OrderItem(BaseModel):
    """订单商品"""
    product_id: int = Field(description="产品ID", gt=0)
    quantity: int = Field(description="数量", ge=1)
    price: float = Field(description="单价", gt=0)

class CreateOrderInput(BaseModel):
    """创建订单参数"""
    user_id: int = Field(description="用户ID")
    items: List[OrderItem] = Field(
        description="商品列表",
        min_items=1
    )
    shipping_address: Address = Field(description="收货地址")
    note: Optional[str] = Field(
        default=None,
        description="订单备注",
        max_length=200
    )

@tool(args_schema=CreateOrderInput)
def create_order(
    user_id: int,
    items: List[dict],
    shipping_address: dict,
    note: Optional[str] = None
) -> str:
    """创建订单"""
    total = sum(item['quantity'] * item['price'] for item in items)
    return f"订单创建成功：用户{user_id}, {len(items)}件商品, 总价¥{total}, 地址{shipping_address['city']}"
```

**验证效果：**
```python
# ✅ 正确
create_order.invoke({
    "user_id": 123,
    "items": [
        {"product_id": 1, "quantity": 2, "price": 99.99},
        {"product_id": 2, "quantity": 1, "price": 199.99}
    ],
    "shipping_address": {
        "street": "中关村大街1号",
        "city": "北京",
        "zipcode": "100000"
    },
    "note": "请尽快发货"
})

# ❌ items为空列表
create_order.invoke({
    "user_id": 123,
    "items": [],  # 至少需要1个商品，验证失败
    "shipping_address": {...}
})

# ❌ zipcode格式错误
create_order.invoke({
    "user_id": 123,
    "items": [{...}],
    "shipping_address": {
        "street": "中关村大街1号",
        "city": "北京",
        "zipcode": "12345"  # 只有5位，验证失败
    }
})

# ❌ quantity为0
create_order.invoke({
    "user_id": 123,
    "items": [
        {"product_id": 1, "quantity": 0, "price": 99.99}  # quantity必须>=1，验证失败
    ],
    "shipping_address": {...}
})
```

---

## 高级验证

### 1. 自定义验证器

```python
from pydantic import BaseModel, Field, validator

class EmailInput(BaseModel):
    """邮件参数"""
    to: str = Field(description="收件人邮箱")
    subject: str = Field(description="邮件主题")
    body: str = Field(description="邮件内容")

    @validator('to')
    def validate_email(cls, v):
        """验证邮箱格式"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, v):
            raise ValueError(f"无效的邮箱地址: {v}")
        return v

    @validator('subject')
    def validate_subject(cls, v):
        """验证主题不为空"""
        if not v.strip():
            raise ValueError("邮件主题不能为空")
        return v.strip()

    @validator('body')
    def validate_body(cls, v):
        """验证内容长度"""
        if len(v) < 10:
            raise ValueError("邮件内容至少10个字符")
        if len(v) > 10000:
            raise ValueError("邮件内容不能超过10000个字符")
        return v

@tool(args_schema=EmailInput)
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件"""
    return f"邮件已发送到{to}，主题：{subject}"
```

### 2. 字段依赖验证

```python
from pydantic import BaseModel, Field, root_validator

class DateRangeInput(BaseModel):
    """日期范围参数"""
    start_date: str = Field(description="开始日期，格式YYYY-MM-DD")
    end_date: str = Field(description="结束日期，格式YYYY-MM-DD")

    @root_validator
    def validate_date_range(cls, values):
        """验证日期范围"""
        from datetime import datetime

        start = values.get('start_date')
        end = values.get('end_date')

        if start and end:
            start_dt = datetime.strptime(start, '%Y-%m-%d')
            end_dt = datetime.strptime(end, '%Y-%m-%d')

            if start_dt > end_dt:
                raise ValueError("开始日期不能晚于结束日期")

            # 限制查询范围不超过1年
            if (end_dt - start_dt).days > 365:
                raise ValueError("查询范围不能超过1年")

        return values

@tool(args_schema=DateRangeInput)
def query_sales(start_date: str, end_date: str) -> str:
    """查询销售数据"""
    return f"查询{start_date}到{end_date}的销售数据"
```

### 3. 条件验证

```python
from pydantic import BaseModel, Field, root_validator
from typing import Optional

class PaymentInput(BaseModel):
    """支付参数"""
    amount: float = Field(description="支付金额", gt=0)
    payment_method: str = Field(description="支付方式：card, alipay, wechat")

    # 信用卡相关字段（仅当payment_method=card时需要）
    card_number: Optional[str] = Field(default=None, description="卡号")
    cvv: Optional[str] = Field(default=None, description="CVV")

    @root_validator
    def validate_card_info(cls, values):
        """验证信用卡信息"""
        method = values.get('payment_method')
        card_number = values.get('card_number')
        cvv = values.get('cvv')

        if method == 'card':
            if not card_number:
                raise ValueError("使用信用卡支付时必须提供卡号")
            if not cvv:
                raise ValueError("使用信用卡支付时必须提供CVV")

            # 验证卡号格式（简化版）
            if not card_number.isdigit() or len(card_number) != 16:
                raise ValueError("卡号必须是16位数字")

            # 验证CVV格式
            if not cvv.isdigit() or len(cvv) != 3:
                raise ValueError("CVV必须是3位数字")

        return values

@tool(args_schema=PaymentInput)
def process_payment(
    amount: float,
    payment_method: str,
    card_number: Optional[str] = None,
    cvv: Optional[str] = None
) -> str:
    """处理支付"""
    if payment_method == 'card':
        return f"信用卡支付¥{amount}，卡号{card_number[:4]}****{card_number[-4:]}"
    else:
        return f"{payment_method}支付¥{amount}"
```

---

## 错误处理

### 1. 捕获验证错误

```python
from pydantic import ValidationError

@tool(args_schema=SearchInput)
def search_products(keyword: str, limit: int = 10) -> str:
    """搜索产品"""
    try:
        # Tool逻辑
        results = db.search(keyword, limit)
        return str(results)
    except ValidationError as e:
        # Pydantic会在调用前自动验证
        # 这里不会执行到
        return f"参数验证失败：{e}"
```

**注意：** Pydantic验证失败会在Tool执行前抛出异常，Tool内部不需要处理验证错误。

### 2. 友好的错误信息

```python
from pydantic import BaseModel, Field, validator

class FriendlyInput(BaseModel):
    """友好的错误提示"""
    email: str = Field(description="邮箱地址")

    @validator('email')
    def validate_email(cls, v):
        """验证邮箱"""
        import re
        if not re.match(r'^[^@]+@[^@]+\.[^@]+$', v):
            raise ValueError(
                f"邮箱格式不正确：'{v}'。"
                "正确格式示例：user@example.com"
            )
        return v
```

---

## 最佳实践

### 1. 描述要清晰

```python
# ❌ 描述不清晰
class BadInput(BaseModel):
    param: str = Field(description="参数")

# ✅ 描述清晰
class GoodInput(BaseModel):
    order_id: str = Field(
        description="订单ID，格式为ORD-XXXXXX，例如ORD-123456",
        pattern=r'^ORD-\d{6}$'
    )
```

### 2. 设置合理的默认值

```python
# ✅ 合理的默认值
class SearchInput(BaseModel):
    keyword: str = Field(description="搜索关键词")
    limit: int = Field(
        default=10,      # 默认返回10条
        ge=1,
        le=100,
        description="返回数量"
    )
    sort_by: str = Field(
        default="relevance",  # 默认按相关性排序
        description="排序方式：relevance, price, date"
    )
```

### 3. 使用枚举限制选项

```python
from enum import Enum

class OrderStatus(str, Enum):
    """订单状态枚举"""
    PENDING = "pending"
    PAID = "paid"
    SHIPPED = "shipped"
    COMPLETED = "completed"

class OrderInput(BaseModel):
    order_id: str = Field(description="订单ID")
    status: OrderStatus = Field(description="订单状态")

@tool(args_schema=OrderInput)
def update_order_status(order_id: str, status: OrderStatus) -> str:
    """更新订单状态"""
    return f"订单{order_id}状态更新为{status.value}"
```

### 4. 验证安全性

```python
import os
from pathlib import Path

class FileInput(BaseModel):
    """文件操作参数"""
    file_path: str = Field(description="文件路径")

    @validator('file_path')
    def validate_path(cls, v):
        """验证路径安全性"""
        # 防止路径遍历攻击
        path = Path(v).resolve()
        allowed_dir = Path("/allowed/directory").resolve()

        if not str(path).startswith(str(allowed_dir)):
            raise ValueError(f"不允许访问路径：{v}")

        return v

@tool(args_schema=FileInput)
def read_file(file_path: str) -> str:
    """读取文件"""
    with open(file_path, 'r') as f:
        return f.read()
```

### 5. 性能考虑

```python
# ❌ 复杂的验证逻辑
class SlowInput(BaseModel):
    url: str = Field(description="URL")

    @validator('url')
    def validate_url(cls, v):
        # 发起HTTP请求验证URL是否可访问（慢！）
        import requests
        response = requests.get(v)
        if response.status_code != 200:
            raise ValueError(f"URL不可访问：{v}")
        return v

# ✅ 简单的格式验证
class FastInput(BaseModel):
    url: str = Field(
        description="URL",
        pattern=r'^https?://.+'  # 只验证格式
    )
```

---

## 总结

### 参数验证的核心价值

1. **拦截无效输入**：在Tool执行前就发现问题
2. **节省成本**：避免无效的API调用和数据库查询
3. **提升安全性**：防止注入攻击和路径遍历
4. **改善体验**：提供清晰的错误提示

### 验证层次

```
第1层：类型验证（str, int, float, bool）
    ↓
第2层：范围验证（min_length, max_length, ge, le）
    ↓
第3层：格式验证（pattern正则表达式）
    ↓
第4层：业务验证（自定义validator）
    ↓
第5层：安全验证（路径、SQL注入防护）
```

### 快速参考

| 验证类型 | Pydantic语法 | 示例 |
|---------|-------------|------|
| 类型 | `: str`, `: int` | `name: str` |
| 长度 | `min_length`, `max_length` | `Field(min_length=1)` |
| 范围 | `ge`, `le`, `gt`, `lt` | `Field(ge=0, le=100)` |
| 格式 | `pattern` | `Field(pattern=r'^\d{6}$')` |
| 默认值 | `default` | `Field(default=10)` |
| 可选 | `Optional[type]` | `Optional[str]` |
| 列表 | `List[type]` | `List[str]` |
| 字典 | `Dict[key, value]` | `Dict[str, int]` |
| 自定义 | `@validator` | `@validator('field')` |

---

**记住：** 参数验证是Tool可靠性的第一道防线，投入时间设计好Schema比事后调试更高效！
