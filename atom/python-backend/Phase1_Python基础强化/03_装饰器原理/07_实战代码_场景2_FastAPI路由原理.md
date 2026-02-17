# 实战代码 - 场景2：FastAPI 路由装饰器原理

> 手写简化版 FastAPI，理解路由装饰器的工作原理

---

## 场景概述

本场景通过手写简化版 FastAPI 路由系统，深入理解装饰器在 Web 框架中的应用。

**学习目标：**
1. 理解路由装饰器的注册机制
2. 掌握装饰器工厂模式
3. 理解依赖注入的实现原理
4. 构建完整的 API 应用

---

## 完整代码示例

```python
"""
手写简化版 FastAPI 路由系统
演示：路由装饰器的工作原理
"""

from functools import wraps
from typing import Callable, Dict, Any, Optional, List
from dataclasses import dataclass
import json
import inspect

# ===== 1. 路由数据结构 =====

@dataclass
class Route:
    """路由信息"""
    path: str
    method: str
    handler: Callable
    status_code: int = 200
    dependencies: List[Callable] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


# ===== 2. 简化版 FastAPI 类 =====

class SimpleFastAPI:
    """
    简化版 FastAPI 框架

    功能：
    - 路由注册
    - 请求处理
    - 依赖注入
    - 响应生成
    """

    def __init__(self, title: str = "SimpleFastAPI"):
        self.title = title
        self.routes: Dict[str, Dict[str, Route]] = {}
        print(f"✨ {title} 应用已创建")

    def _register_route(
        self,
        path: str,
        method: str,
        handler: Callable,
        status_code: int = 200,
        dependencies: List[Callable] = None
    ):
        """
        注册路由

        参数:
            path: 路由路径
            method: HTTP 方法
            handler: 处理函数
            status_code: 状态码
            dependencies: 依赖列表
        """
        # 初始化路径
        if path not in self.routes:
            self.routes[path] = {}

        # 创建路由对象
        route = Route(
            path=path,
            method=method,
            handler=handler,
            status_code=status_code,
            dependencies=dependencies or []
        )

        # 注册路由
        self.routes[path][method] = route
        print(f"📝 注册路由: {method} {path} -> {handler.__name__}")

    def get(self, path: str, status_code: int = 200, dependencies: List[Callable] = None):
        """
        GET 路由装饰器

        用法:
            @app.get("/users")
            def get_users():
                return [{"id": 1, "name": "Alice"}]
        """
        def decorator(func: Callable) -> Callable:
            # 注册路由（在定义时执行）
            self._register_route(path, "GET", func, status_code, dependencies)
            return func  # 返回原函数
        return decorator

    def post(self, path: str, status_code: int = 201, dependencies: List[Callable] = None):
        """POST 路由装饰器"""
        def decorator(func: Callable) -> Callable:
            self._register_route(path, "POST", func, status_code, dependencies)
            return func
        return decorator

    def put(self, path: str, status_code: int = 200, dependencies: List[Callable] = None):
        """PUT 路由装饰器"""
        def decorator(func: Callable) -> Callable:
            self._register_route(path, "PUT", func, status_code, dependencies)
            return func
        return decorator

    def delete(self, path: str, status_code: int = 204, dependencies: List[Callable] = None):
        """DELETE 路由装饰器"""
        def decorator(func: Callable) -> Callable:
            self._register_route(path, "DELETE", func, status_code, dependencies)
            return func
        return decorator

    def _resolve_dependencies(self, dependencies: List[Callable], request: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析依赖注入

        参数:
            dependencies: 依赖函数列表
            request: 请求对象

        返回:
            依赖结果字典
        """
        resolved = {}
        for dep in dependencies:
            dep_name = dep.__name__
            dep_result = dep(request)
            resolved[dep_name] = dep_result
        return resolved

    def handle_request(self, method: str, path: str, request: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        处理请求

        参数:
            method: HTTP 方法
            path: 请求路径
            request: 请求数据

        返回:
            响应数据
        """
        if request is None:
            request = {}

        print(f"\n🌐 收到请求: {method} {path}")

        # 查找路由
        if path not in self.routes or method not in self.routes[path]:
            return {
                "status_code": 404,
                "body": {"error": "Not Found"}
            }

        route = self.routes[path][method]

        try:
            # 解析依赖注入
            dependencies_result = self._resolve_dependencies(route.dependencies, request)

            # 获取处理函数的参数
            sig = inspect.signature(route.handler)
            kwargs = {}

            # 注入依赖
            for param_name, param in sig.parameters.items():
                if param_name in dependencies_result:
                    kwargs[param_name] = dependencies_result[param_name]
                elif param_name in request:
                    kwargs[param_name] = request[param_name]

            # 调用处理函数
            result = route.handler(**kwargs)

            # 返回响应
            return {
                "status_code": route.status_code,
                "body": result
            }

        except Exception as e:
            print(f"❌ 错误: {e}")
            return {
                "status_code": 500,
                "body": {"error": str(e)}
            }

    def list_routes(self):
        """列出所有路由"""
        print(f"\n📋 {self.title} 路由列表:")
        for path, methods in self.routes.items():
            for method, route in methods.items():
                print(f"  {method:6} {path:20} -> {route.handler.__name__}")


# ===== 3. 依赖注入示例 =====

def get_current_user(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    依赖函数：获取当前用户

    模拟从请求中提取用户信息
    """
    token = request.get("token")
    if not token:
        raise Exception("未提供认证令牌")

    # 模拟验证 token
    if token == "valid_token":
        return {"id": 1, "name": "Alice", "role": "admin"}
    else:
        raise Exception("无效的认证令牌")


def require_admin(request: Dict[str, Any]) -> bool:
    """
    依赖函数：检查管理员权限

    模拟权限检查
    """
    user = get_current_user(request)
    if user["role"] != "admin":
        raise Exception("需要管理员权限")
    return True


# ===== 4. 完整应用示例 =====

if __name__ == "__main__":
    print("=" * 60)
    print("SimpleFastAPI 路由装饰器示例")
    print("=" * 60)

    # 创建应用
    app = SimpleFastAPI(title="我的 API")

    # ===== 定义路由 =====

    @app.get("/")
    def root():
        """根路径"""
        return {"message": "欢迎使用 SimpleFastAPI"}

    @app.get("/users")
    def get_users():
        """获取用户列表"""
        return [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"}
        ]

    @app.get("/users/{user_id}")
    def get_user(user_id: int):
        """获取单个用户"""
        return {"id": user_id, "name": f"User{user_id}"}

    @app.post("/users", status_code=201)
    def create_user(name: str):
        """创建用户"""
        return {"id": 3, "name": name, "created": True}

    @app.put("/users/{user_id}")
    def update_user(user_id: int, name: str):
        """更新用户"""
        return {"id": user_id, "name": name, "updated": True}

    @app.delete("/users/{user_id}", status_code=204)
    def delete_user(user_id: int):
        """删除用户"""
        return {"deleted": True}

    # ===== 带依赖注入的路由 =====

    @app.get("/profile", dependencies=[get_current_user])
    def get_profile(get_current_user: Dict[str, Any]):
        """获取当前用户资料（需要认证）"""
        return {
            "user": get_current_user,
            "profile": "这是用户资料"
        }

    @app.delete("/admin/users/{user_id}", dependencies=[require_admin])
    def admin_delete_user(user_id: int, require_admin: bool):
        """管理员删除用户（需要管理员权限）"""
        return {"admin_deleted": True, "user_id": user_id}

    # ===== 列出所有路由 =====
    app.list_routes()

    # ===== 测试请求 =====

    print("\n" + "=" * 60)
    print("测试请求")
    print("=" * 60)

    # 测试1：根路径
    response = app.handle_request("GET", "/")
    print(f"响应: {response}")

    # 测试2：获取用户列表
    response = app.handle_request("GET", "/users")
    print(f"响应: {response}")

    # 测试3：获取单个用户
    response = app.handle_request("GET", "/users/{user_id}", {"user_id": 1})
    print(f"响应: {response}")

    # 测试4：创建用户
    response = app.handle_request("POST", "/users", {"name": "Charlie"})
    print(f"响应: {response}")

    # 测试5：更新用户
    response = app.handle_request("PUT", "/users/{user_id}", {"user_id": 2, "name": "Bob Updated"})
    print(f"响应: {response}")

    # 测试6：删除用户
    response = app.handle_request("DELETE", "/users/{user_id}", {"user_id": 3})
    print(f"响应: {response}")

    # 测试7：获取资料（需要认证）
    print("\n--- 测试认证 ---")

    # 无 token
    response = app.handle_request("GET", "/profile")
    print(f"响应: {response}")

    # 无效 token
    response = app.handle_request("GET", "/profile", {"token": "invalid"})
    print(f"响应: {response}")

    # 有效 token
    response = app.handle_request("GET", "/profile", {"token": "valid_token"})
    print(f"响应: {response}")

    # 测试8：管理员删除用户（需要管理员权限）
    print("\n--- 测试权限 ---")

    # 有效 token（管理员）
    response = app.handle_request(
        "DELETE",
        "/admin/users/{user_id}",
        {"user_id": 5, "token": "valid_token"}
    )
    print(f"响应: {response}")

    # 测试9：404 错误
    print("\n--- 测试错误处理 ---")
    response = app.handle_request("GET", "/not-found")
    print(f"响应: {response}")

    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)


# ===== 5. 进阶示例：中间件装饰器 =====

class MiddlewareApp(SimpleFastAPI):
    """
    带中间件支持的 FastAPI

    中间件：在请求处理前后执行的函数
    """

    def __init__(self, title: str = "MiddlewareApp"):
        super().__init__(title)
        self.middlewares: List[Callable] = []

    def middleware(self, func: Callable) -> Callable:
        """
        中间件装饰器

        用法:
            @app.middleware
            def log_middleware(request, call_next):
                print("请求前")
                response = call_next(request)
                print("请求后")
                return response
        """
        self.middlewares.append(func)
        print(f"🔧 注册中间件: {func.__name__}")
        return func

    def handle_request(self, method: str, path: str, request: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        处理请求（带中间件）

        中间件执行顺序：
        1. 中间件1 前
        2. 中间件2 前
        3. 路由处理
        4. 中间件2 后
        5. 中间件1 后
        """
        if request is None:
            request = {}

        # 构建中间件链
        def call_next(req):
            return super(MiddlewareApp, self).handle_request(method, path, req)

        # 应用中间件（从后往前）
        handler = call_next
        for middleware in reversed(self.middlewares):
            current_handler = handler
            handler = lambda req, mw=middleware, h=current_handler: mw(req, lambda r: h(r))

        # 执行中间件链
        return handler(request)


# ===== 6. 中间件示例 =====

def example_middleware():
    """中间件使用示例"""
    print("\n" + "=" * 60)
    print("中间件示例")
    print("=" * 60)

    app = MiddlewareApp(title="中间件应用")

    # 定义中间件
    @app.middleware
    def logging_middleware(request, call_next):
        """日志中间件"""
        print("📝 [日志中间件] 请求前")
        response = call_next(request)
        print("📝 [日志中间件] 请求后")
        return response

    @app.middleware
    def timing_middleware(request, call_next):
        """计时中间件"""
        import time
        print("⏱️  [计时中间件] 开始计时")
        start = time.time()
        response = call_next(request)
        elapsed = time.time() - start
        print(f"⏱️  [计时中间件] 耗时: {elapsed:.4f}秒")
        return response

    # 定义路由
    @app.get("/test")
    def test_endpoint():
        """测试端点"""
        print("  🎯 [路由处理] 执行业务逻辑")
        import time
        time.sleep(0.1)
        return {"message": "测试成功"}

    # 测试请求
    app.list_routes()
    response = app.handle_request("GET", "/test")
    print(f"\n最终响应: {response}")


if __name__ == "__main__":
    example_middleware()
```

---

## 运行输出示例

```
============================================================
SimpleFastAPI 路由装饰器示例
============================================================
✨ 我的 API 应用已创建
📝 注册路由: GET / -> root
📝 注册路由: GET /users -> get_users
📝 注册路由: GET /users/{user_id} -> get_user
📝 注册路由: POST /users -> create_user
📝 注册路由: PUT /users/{user_id} -> update_user
📝 注册路由: DELETE /users/{user_id} -> delete_user
📝 注册路由: GET /profile -> get_profile
📝 注册路由: DELETE /admin/users/{user_id} -> admin_delete_user

📋 我的 API 路由列表:
  GET    /                    -> root
  GET    /users               -> get_users
  GET    /users/{user_id}     -> get_user
  POST   /users               -> create_user
  PUT    /users/{user_id}     -> update_user
  DELETE /users/{user_id}     -> delete_user
  GET    /profile             -> get_profile
  DELETE /admin/users/{user_id} -> admin_delete_user

============================================================
测试请求
============================================================

🌐 收到请求: GET /
响应: {'status_code': 200, 'body': {'message': '欢迎使用 SimpleFastAPI'}}

🌐 收到请求: GET /users
响应: {'status_code': 200, 'body': [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}]}

🌐 收到请求: GET /users/{user_id}
响应: {'status_code': 200, 'body': {'id': 1, 'name': 'User1'}}

🌐 收到请求: POST /users
响应: {'status_code': 201, 'body': {'id': 3, 'name': 'Charlie', 'created': True}}

🌐 收到请求: PUT /users/{user_id}
响应: {'status_code': 200, 'body': {'id': 2, 'name': 'Bob Updated', 'updated': True}}

🌐 收到请求: DELETE /users/{user_id}
响应: {'status_code': 204, 'body': {'deleted': True}}

--- 测试认证 ---

🌐 收到请求: GET /profile
❌ 错误: 未提供认证令牌
响应: {'status_code': 500, 'body': {'error': '未提供认证令牌'}}

🌐 收到请求: GET /profile
❌ 错误: 无效的认证令牌
响应: {'status_code': 500, 'body': {'error': '无效的认证令牌'}}

🌐 收到请求: GET /profile
响应: {'status_code': 200, 'body': {'user': {'id': 1, 'name': 'Alice', 'role': 'admin'}, 'profile': '这是用户资料'}}

--- 测试权限 ---

🌐 收到请求: DELETE /admin/users/{user_id}
响应: {'status_code': 200, 'body': {'admin_deleted': True, 'user_id': 5}}

--- 测试错误处理 ---

🌐 收到请求: GET /not-found
响应: {'status_code': 404, 'body': {'error': 'Not Found'}}

============================================================
示例完成
============================================================

============================================================
中间件示例
============================================================
✨ 中间件应用 应用已创建
🔧 注册中间件: logging_middleware
🔧 注册中间件: timing_middleware
📝 注册路由: GET /test -> test_endpoint

📋 中间件应用 路由列表:
  GET    /test                -> test_endpoint

🌐 收到请求: GET /test
📝 [日志中间件] 请求前
⏱️  [计时中间件] 开始计时
  🎯 [路由处理] 执行业务逻辑
⏱️  [计时中间件] 耗时: 0.1005秒
📝 [日志中间件] 请求后

最终响应: {'status_code': 200, 'body': {'message': '测试成功'}}
```

---

## 关键知识点

### 1. 路由装饰器的本质

```python
# 路由装饰器做了什么？
@app.get("/users")
def get_users():
    return [{"id": 1}]

# 等价于：
def get_users():
    return [{"id": 1}]
get_users = app.get("/users")(get_users)

# 执行流程：
# 1. app.get("/users") 返回 decorator 函数
# 2. decorator(get_users) 注册路由并返回 get_users
# 3. get_users 现在仍然是原函数（没有被包装）
```

### 2. 装饰器工厂模式

```python
def route_decorator(path, method, status_code=200):
    """装饰器工厂：根据参数创建装饰器"""
    def decorator(func):
        # 注册路由
        register_route(path, method, func, status_code)
        return func  # 返回原函数
    return decorator

# 使用
@route_decorator("/users", "GET", 200)
def get_users():
    pass
```

### 3. 依赖注入的实现

```python
# 依赖注入的核心思想：
# 1. 定义依赖函数
def get_current_user(request):
    return {"id": 1, "name": "Alice"}

# 2. 在路由中声明依赖
@app.get("/profile", dependencies=[get_current_user])
def get_profile(get_current_user):  # 参数名与依赖函数名相同
    return {"user": get_current_user}

# 3. 框架自动注入
# - 调用依赖函数：user = get_current_user(request)
# - 注入到路由函数：get_profile(get_current_user=user)
```

### 4. 中间件的洋葱模型

```
请求 → 中间件1前 → 中间件2前 → 路由处理 → 中间件2后 → 中间件1后 → 响应
```

---

## 与真实 FastAPI 的对比

### 相似之处

1. **路由装饰器**：`@app.get(path)` 语法相同
2. **依赖注入**：通过函数参数注入依赖
3. **状态码配置**：`status_code` 参数
4. **中间件模式**：洋葱模型

### 差异之处

1. **异步支持**：真实 FastAPI 支持 `async/await`
2. **自动文档**：真实 FastAPI 自动生成 OpenAPI 文档
3. **数据验证**：真实 FastAPI 使用 Pydantic 自动验证
4. **路径参数**：真实 FastAPI 自动解析路径参数
5. **性能优化**：真实 FastAPI 基于 Starlette，性能更高

---

## 实际应用

### 在真实 FastAPI 中使用

```python
from fastapi import FastAPI, Depends, HTTPException

app = FastAPI()

# 依赖函数
def get_current_user(token: str):
    if token != "valid":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"id": 1, "name": "Alice"}

# 路由（带依赖注入）
@app.get("/profile")
def get_profile(user: dict = Depends(get_current_user)):
    return {"user": user}

# 运行：uvicorn main:app --reload
```

---

## 扩展练习

1. **练习1：添加路径参数解析**
   - 支持 `/users/{user_id}` 格式
   - 自动提取路径参数
   - 类型转换

2. **练习2：添加查询参数支持**
   - 支持 `/users?page=1&size=10`
   - 自动解析查询参数
   - 默认值处理

3. **练习3：添加请求体验证**
   - 使用 Pydantic 验证请求体
   - 自动返回验证错误
   - 类型转换

4. **练习4：添加异步支持**
   - 支持 `async def` 路由函数
   - 异步依赖注入
   - 异步中间件

---

## 总结

**路由装饰器的核心原理：**

1. **装饰器工厂**：`app.get(path)` 返回装饰器
2. **路由注册**：装饰器在定义时注册路由
3. **返回原函数**：装饰器返回原函数（不包装）
4. **依赖注入**：通过函数参数自动注入依赖
5. **中间件链**：洋葱模型处理请求

**关键点：**
- 路由装饰器在定义时执行（注册路由）
- 装饰器返回原函数（不影响函数调用）
- 依赖注入通过参数名匹配
- 中间件按注册顺序执行（洋葱模型）

**下一步：**
- 场景3：权限与认证装饰器
- 场景4：缓存与性能优化
- 场景5：AI Agent 专用装饰器
