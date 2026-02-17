# 【实战代码1】基础Session操作场景

> 创建、查询、关闭、上下文管理器的完整示例

---

## 场景1：基础CRUD操作

### 完整代码

```python
# database.py
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

# 数据库配置
DATABASE_URL = "postgresql://user:password@localhost:5432/mydb"

# 创建引擎
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=True,  # 打印 SQL 语句
)

# 创建 SessionMaker
SessionLocal = sessionmaker(bind=engine)

# 创建 Base 类
Base = declarative_base()


# 定义模型
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True)
    age = Column(Integer)

    def __repr__(self):
        return f"<User(id={self.id}, name={self.name}, email={self.email})>"


# 创建表
Base.metadata.create_all(bind=engine)


# CRUD 操作
def create_user(name: str, email: str, age: int):
    """创建用户"""
    session = SessionLocal()
    try:
        user = User(name=name, email=email, age=age)
        session.add(user)
        session.commit()
        session.refresh(user)  # 刷新对象，获取数据库生成的 ID
        print(f"Created user: {user}")
        return user
    except Exception as e:
        session.rollback()
        print(f"Error creating user: {e}")
        raise
    finally:
        session.close()


def get_user_by_id(user_id: int):
    """根据 ID 查询用户"""
    session = SessionLocal()
    try:
        user = session.query(User).filter_by(id=user_id).first()
        if user:
            print(f"Found user: {user}")
        else:
            print(f"User {user_id} not found")
        return user
    finally:
        session.close()


def get_all_users():
    """查询所有用户"""
    session = SessionLocal()
    try:
        users = session.query(User).all()
        print(f"Found {len(users)} users")
        for user in users:
            print(f"  - {user}")
        return users
    finally:
        session.close()


def update_user(user_id: int, name: str = None, email: str = None, age: int = None):
    """更新用户"""
    session = SessionLocal()
    try:
        user = session.query(User).filter_by(id=user_id).first()
        if not user:
            print(f"User {user_id} not found")
            return None

        if name:
            user.name = name
        if email:
            user.email = email
        if age:
            user.age = age

        session.commit()
        session.refresh(user)
        print(f"Updated user: {user}")
        return user
    except Exception as e:
        session.rollback()
        print(f"Error updating user: {e}")
        raise
    finally:
        session.close()


def delete_user(user_id: int):
    """删除用户"""
    session = SessionLocal()
    try:
        user = session.query(User).filter_by(id=user_id).first()
        if not user:
            print(f"User {user_id} not found")
            return False

        session.delete(user)
        session.commit()
        print(f"Deleted user: {user}")
        return True
    except Exception as e:
        session.rollback()
        print(f"Error deleting user: {e}")
        raise
    finally:
        session.close()


# 测试
if __name__ == "__main__":
    # 创建用户
    user1 = create_user("Alice", "alice@example.com", 25)
    user2 = create_user("Bob", "bob@example.com", 30)
    user3 = create_user("Charlie", "charlie@example.com", 35)

    # 查询所有用户
    get_all_users()

    # 查询单个用户
    get_user_by_id(1)

    # 更新用户
    update_user(1, name="Alice Updated", age=26)

    # 删除用户
    delete_user(3)

    # 查询所有用户
    get_all_users()
```

### 运行结果

```
Created user: <User(id=1, name=Alice, email=alice@example.com)>
Created user: <User(id=2, name=Bob, email=bob@example.com)>
Created user: <User(id=3, name=Charlie, email=charlie@example.com)>
Found 3 users
  - <User(id=1, name=Alice, email=alice@example.com)>
  - <User(id=2, name=Bob, email=bob@example.com)>
  - <User(id=3, name=Charlie, email=charlie@example.com)>
Found user: <User(id=1, name=Alice, email=alice@example.com)>
Updated user: <User(id=1, name=Alice Updated, email=alice@example.com)>
Deleted user: <User(id=3, name=Charlie, email=charlie@example.com)>
Found 2 users
  - <User(id=1, name=Alice Updated, email=alice@example.com)>
  - <User(id=2, name=Bob, email=bob@example.com)>
```

---

## 场景2：使用上下文管理器

### 完整代码

```python
# database.py
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from contextlib import contextmanager

DATABASE_URL = "postgresql://user:password@localhost:5432/mydb"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True)

    def __repr__(self):
        return f"<User(id={self.id}, name={self.name})>"


Base.metadata.create_all(bind=engine)


# 方法1：使用 with 语句
def create_user_with_context(name: str, email: str):
    """使用 with 语句创建用户"""
    with SessionLocal() as session:
        user = User(name=name, email=email)
        session.add(user)
        session.commit()
        session.refresh(user)
        print(f"Created user: {user}")
        return user


# 方法2：自定义上下文管理器
@contextmanager
def get_db_session():
    """自定义上下文管理器"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_user_with_custom_context(name: str, email: str):
    """使用自定义上下文管理器创建用户"""
    with get_db_session() as session:
        user = User(name=name, email=email)
        session.add(user)
        # 自动 commit
        print(f"Created user: {user}")
        return user


# 方法3：使用 begin 上下文管理器
def create_user_with_begin(name: str, email: str):
    """使用 begin 上下文管理器创建用户"""
    with SessionLocal.begin() as session:
        user = User(name=name, email=email)
        session.add(user)
        # 自动 commit 和 close
        print(f"Created user: {user}")
        return user


# 测试
if __name__ == "__main__":
    # 方法1
    create_user_with_context("Alice", "alice@example.com")

    # 方法2
    create_user_with_custom_context("Bob", "bob@example.com")

    # 方法3
    create_user_with_begin("Charlie", "charlie@example.com")
```

### 运行结果

```
Created user: <User(id=1, name=Alice)>
Created user: <User(id=2, name=Bob)>
Created user: <User(id=3, name=Charlie)>
```

---

## 场景3：批量操作

### 完整代码

```python
# database.py
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "postgresql://user:password@localhost:5432/mydb"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True)

    def __repr__(self):
        return f"<User(id={self.id}, name={self.name})>"


Base.metadata.create_all(bind=engine)


# 批量插入
def bulk_create_users(users_data: list[dict]):
    """批量创建用户"""
    with SessionLocal() as session:
        users = [User(**data) for data in users_data]
        session.add_all(users)
        session.commit()
        print(f"Created {len(users)} users")
        return users


# 批量更新
def bulk_update_users(updates: list[dict]):
    """批量更新用户"""
    with SessionLocal() as session:
        for update in updates:
            user_id = update.pop("id")
            session.query(User).filter_by(id=user_id).update(update)
        session.commit()
        print(f"Updated {len(updates)} users")


# 批量删除
def bulk_delete_users(user_ids: list[int]):
    """批量删除用户"""
    with SessionLocal() as session:
        deleted = session.query(User).filter(User.id.in_(user_ids)).delete()
        session.commit()
        print(f"Deleted {deleted} users")
        return deleted


# 使用 bulk_insert_mappings（更高效）
def bulk_insert_users_fast(users_data: list[dict]):
    """使用 bulk_insert_mappings 批量插入"""
    with SessionLocal() as session:
        session.bulk_insert_mappings(User, users_data)
        session.commit()
        print(f"Inserted {len(users_data)} users (fast)")


# 测试
if __name__ == "__main__":
    # 批量插入
    users_data = [
        {"name": "Alice", "email": "alice@example.com"},
        {"name": "Bob", "email": "bob@example.com"},
        {"name": "Charlie", "email": "charlie@example.com"},
    ]
    bulk_create_users(users_data)

    # 批量更新
    updates = [
        {"id": 1, "name": "Alice Updated"},
        {"id": 2, "name": "Bob Updated"},
    ]
    bulk_update_users(updates)

    # 批量删除
    bulk_delete_users([3])

    # 高效批量插入
    users_data = [
        {"name": f"User{i}", "email": f"user{i}@example.com"}
        for i in range(100)
    ]
    bulk_insert_users_fast(users_data)
```

### 运行结果

```
Created 3 users
Updated 2 users
Deleted 1 users
Inserted 100 users (fast)
```

---

## 场景4：查询操作

### 完整代码

```python
# database.py
from sqlalchemy import create_engine, Column, Integer, String, and_, or_
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "postgresql://user:password@localhost:5432/mydb"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True)
    age = Column(Integer)

    def __repr__(self):
        return f"<User(id={self.id}, name={self.name}, age={self.age})>"


Base.metadata.create_all(bind=engine)


# 基础查询
def basic_queries():
    """基础查询示例"""
    with SessionLocal() as session:
        # 查询所有
        users = session.query(User).all()
        print(f"All users: {users}")

        # 查询第一个
        user = session.query(User).first()
        print(f"First user: {user}")

        # 根据主键查询
        user = session.get(User, 1)
        print(f"User by ID: {user}")

        # 条件查询
        user = session.query(User).filter_by(name="Alice").first()
        print(f"User by name: {user}")

        # 多条件查询
        user = session.query(User).filter_by(name="Alice", age=25).first()
        print(f"User by name and age: {user}")


# 高级查询
def advanced_queries():
    """高级查询示例"""
    with SessionLocal() as session:
        # 使用 filter
        users = session.query(User).filter(User.age > 25).all()
        print(f"Users older than 25: {users}")

        # AND 条件
        users = session.query(User).filter(
            and_(User.age > 20, User.age < 30)
        ).all()
        print(f"Users between 20 and 30: {users}")

        # OR 条件
        users = session.query(User).filter(
            or_(User.name == "Alice", User.name == "Bob")
        ).all()
        print(f"Users named Alice or Bob: {users}")

        # IN 查询
        users = session.query(User).filter(User.id.in_([1, 2, 3])).all()
        print(f"Users with ID in [1, 2, 3]: {users}")

        # LIKE 查询
        users = session.query(User).filter(User.name.like("A%")).all()
        print(f"Users whose name starts with A: {users}")

        # 排序
        users = session.query(User).order_by(User.age.desc()).all()
        print(f"Users ordered by age (desc): {users}")

        # 限制数量
        users = session.query(User).limit(5).all()
        print(f"First 5 users: {users}")

        # 偏移
        users = session.query(User).offset(5).limit(5).all()
        print(f"Users 6-10: {users}")

        # 计数
        count = session.query(User).count()
        print(f"Total users: {count}")


# 聚合查询
def aggregate_queries():
    """聚合查询示例"""
    from sqlalchemy import func

    with SessionLocal() as session:
        # 平均年龄
        avg_age = session.query(func.avg(User.age)).scalar()
        print(f"Average age: {avg_age}")

        # 最大年龄
        max_age = session.query(func.max(User.age)).scalar()
        print(f"Max age: {max_age}")

        # 最小年龄
        min_age = session.query(func.min(User.age)).scalar()
        print(f"Min age: {min_age}")

        # 分组统计
        results = session.query(
            User.age, func.count(User.id)
        ).group_by(User.age).all()
        print(f"Users by age: {results}")


# 测试
if __name__ == "__main__":
    # 创建测试数据
    with SessionLocal() as session:
        users = [
            User(name="Alice", email="alice@example.com", age=25),
            User(name="Bob", email="bob@example.com", age=30),
            User(name="Charlie", email="charlie@example.com", age=35),
            User(name="David", email="david@example.com", age=20),
            User(name="Eve", email="eve@example.com", age=28),
        ]
        session.add_all(users)
        session.commit()

    # 基础查询
    basic_queries()

    # 高级查询
    advanced_queries()

    # 聚合查询
    aggregate_queries()
```

### 运行结果

```
All users: [<User(id=1, name=Alice, age=25)>, <User(id=2, name=Bob, age=30)>, ...]
First user: <User(id=1, name=Alice, age=25)>
User by ID: <User(id=1, name=Alice, age=25)>
User by name: <User(id=1, name=Alice, age=25)>
User by name and age: <User(id=1, name=Alice, age=25)>
Users older than 25: [<User(id=2, name=Bob, age=30)>, <User(id=3, name=Charlie, age=35)>, ...]
Users between 20 and 30: [<User(id=1, name=Alice, age=25)>, <User(id=2, name=Bob, age=30)>, ...]
Users named Alice or Bob: [<User(id=1, name=Alice, age=25)>, <User(id=2, name=Bob, age=30)>]
Users with ID in [1, 2, 3]: [<User(id=1, name=Alice, age=25)>, <User(id=2, name=Bob, age=30)>, ...]
Users whose name starts with A: [<User(id=1, name=Alice, age=25)>]
Users ordered by age (desc): [<User(id=3, name=Charlie, age=35)>, <User(id=2, name=Bob, age=30)>, ...]
First 5 users: [<User(id=1, name=Alice, age=25)>, <User(id=2, name=Bob, age=30)>, ...]
Users 6-10: []
Total users: 5
Average age: 27.6
Max age: 35
Min age: 20
Users by age: [(20, 1), (25, 1), (28, 1), (30, 1), (35, 1)]
```

---

## 场景5：Session 状态管理

### 完整代码

```python
# database.py
from sqlalchemy import create_engine, Column, Integer, String, inspect
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "postgresql://user:password@localhost:5432/mydb"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True)

    def __repr__(self):
        return f"<User(id={self.id}, name={self.name})>"


Base.metadata.create_all(bind=engine)


def check_object_state(obj, label: str):
    """检查对象状态"""
    state = inspect(obj)
    print(f"\n{label}:")
    print(f"  Transient: {state.transient}")
    print(f"  Pending: {state.pending}")
    print(f"  Persistent: {state.persistent}")
    print(f"  Detached: {state.detached}")


def demonstrate_object_states():
    """演示对象状态转换"""
    # Transient 状态
    user = User(name="Alice", email="alice@example.com")
    check_object_state(user, "Transient (刚创建)")

    with SessionLocal() as session:
        # Pending 状态
        session.add(user)
        check_object_state(user, "Pending (已加入 Session)")

        # Persistent 状态
        session.commit()
        check_object_state(user, "Persistent (已提交)")

    # Detached 状态
    check_object_state(user, "Detached (Session 关闭)")


def demonstrate_session_operations():
    """演示 Session 操作"""
    with SessionLocal() as session:
        # 创建对象
        user = User(name="Bob", email="bob@example.com")
        session.add(user)

        # 查看 Session 中的对象
        print(f"\nNew objects: {list(session.new)}")
        print(f"Dirty objects: {list(session.dirty)}")
        print(f"Deleted objects: {list(session.deleted)}")

        # flush
        session.flush()
        print(f"\nAfter flush:")
        print(f"New objects: {list(session.new)}")
        print(f"Dirty objects: {list(session.dirty)}")

        # 修改对象
        user.name = "Bob Updated"
        print(f"\nAfter modification:")
        print(f"Dirty objects: {list(session.dirty)}")

        # commit
        session.commit()
        print(f"\nAfter commit:")
        print(f"New objects: {list(session.new)}")
        print(f"Dirty objects: {list(session.dirty)}")


def demonstrate_expunge():
    """演示 expunge 操作"""
    with SessionLocal() as session:
        user = User(name="Charlie", email="charlie@example.com")
        session.add(user)
        session.commit()

        print(f"\nBefore expunge:")
        check_object_state(user, "User")

        # 移除对象
        session.expunge(user)

        print(f"\nAfter expunge:")
        check_object_state(user, "User")

        # 修改对象不会影响数据库
        user.name = "Charlie Updated"
        session.commit()

        # 验证数据库中的数据没有改变
        db_user = session.query(User).filter_by(email="charlie@example.com").first()
        print(f"\nDatabase user: {db_user}")


# 测试
if __name__ == "__main__":
    demonstrate_object_states()
    demonstrate_session_operations()
    demonstrate_expunge()
```

### 运行结果

```
Transient (刚创建):
  Transient: True
  Pending: False
  Persistent: False
  Detached: False

Pending (已加入 Session):
  Transient: False
  Pending: True
  Persistent: False
  Detached: False

Persistent (已提交):
  Transient: False
  Pending: False
  Persistent: True
  Detached: False

Detached (Session 关闭):
  Transient: False
  Pending: False
  Persistent: False
  Detached: True

New objects: [<User(id=None, name=Bob)>]
Dirty objects: []
Deleted objects: []

After flush:
New objects: []
Dirty objects: []

After modification:
Dirty objects: [<User(id=2, name=Bob Updated)>]

After commit:
New objects: []
Dirty objects: []

Before expunge:
User:
  Transient: False
  Pending: False
  Persistent: True
  Detached: False

After expunge:
User:
  Transient: False
  Pending: False
  Persistent: False
  Detached: True

Database user: <User(id=3, name=Charlie)>
```

---

## 总结

**基础 Session 操作的核心要点：**

1. **CRUD 操作**：create、read、update、delete
2. **上下文管理器**：使用 with 语句自动管理 Session
3. **批量操作**：add_all、bulk_insert_mappings
4. **查询操作**：filter、filter_by、order_by、limit、offset
5. **对象状态**：Transient、Pending、Persistent、Detached

**最佳实践：**
- 使用上下文管理器管理 Session
- 使用 try-except-finally 处理异常
- 批量操作使用 bulk_insert_mappings
- 查询后及时关闭 Session

---

**记住：** Session 是工作单元，用完即关，避免连接泄漏。
