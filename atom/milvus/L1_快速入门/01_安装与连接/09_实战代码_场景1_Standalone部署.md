# 09_实战代码_场景1_Standalone部署

> 完整的 Docker Standalone 部署实战示例

---

## 场景说明

**适用场景：** 本地开发、快速测试、学习 Milvus

**学习目标：**
- 掌握 Docker 部署 Milvus Standalone 的完整流程
- 学会使用 Python 自动化部署和验证
- 理解 Milvus 的启动和健康检查机制

**前置要求：**
- Docker 已安装并运行
- Python 3.9+
- pymilvus 已安装

---

## 完整代码实现

```python
"""
Milvus Standalone 部署实战
演示：使用 Docker 部署 Milvus 并验证连接
"""

import subprocess
import time
import sys
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

class MilvusStandaloneDeployer:
    """Milvus Standalone 部署器"""
    
    def __init__(self, container_name="milvus-standalone", port=19530):
        self.container_name = container_name
        self.port = port
        self.host = "localhost"
    
    def check_docker(self):
        """检查 Docker 是否安装"""
        print("=== 1. 检查 Docker 环境 ===")
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ Docker 已安装: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Docker 未安装或未运行")
            return False
    
    def stop_existing_container(self):
        """停止并删除已存在的容器"""
        print(f"\n=== 2. 清理已存在的容器 ===")
        try:
            # 检查容器是否存在
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True
            )
            
            if self.container_name in result.stdout:
                print(f"发现已存在的容器: {self.container_name}")
                
                # 停止容器
                subprocess.run(["docker", "stop", self.container_name], check=True)
                print(f"✅ 已停止容器: {self.container_name}")
                
                # 删除容器
                subprocess.run(["docker", "rm", self.container_name], check=True)
                print(f"✅ 已删除容器: {self.container_name}")
            else:
                print("✅ 没有需要清理的容器")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ 清理容器时出错: {e}")
    
    def pull_image(self):
        """拉取 Milvus 镜像"""
        print("\n=== 3. 拉取 Milvus 镜像 ===")
        try:
            print("正在拉取 milvusdb/milvus:latest...")
            subprocess.run(
                ["docker", "pull", "milvusdb/milvus:latest"],
                check=True
            )
            print("✅ 镜像拉取成功")
            return True
        except subprocess.CalledProcessError:
            print("❌ 镜像拉取失败")
            return False
    
    def start_container(self):
        """启动 Milvus 容器"""
        print("\n=== 4. 启动 Milvus 容器 ===")
        try:
            cmd = [
                "docker", "run", "-d",
                "--name", self.container_name,
                "-p", f"{self.port}:19530",
                "-p", "9091:9091",
                "milvusdb/milvus:latest"
            ]
            
            subprocess.run(cmd, check=True)
            print(f"✅ 容器启动成功: {self.container_name}")
            return True
        except subprocess.CalledProcessError:
            print("❌ 容器启动失败")
            return False
    
    def wait_for_ready(self, max_wait=60):
        """等待 Milvus 就绪"""
        print("\n=== 5. 等待 Milvus 就绪 ===")
        print(f"最多等待 {max_wait} 秒...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=str(self.port),
                    timeout=5
                )
                version = utility.get_server_version()
                print(f"✅ Milvus 已就绪！版本: {version}")
                connections.disconnect("default")
                return True
            except Exception:
                elapsed = int(time.time() - start_time)
                print(f"⏳ 等待中... ({elapsed}s)", end="\r")
                time.sleep(2)
        
        print("\n❌ Milvus 启动超时")
        return False
    
    def verify_connection(self):
        """验证连接"""
        print("\n=== 6. 验证连接 ===")
        try:
            # 建立连接
            connections.connect(
                alias="default",
                host=self.host,
                port=str(self.port)
            )
            print("✅ 连接建立成功")
            
            # 获取服务器版本
            version = utility.get_server_version()
            print(f"✅ 服务器版本: {version}")
            
            # 列出所有 Collection
            collections = utility.list_collections()
            print(f"✅ 当前 Collection 数量: {len(collections)}")
            
            return True
        except Exception as e:
            print(f"❌ 连接验证失败: {e}")
            return False
    
    def create_test_collection(self):
        """创建测试 Collection"""
        print("\n=== 7. 创建测试 Collection ===")
        try:
            collection_name = "test_standalone"
            
            # 检查是否已存在
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                print(f"✅ 已删除旧的 Collection: {collection_name}")
            
            # 定义 Schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
            ]
            schema = CollectionSchema(fields, description="Test collection")
            
            # 创建 Collection
            collection = Collection(collection_name, schema)
            print(f"✅ Collection 创建成功: {collection_name}")
            
            # 插入测试数据
            import random
            texts = [f"Test document {i}" for i in range(10)]
            embeddings = [[random.random() for _ in range(128)] for _ in range(10)]
            
            collection.insert([texts, embeddings])
            collection.flush()
            print(f"✅ 插入了 {collection.num_entities} 条测试数据")
            
            # 创建索引
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128}
            }
            collection.create_index("embedding", index_params)
            print("✅ 索引创建成功")
            
            # 加载 Collection
            collection.load()
            print("✅ Collection 已加载到内存")
            
            # 执行检索测试
            query_embedding = [random.random() for _ in range(128)]
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=3,
                output_fields=["text"]
            )
            
            print(f"✅ 检索测试成功，返回 {len(results[0])} 条结果")
            for i, hit in enumerate(results[0]):
                print(f"   结果 {i+1}: ID={hit.id}, Distance={hit.distance:.4f}, Text={hit.entity.get('text')}")
            
            return True
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False
    
    def show_container_info(self):
        """显示容器信息"""
        print("\n=== 8. 容器信息 ===")
        try:
            # 容器状态
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}"],
                capture_output=True,
                text=True
            )
            print(result.stdout)
            
            # 容器日志（最后10行）
            print("\n最近日志：")
            result = subprocess.run(
                ["docker", "logs", "--tail", "10", self.container_name],
                capture_output=True,
                text=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"⚠️ 获取容器信息失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        print("\n=== 9. 清理资源 ===")
        try:
            connections.disconnect("default")
            print("✅ 已断开连接")
        except:
            pass
    
    def deploy(self):
        """完整部署流程"""
        print("=" * 50)
        print("Milvus Standalone 自动化部署")
        print("=" * 50)
        
        # 1. 检查 Docker
        if not self.check_docker():
            return False
        
        # 2. 清理已存在的容器
        self.stop_existing_container()
        
        # 3. 拉取镜像
        if not self.pull_image():
            return False
        
        # 4. 启动容器
        if not self.start_container():
            return False
        
        # 5. 等待就绪
        if not self.wait_for_ready():
            return False
        
        # 6. 验证连接
        if not self.verify_connection():
            return False
        
        # 7. 创建测试 Collection
        if not self.create_test_collection():
            return False
        
        # 8. 显示容器信息
        self.show_container_info()
        
        # 9. 清理
        self.cleanup()
        
        print("\n" + "=" * 50)
        print("✅ 部署完成！")
        print("=" * 50)
        print(f"\n连接信息：")
        print(f"  Host: {self.host}")
        print(f"  Port: {self.port}")
        print(f"\n使用以下命令连接：")
        print(f"  from pymilvus import connections")
        print(f"  connections.connect('default', host='{self.host}', port='{self.port}')")
        print(f"\n停止容器：")
        print(f"  docker stop {self.container_name}")
        print(f"\n删除容器：")
        print(f"  docker rm {self.container_name}")
        
        return True

# ===== 主程序 =====
if __name__ == "__main__":
    deployer = MilvusStandaloneDeployer()
    
    try:
        success = deployer.deploy()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ 部署被用户中断")
        deployer.cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 部署失败: {e}")
        deployer.cleanup()
        sys.exit(1)
```

---

## 运行输出示例

```
==================================================
Milvus Standalone 自动化部署
==================================================
=== 1. 检查 Docker 环境 ===
✅ Docker 已安装: Docker version 24.0.7, build afdd53b

=== 2. 清理已存在的容器 ===
✅ 没有需要清理的容器

=== 3. 拉取 Milvus 镜像 ===
正在拉取 milvusdb/milvus:latest...
✅ 镜像拉取成功

=== 4. 启动 Milvus 容器 ===
✅ 容器启动成功: milvus-standalone

=== 5. 等待 Milvus 就绪 ===
最多等待 60 秒...
✅ Milvus 已就绪！版本: 2.4.0

=== 6. 验证连接 ===
✅ 连接建立成功
✅ 服务器版本: 2.4.0
✅ 当前 Collection 数量: 0

=== 7. 创建测试 Collection ===
✅ Collection 创建成功: test_standalone
✅ 插入了 10 条测试数据
✅ 索引创建成功
✅ Collection 已加载到内存
✅ 检索测试成功，返回 3 条结果
   结果 1: ID=1, Distance=12.3456, Text=Test document 1
   结果 2: ID=5, Distance=15.7890, Text=Test document 5
   结果 3: ID=3, Distance=18.2345, Text=Test document 3

=== 8. 容器信息 ===
NAMES               STATUS              PORTS
milvus-standalone   Up 2 minutes        0.0.0.0:19530->19530/tcp, 0.0.0.0:9091->9091/tcp

最近日志：
[2026-02-09 12:34:56] Milvus Proxy successfully started
[2026-02-09 12:34:57] Listening on 0.0.0.0:19530

=== 9. 清理资源 ===
✅ 已断开连接

==================================================
✅ 部署完成！
==================================================

连接信息：
  Host: localhost
  Port: 19530

使用以下命令连接：
  from pymilvus import connections
  connections.connect('default', host='localhost', port='19530')

停止容器：
  docker stop milvus-standalone

删除容器：
  docker rm milvus-standalone
```

---

## 代码详解

### 1. 部署器类设计

```python
class MilvusStandaloneDeployer:
    """封装了完整的部署流程"""
    
    def __init__(self, container_name="milvus-standalone", port=19530):
        # 可配置的容器名称和端口
        self.container_name = container_name
        self.port = port
```

**设计优势：**
- 可复用：可以部署多个 Milvus 实例
- 可配置：容器名称和端口可自定义
- 易测试：每个方法独立，便于单元测试

### 2. Docker 环境检查

```python
def check_docker(self):
    """使用 subprocess 调用 docker 命令"""
    result = subprocess.run(
        ["docker", "--version"],
        capture_output=True,  # 捕获输出
        text=True,            # 以文本形式返回
        check=True            # 失败时抛出异常
    )
```

**关键点：**
- `capture_output=True`：捕获标准输出和错误输出
- `text=True`：返回字符串而非字节
- `check=True`：命令失败时抛出 `CalledProcessError`

### 3. 等待服务就绪

```python
def wait_for_ready(self, max_wait=60):
    """轮询检查 Milvus 是否就绪"""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            connections.connect(...)
            version = utility.get_server_version()
            return True  # 连接成功，服务就绪
        except Exception:
            time.sleep(2)  # 等待2秒后重试
```

**为什么需要等待？**
- Milvus 容器启动后需要时间初始化
- 过早连接会导致连接失败
- 轮询检查确保服务完全就绪

### 4. 完整的测试流程

```python
def create_test_collection(self):
    """创建 → 插入 → 索引 → 检索"""
    # 1. 创建 Collection
    collection = Collection(collection_name, schema)
    
    # 2. 插入数据
    collection.insert([texts, embeddings])
    collection.flush()  # 确保数据持久化
    
    # 3. 创建索引
    collection.create_index("embedding", index_params)
    
    # 4. 加载到内存
    collection.load()
    
    # 5. 执行检索
    results = collection.search(...)
```

**测试覆盖：**
- ✅ Collection 创建
- ✅ 数据插入
- ✅ 索引创建
- ✅ 数据检索
- ✅ 完整的 CRUD 流程

---

## 在 RAG 中的应用

### 快速搭建 RAG 开发环境

```python
# 1. 部署 Milvus
deployer = MilvusStandaloneDeployer()
deployer.deploy()

# 2. 创建 RAG Collection
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

connections.connect("default", host="localhost", port="19530")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=5000),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
]
schema = CollectionSchema(fields, description="RAG documents")
collection = Collection("rag_docs", schema)

# 3. 开始开发 RAG 应用
# ...
```

---

## 常见问题排查

### 问题1：端口被占用

```bash
# 错误信息
docker: Error response from daemon: driver failed programming external connectivity on endpoint milvus-standalone: Bind for 0.0.0.0:19530 failed: port is already allocated.

# 解决方案
# 方案1：停止占用端口的进程
lsof -i :19530
kill -9 <PID>

# 方案2：使用其他端口
deployer = MilvusStandaloneDeployer(port=19531)
```

### 问题2：容器启动失败

```bash
# 查看容器日志
docker logs milvus-standalone

# 常见原因：
# 1. 内存不足（需要至少 4GB）
# 2. Docker 版本过旧（需要 19.03+）
# 3. 镜像损坏（重新拉取）
```

### 问题3：连接超时

```python
# 增加等待时间
deployer.wait_for_ready(max_wait=120)  # 等待2分钟

# 或手动检查
docker logs -f milvus-standalone
# 看到 "Milvus Proxy successfully started" 表示就绪
```

---

## 检查清单

完成本实战后，你应该能够：

- [ ] 使用 Python 自动化部署 Milvus Standalone
- [ ] 理解 Docker 容器的生命周期管理
- [ ] 实现健康检查和等待逻辑
- [ ] 创建测试 Collection 并验证功能
- [ ] 排查常见的部署问题
- [ ] 将部署流程集成到 RAG 开发环境

---

## 下一步学习

- **场景2：Docker Compose 部署**（10_实战代码_场景2_Compose部署.md）
  - 学习生产环境的部署方式

- **场景3：连接管理**（11_实战代码_场景3_连接管理.md）
  - 学习连接池和异常处理

---

**记住：** 自动化部署脚本是开发效率的关键，掌握这个技能能大大提升你的工作效率！
