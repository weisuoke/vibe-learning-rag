# 10_实战代码_场景2_Compose部署

> Docker Compose 生产级部署实战

---

## 场景说明

**适用场景：** 生产环境、团队协作、持久化需求

**学习目标：**
- 掌握 docker-compose.yml 配置
- 理解 Milvus 的依赖服务（etcd、MinIO）
- 学会数据持久化和网络配置
- 能够管理多容器应用

**前置要求：**
- Docker Compose 已安装
- 理解 YAML 语法
- 完成场景1学习

---

## docker-compose.yml 完整配置

```yaml
version: '3.8'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9001:9001"
      - "9000:9000"
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "standalone"]
    security_opt:
      - seccomp:unconfined
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

networks:
  default:
    name: milvus
```

---

## 完整代码实现

```python
"""
Docker Compose 部署管理
演示：使用 Python 管理 docker-compose 部署
"""

import subprocess
import time
import os
from pathlib import Path
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

class MilvusComposeDeployer:
    """Docker Compose 部署管理器"""
    
    def __init__(self, project_dir="./milvus-compose"):
        self.project_dir = Path(project_dir)
        self.compose_file = self.project_dir / "docker-compose.yml"
        self.env_file = self.project_dir / ".env"
    
    def create_project_structure(self):
        """创建项目目录结构"""
        print("=== 1. 创建项目结构 ===")
        
        # 创建目录
        self.project_dir.mkdir(exist_ok=True)
        (self.project_dir / "volumes").mkdir(exist_ok=True)
        
        print(f"✅ 项目目录: {self.project_dir}")
        print(f"✅ 数据目录: {self.project_dir / 'volumes'}")
    
    def create_compose_file(self):
        """创建 docker-compose.yml"""
        print("\n=== 2. 创建 docker-compose.yml ===")
        
        compose_content = """version: '3.8'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9001:9001"
      - "9000:9000"
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

networks:
  default:
    name: milvus
"""
        
        with open(self.compose_file, 'w') as f:
            f.write(compose_content)
        
        print(f"✅ 已创建: {self.compose_file}")
    
    def create_env_file(self):
        """创建 .env 文件"""
        print("\n=== 3. 创建 .env 文件 ===")
        
        env_content = """# Milvus Docker Compose 配置
DOCKER_VOLUME_DIRECTORY=./volumes
MILVUS_VERSION=v2.4.0
"""
        
        with open(self.env_file, 'w') as f:
            f.write(env_content)
        
        print(f"✅ 已创建: {self.env_file}")
    
    def start_services(self):
        """启动所有服务"""
        print("\n=== 4. 启动服务 ===")
        
        try:
            # 切换到项目目录
            os.chdir(self.project_dir)
            
            # 启动服务
            print("正在启动服务...")
            subprocess.run(
                ["docker-compose", "up", "-d"],
                check=True
            )
            
            print("✅ 服务启动成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ 服务启动失败: {e}")
            return False
    
    def wait_for_services(self, max_wait=120):
        """等待所有服务就绪"""
        print("\n=== 5. 等待服务就绪 ===")
        print(f"最多等待 {max_wait} 秒...")
        
        start_time = time.time()
        services = ["milvus-etcd", "milvus-minio", "milvus-standalone"]
        
        while time.time() - start_time < max_wait:
            try:
                # 检查所有容器状态
                result = subprocess.run(
                    ["docker-compose", "ps", "--services", "--filter", "status=running"],
                    capture_output=True,
                    text=True,
                    cwd=self.project_dir
                )
                
                running_services = result.stdout.strip().split('\n')
                
                if len(running_services) >= 3:
                    # 尝试连接 Milvus
                    try:
                        connections.connect(
                            alias="default",
                            host="localhost",
                            port="19530",
                            timeout=5
                        )
                        version = utility.get_server_version()
                        print(f"✅ 所有服务就绪！Milvus 版本: {version}")
                        connections.disconnect("default")
                        return True
                    except:
                        pass
                
                elapsed = int(time.time() - start_time)
                print(f"⏳ 等待中... ({elapsed}s)", end="\r")
                time.sleep(3)
            except Exception as e:
                print(f"检查服务状态时出错: {e}")
                time.sleep(3)
        
        print("\n❌ 服务启动超时")
        return False
    
    def show_services_status(self):
        """显示服务状态"""
        print("\n=== 6. 服务状态 ===")
        
        try:
            result = subprocess.run(
                ["docker-compose", "ps"],
                capture_output=True,
                text=True,
                cwd=self.project_dir
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"获取服务状态失败: {e}")
    
    def test_persistence(self):
        """测试数据持久化"""
        print("\n=== 7. 测试数据持久化 ===")
        
        try:
            # 连接
            connections.connect("default", host="localhost", port="19530")
            
            # 创建 Collection
            collection_name = "test_persistence"
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
            
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
            ]
            schema = CollectionSchema(fields)
            collection = Collection(collection_name, schema)
            
            # 插入数据
            import random
            texts = [f"Document {i}" for i in range(100)]
            embeddings = [[random.random() for _ in range(128)] for _ in range(100)]
            collection.insert([texts, embeddings])
            collection.flush()
            
            print(f"✅ 插入了 {collection.num_entities} 条数据")
            
            # 重启服务测试持久化
            print("\n重启服务以测试持久化...")
            self.restart_services()
            
            # 等待服务就绪
            time.sleep(10)
            
            # 重新连接
            connections.connect("default", host="localhost", port="19530")
            collection = Collection(collection_name)
            
            print(f"✅ 数据持久化成功！数据量: {collection.num_entities}")
            
            return True
        except Exception as e:
            print(f"❌ 持久化测试失败: {e}")
            return False
        finally:
            try:
                connections.disconnect("default")
            except:
                pass
    
    def restart_services(self):
        """重启服务"""
        print("\n=== 重启服务 ===")
        
        try:
            # 停止服务
            subprocess.run(
                ["docker-compose", "stop"],
                check=True,
                cwd=self.project_dir
            )
            print("✅ 服务已停止")
            
            time.sleep(2)
            
            # 启动服务
            subprocess.run(
                ["docker-compose", "start"],
                check=True,
                cwd=self.project_dir
            )
            print("✅ 服务已启动")
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ 重启失败: {e}")
            return False
    
    def stop_services(self):
        """停止服务"""
        print("\n=== 停止服务 ===")
        
        try:
            subprocess.run(
                ["docker-compose", "stop"],
                check=True,
                cwd=self.project_dir
            )
            print("✅ 服务已停止")
        except subprocess.CalledProcessError as e:
            print(f"停止服务失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        print("\n=== 清理资源 ===")
        
        try:
            subprocess.run(
                ["docker-compose", "down", "-v"],
                check=True,
                cwd=self.project_dir
            )
            print("✅ 资源已清理")
        except subprocess.CalledProcessError as e:
            print(f"清理失败: {e}")
    
    def deploy(self):
        """完整部署流程"""
        print("=" * 50)
        print("Milvus Docker Compose 部署")
        print("=" * 50)
        
        # 1. 创建项目结构
        self.create_project_structure()
        
        # 2. 创建配置文件
        self.create_compose_file()
        self.create_env_file()
        
        # 3. 启动服务
        if not self.start_services():
            return False
        
        # 4. 等待服务就绪
        if not self.wait_for_services():
            return False
        
        # 5. 显示服务状态
        self.show_services_status()
        
        # 6. 测试持久化
        self.test_persistence()
        
        print("\n" + "=" * 50)
        print("✅ 部署完成！")
        print("=" * 50)
        print(f"\n项目目录: {self.project_dir}")
        print(f"Milvus 连接: localhost:19530")
        print(f"MinIO 控制台: http://localhost:9001")
        print(f"\n管理命令：")
        print(f"  启动: cd {self.project_dir} && docker-compose up -d")
        print(f"  停止: cd {self.project_dir} && docker-compose stop")
        print(f"  查看日志: cd {self.project_dir} && docker-compose logs -f")
        print(f"  清理: cd {self.project_dir} && docker-compose down -v")
        
        return True

# ===== 主程序 =====
if __name__ == "__main__":
    deployer = MilvusComposeDeployer()
    
    try:
        deployer.deploy()
    except KeyboardInterrupt:
        print("\n\n⚠️ 部署被用户中断")
        deployer.stop_services()
    except Exception as e:
        print(f"\n\n❌ 部署失败: {e}")
        deployer.stop_services()
```

---

## 配置详解

### etcd 配置

```yaml
etcd:
  environment:
    - ETCD_AUTO_COMPACTION_MODE=revision  # 自动压缩模式
    - ETCD_AUTO_COMPACTION_RETENTION=1000  # 保留最近1000个版本
    - ETCD_QUOTA_BACKEND_BYTES=4294967296  # 后端存储配额（4GB）
```

**作用：**
- etcd 存储 Milvus 的元数据
- 自动压缩防止数据膨胀
- 配额限制防止磁盘占满

### MinIO 配置

```yaml
minio:
  environment:
    MINIO_ACCESS_KEY: minioadmin  # 访问密钥
    MINIO_SECRET_KEY: minioadmin  # 密钥
  ports:
    - "9001:9001"  # Web 控制台
    - "9000:9000"  # API 端口
```

**作用：**
- MinIO 存储向量数据和日志
- 控制台可视化管理
- S3 兼容 API

### Milvus 配置

```yaml
standalone:
  environment:
    ETCD_ENDPOINTS: etcd:2379      # etcd 地址
    MINIO_ADDRESS: minio:9000      # MinIO 地址
  depends_on:
    - "etcd"
    - "minio"
```

**作用：**
- 通过服务名访问依赖服务
- depends_on 确保启动顺序
- 健康检查确保服务就绪

---

## 在 RAG 中的应用

### 生产环境部署

```yaml
# 生产环境 docker-compose.yml
version: '3.8'

services:
  # ... etcd, minio, milvus ...
  
  rag-api:
    build: ./rag-api
    ports:
      - "8000:8000"
    environment:
      MILVUS_HOST: standalone
      MILVUS_PORT: 19530
    depends_on:
      - standalone
    restart: always

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - rag-api
```

---

## 检查清单

- [ ] 理解 docker-compose.yml 结构
- [ ] 掌握服务依赖配置
- [ ] 配置数据持久化
- [ ] 测试服务重启后数据保留
- [ ] 能够管理多容器应用
- [ ] 理解网络配置

---

**记住：** Docker Compose 是生产环境的推荐部署方式，配置文件化便于版本控制和团队协作！
