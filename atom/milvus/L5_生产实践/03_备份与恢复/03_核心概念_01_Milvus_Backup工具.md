# 核心概念1：Milvus Backup 工具

> 官方备份工具的完整使用指南

---

## 什么是 Milvus Backup？

**Milvus Backup** 是 Milvus 官方提供的备份恢复工具，支持 Collection 级别的数据备份和恢复。

**核心特性：**
- ✅ 全量备份和增量备份
- ✅ Collection 级别的备份
- ✅ 支持压缩和加密
- ✅ 跨版本恢复
- ✅ 命令行和 API 两种使用方式

---

## 架构原理

### 1. 整体架构

```
┌─────────────────────────────────────────┐
│         Milvus Backup 工具               │
├─────────────────────────────────────────┤
│  命令行接口 (CLI)  │  HTTP API 接口      │
├─────────────────────────────────────────┤
│         备份管理器 (Backup Manager)      │
│  - 备份创建                              │
│  - 备份恢复                              │
│  - 备份验证                              │
│  - 备份列表                              │
├─────────────────────────────────────────┤
│      存储适配器 (Storage Adapter)        │
│  - 本地存储                              │
│  - MinIO/S3                             │
│  - 阿里云 OSS                            │
│  - 腾讯云 COS                            │
└─────────────────────────────────────────┘
         ↓                    ↓
    ┌────────┐          ┌──────────┐
    │ Milvus │          │  存储后端  │
    └────────┘          └──────────┘
```

### 2. 备份流程

```
创建备份：
1. 连接 Milvus → 2. 读取 Collection 元数据
   ↓
3. 导出数据 → 4. 压缩（可选）
   ↓
5. 上传到存储 → 6. 记录备份元数据

恢复备份：
1. 从存储下载 → 2. 解压（如果压缩）
   ↓
3. 验证完整性 → 4. 创建 Collection
   ↓
5. 导入数据 → 6. 重建索引
```

---

## 安装和配置

### 1. 安装 Milvus Backup

**方式1：使用 Docker（推荐）**

```bash
# 拉取镜像
docker pull milvusdb/milvus-backup:latest

# 运行容器
docker run -d \
  --name milvus-backup \
  -p 8080:8080 \
  -v $(pwd)/backup_data:/backup \
  -v $(pwd)/config.yaml:/etc/milvus-backup/config.yaml \
  milvusdb/milvus-backup:latest
```

**方式2：从源码编译**

```bash
# 克隆仓库
git clone https://github.com/zilliztech/milvus-backup.git
cd milvus-backup

# 编译
go build -o milvus-backup cmd/backup/main.go

# 安装
sudo mv milvus-backup /usr/local/bin/
```

### 2. 配置文件

创建 `config.yaml`：

```yaml
# Milvus 连接配置
milvus:
  address: localhost
  port: 19530
  username: ""
  password: ""
  authorizationEnabled: false

# 备份存储配置
storage:
  # 存储类型：local, minio, s3, oss, cos
  storageType: local

  # 本地存储配置
  local:
    path: /backup

  # MinIO/S3 配置
  minio:
    address: localhost
    port: 9000
    accessKeyID: minioadmin
    secretAccessKey: minioadmin
    useSSL: false
    bucketName: milvus-backup
    rootPath: backup

  # 阿里云 OSS 配置
  oss:
    endpoint: oss-cn-hangzhou.aliyuncs.com
    accessKeyID: your-access-key
    accessKeySecret: your-access-secret
    bucketName: milvus-backup
    rootPath: backup

# 备份配置
backup:
  # 最大备份数量
  maxBackupNum: 10

  # 备份保留天数
  retentionDays: 30

  # 是否压缩
  compression: true

  # 压缩算法：gzip, zstd
  compressionAlgorithm: zstd

  # 压缩级别：1-9
  compressionLevel: 3

# HTTP 服务配置
http:
  # 监听地址
  address: 0.0.0.0

  # 监听端口
  port: 8080

# 日志配置
log:
  # 日志级别：debug, info, warn, error
  level: info

  # 日志文件路径
  file: /var/log/milvus-backup.log
```

---

## 命令行使用

### 1. 创建备份

**全量备份：**

```bash
# 备份单个 Collection
milvus-backup create \
  --collection my_collection \
  --backup-name backup_20260210

# 备份多个 Collection
milvus-backup create \
  --collection collection1,collection2,collection3 \
  --backup-name backup_20260210

# 备份所有 Collection
milvus-backup create \
  --all \
  --backup-name backup_20260210
```

**增量备份：**

```bash
# 基于上次备份创建增量备份
milvus-backup create \
  --collection my_collection \
  --backup-name backup_20260210_incremental \
  --base-backup backup_20260209
```

**输出示例：**

```
Creating backup...
[1/5] Connecting to Milvus...
[2/5] Reading collection metadata...
[3/5] Exporting data...
  Progress: 100% (1000000/1000000 entities)
[4/5] Compressing backup...
  Compression ratio: 75%
[5/5] Uploading to storage...
  Upload speed: 50 MB/s

✅ Backup created successfully!
  Backup name: backup_20260210
  Collections: my_collection
  Total entities: 1000000
  Backup size: 2.5 GB (compressed)
  Duration: 5m 30s
```

### 2. 列出备份

```bash
# 列出所有备份
milvus-backup list

# 列出特定 Collection 的备份
milvus-backup list --collection my_collection

# 显示详细信息
milvus-backup list --verbose
```

**输出示例：**

```
Backup Name              Collections         Entities    Size      Created At
backup_20260210          my_collection       1000000     2.5 GB    2026-02-10 10:00:00
backup_20260209          my_collection       950000      2.4 GB    2026-02-09 10:00:00
backup_20260208          my_collection       900000      2.3 GB    2026-02-08 10:00:00
```

### 3. 恢复备份

**恢复到原 Collection：**

```bash
# 恢复备份（覆盖原 Collection）
milvus-backup restore \
  --backup-name backup_20260210 \
  --collection my_collection
```

**恢复到新 Collection：**

```bash
# 恢复到新 Collection
milvus-backup restore \
  --backup-name backup_20260210 \
  --collection my_collection \
  --target-collection my_collection_restored
```

**恢复多个 Collection：**

```bash
# 恢复所有 Collection
milvus-backup restore \
  --backup-name backup_20260210 \
  --all
```

**输出示例：**

```
Restoring backup...
[1/6] Downloading backup from storage...
  Download speed: 100 MB/s
[2/6] Decompressing backup...
[3/6] Verifying backup integrity...
[4/6] Creating collection...
[5/6] Importing data...
  Progress: 100% (1000000/1000000 entities)
[6/6] Building index...
  Index type: HNSW
  Progress: 100%

✅ Backup restored successfully!
  Collection: my_collection_restored
  Total entities: 1000000
  Duration: 8m 20s
```

### 4. 删除备份

```bash
# 删除单个备份
milvus-backup delete --backup-name backup_20260208

# 删除所有备份
milvus-backup delete --all

# 删除过期备份（超过 30 天）
milvus-backup delete --older-than 30d
```

### 5. 验证备份

```bash
# 验证备份完整性
milvus-backup verify --backup-name backup_20260210

# 验证所有备份
milvus-backup verify --all
```

**输出示例：**

```
Verifying backup...
[1/3] Checking backup metadata...
[2/3] Verifying checksum...
[3/3] Testing restore (dry run)...

✅ Backup is valid!
  Backup name: backup_20260210
  Checksum: OK
  Metadata: OK
  Restore test: OK
```

---

## HTTP API 使用

### 1. 启动 HTTP 服务

```bash
# 启动服务
milvus-backup server --config config.yaml

# 或使用 Docker
docker run -d \
  --name milvus-backup \
  -p 8080:8080 \
  -v $(pwd)/config.yaml:/etc/milvus-backup/config.yaml \
  milvusdb/milvus-backup:latest
```

### 2. API 接口

**创建备份：**

```bash
curl -X POST http://localhost:8080/api/v1/backup/create \
  -H "Content-Type: application/json" \
  -d '{
    "backup_name": "backup_20260210",
    "collections": ["my_collection"],
    "compression": true
  }'
```

**响应：**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "backup_id": "backup_20260210",
    "status": "completed",
    "collections": ["my_collection"],
    "total_entities": 1000000,
    "backup_size": 2684354560,
    "created_at": "2026-02-10T10:00:00Z",
    "duration": 330
  }
}
```

**列出备份：**

```bash
curl -X GET http://localhost:8080/api/v1/backup/list
```

**恢复备份：**

```bash
curl -X POST http://localhost:8080/api/v1/backup/restore \
  -H "Content-Type: application/json" \
  -d '{
    "backup_name": "backup_20260210",
    "collections": ["my_collection"],
    "target_collection": "my_collection_restored"
  }'
```

**删除备份：**

```bash
curl -X DELETE http://localhost:8080/api/v1/backup/delete \
  -H "Content-Type: application/json" \
  -d '{
    "backup_name": "backup_20260210"
  }'
```

---

## Python SDK 使用

### 1. 安装 SDK

```bash
pip install milvus-backup-sdk
```

### 2. 基础使用

```python
from milvus_backup import BackupClient

# 创建客户端
client = BackupClient(
    milvus_host="localhost",
    milvus_port=19530,
    backup_host="localhost",
    backup_port=8080
)

# 创建备份
backup_id = client.create_backup(
    backup_name="backup_20260210",
    collections=["my_collection"],
    compression=True
)

print(f"Backup created: {backup_id}")

# 列出备份
backups = client.list_backups()
for backup in backups:
    print(f"{backup['name']}: {backup['size']} bytes")

# 恢复备份
client.restore_backup(
    backup_name="backup_20260210",
    target_collection="my_collection_restored"
)

print("Backup restored successfully!")

# 删除备份
client.delete_backup(backup_name="backup_20260208")
```

### 3. 高级用法

```python
# 异步备份
import asyncio

async def async_backup():
    """异步创建备份"""
    backup_id = await client.create_backup_async(
        backup_name="backup_20260210",
        collections=["my_collection"]
    )

    # 监控备份进度
    while True:
        status = await client.get_backup_status(backup_id)
        print(f"Progress: {status['progress']}%")

        if status['status'] == 'completed':
            break

        await asyncio.sleep(1)

# 运行异步任务
asyncio.run(async_backup())

# 增量备份
client.create_incremental_backup(
    backup_name="backup_20260210_incremental",
    base_backup="backup_20260209",
    collections=["my_collection"]
)

# 验证备份
is_valid = client.verify_backup(backup_name="backup_20260210")
if is_valid:
    print("Backup is valid!")
else:
    print("Backup is corrupted!")
```

---

## 在 RAG 系统中的应用

### 场景1：知识库定期备份

```python
from milvus_backup import BackupClient
from datetime import datetime
import schedule

class RAGBackupManager:
    """RAG 知识库备份管理器"""

    def __init__(self):
        self.client = BackupClient(
            milvus_host="localhost",
            milvus_port=19530,
            backup_host="localhost",
            backup_port=8080
        )

    def backup_knowledge_base(self):
        """备份知识库"""
        # 生成备份名称
        backup_name = f"rag_kb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 创建备份
        print(f"Creating backup: {backup_name}")
        backup_id = self.client.create_backup(
            backup_name=backup_name,
            collections=["documents", "embeddings"],
            compression=True
        )

        # 验证备份
        if self.client.verify_backup(backup_name):
            print(f"✅ Backup {backup_name} created and verified")
        else:
            print(f"❌ Backup {backup_name} verification failed")

        # 清理旧备份（保留最近 7 天）
        self.cleanup_old_backups(days=7)

    def cleanup_old_backups(self, days=7):
        """清理旧备份"""
        backups = self.client.list_backups()
        cutoff_time = datetime.now().timestamp() - (days * 86400)

        for backup in backups:
            if backup['created_at'] < cutoff_time:
                print(f"Deleting old backup: {backup['name']}")
                self.client.delete_backup(backup['name'])

# 使用示例
manager = RAGBackupManager()

# 每天凌晨 2 点备份
schedule.every().day.at("02:00").do(manager.backup_knowledge_base)

# 运行调度器
while True:
    schedule.run_pending()
    time.sleep(60)
```

### 场景2：知识库版本管理

```python
class RAGVersionManager:
    """RAG 知识库版本管理"""

    def __init__(self):
        self.client = BackupClient()
        self.versions = {}

    def create_version(self, version_name, description=""):
        """创建知识库版本"""
        backup_name = f"rag_version_{version_name}"

        # 创建备份
        backup_id = self.client.create_backup(
            backup_name=backup_name,
            collections=["documents", "embeddings"]
        )

        # 记录版本信息
        self.versions[version_name] = {
            "backup_id": backup_id,
            "description": description,
            "created_at": datetime.now(),
            "backup_name": backup_name
        }

        print(f"✅ Version {version_name} created")

    def rollback_to_version(self, version_name):
        """回滚到指定版本"""
        if version_name not in self.versions:
            raise ValueError(f"Version {version_name} not found")

        version = self.versions[version_name]

        # 恢复备份
        self.client.restore_backup(
            backup_name=version["backup_name"],
            collections=["documents", "embeddings"]
        )

        print(f"✅ Rolled back to version {version_name}")

    def list_versions(self):
        """列出所有版本"""
        for name, info in self.versions.items():
            print(f"{name}: {info['description']} ({info['created_at']})")

# 使用示例
manager = RAGVersionManager()

# 创建版本
manager.create_version("v1.0", "Initial knowledge base")

# 更新知识库...

# 创建新版本
manager.create_version("v1.1", "Added new documents")

# 如果有问题，回滚
manager.rollback_to_version("v1.0")
```

---

## 最佳实践

### 1. 备份策略

```yaml
# 推荐的备份策略
backup_strategy:
  # 全量备份：每周日
  full_backup:
    schedule: "0 2 * * 0"
    retention: 4  # 保留 4 周

  # 增量备份：每天
  incremental_backup:
    schedule: "0 2 * * 1-6"
    retention: 7  # 保留 7 天

  # 存储配置
  storage:
    primary: s3://backups-us-west/
    secondary: s3://backups-eu-central/

  # 验证
  verification:
    enabled: true
    schedule: "0 3 * * *"  # 每天凌晨 3 点
```

### 2. 性能优化

```python
# 并行备份多个 Collection
from concurrent.futures import ThreadPoolExecutor

def parallel_backup(collections):
    """并行备份多个 Collection"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for collection in collections:
            future = executor.submit(
                client.create_backup,
                backup_name=f"backup_{collection}_{datetime.now().strftime('%Y%m%d')}",
                collections=[collection]
            )
            futures.append(future)

        # 等待所有备份完成
        for future in futures:
            backup_id = future.result()
            print(f"Backup completed: {backup_id}")

# 使用示例
collections = ["collection1", "collection2", "collection3"]
parallel_backup(collections)
```

### 3. 监控和告警

```python
class BackupMonitor:
    """备份监控"""

    def __init__(self, client):
        self.client = client

    def check_backup_health(self):
        """检查备份健康状态"""
        backups = self.client.list_backups()

        # 检查最近备份时间
        if backups:
            latest_backup = max(backups, key=lambda x: x['created_at'])
            time_since_last = datetime.now().timestamp() - latest_backup['created_at']

            if time_since_last > 86400:  # 24 小时
                self.alert("No backup in last 24 hours!")

        # 检查备份完整性
        for backup in backups:
            if not self.client.verify_backup(backup['name']):
                self.alert(f"Backup {backup['name']} is corrupted!")

    def alert(self, message):
        """发送告警"""
        print(f"🚨 ALERT: {message}")
        # 发送邮件、Slack 通知等

# 使用示例
monitor = BackupMonitor(client)
schedule.every().hour.do(monitor.check_backup_health)
```

---

## 故障排查

### 常见问题

**问题1：备份失败 - 连接超时**

```
Error: Failed to connect to Milvus: connection timeout
```

**解决方案：**
```bash
# 检查 Milvus 是否运行
docker ps | grep milvus

# 检查网络连接
telnet localhost 19530

# 检查配置文件
cat config.yaml | grep address
```

**问题2：备份文件过大**

```
Error: Backup size exceeds storage quota
```

**解决方案：**
```yaml
# 启用压缩
backup:
  compression: true
  compressionAlgorithm: zstd
  compressionLevel: 9  # 最高压缩率
```

**问题3：恢复失败 - 索引重建错误**

```
Error: Failed to build index: out of memory
```

**解决方案：**
```python
# 使用更小的索引参数
index_params = {
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}  # 减小 nlist
}
```

---

## 总结

### 核心要点

1. **Milvus Backup 是官方推荐的备份工具**
2. **支持命令行、HTTP API 和 Python SDK**
3. **支持全量和增量备份**
4. **支持多种存储后端**
5. **内置压缩和验证功能**

### 适用场景

- ✅ 生产环境的定期备份
- ✅ 知识库版本管理
- ✅ 跨集群数据迁移
- ✅ 灾难恢复

### 下一步

- 学习 [Collection 导出导入](./03_核心概念_02_Collection导出导入.md)
- 学习 [数据迁移策略](./03_核心概念_03_数据迁移策略.md)
