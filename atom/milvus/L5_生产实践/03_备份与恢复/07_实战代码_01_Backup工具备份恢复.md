# 实战代码1：使用 Backup 工具备份恢复

> 完整的 Milvus Backup 工具使用示例

---

## 场景概述

本场景演示如何使用 Milvus Backup 工具进行：
- Collection 的完整备份
- 备份的验证和管理
- 数据的恢复
- 自动化备份脚本

**适用场景：**
- 生产环境定期备份
- 灾难恢复演练
- 数据迁移准备

---

## 环境准备

### 1. 安装 Milvus Backup

```bash
# 使用 Docker 安装（推荐）
docker pull milvusdb/milvus-backup:latest

# 或从源码编译
git clone https://github.com/zilliztech/milvus-backup.git
cd milvus-backup
go build -o milvus-backup cmd/backup/main.go
```

### 2. 配置文件

创建 `backup_config.yaml`：

```yaml
# Milvus 连接配置
milvus:
  address: localhost
  port: 19530
  username: ""
  password: ""

# 备份存储配置
storage:
  storageType: local
  local:
    path: /data/milvus-backup

# 备份配置
backup:
  maxBackupNum: 10
  retentionDays: 30
  compression: true
  compressionAlgorithm: zstd
  compressionLevel: 3

# HTTP 服务配置
http:
  address: 0.0.0.0
  port: 8080

# 日志配置
log:
  level: info
  file: /var/log/milvus-backup.log
```

### 3. 启动 Backup 服务

```bash
# 使用 Docker 启动
docker run -d \
  --name milvus-backup \
  -p 8080:8080 \
  -v $(pwd)/backup_data:/data/milvus-backup \
  -v $(pwd)/backup_config.yaml:/etc/milvus-backup/config.yaml \
  milvusdb/milvus-backup:latest

# 验证服务
curl http://localhost:8080/api/v1/health
```

---

## 完整示例代码

### 示例1：基础备份恢复

```python
#!/usr/bin/env python3
"""
Milvus Backup 工具基础使用示例
"""

import requests
import time
import json
from typing import Dict, List, Optional

class MilvusBackupClient:
    """Milvus Backup 客户端"""

    def __init__(self, host: str = "localhost", port: int = 8080):
        """初始化客户端"""
        self.base_url = f"http://{host}:{port}/api/v1"

    def create_backup(
        self,
        backup_name: str,
        collections: List[str],
        compression: bool = True
    ) -> Dict:
        """创建备份"""
        url = f"{self.base_url}/backup/create"
        payload = {
            "backup_name": backup_name,
            "collections": collections,
            "compression": compression
        }

        print(f"创建备份: {backup_name}")
        print(f"Collection: {collections}")

        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        if result["code"] != 0:
            raise Exception(f"备份失败: {result['message']}")

        print(f"✅ 备份创建成功")
        print(f"  备份ID: {result['data']['backup_id']}")
        print(f"  数据量: {result['data']['total_entities']}")
        print(f"  大小: {result['data']['backup_size'] / 1024 / 1024:.2f} MB")
        print(f"  耗时: {result['data']['duration']} 秒")

        return result["data"]

    def list_backups(self) -> List[Dict]:
        """列出所有备份"""
        url = f"{self.base_url}/backup/list"
        response = requests.get(url)
        response.raise_for_status()

        result = response.json()
        if result["code"] != 0:
            raise Exception(f"获取备份列表失败: {result['message']}")

        backups = result["data"]["backups"]
        print(f"\n备份列表 (共 {len(backups)} 个):")
        print("-" * 80)
        print(f"{'备份名称':<30} {'Collection':<20} {'大小':<15} {'创建时间':<20}")
        print("-" * 80)

        for backup in backups:
            size_mb = backup["backup_size"] / 1024 / 1024
            created_at = time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(backup["created_at"])
            )
            collections = ", ".join(backup["collections"])
            print(f"{backup['backup_id']:<30} {collections:<20} {size_mb:>10.2f} MB {created_at:<20}")

        return backups

    def restore_backup(
        self,
        backup_name: str,
        collections: Optional[List[str]] = None,
        target_collection: Optional[str] = None
    ) -> Dict:
        """恢复备份"""
        url = f"{self.base_url}/backup/restore"
        payload = {
            "backup_name": backup_name
        }

        if collections:
            payload["collections"] = collections

        if target_collection:
            payload["target_collection"] = target_collection

        print(f"\n恢复备份: {backup_name}")
        if target_collection:
            print(f"目标 Collection: {target_collection}")

        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        if result["code"] != 0:
            raise Exception(f"恢复失败: {result['message']}")

        print(f"✅ 备份恢复成功")
        print(f"  数据量: {result['data']['total_entities']}")
        print(f"  耗时: {result['data']['duration']} 秒")

        return result["data"]

    def delete_backup(self, backup_name: str):
        """删除备份"""
        url = f"{self.base_url}/backup/delete"
        payload = {"backup_name": backup_name}

        print(f"\n删除备份: {backup_name}")

        response = requests.delete(url, json=payload)
        response.raise_for_status()

        result = response.json()
        if result["code"] != 0:
            raise Exception(f"删除失败: {result['message']}")

        print(f"✅ 备份已删除")

    def verify_backup(self, backup_name: str) -> bool:
        """验证备份"""
        url = f"{self.base_url}/backup/verify"
        payload = {"backup_name": backup_name}

        print(f"\n验证备份: {backup_name}")

        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        if result["code"] != 0:
            raise Exception(f"验证失败: {result['message']}")

        is_valid = result["data"]["is_valid"]
        if is_valid:
            print(f"✅ 备份验证通过")
        else:
            print(f"❌ 备份验证失败: {result['data']['error']}")

        return is_valid


def main():
    """主函数"""
    # 创建客户端
    client = MilvusBackupClient(host="localhost", port=8080)

    # 1. 创建备份
    backup_name = f"backup_{time.strftime('%Y%m%d_%H%M%S')}"
    backup_info = client.create_backup(
        backup_name=backup_name,
        collections=["my_collection"],
        compression=True
    )

    # 2. 列出所有备份
    backups = client.list_backups()

    # 3. 验证备份
    is_valid = client.verify_backup(backup_name)

    if is_valid:
        # 4. 恢复备份到新 Collection
        client.restore_backup(
            backup_name=backup_name,
            target_collection="my_collection_restored"
        )

    # 5. 清理旧备份（可选）
    # client.delete_backup("old_backup_name")


if __name__ == "__main__":
    main()
```

**运行示例：**

```bash
python backup_basic.py
```

**输出：**

```
创建备份: backup_20260210_100000
Collection: ['my_collection']
✅ 备份创建成功
  备份ID: backup_20260210_100000
  数据量: 1000000
  大小: 2500.00 MB
  耗时: 330 秒

备份列表 (共 3 个):
--------------------------------------------------------------------------------
备份名称                        Collection           大小            创建时间
--------------------------------------------------------------------------------
backup_20260210_100000          my_collection        2500.00 MB      2026-02-10 10:00:00
backup_20260209_100000          my_collection        2400.00 MB      2026-02-09 10:00:00
backup_20260208_100000          my_collection        2300.00 MB      2026-02-08 10:00:00

验证备份: backup_20260210_100000
✅ 备份验证通过

恢复备份: backup_20260210_100000
目标 Collection: my_collection_restored
✅ 备份恢复成功
  数据量: 1000000
  耗时: 500 秒
```

---

### 示例2：增量备份

```python
#!/usr/bin/env python3
"""
增量备份示例
"""

import requests
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

class IncrementalBackup:
    """增量备份管理器"""

    def __init__(self, host: str = "localhost", port: int = 8080):
        """初始化"""
        self.base_url = f"http://{host}:{port}/api/v1"
        self.last_full_backup = None
        self.incremental_backups = []

    def create_full_backup(self, collection_name: str) -> str:
        """创建全量备份"""
        backup_name = f"full_{collection_name}_{time.strftime('%Y%m%d_%H%M%S')}"

        print(f"创建全量备份: {backup_name}")

        url = f"{self.base_url}/backup/create"
        payload = {
            "backup_name": backup_name,
            "collections": [collection_name],
            "compression": True
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        if result["code"] != 0:
            raise Exception(f"全量备份失败: {result['message']}")

        self.last_full_backup = backup_name
        print(f"✅ 全量备份完成: {backup_name}")

        return backup_name

    def create_incremental_backup(
        self,
        collection_name: str,
        base_backup: Optional[str] = None
    ) -> str:
        """创建增量备份"""
        if not base_backup:
            base_backup = self.last_full_backup

        if not base_backup:
            raise Exception("没有基准备份，请先创建全量备份")

        backup_name = f"incr_{collection_name}_{time.strftime('%Y%m%d_%H%M%S')}"

        print(f"创建增量备份: {backup_name}")
        print(f"基于: {base_backup}")

        url = f"{self.base_url}/backup/create"
        payload = {
            "backup_name": backup_name,
            "collections": [collection_name],
            "base_backup": base_backup,
            "incremental": True,
            "compression": True
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        if result["code"] != 0:
            raise Exception(f"增量备份失败: {result['message']}")

        self.incremental_backups.append(backup_name)
        print(f"✅ 增量备份完成: {backup_name}")
        print(f"  增量数据量: {result['data']['total_entities']}")
        print(f"  大小: {result['data']['backup_size'] / 1024 / 1024:.2f} MB")

        return backup_name

    def restore_with_incremental(
        self,
        collection_name: str,
        target_collection: str
    ):
        """恢复全量 + 增量备份"""
        print(f"\n恢复备份链:")
        print(f"  全量备份: {self.last_full_backup}")
        print(f"  增量备份: {len(self.incremental_backups)} 个")

        # 1. 恢复全量备份
        print(f"\n[1/2] 恢复全量备份...")
        url = f"{self.base_url}/backup/restore"
        payload = {
            "backup_name": self.last_full_backup,
            "target_collection": target_collection
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        # 2. 依次恢复增量备份
        print(f"\n[2/2] 恢复增量备份...")
        for i, incr_backup in enumerate(self.incremental_backups, 1):
            print(f"  [{i}/{len(self.incremental_backups)}] {incr_backup}")

            payload = {
                "backup_name": incr_backup,
                "target_collection": target_collection,
                "merge": True  # 合并到已有数据
            }

            response = requests.post(url, json=payload)
            response.raise_for_status()

        print(f"\n✅ 备份链恢复完成")


def main():
    """主函数"""
    manager = IncrementalBackup()
    collection_name = "my_collection"

    # 1. 创建全量备份（每周一次）
    full_backup = manager.create_full_backup(collection_name)

    # 2. 模拟每天的增量备份
    for day in range(1, 8):
        print(f"\n--- 第 {day} 天 ---")

        # 模拟数据变化
        time.sleep(1)

        # 创建增量备份
        incr_backup = manager.create_incremental_backup(collection_name)

    # 3. 恢复完整数据
    manager.restore_with_incremental(
        collection_name=collection_name,
        target_collection="my_collection_restored"
    )


if __name__ == "__main__":
    main()
```

---

### 示例3：自动化备份脚本

```python
#!/usr/bin/env python3
"""
自动化备份脚本
"""

import requests
import time
import schedule
import logging
from datetime import datetime, timedelta
from typing import List, Dict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/milvus-backup.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class AutoBackupManager:
    """自动化备份管理器"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        collections: List[str] = None,
        retention_days: int = 7
    ):
        """初始化"""
        self.base_url = f"http://{host}:{port}/api/v1"
        self.collections = collections or []
        self.retention_days = retention_days

    def backup_all_collections(self):
        """备份所有 Collection"""
        logger.info("开始自动备份...")

        backup_name = f"auto_{time.strftime('%Y%m%d_%H%M%S')}"

        try:
            # 创建备份
            url = f"{self.base_url}/backup/create"
            payload = {
                "backup_name": backup_name,
                "collections": self.collections,
                "compression": True
            }

            response = requests.post(url, json=payload, timeout=3600)
            response.raise_for_status()

            result = response.json()
            if result["code"] != 0:
                raise Exception(f"备份失败: {result['message']}")

            logger.info(f"✅ 备份成功: {backup_name}")
            logger.info(f"  数据量: {result['data']['total_entities']}")
            logger.info(f"  大小: {result['data']['backup_size'] / 1024 / 1024:.2f} MB")

            # 验证备份
            self.verify_backup(backup_name)

            # 清理旧备份
            self.cleanup_old_backups()

        except Exception as e:
            logger.error(f"❌ 备份失败: {e}")
            self.send_alert(f"备份失败: {e}")

    def verify_backup(self, backup_name: str):
        """验证备份"""
        logger.info(f"验证备份: {backup_name}")

        try:
            url = f"{self.base_url}/backup/verify"
            payload = {"backup_name": backup_name}

            response = requests.post(url, json=payload, timeout=600)
            response.raise_for_status()

            result = response.json()
            if result["code"] != 0 or not result["data"]["is_valid"]:
                raise Exception("备份验证失败")

            logger.info(f"✅ 备份验证通过")

        except Exception as e:
            logger.error(f"❌ 备份验证失败: {e}")
            self.send_alert(f"备份验证失败: {e}")

    def cleanup_old_backups(self):
        """清理旧备份"""
        logger.info(f"清理超过 {self.retention_days} 天的备份...")

        try:
            # 获取备份列表
            url = f"{self.base_url}/backup/list"
            response = requests.get(url)
            response.raise_for_status()

            result = response.json()
            if result["code"] != 0:
                raise Exception("获取备份列表失败")

            backups = result["data"]["backups"]
            cutoff_time = time.time() - (self.retention_days * 86400)

            # 删除旧备份
            deleted_count = 0
            for backup in backups:
                if backup["created_at"] < cutoff_time:
                    self.delete_backup(backup["backup_id"])
                    deleted_count += 1

            logger.info(f"✅ 清理完成，删除 {deleted_count} 个旧备份")

        except Exception as e:
            logger.error(f"❌ 清理失败: {e}")

    def delete_backup(self, backup_name: str):
        """删除备份"""
        url = f"{self.base_url}/backup/delete"
        payload = {"backup_name": backup_name}

        response = requests.delete(url, json=payload)
        response.raise_for_status()

        logger.info(f"删除备份: {backup_name}")

    def send_alert(self, message: str):
        """发送告警"""
        # 这里可以集成邮件、Slack、钉钉等告警方式
        logger.error(f"🚨 告警: {message}")

        # 示例：发送邮件
        # send_email(
        #     to="admin@example.com",
        #     subject="Milvus 备份告警",
        #     body=message
        # )

    def run_scheduler(self):
        """运行调度器"""
        logger.info("启动自动备份调度器...")

        # 每天凌晨 2 点备份
        schedule.every().day.at("02:00").do(self.backup_all_collections)

        # 每小时检查一次
        schedule.every().hour.do(self.check_backup_health)

        while True:
            schedule.run_pending()
            time.sleep(60)

    def check_backup_health(self):
        """检查备份健康状态"""
        try:
            # 获取最近的备份
            url = f"{self.base_url}/backup/list"
            response = requests.get(url)
            response.raise_for_status()

            result = response.json()
            if result["code"] != 0:
                raise Exception("获取备份列表失败")

            backups = result["data"]["backups"]

            if not backups:
                self.send_alert("没有任何备份！")
                return

            # 检查最近备份时间
            latest_backup = max(backups, key=lambda x: x["created_at"])
            time_since_last = time.time() - latest_backup["created_at"]

            if time_since_last > 86400:  # 24 小时
                self.send_alert(f"最近备份时间超过 24 小时: {time_since_last / 3600:.1f} 小时")

        except Exception as e:
            logger.error(f"健康检查失败: {e}")


def main():
    """主函数"""
    # 创建备份管理器
    manager = AutoBackupManager(
        host="localhost",
        port=8080,
        collections=["collection1", "collection2", "collection3"],
        retention_days=7
    )

    # 运行调度器
    manager.run_scheduler()


if __name__ == "__main__":
    main()
```

**部署为系统服务：**

创建 `/etc/systemd/system/milvus-backup.service`：

```ini
[Unit]
Description=Milvus Auto Backup Service
After=network.target

[Service]
Type=simple
User=milvus
WorkingDirectory=/opt/milvus-backup
ExecStart=/usr/bin/python3 /opt/milvus-backup/auto_backup.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable milvus-backup
sudo systemctl start milvus-backup
sudo systemctl status milvus-backup
```

---

## 总结

### 核心要点

1. **Milvus Backup 工具简单易用**：通过 HTTP API 或命令行即可使用
2. **支持增量备份**：减少备份时间和存储空间
3. **自动化备份**：使用 schedule 库实现定时备份
4. **备份验证**：每次备份后都要验证
5. **清理策略**：定期清理旧备份

### 适用场景

- ✅ 生产环境定期备份
- ✅ 灾难恢复演练
- ✅ 数据迁移准备
- ✅ 知识库版本管理

### 下一步

- 学习 [Collection 导出导入](./07_实战代码_02_Collection导出导入.md)
- 学习 [跨集群数据迁移](./07_实战代码_03_跨集群数据迁移.md)
