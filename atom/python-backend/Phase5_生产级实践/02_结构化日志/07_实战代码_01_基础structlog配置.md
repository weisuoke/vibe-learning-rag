# 实战代码01：基础structlog配置

## 学习目标

从零开始配置structlog，理解每个配置项的作用，掌握基础的结构化日志使用。

---

## 第一步：安装structlog

```bash
# 使用uv安装
uv add structlog

# 或使用pip
pip install structlog
```

**验证安装：**
```bash
python -c "import structlog; print(structlog.__version__)"
```

---

## 第二步：最简单的配置

### 示例1：零配置使用

```python
# examples/logging/01_basic_structlog.py
import structlog

# 直接使用，无需配置
logger = structlog.get_logger()

# 记录日志
logger.info("hello", name="world")
logger.warning("temperature_high", value=95)
logger.error("connection_failed", host="localhost", port=5432)
```

**输出（默认格式）：**
```
2024-01-15 10:30:45 [info     ] hello                          name=world
2024-01-15 10:30:46 [warning  ] temperature_high               value=95
2024-01-15 10:30:47 [error    ] connection_failed              host=localhost port=5432
```

**特点：**
- 人类可读
- 彩色输出（终端支持时）
- 适合开发环境

---

## 第三步：JSON格式配置

### 示例2：输出JSON格式

```python
# examples/logging/02_json_output.py
import structlog

# 配置JSON输出
structlog.configure(
    processors=[
        structlog.processors.JSONRenderer()  # JSON格式
    ]
)

logger = structlog.get_logger()

logger.info("user_login", user_id="user_123", ip="192.168.1.1")
```

**输出：**
```json
{"event": "user_login", "user_id": "user_123", "ip": "192.168.1.1"}
```

**用途：** 生产环境，便于日志平台解析

---

## 第四步：添加时间戳

### 示例3：添加ISO格式时间戳

```python
# examples/logging/03_with_timestamp.py
import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),  # ISO 8601格式
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()

logger.info("api_call", endpoint="/users", method="GET")
```

**输出：**
```json
{
  "event": "api_call",
  "endpoint": "/users",
  "method": "GET",
  "timestamp": "2024-01-15T10:30:45.123456Z"
}
```

**时间戳格式选项：**
```python
# ISO 8601格式（推荐）
structlog.processors.TimeStamper(fmt="iso")
# 输出：2024-01-15T10:30:45.123456Z

# Unix时间戳
structlog.processors.TimeStamper(fmt="unix")
# 输出：1705318245.123456

# 自定义格式
structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S")
# 输出：2024-01-15 10:30:45
```

---

## 第五步：添加日志级别

### 示例4：添加日志级别字段

```python
# examples/logging/04_with_log_level.py
import structlog

structlog.configure(
    processors=[
        structlog.processors.add_log_level,  # 添加日志级别
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()

logger.debug("debug_info", variable="x")
logger.info("user_login", user_id="user_123")
logger.warning("slow_query", duration_ms=3000)
logger.error("api_failed", error="timeout")
```

**输出：**
```json
{"level": "debug", "event": "debug_info", "variable": "x", "timestamp": "..."}
{"level": "info", "event": "user_login", "user_id": "user_123", "timestamp": "..."}
{"level": "warning", "event": "slow_query", "duration_ms": 3000, "timestamp": "..."}
{"level": "error", "event": "api_failed", "error": "timeout", "timestamp": "..."}
```

---

## 第六步：完整的基础配置

### 示例5：生产环境配置

```python
# examples/logging/05_production_config.py
import structlog
import logging

def setup_logging(log_level: str = "INFO", json_output: bool = True):
    """
    配置structlog用于生产环境

    Args:
        log_level: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
        json_output: 是否输出JSON格式（True=生产环境，False=开发环境）
    """

    # 1. 配置标准库logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
    )

    # 2. 选择渲染器
    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()  # 彩色、易读

    # 3. 配置structlog
    structlog.configure(
        processors=[
            # 添加日志级别
            structlog.processors.add_log_level,

            # 添加时间戳
            structlog.processors.TimeStamper(fmt="iso"),

            # 添加调用者信息（可选，开发环境有用）
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ) if not json_output else structlog.processors.CallsiteParameterAdder(
                parameters=[]
            ),

            # 渲染器
            renderer
        ],

        # 日志级别过滤
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),

        # 上下文类
        context_class=dict,

        # Logger工厂
        logger_factory=structlog.PrintLoggerFactory(),

        # 缓存logger
        cache_logger_on_first_use=True,
    )


# 使用示例
if __name__ == "__main__":
    # 开发环境
    setup_logging(log_level="DEBUG", json_output=False)

    logger = structlog.get_logger()

    logger.debug("debug_message", x=1, y=2)
    logger.info("user_login", user_id="user_123")
    logger.warning("slow_query", duration_ms=3000)
    logger.error("api_failed", error="timeout")
```

**开发环境输出：**
```
2024-01-15 10:30:45 [debug    ] debug_message                  x=1 y=2 [05_production_config.py:52]
2024-01-15 10:30:46 [info     ] user_login                     user_id=user_123 [05_production_config.py:53]
2024-01-15 10:30:47 [warning  ] slow_query                     duration_ms=3000 [05_production_config.py:54]
2024-01-15 10:30:48 [error    ] api_failed                     error=timeout [05_production_config.py:55]
```

**生产环境输出：**
```json
{"level": "info", "event": "user_login", "user_id": "user_123", "timestamp": "2024-01-15T10:30:46Z"}
{"level": "warning", "event": "slow_query", "duration_ms": 3000, "timestamp": "2024-01-15T10:30:47Z"}
{"level": "error", "event": "api_failed", "error": "timeout", "timestamp": "2024-01-15T10:30:48Z"}
```

---

## 第七步：环境自适应配置

### 示例6：根据环境自动选择配置

```python
# examples/logging/06_env_adaptive.py
import os
import sys
import structlog
import logging

def setup_logging():
    """根据环境自动配置日志"""

    # 判断环境
    env = os.getenv("ENV", "development")
    is_dev = env == "development" or sys.stdout.isatty()

    # 开发环境配置
    if is_dev:
        log_level = "DEBUG"
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    # 生产环境配置
    else:
        log_level = "INFO"
        renderer = structlog.processors.JSONRenderer()

    # 配置标准库logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level),
    )

    # 配置structlog
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger()


# 使用
if __name__ == "__main__":
    logger = setup_logging()

    logger.debug("debug_info")  # 开发环境显示，生产环境不显示
    logger.info("user_login", user_id="user_123")
    logger.error("api_failed", error="timeout")
```

**测试不同环境：**
```bash
# 开发环境
python 06_env_adaptive.py

# 生产环境
ENV=production python 06_env_adaptive.py
```

---

## 第八步：日志级别过滤

### 示例7：只记录特定级别的日志

```python
# examples/logging/07_level_filtering.py
import structlog
import logging

# 配置：只记录INFO及以上级别
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO)
)

logger = structlog.get_logger()

# 这条不会输出（DEBUG < INFO）
logger.debug("debug_message", x=1)

# 这些会输出
logger.info("info_message", y=2)
logger.warning("warning_message", z=3)
logger.error("error_message", error="test")
```

**输出：**
```json
{"level": "info", "event": "info_message", "y": 2, "timestamp": "..."}
{"level": "warning", "event": "warning_message", "z": 3, "timestamp": "..."}
{"level": "error", "event": "error_message", "error": "test", "timestamp": "..."}
```

---

## 第九步：输出到文件

### 示例8：日志输出到文件

```python
# examples/logging/08_file_output.py
import structlog
import logging
from logging.handlers import RotatingFileHandler

# 配置文件handler
file_handler = RotatingFileHandler(
    "app.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)

# 配置标准库logging
logging.basicConfig(
    handlers=[file_handler],
    level=logging.INFO
)

# 配置structlog使用标准库logging
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory()
)

logger = structlog.get_logger()

# 日志会写入app.log文件
logger.info("user_login", user_id="user_123")
logger.error("api_failed", error="timeout")

print("日志已写入 app.log")
```

**查看日志：**
```bash
cat app.log
```

---

## 第十步：同时输出到控制台和文件

### 示例9：多个输出目标

```python
# examples/logging/09_multiple_outputs.py
import structlog
import logging
from logging.handlers import RotatingFileHandler

# 控制台handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 文件handler
file_handler = RotatingFileHandler(
    "app.log",
    maxBytes=10*1024*1024,
    backupCount=5
)
file_handler.setLevel(logging.INFO)

# 错误文件handler（只记录ERROR及以上）
error_handler = RotatingFileHandler(
    "error.log",
    maxBytes=10*1024*1024,
    backupCount=5
)
error_handler.setLevel(logging.ERROR)

# 配置标准库logging
logging.basicConfig(
    handlers=[console_handler, file_handler, error_handler],
    level=logging.INFO
)

# 配置structlog
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory()
)

logger = structlog.get_logger()

# 这条会输出到：控制台 + app.log
logger.info("user_login", user_id="user_123")

# 这条会输出到：控制台 + app.log + error.log
logger.error("api_failed", error="timeout")

print("\n日志已输出到：")
print("- 控制台")
print("- app.log")
print("- error.log（仅错误）")
```

---

## 完整示例：可复用的日志配置模块

### 示例10：生产级配置模块

```python
# config/logging.py
"""
结构化日志配置模块

使用方法：
    from config.logging import setup_logging

    # 开发环境
    setup_logging()

    # 生产环境
    setup_logging(env="production")
"""

import os
import sys
import structlog
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    env: str = None,
    log_level: str = None,
    log_dir: str = "logs",
    max_bytes: int = 10*1024*1024,  # 10MB
    backup_count: int = 5
):
    """
    配置结构化日志

    Args:
        env: 环境（development/production），默认从ENV环境变量读取
        log_level: 日志级别，默认根据环境自动选择
        log_dir: 日志文件目录
        max_bytes: 单个日志文件最大大小
        backup_count: 保留的日志文件数量
    """

    # 1. 确定环境
    if env is None:
        env = os.getenv("ENV", "development")

    is_dev = env == "development" or sys.stdout.isatty()

    # 2. 确定日志级别
    if log_level is None:
        log_level = "DEBUG" if is_dev else "INFO"

    # 3. 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # 4. 配置handlers
    handlers = []

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    handlers.append(console_handler)

    # 生产环境：添加文件handler
    if not is_dev:
        # 所有日志
        file_handler = RotatingFileHandler(
            log_path / "app.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.INFO)
        handlers.append(file_handler)

        # 错误日志
        error_handler = RotatingFileHandler(
            log_path / "error.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        handlers.append(error_handler)

    # 5. 配置标准库logging
    logging.basicConfig(
        handlers=handlers,
        level=getattr(logging, log_level),
        format="%(message)s"
    )

    # 6. 选择渲染器
    if is_dev:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    # 7. 配置structlog
    structlog.configure(
        processors=[
            # 添加日志级别
            structlog.processors.add_log_level,

            # 添加时间戳
            structlog.processors.TimeStamper(fmt="iso"),

            # 开发环境：添加调用者信息
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ) if is_dev else structlog.processors.CallsiteParameterAdder(
                parameters=[]
            ),

            # 渲染器
            renderer
        ],

        # 日志级别过滤
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level)
        ),

        # 使用标准库logging
        logger_factory=structlog.stdlib.LoggerFactory(),

        # 缓存logger
        cache_logger_on_first_use=True,
    )

    # 8. 记录配置信息
    logger = structlog.get_logger()
    logger.info("logging_configured",
        env=env,
        log_level=log_level,
        log_dir=str(log_path),
        handlers_count=len(handlers)
    )


# 使用示例
if __name__ == "__main__":
    # 配置日志
    setup_logging()

    # 获取logger
    logger = structlog.get_logger()

    # 使用
    logger.debug("debug_info", x=1)
    logger.info("user_login", user_id="user_123")
    logger.warning("slow_query", duration_ms=3000)
    logger.error("api_failed", error="timeout")
```

**使用方法：**

```python
# main.py
from config.logging import setup_logging
import structlog

# 初始化日志
setup_logging()

# 获取logger
logger = structlog.get_logger()

# 使用
logger.info("app_started")
```

---

## 总结

### 核心配置项

1. **Processors（处理器）**
   - `add_log_level`：添加日志级别
   - `TimeStamper`：添加时间戳
   - `JSONRenderer`：JSON格式输出
   - `ConsoleRenderer`：人类可读输出

2. **Logger Factory**
   - `PrintLoggerFactory`：直接打印
   - `LoggerFactory`：使用标准库logging

3. **日志级别过滤**
   - `make_filtering_bound_logger`：过滤日志级别

### 最佳实践

1. 开发环境用人类可读格式
2. 生产环境用JSON格式
3. 根据环境自动选择配置
4. 日志文件使用轮转
5. 错误日志单独存储

### 下一步

- 【实战代码02】：FastAPI日志集成
- 【实战代码03】：请求ID追踪
