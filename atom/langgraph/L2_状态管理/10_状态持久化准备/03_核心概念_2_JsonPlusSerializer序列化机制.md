# 核心概念 2：JsonPlusSerializer 序列化机制

> LangGraph 的默认序列化器名叫 "JsonPlus"，但实际主力是 msgpack。理解它的类型标签系统和扩展类型机制，你就能预判哪些状态能存、怎么存、存多大。

---

## 引用来源

**源码分析**:
- `libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py` — JsonPlusSerializer 完整实现
- `libs/checkpoint/langgraph/checkpoint/serde/base.py` — SerializerProtocol 协议

[来源: sourcecode/langgraph/libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py]

**官方文档**:
- Context7 LangGraph 文档 (2026-02-27)

---

## 一句话定义

**JsonPlusSerializer 是 LangGraph 的默认序列化器，通过类型标签 + msgpack 扩展类型系统，将 Python 对象转换为紧凑的二进制格式，支持远超标准 JSON 的类型范围。**

---

## 为什么叫 "JsonPlus" 但用的是 msgpack？

### 历史原因

<!-- PLACEHOLDER_SECTION_1 -->
