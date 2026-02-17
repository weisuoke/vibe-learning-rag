# L1_核心抽象 - 知识点列表

**层级说明**: LangChain 的核心抽象层，理解 Runnable 协议和 LCEL 基础

**知识点数量**: 10个

**学习目标**: 掌握 LangChain 的核心设计理念和基础抽象

---

## 核心知识点（20%核心）

1. **Runnable接口与LCEL基础** ⭐⭐⭐
   - Runnable 协议定义
   - invoke、batch、stream 方法
   - LCEL 设计哲学

2. **ChatModel与PromptTemplate** ⭐⭐⭐
   - ChatModel 抽象
   - PromptTemplate 模板系统
   - 消息格式与角色

3. **OutputParser与结构化输出** ⭐⭐⭐
   - OutputParser 基类
   - 结构化输出解析
   - Pydantic 集成

4. **链式组合（管道操作符）** ⭐⭐⭐
   - | 操作符原理
   - 链式调用机制
   - 数据流转

---

## 扩展知识点

5. **RunnablePassthrough与RunnableLambda**
   - 透传机制
   - 自定义函数包装
   - 数据转换

6. **RunnableParallel并行执行**
   - 并行执行原理
   - 结果合并
   - 性能优化

7. **RunnableBranch条件路由**
   - 条件分支
   - 动态路由
   - 决策逻辑

8. **RunnableSequence序列执行**
   - 序列组合
   - 错误传播
   - 中间结果

9. **Runnable配置与回调**
   - 配置传递
   - 回调系统
   - 可观测性钩子

10. **类型系统与泛型**
    - 输入输出类型
    - 泛型约束
    - 类型推断

---

**前置知识**: Python 基础、类型注解、异步编程
**后续层级**: L2_LCEL表达式
**预计学习时长**: 3-4天
