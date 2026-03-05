# PromptTemplate高级用法 - 生成计划

## 数据来源记录

### 源码分析
- ✓ reference/source_prompttemplate_01.md - PromptTemplate 核心实现分析

### Context7 官方文档
- ✓ reference/context7_langchain_01.md - 基础用法
- ✓ reference/context7_langchain_02.md - Partial Variables
- ✓ reference/context7_langchain_03.md - Template Composition

### 网络搜索
- ✓ reference/search_partial_01.md - Partial Variables 社区资料（9 个链接）
- ✓ reference/search_composition_01.md - Template Composition 社区资料（8 个链接）

### 待抓取链接（将由第三方工具自动保存到 reference/）

**高优先级**（2025-2026 年资料）：
- [ ] https://langchain-opentutorial.gitbook.io/langchain-opentutorial/02-prompt/01-prompttemplate
- [ ] https://latenode.com/blog/ai-frameworks-technical-infrastructure/langchain-setup-tools-agents-memory/langchain-prompt-templates-complete-guide-with-examples
- [ ] https://www.newline.co/@zaoyang/dynamic-prompts-with-langchain-templates--71d0c244
- [ ] https://medium.com/@shoaibahamedshafi/the-only-langchain-prompt-templates-guide-youll-ever-need-2219293708eb
- [ ] https://mirascope.com/blog/langchain-prompt-template
- [ ] https://www.youtube.com/watch?v=i8XCnz0PM8g

**中优先级**（社区讨论和教程）：
- [ ] https://bhavikjikadara.medium.com/mastering-prompttemplates-in-langchain-74f679c467ec
- [ ] https://github.com/langchain-ai/langchain/issues/17560
- [ ] https://langchain-tutorials.com/lessons/langchain-essentials/lesson-6
- [ ] https://www.voiceflow.com/blog/langchain-prompt-template
- [ ] https://www.ibm.com/think/tutorials/prompt-chaining-langchain

**低优先级**（旧版文档）：
- [ ] https://lagnchain.readthedocs.io/en/latest/modules/prompts/prompt_templates/examples/partial.html

## 文件清单

### 基础维度文件
- [ ] 00_概览.md
- [ ] 01_30字核心.md
- [ ] 02_第一性原理.md

### 核心概念文件（基于源码 + Context7 + 网络调研）

#### 1. Partial Variables（部分变量）
- [ ] 03_核心概念_1_Partial_Variables基础.md - 预填充模板变量的核心机制 [来源: 源码 + Context7 + 网络]
  - `partial_variables` 参数的使用
  - 字符串固定值 vs 函数动态值
  - 减少重复传递的优势

- [ ] 03_核心概念_2_Partial_Variables高级用法.md - 动态值与函数应用 [来源: 网络]
  - 函数动态值的实现（如当前日期）
  - 延迟计算与懒加载
  - 与 ChatPromptTemplate 的兼容性问题

#### 2. Template Composition（模板组合）
- [ ] 03_核心概念_3_Template_Composition基础.md - 使用 + 操作符组合模板 [来源: 源码 + 网络]
  - `__add__` 方法的实现原理
  - 自动合并 input_variables 和 partial_variables
  - PromptTemplate + str 的组合

- [ ] 03_核心概念_4_Template_Composition高级模式.md - 复杂组合场景 [来源: 网络]
  - 组合时的变量冲突检查
  - template_format 一致性要求
  - 与 PipelinePromptTemplate 的对比

#### 3. Template Formats（模板格式）
- [ ] 03_核心概念_5_Template_Formats对比.md - 三种模板格式详解 [来源: 源码 + Context7]
  - f-string（默认）：简单快速
  - mustache：支持嵌套和 section
  - jinja2：功能强大但有安全风险

- [ ] 03_核心概念_6_Mustache高级特性.md - Mustache 模板深入 [来源: 源码]
  - 嵌套变量（`{{obj.bar}}`）
  - Section 变量（`{{#foo}} {{bar}} {{/foo}}`）
  - No escape（`{{{foo}}}`）
  - Mustache Schema 生成

#### 4. Template Validation（模板验证）
- [ ] 03_核心概念_7_Template_Validation.md - 模板验证机制 [来源: 源码]
  - `validate_template` 参数
  - 变量完整性检查
  - Mustache 模板的验证限制

#### 5. 从文件加载
- [ ] 03_核心概念_8_从文件加载模板.md - 文件加载与编码处理 [来源: 源码]
  - `from_file` 方法
  - 多种编码支持（UTF-8、CP-1252 等）
  - 自动提取变量

#### 6. 与其他组件的集成
- [ ] 03_核心概念_9_与LCEL链式组合.md - LCEL 表达式集成 [来源: Context7 + 网络]
  - 使用 `|` 操作符进行链式组合
  - 与 ChatModel 的集成
  - 与 OutputParser 的集成

- [ ] 03_核心概念_10_与ChatPromptTemplate关系.md - 对话模板集成 [来源: Context7 + 网络]
  - ChatPromptTemplate 的组合方式
  - 消息模板的构建
  - partial_variables 的兼容性问题

### 基础维度文件（续）
- [ ] 04_最小可用.md
- [ ] 05_双重类比.md
- [ ] 06_反直觉点.md

### 实战代码文件（基于源码 + Context7 + 网络调研）

#### Partial Variables 实战
- [ ] 07_实战代码_场景1_字符串固定值Partial.md - 预填充常量变量 [来源: 网络]
  - 系统设置、日期、常量的预填充
  - 减少重复传递的实践

- [ ] 07_实战代码_场景2_函数动态值Partial.md - 动态计算变量 [来源: 网络]
  - 当前日期时间的动态获取
  - 延迟计算的实现
  - 函数参数的传递

#### Template Composition 实战
- [ ] 07_实战代码_场景3_基础模板组合.md - 使用 + 操作符组合 [来源: 源码 + 网络]
  - PromptTemplate + PromptTemplate
  - PromptTemplate + str
  - 变量自动合并

- [ ] 07_实战代码_场景4_复杂模板组合.md - 多层次模板组合 [来源: 网络]
  - 多个模板的链式组合
  - partial_variables 的合并
  - 组合时的错误处理

#### Template Formats 实战
- [ ] 07_实战代码_场景5_Mustache模板实战.md - Mustache 高级用法 [来源: 源码]
  - 嵌套变量的使用
  - Section 变量的实现
  - Schema 生成与验证

- [ ] 07_实战代码_场景6_Jinja2模板实战.md - Jinja2 模板应用 [来源: 源码]
  - Jinja2 模板的创建
  - 安全性考虑
  - SandboxedEnvironment 的使用

#### 综合实战
- [ ] 07_实战代码_场景7_RAG系统模板管理.md - RAG 场景应用 [来源: 网络]
  - 多语言模板管理
  - 模板复用与组合
  - 动态变量注入

- [ ] 07_实战代码_场景8_Agent系统模板设计.md - Agent 场景应用 [来源: 网络]
  - Agent 提示词模板
  - 工具调用模板
  - 多步推理模板

### 基础维度文件（续）
- [ ] 08_面试必问.md
- [ ] 09_化骨绵掌.md
- [ ] 10_一句话总结.md

## 核心概念总结

基于源码分析、Context7 文档和网络搜索，识别出 6 个核心概念：

1. **Partial Variables（部分变量）**：
   - 预填充模板变量
   - 支持字符串固定值和函数动态值
   - 减少重复传递相同的变量
   - 适用场景：日期时间、系统设置、共享常量

2. **Template Composition（模板组合）**：
   - 使用 `+` 操作符组合模板
   - 自动合并 input_variables 和 partial_variables
   - 支持 PromptTemplate + str 的组合
   - 要求两个模板的 template_format 必须一致

3. **Template Formats（模板格式）**：
   - f-string（默认）：简单快速
   - mustache：支持嵌套和 section
   - jinja2：功能强大但有安全风险

4. **Template Validation（模板验证）**：
   - 可选的模板验证
   - 检查变量完整性
   - Mustache 模板不支持验证

5. **从文件加载**：
   - 支持多种编码
   - 自动提取变量
   - `from_file` 方法

6. **与其他组件的集成**：
   - 与 ChatPromptTemplate 的关系
   - 与 LCEL 的链式组合
   - 与 SequentialChain 的组合

## 实战场景总结

基于网络搜索和社区资料，识别出 8 个实战场景：

1. **字符串固定值 Partial**：预填充常量变量
2. **函数动态值 Partial**：动态计算变量
3. **基础模板组合**：使用 + 操作符组合
4. **复杂模板组合**：多层次模板组合
5. **Mustache 模板实战**：Mustache 高级用法
6. **Jinja2 模板实战**：Jinja2 模板应用
7. **RAG 系统模板管理**：RAG 场景应用
8. **Agent 系统模板设计**：Agent 场景应用

## 生成进度

- [x] 阶段一：Plan 生成
  - [x] 1.1 Brainstorm 分析
  - [x] 1.2 多源数据收集（源码 + Context7 + 网络）
  - [x] 1.3 用户确认拆解方案
  - [x] 1.4 Plan 最终确定
- [ ] 阶段二：补充调研（针对需要更多资料的部分）
- [ ] 阶段三：文档生成（读取 reference/ 中的所有资料）

## 文件长度控制

- **目标长度**：每个文件 300-500 行
- **超长处理**：单文件超过 500 行时，自动拆分成更小的文件
- **代码示例**：每个示例 100-200 行，必须完整可运行

## 质量标准

- **代码语言**：Python 3.13+
- **代码完整性**：所有代码必须完整可运行
- **技术深度**：每个技术包含
  - 原理讲解
  - 源码分析
  - 实际应用场景
- **引用规范**：
  - 源码引用：`[来源: sourcecode/langchain/libs/core/langchain_core/prompts/prompt.py:142-184]`
  - Context7 引用：`[来源: reference/context7_langchain_01.md | LangChain 官方文档]`
  - 网络搜索引用：`[来源: reference/search_partial_01.md]`
  - 抓取内容引用：`[来源: reference/fetch_xxx.md | 原始URL]`

## 下一步操作

### 阶段二：补充调研（可选）

如果现有资料已足够，可以跳过阶段二，直接进入阶段三。

如果需要更多资料，请使用第三方抓取工具处理待抓取链接。

### 阶段三：文档生成

使用 subagent 批量生成文档，每个文件生成后更新本 PLAN.md 中的文件清单（标记 ✓）。

**生成顺序**：
1. 基础维度文件（第一部分）：00_概览.md、01_30字核心.md、02_第一性原理.md
2. 核心概念文件：03_核心概念_1_xxx.md ~ 03_核心概念_10_xxx.md
3. 基础维度文件（第二部分）：04_最小可用.md、05_双重类比.md、06_反直觉点.md
4. 实战代码文件：07_实战代码_场景1_xxx.md ~ 07_实战代码_场景8_xxx.md
5. 基础维度文件（第三部分）：08_面试必问.md、09_化骨绵掌.md、10_一句话总结.md

---

**生成时间**：2026-02-26
**知识点**：PromptTemplate高级用法
**层级**：L3_组件生态
**编号**：10
