---
type: source_code_analysis
source: sourcecode/langchain/libs/core/langchain_core/prompts/
analyzed_files:
  - prompt.py
  - base.py
  - string.py
  - test_prompt.py
analyzed_at: 2026-02-26
knowledge_point: PromptTemplate高级用法
---

# 源码分析：PromptTemplate 核心实现

## 分析的文件

- `sourcecode/langchain/libs/core/langchain_core/prompts/prompt.py` - 核心 PromptTemplate 实现
- `sourcecode/langchain/libs/core/langchain_core/prompts/base.py` - BasePromptTemplate 基类
- `sourcecode/langchain/libs/core/langchain_core/prompts/string.py` - StringPromptTemplate 和格式化器
- `sourcecode/langchain/libs/core/tests/unit_tests/prompts/test_prompt.py` - 测试用例

## 关键发现

### 1. 模板组合（Template Composition）

**源码位置**: `prompt.py:142-184`

```python
def __add__(self, other: Any) -> PromptTemplate:
    """Override the `+` operator to allow for combining prompt templates."""
    if isinstance(other, PromptTemplate):
        if self.template_format != other.template_format:
            msg = "Cannot add templates of different formats"
            raise ValueError(msg)
        input_variables = list(
            set(self.input_variables) | set(other.input_variables)
        )
        template = self.template + other.template
        validate_template = self.validate_template and other.validate_template
        partial_variables = dict(self.partial_variables.items())
        for k, v in other.partial_variables.items():
            if k in partial_variables:
                msg = "Cannot have same variable partialed twice."
                raise ValueError(msg)
            partial_variables[k] = v
        return PromptTemplate(
            template=template,
            input_variables=input_variables,
            partial_variables=partial_variables,
            template_format=self.template_format,
            validate_template=validate_template,
        )
    if isinstance(other, str):
        prompt = PromptTemplate.from_template(
            other,
            template_format=self.template_format,
        )
        return self + prompt
```

**关键特性**：
- 支持使用 `+` 操作符组合两个 PromptTemplate
- 自动合并 `input_variables`（取并集）
- 自动合并 `partial_variables`（检查冲突）
- 支持 PromptTemplate + str 的组合
- 要求两个模板的 `template_format` 必须一致

### 2. 部分变量（Partial Variables）

**源码位置**: `base.py:65-70`

```python
partial_variables: Mapping[str, Any] = Field(default_factory=dict)
"""A dictionary of the partial variables the prompt template carries.

Partial variables populate the template so that you don't need to pass them in every
time you call the prompt.
"""
```

**关键特性**：
- `partial_variables` 是一个字典，存储预填充的变量
- 在模板格式化时，会自动合并 `partial_variables` 和用户提供的变量
- 用于减少重复传递相同的变量
- 在 `__add__` 组合时会检查冲突

**源码位置**: `prompt.py:99-100`

```python
values.setdefault("partial_variables", {})
```

**源码位置**: `prompt.py:111`

```python
all_inputs = values["input_variables"] + list(values["partial_variables"])
```

### 3. 多种模板格式（Template Formats）

**源码位置**: `prompt.py:80-84`

```python
template_format: PromptTemplateFormat = "f-string"
"""The format of the prompt template.

Options are: `'f-string'`, `'mustache'`, `'jinja2'`.
"""
```

**支持的格式**：

#### a. f-string（默认）
```python
template = "Say {foo}"
prompt = PromptTemplate.from_template(template)
prompt.format(foo="bar")  # "Say bar"
```

#### b. mustache
```python
template = "This is a {{foo}} test."
prompt = PromptTemplate.from_template(template, template_format="mustache")
prompt.format(foo="bar")  # "This is a bar test."
```

**Mustache 高级特性**（从测试文件）：
- 嵌套变量：`{{obj.bar}}`
- Section 变量：`{{#foo}} {{bar}} {{/foo}}`
- No escape：`{{{foo}}}`（不转义 HTML）

#### c. jinja2
```python
template = "Say {{ foo }}"
prompt = PromptTemplate.from_template(template, template_format="jinja2")
```

**安全警告**：
- Jinja2 使用 `SandboxedEnvironment`
- 不要接受来自不可信源的 jinja2 模板
- 可能导致任意 Python 代码执行

### 4. 模板验证（Template Validation）

**源码位置**: `prompt.py:86-87`

```python
validate_template: bool = False
"""Whether or not to try validating the template."""
```

**验证逻辑**（`prompt.py:102-114`）：
```python
if values.get("validate_template"):
    if values["template_format"] == "mustache":
        msg = "Mustache templates cannot be validated."
        raise ValueError(msg)

    if "input_variables" not in values:
        msg = "Input variables must be provided to validate the template."
        raise ValueError(msg)

    all_inputs = values["input_variables"] + list(values["partial_variables"])
    check_valid_template(
        values["template"], values["template_format"], all_inputs
    )
```

**关键特性**：
- Mustache 模板不支持验证
- 需要提供 `input_variables` 才能验证
- 验证时会检查 `input_variables` 和 `partial_variables` 的总和

### 5. 从文件加载模板

**源码位置**: 测试文件 `test_prompt.py:30-48`

```python
def test_from_file_encoding() -> None:
    """Test that we can load a template from a file with a non utf-8 encoding."""
    template = "This is a {foo} test with special character €."
    input_variables = ["foo"]

    # First write to a file using CP-1252 encoding.
    with NamedTemporaryFile(delete=True, mode="w", encoding="cp1252") as f:
        f.write(template)
        f.flush()
        file_name = f.name

        # Now read from the file using CP-1252 encoding and test
        prompt = PromptTemplate.from_file(file_name, encoding="cp1252")
        assert prompt.template == template
        assert prompt.input_variables == input_variables
```

**关键特性**：
- 支持从文件加载模板
- 支持指定编码（如 CP-1252、UTF-8）
- 自动提取 `input_variables`

### 6. 输入变量自动提取

**源码位置**: `prompt.py:116-123`

```python
if values["template_format"]:
    values["input_variables"] = [
        var
        for var in get_template_variables(
            values["template"], values["template_format"]
        )
        if var not in values["partial_variables"]
    ]
```

**关键特性**：
- 自动从模板中提取变量名
- 排除 `partial_variables` 中的变量
- 支持所有三种模板格式

### 7. Mustache Schema 生成

**源码位置**: `string.py:158-194`

```python
def mustache_schema(template: str) -> type[BaseModel]:
    """Get the variables from a mustache template.

    Args:
        template: The template string.

    Returns:
        The variables from the template as a Pydantic model.
    """
    fields = {}
    prefix: tuple[str, ...] = ()
    section_stack: list[tuple[str, ...]] = []
    for type_, key in mustache.tokenize(template):
        if key == ".":
            continue
        if type_ == "end":
            if section_stack:
                prefix = section_stack.pop()
        elif type_ in {"section", "inverted section"}:
            section_stack.append(prefix)
            prefix += tuple(key.split("."))
            fields[prefix] = False
        elif type_ in {"variable", "no escape"}:
            fields[prefix + tuple(key.split("."))] = True
    # ... (省略后续代码)
```

**关键特性**：
- 从 Mustache 模板生成 Pydantic 模型
- 支持嵌套变量和 section 变量
- 用于类型验证和 schema 生成

## 高级用法总结

基于源码分析，PromptTemplate 的高级用法包括：

1. **模板组合**：
   - 使用 `+` 操作符组合多个模板
   - 自动合并变量和部分变量
   - 支持字符串直接组合

2. **部分变量**：
   - 预填充常用变量
   - 减少重复传递
   - 支持在组合时合并

3. **多种模板格式**：
   - f-string：简单快速
   - mustache：支持嵌套和 section
   - jinja2：功能强大但有安全风险

4. **模板验证**：
   - 可选的模板验证
   - 检查变量完整性

5. **从文件加载**：
   - 支持多种编码
   - 自动提取变量

6. **Schema 生成**：
   - Mustache 模板的 Pydantic 模型生成
   - 支持类型验证

## 需要进一步调研的内容

1. **模板继承**：
   - 源码中没有直接看到"继承"的实现
   - 可能指的是模板组合和扩展
   - 需要查询官方文档确认

2. **实际应用场景**：
   - 需要查找社区实践案例
   - 了解在 RAG、Agent 等场景中的应用

3. **性能优化**：
   - 不同模板格式的性能对比
   - 大规模模板组合的最佳实践

4. **与其他组件的集成**：
   - 与 ChatPromptTemplate 的关系
   - 与 FewShotPromptTemplate 的关系
