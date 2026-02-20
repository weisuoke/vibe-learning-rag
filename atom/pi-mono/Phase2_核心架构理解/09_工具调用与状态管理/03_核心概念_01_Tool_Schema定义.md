# 核心概念 01：Tool Schema 定义

## 概述

Tool Schema 是工具调用的"契约"，定义了工具的参数类型、约束和描述。它需要同时满足两个需求：
1. **给 LLM 看**：生成 JSON Schema，让 LLM 理解参数格式
2. **给代码用**：推导 TypeScript 类型，提供类型安全

Pi-mono 使用 **TypeBox** 而非 Zod，因为它更轻量、更快，且直接生成 JSON Schema [4]。

---

## 为什么需要 Schema？

### 问题：LLM 如何知道工具的参数格式？

LLM 只能生成文本，它需要知道：
- 工具叫什么名字？
- 需要哪些参数？
- 参数是什么类型？
- 有什么约束？

**没有 Schema 的情况：**
```typescript
// LLM 随意猜测参数格式
{
  "tool": "readFile",  // 工具名错误
  "file": "/etc/passwd",  // 参数名错误
  "format": "json"  // 多余的参数
}
```

**有 Schema 的情况：**
```typescript
// LLM 根据 Schema 生成正确格式
{
  "name": "read",
  "parameters": {
    "path": "./config.json"
  }
}
```

---

## TypeBox vs Zod：为什么选择 TypeBox？

### 对比表

| 特性 | TypeBox | Zod | 说明 |
|------|---------|-----|------|
| **Bundle Size** | 6KB | 60KB | TypeBox 小 10 倍 [4] |
| **验证速度** | 快 2-3 倍 | 基准 | TypeBox 使用 AJV，性能更好 |
| **JSON Schema** | 直接生成 | 需要转换 | TypeBox 本身就是 JSON Schema |
| **类型推导** | `Static<T>` | `z.infer<T>` | 都支持，语法略有不同 |
| **生态** | 较小 | 较大 | Zod 社区更活跃 |
| **学习曲线** | 平缓 | 平缓 | 都很容易上手 |

### Pi-mono 选择 TypeBox 的原因

1. **轻量级**：CLI 工具需要快速启动，bundle size 很重要
2. **性能**：工具调用频繁，验证速度影响用户体验
3. **直接生成 JSON Schema**：不需要额外转换步骤

**代码对比：**

```typescript
// Zod
import { z } from 'zod';

const schema = z.object({
  path: z.string().min(1),
  lines: z.object({
    start: z.number().min(1),
    end: z.number().min(1)
  }).optional()
});

type Params = z.infer<typeof schema>;

// 需要额外转换为 JSON Schema
import { zodToJsonSchema } from 'zod-to-json-schema';
const jsonSchema = zodToJsonSchema(schema);

// TypeBox
import { Type, Static } from '@sinclair/typebox';

const schema = Type.Object({
  path: Type.String({ minLength: 1 }),
  lines: Type.Optional(Type.Object({
    start: Type.Number({ minimum: 1 }),
    end: Type.Number({ minimum: 1 })
  }))
});

type Params = Static<typeof schema>;

// 直接就是 JSON Schema
const jsonSchema = schema;
```

---

## TypeBox 基础语法

### 1. 基本类型

```typescript
import { Type } from '@sinclair/typebox';

// 字符串
Type.String()
Type.String({ minLength: 1, maxLength: 100 })
Type.String({ pattern: '^[a-z]+$' })

// 数字
Type.Number()
Type.Number({ minimum: 0, maximum: 100 })
Type.Number({ multipleOf: 5 })

// 布尔
Type.Boolean()

// 字面量
Type.Literal('read')
Type.Literal(42)
Type.Literal(true)

// 枚举
Type.Union([
  Type.Literal('utf-8'),
  Type.Literal('ascii'),
  Type.Literal('utf-16')
])
```

### 2. 复合类型

```typescript
// 对象
Type.Object({
  name: Type.String(),
  age: Type.Number()
})

// 数组
Type.Array(Type.String())
Type.Array(Type.Number(), { minItems: 1, maxItems: 10 })

// 可选字段
Type.Object({
  required: Type.String(),
  optional: Type.Optional(Type.String())
})

// 联合类型
Type.Union([
  Type.String(),
  Type.Number()
])

// 元组
Type.Tuple([
  Type.String(),
  Type.Number(),
  Type.Boolean()
])

// Record（字典）
Type.Record(Type.String(), Type.Number())
// { [key: string]: number }
```

### 3. 高级特性

```typescript
// 递归类型
const Node = Type.Recursive(This => Type.Object({
  value: Type.String(),
  children: Type.Array(This)
}));

// 引用类型
const Address = Type.Object({
  street: Type.String(),
  city: Type.String()
});

const Person = Type.Object({
  name: Type.String(),
  address: Type.Ref(Address)
});

// 条件类型
Type.Object({
  type: Type.Literal('file'),
  path: Type.String()
}, {
  $id: 'FileInput'
});
```

---

## Pi-mono 的 Tool Schema 设计

### Read 工具

```typescript
import { Type, Static } from '@sinclair/typebox';

export const ReadToolSchema = Type.Object({
  path: Type.String({
    minLength: 1,
    description: '要读取的文件路径（相对于工作目录）'
  }),
  lines: Type.Optional(Type.Object({
    start: Type.Number({
      minimum: 1,
      description: '起始行号（从 1 开始）'
    }),
    end: Type.Number({
      minimum: 1,
      description: '结束行号（包含）'
    })
  }, {
    description: '可选：只读取指定行范围'
  }))
}, {
  $id: 'ReadTool',
  description: '读取文件内容'
});

export type ReadToolParams = Static<typeof ReadToolSchema>;
```

**生成的 JSON Schema（给 LLM）：**
```json
{
  "$id": "ReadTool",
  "type": "object",
  "description": "读取文件内容",
  "properties": {
    "path": {
      "type": "string",
      "minLength": 1,
      "description": "要读取的文件路径（相对于工作目录）"
    },
    "lines": {
      "type": "object",
      "description": "可选：只读取指定行范围",
      "properties": {
        "start": {
          "type": "number",
          "minimum": 1,
          "description": "起始行号（从 1 开始）"
        },
        "end": {
          "type": "number",
          "minimum": 1,
          "description": "结束行号（包含）"
        }
      },
      "required": ["start", "end"]
    }
  },
  "required": ["path"]
}
```

**推导的 TypeScript 类型（给代码）：**
```typescript
type ReadToolParams = {
  path: string;
  lines?: {
    start: number;
    end: number;
  };
};
```

### Write 工具

```typescript
export const WriteToolSchema = Type.Object({
  path: Type.String({
    minLength: 1,
    description: '要写入的文件路径'
  }),
  content: Type.String({
    description: '文件内容'
  }),
  createDirs: Type.Optional(Type.Boolean({
    description: '是否自动创建父目录（默认 false）'
  }))
}, {
  $id: 'WriteTool',
  description: '写入文件内容（覆盖现有文件）'
});

export type WriteToolParams = Static<typeof WriteToolSchema>;
```

### Edit 工具

```typescript
export const EditToolSchema = Type.Object({
  path: Type.String({
    minLength: 1,
    description: '要编辑的文件路径'
  }),
  oldText: Type.String({
    minLength: 1,
    description: '要替换的文本（必须完全匹配）'
  }),
  newText: Type.String({
    description: '新文本（可以为空字符串表示删除）'
  })
}, {
  $id: 'EditTool',
  description: '编辑文件内容（精确替换）'
});

export type EditToolParams = Static<typeof EditToolSchema>;
```

### Bash 工具

```typescript
export const BashToolSchema = Type.Object({
  command: Type.String({
    minLength: 1,
    description: '要执行的 shell 命令'
  }),
  timeout: Type.Optional(Type.Number({
    minimum: 1000,
    maximum: 600000,
    description: '超时时间（毫秒，默认 120000）'
  }))
}, {
  $id: 'BashTool',
  description: '执行 shell 命令'
});

export type BashToolParams = Static<typeof BashToolSchema>;
```

---

## Schema 设计最佳实践

### 1. 适度宽松的约束

**❌ 过度严格：**
```typescript
Type.String({
  minLength: 1,
  maxLength: 255,
  pattern: '^[a-zA-Z0-9/_.-]+$',  // 太严格
  description: '文件路径，必须是相对路径，不能包含 ..'
})
```

**✅ 适度宽松：**
```typescript
Type.String({
  minLength: 1,
  description: '文件路径（相对于工作目录）'
})
// 安全检查在执行时进行
```

**原因：**
- LLM 很难记住复杂的正则表达式
- 过度约束会增加验证失败率
- 运行时检查更灵活（可以给出更清晰的错误信息）

### 2. 清晰的描述

**❌ 模糊描述：**
```typescript
Type.String({ description: 'path' })
```

**✅ 清晰描述：**
```typescript
Type.String({
  description: '要读取的文件路径（相对于工作目录，如 ./config.json）'
})
```

**原因：**
- LLM 依赖描述理解参数含义
- 清晰的描述能减少错误
- 提供示例更有帮助

### 3. 合理的默认值

```typescript
Type.Object({
  encoding: Type.Optional(Type.String({
    default: 'utf-8',
    description: '文件编码（默认 utf-8）'
  })),
  timeout: Type.Optional(Type.Number({
    default: 120000,
    minimum: 1000,
    description: '超时时间（毫秒，默认 120000）'
  }))
})
```

**原因：**
- 减少 LLM 需要生成的参数
- 提供合理的默认行为
- 降低验证失败率

### 4. 避免嵌套过深

**❌ 嵌套过深：**
```typescript
Type.Object({
  config: Type.Object({
    server: Type.Object({
      host: Type.String(),
      port: Type.Number(),
      ssl: Type.Object({
        enabled: Type.Boolean(),
        cert: Type.String(),
        key: Type.String()
      })
    })
  })
})
```

**✅ 扁平化：**
```typescript
Type.Object({
  host: Type.String(),
  port: Type.Number(),
  sslEnabled: Type.Boolean(),
  sslCert: Type.Optional(Type.String()),
  sslKey: Type.Optional(Type.String())
})
```

**原因：**
- LLM 更容易理解扁平结构
- 减少嵌套错误
- 提高验证成功率

---

## 完整示例：手写工具 Schema

```typescript
import { Type, Static } from '@sinclair/typebox';

// 1. 定义 Schema
export const SearchToolSchema = Type.Object({
  pattern: Type.String({
    minLength: 1,
    description: '搜索模式（支持正则表达式）'
  }),
  path: Type.Optional(Type.String({
    description: '搜索路径（默认当前目录）'
  })),
  filePattern: Type.Optional(Type.String({
    description: '文件名模式（如 *.ts）'
  })),
  caseSensitive: Type.Optional(Type.Boolean({
    default: false,
    description: '是否区分大小写（默认 false）'
  })),
  maxResults: Type.Optional(Type.Number({
    minimum: 1,
    maximum: 1000,
    default: 100,
    description: '最大结果数（默认 100）'
  }))
}, {
  $id: 'SearchTool',
  description: '在文件中搜索文本'
});

// 2. 推导类型
export type SearchToolParams = Static<typeof SearchToolSchema>;

// 3. 实现工具
export async function executeSearch(params: SearchToolParams): Promise<string> {
  const {
    pattern,
    path = '.',
    filePattern = '*',
    caseSensitive = false,
    maxResults = 100
  } = params;

  // 实现搜索逻辑
  // ...

  return `Found ${results.length} matches`;
}

// 4. 注册工具
registry.register({
  name: 'search',
  schema: SearchToolSchema,
  execute: executeSearch
});
```

---

## 与 LLM API 的集成

### Anthropic Claude API

```typescript
import Anthropic from '@anthropic-ai/sdk';

const client = new Anthropic();

// 将 TypeBox Schema 转换为 Claude 工具格式
const tools = [
  {
    name: 'read',
    description: 'Read file content',
    input_schema: ReadToolSchema  // 直接使用 TypeBox Schema
  },
  {
    name: 'write',
    description: 'Write file content',
    input_schema: WriteToolSchema
  }
];

// 调用 Claude
const response = await client.messages.create({
  model: 'claude-3-5-sonnet-20241022',
  max_tokens: 4096,
  tools: tools,
  messages: [
    { role: 'user', content: 'Read config.json' }
  ]
});

// Claude 返回工具调用
// {
//   type: 'tool_use',
//   name: 'read',
//   input: { path: './config.json' }
// }
```

### OpenAI GPT API

```typescript
import OpenAI from 'openai';

const client = new OpenAI();

// OpenAI 使用 JSON Schema 格式
const tools = [
  {
    type: 'function',
    function: {
      name: 'read',
      description: 'Read file content',
      parameters: ReadToolSchema  // 直接使用 TypeBox Schema
    }
  }
];

const response = await client.chat.completions.create({
  model: 'gpt-4-turbo',
  messages: [
    { role: 'user', content: 'Read config.json' }
  ],
  tools: tools
});
```

---

## 调试技巧

### 1. 查看生成的 JSON Schema

```typescript
import { Type } from '@sinclair/typebox';

const schema = Type.Object({
  path: Type.String()
});

console.log(JSON.stringify(schema, null, 2));
// 输出完整的 JSON Schema
```

### 2. 测试类型推导

```typescript
import { Static } from '@sinclair/typebox';

type Params = Static<typeof schema>;

// 测试类型是否正确
const params: Params = {
  path: './test.txt'
};
```

### 3. 验证示例数据

```typescript
import Ajv from 'ajv';

const ajv = new Ajv();
const validate = ajv.compile(schema);

const testData = { path: './test.txt' };
const valid = validate(testData);

if (!valid) {
  console.error('Validation errors:', validate.errors);
}
```

---

## 总结

**Tool Schema 的三个关键点：**

1. **双重生成**：TypeBox 同时生成 JSON Schema（给 LLM）和 TypeScript 类型（给代码）
2. **适度宽松**：Schema 应该引导而非限制，运行时再做安全检查
3. **清晰描述**：LLM 依赖描述理解参数，描述越清晰，成功率越高

**Pi-mono 的设计哲学：**
- 轻量级（TypeBox 6KB vs Zod 60KB）
- 高性能（验证速度快 2-3 倍）
- 开发体验好（直接生成 JSON Schema，无需转换）

---

**参考文献：**
- [1] Anthropic SDK TypeScript: https://github.com/anthropics/anthropic-sdk-typescript
- [4] TypeBox vs Zod: https://github.com/colinhacks/zod/issues/2482
