# 核心概念 02：Schema 验证与 TypeBox

> **类型安全的参数定义 - TypeBox 的三重作用**

## 概述

Schema 验证是自定义工具注册的关键环节，它确保 LLM 传递的参数是正确的。Pi-mono 选择 TypeBox 作为 Schema 库，提供编译时类型安全、运行时验证和 LLM 友好的 JSON Schema 生成。

---

## 为什么需要 Schema 验证？

### 问题场景

```typescript
// 没有 Schema 验证的工具
const tool = {
  name: 'divide',
  execute: async ({ a, b }) => {
    // 问题 1: a 和 b 可能不是数字
    // 问题 2: b 可能是 0（除零错误）
    // 问题 3: 参数可能缺失
    return { content: String(a / b) };
  }
};

// LLM 可能传入错误的参数
// { a: "hello", b: "world" }  → NaN
// { a: 10, b: 0 }             → Infinity
// { a: 10 }                   → NaN (b 是 undefined)
```

### 解决方案

```typescript
// 使用 TypeBox Schema 验证
import { Type } from '@sinclair/typebox';

const tool = {
  name: 'divide',
  parameters: Type.Object({
    a: Type.Number({ description: 'Dividend' }),
    b: Type.Number({ 
      description: 'Divisor',
      minimum: 0.0001  // 防止除零
    })
  }),
  execute: async ({ a, b }) => {
    // 参数已经验证过了，可以安全使用
    return { content: String(a / b) };
  }
};
```

---

## TypeBox 的三重作用

### 1. LLM 理解（静态）

**生成 JSON Schema**

```typescript
const params = Type.Object({
  city: Type.String({ description: 'City name' }),
  units: Type.Union([
    Type.Literal('celsius'),
    Type.Literal('fahrenheit')
  ], { description: 'Temperature units' })
});

// 生成的 JSON Schema（发送给 LLM）
{
  "type": "object",
  "properties": {
    "city": {
      "type": "string",
      "description": "City name"
    },
    "units": {
      "enum": ["celsius", "fahrenheit"],
      "description": "Temperature units"
    }
  },
  "required": ["city", "units"]
}
```

**LLM 如何使用：**
- 理解参数的类型和含义
- 根据 description 决定如何传参
- 根据约束生成正确的值

### 2. 运行时验证（动态）

**自动参数验证**

```typescript
import { Value } from '@sinclair/typebox/value';

const params = Type.Object({
  age: Type.Number({ minimum: 0, maximum: 120 })
});

// 验证成功
Value.Check(params, { age: 25 });  // true

// 验证失败
Value.Check(params, { age: 200 });  // false
Value.Check(params, { age: "25" });  // false
Value.Check(params, {});  // false (缺少 age)
```

**框架自动验证：**
```typescript
// Pi-mono 框架内部实现
async function callTool(tool, params) {
  // 1. 验证参数
  if (!Value.Check(tool.parameters, params)) {
    const errors = [...Value.Errors(tool.parameters, params)];
    throw new Error(`Invalid parameters: ${errors[0].message}`);
  }
  
  // 2. 调用工具
  return await tool.execute(params, context);
}
```

### 3. TypeScript 类型推导（编译时）

**类型安全**

```typescript
import { Type, Static } from '@sinclair/typebox';

const params = Type.Object({
  name: Type.String(),
  age: Type.Number(),
  email: Type.Optional(Type.String())
});

// 推导类型
type Params = Static<typeof params>;
// Params = { name: string; age: number; email?: string }

// 类型安全的 execute 函数
const execute = async (params: Params, context) => {
  // IDE 自动补全
  const { name, age, email } = params;
  //      ^     ^    ^
  //      string number string | undefined
  
  return { content: `${name} is ${age} years old` };
};
```

---

## TypeBox vs Zod

### 对比表

| 特性 | TypeBox | Zod |
|------|---------|-----|
| **JSON Schema 生成** | ✅ 原生支持 | ⚠️ 需要额外库 |
| **性能** | ✅ 更快（编译时优化） | ⚠️ 较慢 |
| **包体积** | ✅ 更小 | ⚠️ 较大 |
| **API 风格** | 函数式 | 链式调用 |
| **社区** | ⚠️ 较小 | ✅ 更大 |
| **错误信息** | ⚠️ 较简单 | ✅ 更详细 |
| **LLM 集成** | ✅ 完美 | ⚠️ 需要转换 |

### 代码对比

**TypeBox:**
```typescript
const schema = Type.Object({
  name: Type.String({ minLength: 1 }),
  age: Type.Number({ minimum: 0 }),
  email: Type.Optional(Type.String({ format: 'email' }))
});
```

**Zod:**
```typescript
const schema = z.object({
  name: z.string().min(1),
  age: z.number().min(0),
  email: z.string().email().optional()
});
```

### Pi-mono 为什么选择 TypeBox？

1. **LLM 友好**：原生生成 JSON Schema
2. **性能优先**：工具调用频繁，性能很重要
3. **轻量级**：减少包体积
4. **类型推导**：与 TypeScript 完美集成

---

## TypeBox 基础用法

### 基本类型

```typescript
import { Type } from '@sinclair/typebox';

// 字符串
Type.String()
Type.String({ minLength: 1 })
Type.String({ maxLength: 100 })
Type.String({ pattern: '^[a-z]+$' })
Type.String({ format: 'email' })
Type.String({ format: 'uri' })
Type.String({ format: 'date' })

// 数字
Type.Number()
Type.Number({ minimum: 0 })
Type.Number({ maximum: 100 })
Type.Number({ minimum: 0, maximum: 100 })
Type.Number({ multipleOf: 5 })

// 整数
Type.Integer()
Type.Integer({ minimum: 1 })

// 布尔值
Type.Boolean()

// Null
Type.Null()

// Any
Type.Any()
```

### 复合类型

```typescript
// 数组
Type.Array(Type.String())
Type.Array(Type.Number(), { minItems: 1, maxItems: 10 })

// 对象
Type.Object({
  name: Type.String(),
  age: Type.Number()
})

// 可选字段
Type.Object({
  name: Type.String(),
  email: Type.Optional(Type.String())
})

// 嵌套对象
Type.Object({
  user: Type.Object({
    name: Type.String(),
    age: Type.Number()
  }),
  tags: Type.Array(Type.String())
})
```

### 联合类型

```typescript
// 枚举（推荐）
Type.Union([
  Type.Literal('red'),
  Type.Literal('green'),
  Type.Literal('blue')
])

// 多种类型
Type.Union([
  Type.String(),
  Type.Number()
])

// 可选（语法糖）
Type.Optional(Type.String())
// 等价于
Type.Union([Type.String(), Type.Undefined()])
```

### 高级类型

```typescript
// Record
Type.Record(Type.String(), Type.Number())
// { [key: string]: number }

// Tuple
Type.Tuple([Type.String(), Type.Number()])
// [string, number]

// Ref（递归类型）
const Node = Type.Recursive(This => Type.Object({
  value: Type.String(),
  children: Type.Array(This)
}));
```

---

## 实战示例

### 示例 1：文件搜索工具

```typescript
import { Type } from '@sinclair/typebox';

const searchParams = Type.Object({
  // 必需参数：搜索查询
  query: Type.String({
    description: 'Search query (regex pattern)',
    minLength: 1
  }),
  
  // 可选参数：文件类型过滤
  fileType: Type.Optional(Type.Union([
    Type.Literal('js'),
    Type.Literal('ts'),
    Type.Literal('json'),
    Type.Literal('md')
  ], {
    description: 'File type filter'
  })),
  
  // 可选参数：最大结果数
  maxResults: Type.Optional(Type.Number({
    description: 'Maximum number of results',
    minimum: 1,
    maximum: 100,
    default: 10
  })),
  
  // 可选参数：大小写敏感
  caseSensitive: Type.Optional(Type.Boolean({
    description: 'Case sensitive search',
    default: false
  }))
});

// 类型推导
type SearchParams = Static<typeof searchParams>;
// {
//   query: string;
//   fileType?: 'js' | 'ts' | 'json' | 'md';
//   maxResults?: number;
//   caseSensitive?: boolean;
// }
```

### 示例 2：API 调用工具

```typescript
const apiParams = Type.Object({
  // URL
  url: Type.String({
    description: 'API endpoint URL',
    format: 'uri'
  }),
  
  // HTTP 方法
  method: Type.Union([
    Type.Literal('GET'),
    Type.Literal('POST'),
    Type.Literal('PUT'),
    Type.Literal('DELETE')
  ], {
    description: 'HTTP method',
    default: 'GET'
  }),
  
  // 请求头
  headers: Type.Optional(Type.Record(
    Type.String(),
    Type.String()
  ), {
    description: 'HTTP headers'
  }),
  
  // 请求体
  body: Type.Optional(Type.Any({
    description: 'Request body (JSON)'
  })),
  
  // 超时
  timeout: Type.Optional(Type.Number({
    description: 'Timeout in milliseconds',
    minimum: 1000,
    maximum: 60000,
    default: 30000
  }))
});
```

### 示例 3：数据库查询工具

```typescript
const queryParams = Type.Object({
  // 表名
  table: Type.String({
    description: 'Table name',
    pattern: '^[a-zA-Z_][a-zA-Z0-9_]*$'
  }),
  
  // 查询条件
  where: Type.Optional(Type.Record(
    Type.String(),
    Type.Union([
      Type.String(),
      Type.Number(),
      Type.Boolean()
    ])
  ), {
    description: 'WHERE conditions'
  }),
  
  // 排序
  orderBy: Type.Optional(Type.Array(Type.Object({
    field: Type.String(),
    direction: Type.Union([
      Type.Literal('ASC'),
      Type.Literal('DESC')
    ])
  })), {
    description: 'ORDER BY clauses'
  }),
  
  // 分页
  limit: Type.Optional(Type.Integer({
    description: 'Maximum rows to return',
    minimum: 1,
    maximum: 1000,
    default: 100
  })),
  
  offset: Type.Optional(Type.Integer({
    description: 'Number of rows to skip',
    minimum: 0,
    default: 0
  }))
});
```

---

## 验证错误处理

### 获取验证错误

```typescript
import { Value } from '@sinclair/typebox/value';

const schema = Type.Object({
  age: Type.Number({ minimum: 0, maximum: 120 })
});

const data = { age: 200 };

if (!Value.Check(schema, data)) {
  // 获取所有错误
  const errors = [...Value.Errors(schema, data)];
  
  errors.forEach(error => {
    console.log('Path:', error.path);      // '/age'
    console.log('Message:', error.message); // 'Expected number to be less or equal to 120'
    console.log('Value:', error.value);    // 200
  });
}
```

### 自定义错误信息

```typescript
const schema = Type.Object({
  email: Type.String({
    format: 'email',
    description: 'User email address',
    errorMessage: 'Please provide a valid email address'
  })
});
```

---

## 2025-2026 最新趋势

### 1. Zod v4 兼容性

```typescript
// 2025-2026 年，TypeBox 增加了 Zod 兼容层
import { fromZod } from '@sinclair/typebox/zod';
import { z } from 'zod';

const zodSchema = z.object({
  name: z.string(),
  age: z.number()
});

// 转换为 TypeBox
const typeboxSchema = fromZod(zodSchema);
```

### 2. MCP Schema 标准

```typescript
// 2025-2026 年，MCP 定义了标准的 Schema 格式
import { MCPSchema } from '@modelcontextprotocol/sdk';

// TypeBox 可以直接生成 MCP Schema
const mcpSchema: MCPSchema = {
  type: 'object',
  properties: {
    city: {
      type: 'string',
      description: 'City name'
    }
  },
  required: ['city']
};
```

### 3. AI 优化的 Schema

```typescript
// 2025-2026 年，Schema 增加了 AI 友好的元数据
const schema = Type.Object({
  query: Type.String({
    description: 'Search query',
    // AI 提示
    aiHint: 'Use natural language, e.g., "find all async functions"',
    // 示例值
    examples: [
      'function.*async',
      'class.*extends',
      'import.*from'
    ]
  })
});
```

---

## 最佳实践

### 1. 充分利用约束

```typescript
// ✅ 好的做法
Type.Object({
  email: Type.String({ format: 'email' }),
  age: Type.Number({ minimum: 0, maximum: 120 }),
  username: Type.String({ 
    minLength: 3, 
    maxLength: 20,
    pattern: '^[a-zA-Z0-9_]+$'
  })
})

// ❌ 不好的做法（缺少约束）
Type.Object({
  email: Type.String(),
  age: Type.Number(),
  username: Type.String()
})
```

### 2. 每个字段都要有 description

```typescript
// ✅ 好的做法
Type.Object({
  city: Type.String({
    description: 'City name for weather query'
  })
})

// ❌ 不好的做法（缺少 description）
Type.Object({
  city: Type.String()
})
```

### 3. 使用 Type.Optional 标记可选字段

```typescript
// ✅ 好的做法
Type.Object({
  name: Type.String(),
  email: Type.Optional(Type.String())
})

// ❌ 不好的做法（使用 Union）
Type.Object({
  name: Type.String(),
  email: Type.Union([Type.String(), Type.Undefined()])
})
```

### 4. 提供默认值

```typescript
// ✅ 好的做法
Type.Object({
  maxResults: Type.Optional(Type.Number({
    description: 'Maximum results',
    default: 10,
    minimum: 1,
    maximum: 100
  }))
})
```

### 5. 使用枚举而非字符串

```typescript
// ✅ 好的做法
Type.Union([
  Type.Literal('red'),
  Type.Literal('green'),
  Type.Literal('blue')
])

// ❌ 不好的做法
Type.String({ 
  description: 'Color (red, green, or blue)' 
})
```

---

## 常见问题

### Q1: TypeBox 和 Zod 可以混用吗？

**A**: 不建议混用。Pi-mono 统一使用 TypeBox。如果需要使用 Zod schema，可以使用转换工具。

### Q2: 如何验证复杂的业务逻辑？

**A**: Schema 只验证类型和格式，业务逻辑验证在 execute 函数中：

```typescript
parameters: Type.Object({
  startDate: Type.String({ format: 'date' }),
  endDate: Type.String({ format: 'date' })
})

execute: async ({ startDate, endDate }, context) => {
  // 业务逻辑验证
  if (new Date(startDate) > new Date(endDate)) {
    return { content: 'Error: Start date must be before end date' };
  }
  // ...
}
```

### Q3: 如何处理动态 Schema？

**A**: 使用 Type.Any 或 Type.Record：

```typescript
// 动态键值对
Type.Record(Type.String(), Type.Any())

// 动态数组
Type.Array(Type.Any())
```

### Q4: Schema 验证失败会怎样？

**A**: 
1. 工具不会被调用
2. LLM 收到错误信息
3. LLM 可能会重试，传入正确的参数

---

## 总结

**TypeBox 的三重作用：**
1. **LLM 理解**：生成 JSON Schema
2. **运行时验证**：自动参数校验
3. **类型推导**：TypeScript 类型安全

**最佳实践：**
- 充分利用约束（minimum, maximum, pattern, format）
- 每个字段都要有 description
- 使用 Type.Optional 标记可选字段
- 提供默认值
- 使用枚举而非字符串

**记住：**
- Schema 验证类型和格式
- 业务逻辑验证在 execute 中
- 不要在 execute 中重复验证 Schema 已验证的内容

理解 TypeBox 的三重作用，你就能写出类型安全、LLM 友好的工具参数定义！
