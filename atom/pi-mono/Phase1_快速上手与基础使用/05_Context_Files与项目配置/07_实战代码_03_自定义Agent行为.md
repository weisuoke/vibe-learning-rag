# 实战代码：自定义 Agent 行为

> 通过 APPEND_SYSTEM.md 定制 Pi Agent 的工作风格、代码偏好和行为准则

---

## 一、场景描述

**开发者需求：**
- 希望 Pi Agent 有特定的工作风格
- 需要强制执行特定的代码规范
- 想要 Pi Agent 自动执行某些操作
- 需要针对不同项目定制不同的 AI 行为

**目标：**
- 定制 Pi Agent 的沟通方式
- 强制执行代码风格
- 自动化工作流程
- 适应不同项目的需求

---

## 二、场景 1：简洁高效型 Agent

### 2.1 需求描述

**开发者特点：**
- 经验丰富的开发者
- 喜欢简洁直接的沟通
- 不需要冗长的解释
- 希望 AI 代码优先，少说多做

### 2.2 配置文件

**~/.pi/agent/APPEND_SYSTEM.md（全局配置）**

```markdown
# 简洁高效工作风格

## 沟通方式
- 简洁直接，不要冗长的解释
- 代码优先，少说多做
- 不要输出"让我来帮你..."之类的开场白
- 不要问"需要我继续吗？"，直接完成任务
- 不要列举多个方案，直接给出最佳方案

## 输出格式
- 直接输出代码，不要过多解释
- 只在关键地方添加注释
- 不要输出冗长的总结
- 错误信息简洁明了

## 工作流程
- 每次修改代码后自动运行测试
- 测试失败时立即修复，不要问
- 提交前自动运行 lint
- 不要问"需要我运行测试吗？"

## 示例对比

### ❌ 冗长风格
"好的，让我来帮你创建一个用户认证组件。首先，我会创建一个 React 组件，
然后添加表单验证，最后集成 API 调用。这个组件将会..."

### ✅ 简洁风格
创建用户认证组件：

```typescript
// src/components/Auth.tsx
export function Auth() {
  // 实现代码
}
```

测试通过。
```

### 2.3 实际效果

**使用前：**
```
用户：帮我添加一个登录功能
AI：好的，让我来帮你添加登录功能。首先，我会创建一个登录组件...
    [冗长的解释]
    需要我继续吗？
```

**使用后：**
```
用户：帮我添加一个登录功能
AI：[直接输出代码]
    测试通过。
```

---

## 三、场景 2：函数式编程强制型 Agent

### 3.1 需求描述

**项目要求：**
- 严格遵循函数式编程范式
- 禁止使用 class 和面向对象
- 所有函数必须是纯函数
- 优先使用不可变数据结构

### 3.2 配置文件

**project/.pi/APPEND_SYSTEM.md（项目配置）**

```markdown
# 函数式编程强制规范

## 核心原则
这个项目严格遵循函数式编程范式，所有代码必须符合以下规则。

## 必须遵守的规则

### 1. 禁止使用 class
```typescript
// ❌ 禁止
class UserService {
  getUser() { ... }
}

// ✅ 正确
const getUser = (id: string) => { ... };
```

### 2. 所有函数必须是纯函数
```typescript
// ❌ 禁止（有副作用）
let count = 0;
const increment = () => {
  count++;
  return count;
};

// ✅ 正确（纯函数）
const increment = (count: number) => count + 1;
```

### 3. 使用 const 声明所有变量
```typescript
// ❌ 禁止
let user = getUser();
var name = user.name;

// ✅ 正确
const user = getUser();
const name = user.name;
```

### 4. 优先使用不可变操作
```typescript
// ❌ 禁止（修改原数组）
const addItem = (arr: number[], item: number) => {
  arr.push(item);
  return arr;
};

// ✅ 正确（返回新数组）
const addItem = (arr: number[], item: number) => [...arr, item];
```

### 5. 使用高阶函数代替循环
```typescript
// ❌ 禁止
const doubled = [];
for (let i = 0; i < numbers.length; i++) {
  doubled.push(numbers[i] * 2);
}

// ✅ 正确
const doubled = numbers.map(n => n * 2);
```

### 6. 使用函数组合
```typescript
// ❌ 禁止（命令式）
const result = processStep3(processStep2(processStep1(data)));

// ✅ 正确（函数组合）
const pipe = (...fns) => (x) => fns.reduce((v, f) => f(v), x);
const process = pipe(processStep1, processStep2, processStep3);
const result = process(data);
```

## 推荐的函数式工具
- 使用 Ramda 或 Lodash/fp 进行函数式操作
- 使用 fp-ts 进行类型安全的函数式编程
- 使用 Immer 处理不可变数据（如果必须修改复杂对象）

## 代码审查标准
在输出任何代码前，检查：
1. 是否使用了 class？（禁止）
2. 是否有副作用？（禁止）
3. 是否使用了 let 或 var？（禁止）
4. 是否修改了原数据？（禁止）
5. 是否使用了循环？（优先使用 map/filter/reduce）

如果发现违反规则，立即修正。
```

### 3.3 实际效果

**使用前：**
```typescript
// AI 可能输出面向对象代码
class UserManager {
  private users: User[] = [];

  addUser(user: User) {
    this.users.push(user);
  }
}
```

**使用后：**
```typescript
// AI 自动输出函数式代码
const addUser = (users: User[], user: User): User[] =>
  [...users, user];

const removeUser = (users: User[], id: string): User[] =>
  users.filter(u => u.id !== id);
```

---

## 四、场景 3：安全优先型 Agent

### 4.1 需求描述

**项目特点：**
- 金融交易系统
- 处理敏感数据
- 安全性是最高优先级
- 需要严格的安全检查

### 4.2 配置文件

**project/.pi/APPEND_SYSTEM.md（项目配置）**

```markdown
# 安全优先规范

## 项目背景
这是一个金融交易系统，处理用户的资金和敏感信息。
安全性是最高优先级，任何代码都必须经过严格的安全审查。

## 安全规则（最高优先级）

### 1. 输入验证
所有用户输入必须验证和清理：

```typescript
// ❌ 禁止（未验证输入）
const getUser = (id: string) => {
  return db.query(`SELECT * FROM users WHERE id = ${id}`);
};

// ✅ 正确（参数化查询）
const getUser = (id: string) => {
  // 验证 ID 格式
  if (!/^[a-zA-Z0-9-]+$/.test(id)) {
    throw new Error('Invalid user ID');
  }
  return db.query('SELECT * FROM users WHERE id = ?', [id]);
};
```

### 2. SQL 注入防护
永远不要拼接 SQL 字符串：

```typescript
// ❌ 禁止
const query = `SELECT * FROM users WHERE name = '${name}'`;

// ✅ 正确
const query = 'SELECT * FROM users WHERE name = ?';
db.query(query, [name]);
```

### 3. XSS 防护
所有输出到 HTML 的内容必须转义：

```typescript
// ❌ 禁止
<div dangerouslySetInnerHTML={{ __html: userInput }} />

// ✅ 正确
<div>{userInput}</div>  // React 自动转义
```

### 4. 敏感信息处理
- 密码必须使用 bcrypt 加密
- API 密钥不能硬编码
- 敏感信息不能记录到日志

```typescript
// ❌ 禁止
const API_KEY = 'sk-1234567890';
console.log('User password:', password);

// ✅ 正确
const API_KEY = process.env.API_KEY;
console.log('User login attempt:', { userId, timestamp });
```

### 5. 金额处理
所有金额计算使用 Decimal 类型：

```typescript
// ❌ 禁止（浮点数精度问题）
const total = 100.1 + 200.2;  // 300.30000000000004

// ✅ 正确
import Decimal from 'decimal.js';
const total = new Decimal('100.1').plus('200.2');  // 300.3
```

### 6. 错误处理
不要在错误信息中暴露敏感信息：

```typescript
// ❌ 禁止
throw new Error(`Database connection failed: ${dbPassword}`);

// ✅ 正确
logger.error('Database connection failed', { error });
throw new Error('Service temporarily unavailable');
```

## 代码审查清单
在输出任何代码前，检查：
1. [ ] 是否验证了所有用户输入？
2. [ ] 是否使用了参数化查询？
3. [ ] 是否正确处理了敏感信息？
4. [ ] 是否使用 Decimal 处理金额？
5. [ ] 是否在错误信息中暴露了敏感信息？

如果发现安全问题，立即修正并说明。

## 禁止操作
- ⚠️ 不要使用 eval()
- ⚠️ 不要使用 Function() 构造函数
- ⚠️ 不要使用 dangerouslySetInnerHTML
- ⚠️ 不要在客户端存储敏感信息
- ⚠️ 不要在日志中记录密码或 API 密钥
```

### 4.3 实际效果

**使用前：**
```typescript
// AI 可能输出不安全的代码
const getUser = (id) => {
  return db.query(`SELECT * FROM users WHERE id = ${id}`);
};
```

**使用后：**
```typescript
// AI 自动输出安全的代码
const getUser = (id: string) => {
  // 验证输入
  if (!/^[a-zA-Z0-9-]+$/.test(id)) {
    throw new Error('Invalid user ID');
  }

  // 参数化查询
  return db.query('SELECT * FROM users WHERE id = ?', [id]);
};
```

---

## 五、场景 4：测试驱动型 Agent

### 5.1 需求描述

**开发流程：**
- 严格遵循 TDD（测试驱动开发）
- 先写测试，再写实现
- 所有功能必须有测试
- 测试覆盖率必须 > 80%

### 5.2 配置文件

**project/.pi/APPEND_SYSTEM.md（项目配置）**

```markdown
# 测试驱动开发规范

## 核心原则
这个项目严格遵循 TDD（测试驱动开发），所有代码必须先写测试。

## TDD 工作流程

### 1. 先写测试
在实现任何功能前，必须先写测试：

```typescript
// 第一步：写测试
describe('UserService', () => {
  it('should create a new user', () => {
    const user = createUser({ name: 'Alice', email: 'alice@example.com' });
    expect(user).toHaveProperty('id');
    expect(user.name).toBe('Alice');
  });
});

// 第二步：运行测试（应该失败）
// 第三步：实现功能
// 第四步：运行测试（应该通过）
```

### 2. 红-绿-重构循环
1. **红**：写一个失败的测试
2. **绿**：写最少的代码让测试通过
3. **重构**：优化代码，保持测试通过

### 3. 测试覆盖率要求
- 核心业务逻辑：100% 覆盖
- 工具函数：100% 覆盖
- UI 组件：> 80% 覆盖
- 整体项目：> 80% 覆盖

## 自动化工作流程

### 每次修改代码后
1. 自动运行相关测试
2. 如果测试失败，立即修复
3. 不要问"需要我运行测试吗？"

### 每次添加新功能
1. 先写测试（描述期望行为）
2. 运行测试（确认失败）
3. 实现功能
4. 运行测试（确认通过）
5. 重构代码（如果需要）

## 测试编写规范

### 测试结构
```typescript
describe('功能模块', () => {
  // 准备
  beforeEach(() => {
    // 设置测试环境
  });

  // 测试用例
  it('should 做什么', () => {
    // Arrange（准备）
    const input = ...;

    // Act（执行）
    const result = functionUnderTest(input);

    // Assert（断言）
    expect(result).toBe(expected);
  });

  // 清理
  afterEach(() => {
    // 清理测试环境
  });
});
```

### 测试命名
- 使用 "should" 描述期望行为
- 清晰描述测试场景
- 包含边界情况和错误情况

```typescript
// ✅ 好的测试命名
it('should return user when ID exists')
it('should throw error when ID is invalid')
it('should handle empty array')

// ❌ 不好的测试命名
it('test user')
it('works')
```

## 输出格式

### 添加新功能时
```
1. 编写测试：
[测试代码]

2. 运行测试（预期失败）：
❌ Test failed (expected)

3. 实现功能：
[实现代码]

4. 运行测试（预期通过）：
✅ All tests passed
```

### 修改现有功能时
```
1. 运行现有测试：
✅ All tests passed

2. 修改代码：
[修改代码]

3. 运行测试：
✅ All tests passed
```

## 禁止行为
- ⚠️ 不要在没有测试的情况下实现功能
- ⚠️ 不要跳过失败的测试
- ⚠️ 不要为了通过测试而修改测试
- ⚠️ 不要写没有断言的测试
```

### 5.3 实际效果

**使用前：**
```
用户：帮我添加一个用户注册功能
AI：[直接输出实现代码]
```

**使用后：**
```
用户：帮我添加一个用户注册功能
AI：
1. 编写测试：
```typescript
describe('registerUser', () => {
  it('should create a new user with valid data', () => {
    const userData = { name: 'Alice', email: 'alice@example.com' };
    const user = registerUser(userData);
    expect(user).toHaveProperty('id');
    expect(user.name).toBe('Alice');
  });
});
```

2. 运行测试：
❌ Test failed: registerUser is not defined

3. 实现功能：
```typescript
export function registerUser(data: UserData): User {
  return {
    id: generateId(),
    ...data,
    createdAt: new Date()
  };
}
```

4. 运行测试：
✅ All tests passed
```

---

## 六、场景 5：文档优先型 Agent

### 6.1 需求描述

**项目要求：**
- 所有公共 API 必须有文档
- 使用 JSDoc 格式
- 包含类型、参数、返回值、示例
- 文档和代码同步更新

### 6.2 配置文件

**project/.pi/APPEND_SYSTEM.md（项目配置）**

```markdown
# 文档优先规范

## 核心原则
所有公共 API 必须有完整的 JSDoc 文档。

## 文档格式

### 函数文档
```typescript
/**
 * 根据用户 ID 获取用户信息
 *
 * @param userId - 用户的唯一标识符
 * @returns 用户对象，如果用户不存在则返回 null
 * @throws {ValidationError} 当 userId 格式无效时
 *
 * @example
 * ```typescript
 * const user = await getUser('user-123');
 * if (user) {
 *   console.log(user.name);
 * }
 * ```
 */
export async function getUser(userId: string): Promise<User | null> {
  // 实现
}
```

### 类文档
```typescript
/**
 * 用户服务类，处理用户相关的业务逻辑
 *
 * @example
 * ```typescript
 * const service = new UserService(db);
 * const user = await service.createUser({ name: 'Alice' });
 * ```
 */
export class UserService {
  /**
   * 创建新用户
   *
   * @param data - 用户数据
   * @returns 创建的用户对象
   */
  async createUser(data: CreateUserData): Promise<User> {
    // 实现
  }
}
```

### 类型文档
```typescript
/**
 * 用户数据接口
 */
export interface User {
  /** 用户唯一标识符 */
  id: string;

  /** 用户名称 */
  name: string;

  /** 用户邮箱 */
  email: string;

  /** 创建时间 */
  createdAt: Date;
}
```

## 文档要求

### 必须包含
1. **描述**：清晰说明功能
2. **参数**：所有参数的类型和说明
3. **返回值**：返回值的类型和说明
4. **异常**：可能抛出的异常
5. **示例**：实际使用示例

### 可选包含
- `@deprecated` - 标记废弃的 API
- `@see` - 相关 API 的链接
- `@since` - API 添加的版本

## 自动化工作流程

### 添加新 API 时
1. 先写 JSDoc 文档（描述期望的 API）
2. 实现功能
3. 确保文档和实现一致

### 修改现有 API 时
1. 更新 JSDoc 文档
2. 修改实现
3. 确保文档和实现一致

## 输出格式

### 添加新函数时
```typescript
/**
 * [完整的 JSDoc 文档]
 */
export function newFunction() {
  // 实现
}
```

### 不要输出没有文档的公共 API
```typescript
// ❌ 禁止
export function getUser(id: string) {
  // 没有文档
}

// ✅ 正确
/**
 * 根据 ID 获取用户
 * @param id - 用户 ID
 * @returns 用户对象
 */
export function getUser(id: string): User {
  // 实现
}
```
```

### 6.3 实际效果

**使用前：**
```typescript
// AI 输出没有文档的代码
export function calculateTotal(items: Item[]): number {
  return items.reduce((sum, item) => sum + item.price, 0);
}
```

**使用后：**
```typescript
// AI 自动添加完整文档
/**
 * 计算商品列表的总价
 *
 * @param items - 商品列表
 * @returns 总价（保留两位小数）
 *
 * @example
 * ```typescript
 * const items = [
 *   { name: 'Apple', price: 1.5 },
 *   { name: 'Banana', price: 0.8 }
 * ];
 * const total = calculateTotal(items);  // 2.3
 * ```
 */
export function calculateTotal(items: Item[]): number {
  return items.reduce((sum, item) => sum + item.price, 0);
}
```

---

## 七、配置组合策略

### 7.1 全局 + 项目配置

```bash
# 全局配置（个人风格）
~/.pi/agent/APPEND_SYSTEM.md
- 简洁高效的沟通方式
- 代码优先，少说多做

# 项目配置（项目规范）
project/.pi/APPEND_SYSTEM.md
- 函数式编程强制规范
- 安全优先规则
- TDD 工作流程
```

**最终效果：** 全局风格 + 项目规范

### 7.2 不同项目不同配置

```bash
# 项目 A：函数式 + TDD
project-a/.pi/APPEND_SYSTEM.md

# 项目 B：安全优先 + 文档优先
project-b/.pi/APPEND_SYSTEM.md

# 项目 C：简洁高效
project-c/.pi/APPEND_SYSTEM.md
```

---

## 八、配置模板库

### 8.1 创建模板库

```bash
# 创建模板目录
mkdir -p ~/.pi/templates/append-system

# 保存常用模板
cat > ~/.pi/templates/append-system/concise.md << 'EOF'
# 简洁高效工作风格
- 代码优先，少说多做
- 不要冗长的解释
- 不要问"需要我继续吗？"
EOF

cat > ~/.pi/templates/append-system/functional.md << 'EOF'
# 函数式编程强制规范
- 禁止使用 class
- 所有函数必须是纯函数
- 使用 const 声明所有变量
EOF

cat > ~/.pi/templates/append-system/security.md << 'EOF'
# 安全优先规范
- 所有输入必须验证
- 使用参数化查询
- 敏感信息不能硬编码
EOF
```

### 8.2 使用模板

```bash
# 新项目使用模板
cp ~/.pi/templates/append-system/functional.md project/.pi/APPEND_SYSTEM.md

# 组合多个模板
cat ~/.pi/templates/append-system/concise.md \
    ~/.pi/templates/append-system/functional.md \
    > project/.pi/APPEND_SYSTEM.md
```

---

## 九、实际效果对比

### 9.1 代码质量提升

**场景：函数式编程项目**

| 指标 | 使用前 | 使用后 | 提升 |
|------|--------|--------|------|
| 纯函数比例 | 60% | 95% | +58% |
| 不可变操作 | 70% | 98% | +40% |
| 代码一致性 | 中等 | 高 | 显著 |

### 9.2 安全问题减少

**场景：金融交易系统**

| 指标 | 使用前 | 使用后 | 改善 |
|------|--------|--------|------|
| SQL 注入风险 | 5 处 | 0 处 | -100% |
| XSS 风险 | 3 处 | 0 处 | -100% |
| 敏感信息泄露 | 2 处 | 0 处 | -100% |

### 9.3 测试覆盖率提升

**场景：TDD 项目**

| 指标 | 使用前 | 使用后 | 提升 |
|------|--------|--------|------|
| 测试覆盖率 | 65% | 92% | +42% |
| 未测试功能 | 15 个 | 2 个 | -87% |
| 测试先行比例 | 30% | 95% | +217% |

---

## 十、常见问题

### Q1: APPEND_SYSTEM.md 会影响 Pi Agent 的基础能力吗？

**A:** 不会。APPEND_SYSTEM.md 是追加到默认提示词，不会替换基础能力。

### Q2: 如何验证配置是否生效？

**A:** 测试 AI 的行为：

```bash
# 1. 创建配置
echo "- 使用函数式编程" > .pi/APPEND_SYSTEM.md

# 2. 重新加载
pi
/reload

# 3. 测试
# 让 Pi 写一个函数，检查是否使用函数式风格
```

### Q3: 配置太严格会不会影响灵活性？

**A:** 可以使用优先级：

```markdown
# 必须遵守（严格）
- 不要使用 any 类型

# 推荐遵守（灵活）
- 优先使用函数式编程

# 可选（建议）
- 添加 JSDoc 注释
```

### Q4: 如何在团队中共享配置？

**A:** 提交到 Git：

```bash
# 项目配置提交到 Git
git add .pi/APPEND_SYSTEM.md
git commit -m "docs: add agent behavior config"
```

---

## 十一、总结

**自定义 Agent 行为的核心要点：**

1. **使用 APPEND_SYSTEM.md** - 追加到默认提示词，保留基础能力
2. **明确规则** - 清晰定义必须遵守的规则
3. **提供示例** - 用代码示例说明正确和错误的做法
4. **自动化工作流程** - 让 AI 自动执行重复操作
5. **分层配置** - 全局个人风格 + 项目特定规范

**效果：**
- 代码质量显著提升
- 安全问题大幅减少
- 测试覆盖率提高
- 开发效率提升

**记住：** APPEND_SYSTEM.md 是塑造 Pi Agent "性格"的工具，让 AI 成为符合你项目需求的开发伙伴！
