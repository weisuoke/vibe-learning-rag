# write 工具实战

> 完整的 TypeScript 示例展示 write 工具的实际应用场景

---

## 概述

本文档提供 write 工具的完整实战示例，所有代码都可以直接运行。

---

## 场景 1：项目脚手架生成器

**任务：** 自动生成项目基础结构

```typescript
// project-scaffolder.ts
import fs from 'fs'
import path from 'path'

interface ProjectConfig {
  name: string
  type: 'library' | 'application'
  language: 'typescript' | 'javascript'
  packageManager: 'npm' | 'yarn' | 'pnpm'
}

class ProjectScaffolder {
  constructor(private config: ProjectConfig) {}

  /**
   * 生成完整项目结构
   */
  async generateProject(): Promise<void> {
    console.log(`\n🚀 Generating ${this.config.type} project: ${this.config.name}\n`)

    // 创建目录结构
    await this.createDirectories()

    // 生成配置文件
    await this.generatePackageJson()
    await this.generateTSConfig()
    await this.generateGitignore()
    await this.generateReadme()

    // 生成源代码
    await this.generateSourceFiles()

    // 生成测试文件
    await this.generateTestFiles()

    console.log('\n✅ Project generated successfully!')
    console.log(`\nNext steps:`)
    console.log(`  cd ${this.config.name}`)
    console.log(`  ${this.config.packageManager} install`)
    console.log(`  ${this.config.packageManager} test`)
  }

  /**
   * 创建目录结构
   */
  private async createDirectories(): Promise<void> {
    const dirs = [
      this.config.name,
      `${this.config.name}/src`,
      `${this.config.name}/tests`,
      `${this.config.name}/docs`
    ]

    for (const dir of dirs) {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true })
        console.log(`📁 Created: ${dir}`)
      }
    }
  }

  /**
   * 生成 package.json
   */
  private async generatePackageJson(): Promise<void> {
    const pkg = {
      name: this.config.name,
      version: '0.1.0',
      description: `A ${this.config.type} project`,
      main: this.config.type === 'library' ? 'dist/index.js' : undefined,
      scripts: {
        build: 'tsc',
        test: 'jest',
        lint: 'eslint src/**/*.ts',
        format: 'prettier --write "src/**/*.ts"'
      },
      keywords: [],
      author: '',
      license: 'MIT',
      devDependencies: {
        '@types/node': '^20.0.0',
        'typescript': '^5.0.0',
        'jest': '^29.0.0',
        '@types/jest': '^29.0.0',
        'eslint': '^8.0.0',
        'prettier': '^3.0.0'
      }
    }

    const filePath = `${this.config.name}/package.json`
    fs.writeFileSync(filePath, JSON.stringify(pkg, null, 2))
    console.log(`📄 Created: ${filePath}`)
  }

  /**
   * 生成 tsconfig.json
   */
  private async generateTSConfig(): Promise<void> {
    const tsconfig = {
      compilerOptions: {
        target: 'ES2020',
        module: 'commonjs',
        lib: ['ES2020'],
        outDir: './dist',
        rootDir: './src',
        strict: true,
        esModuleInterop: true,
        skipLibCheck: true,
        forceConsistentCasingInFileNames: true,
        declaration: this.config.type === 'library',
        declarationMap: this.config.type === 'library'
      },
      include: ['src/**/*'],
      exclude: ['node_modules', 'dist', 'tests']
    }

    const filePath = `${this.config.name}/tsconfig.json`
    fs.writeFileSync(filePath, JSON.stringify(tsconfig, null, 2))
    console.log(`📄 Created: ${filePath}`)
  }

  /**
   * 生成 .gitignore
   */
  private async generateGitignore(): Promise<void> {
    const content = `
# Dependencies
node_modules/
package-lock.json
yarn.lock
pnpm-lock.yaml

# Build output
dist/
build/
*.tsbuildinfo

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Testing
coverage/
.nyc_output/
`.trim()

    const filePath = `${this.config.name}/.gitignore`
    fs.writeFileSync(filePath, content)
    console.log(`📄 Created: ${filePath}`)
  }

  /**
   * 生成 README.md
   */
  private async generateReadme(): Promise<void> {
    const content = `# ${this.config.name}

> A ${this.config.type} project built with ${this.config.language}

## Installation

\`\`\`bash
${this.config.packageManager} install
\`\`\`

## Usage

${this.config.type === 'library' ? `
\`\`\`typescript
import { example } from '${this.config.name}'

example()
\`\`\`
` : `
\`\`\`bash
${this.config.packageManager} start
\`\`\`
`}

## Development

\`\`\`bash
# Build
${this.config.packageManager} run build

# Test
${this.config.packageManager} test

# Lint
${this.config.packageManager} run lint

# Format
${this.config.packageManager} run format
\`\`\`

## License

MIT
`

    const filePath = `${this.config.name}/README.md`
    fs.writeFileSync(filePath, content)
    console.log(`📄 Created: ${filePath}`)
  }

  /**
   * 生成源代码文件
   */
  private async generateSourceFiles(): Promise<void> {
    if (this.config.type === 'library') {
      // 生成库的入口文件
      const indexContent = `/**
 * ${this.config.name}
 *
 * A TypeScript library
 */

export function example(): string {
  return 'Hello from ${this.config.name}!'
}

export function add(a: number, b: number): number {
  return a + b
}

export function multiply(a: number, b: number): number {
  return a * b
}
`
      fs.writeFileSync(`${this.config.name}/src/index.ts`, indexContent)
      console.log(`📄 Created: ${this.config.name}/src/index.ts`)

    } else {
      // 生成应用的入口文件
      const mainContent = `/**
 * ${this.config.name}
 *
 * A TypeScript application
 */

async function main() {
  console.log('Hello from ${this.config.name}!')

  // Your application logic here
}

main().catch(error => {
  console.error('Error:', error)
  process.exit(1)
})
`
      fs.writeFileSync(`${this.config.name}/src/main.ts`, mainContent)
      console.log(`📄 Created: ${this.config.name}/src/main.ts`)
    }
  }

  /**
   * 生成测试文件
   */
  private async generateTestFiles(): Promise<void> {
    if (this.config.type === 'library') {
      const testContent = `import { example, add, multiply } from '../src/index'

describe('${this.config.name}', () => {
  describe('example', () => {
    it('should return greeting message', () => {
      expect(example()).toBe('Hello from ${this.config.name}!')
    })
  })

  describe('add', () => {
    it('should add two numbers', () => {
      expect(add(2, 3)).toBe(5)
      expect(add(-1, 1)).toBe(0)
    })
  })

  describe('multiply', () => {
    it('should multiply two numbers', () => {
      expect(multiply(2, 3)).toBe(6)
      expect(multiply(-2, 3)).toBe(-6)
    })
  })
})
`
      fs.writeFileSync(`${this.config.name}/tests/index.test.ts`, testContent)
      console.log(`📄 Created: ${this.config.name}/tests/index.test.ts`)
    }
  }
}

// 使用示例
async function main() {
  const config: ProjectConfig = {
    name: 'my-awesome-lib',
    type: 'library',
    language: 'typescript',
    packageManager: 'npm'
  }

  const scaffolder = new ProjectScaffolder(config)
  await scaffolder.generateProject()
}

main()
```

---

## 场景 2：配置文件生成器

**任务：** 根据环境生成不同的配置文件

```typescript
// config-generator.ts
import fs from 'fs'

type Environment = 'development' | 'staging' | 'production'

interface DatabaseConfig {
  host: string
  port: number
  database: string
  username: string
  password: string
}

interface AppConfig {
  env: Environment
  port: number
  database: DatabaseConfig
  redis: {
    host: string
    port: number
  }
  logging: {
    level: string
    format: string
  }
  features: {
    [key: string]: boolean
  }
}

class ConfigGenerator {
  /**
   * 生成环境配置文件
   */
  async generateEnvConfig(env: Environment): Promise<void> {
    console.log(`\n⚙️  Generating ${env} configuration\n`)

    const config = this.getConfigForEnv(env)

    // 生成 .env 文件
    await this.generateEnvFile(env, config)

    // 生成 config.json
    await this.generateConfigJson(env, config)

    // 生成 TypeScript 配置类型
    await this.generateConfigTypes()

    console.log(`\n✅ Configuration generated for ${env}`)
  }

  /**
   * 获取环境配置
   */
  private getConfigForEnv(env: Environment): AppConfig {
    const baseConfig: AppConfig = {
      env,
      port: 3000,
      database: {
        host: 'localhost',
        port: 5432,
        database: 'myapp',
        username: 'user',
        password: 'password'
      },
      redis: {
        host: 'localhost',
        port: 6379
      },
      logging: {
        level: 'info',
        format: 'json'
      },
      features: {
        analytics: false,
        newUI: false
      }
    }

    // 环境特定配置
    switch (env) {
      case 'development':
        return {
          ...baseConfig,
          logging: { level: 'debug', format: 'pretty' },
          features: { analytics: false, newUI: true }
        }

      case 'staging':
        return {
          ...baseConfig,
          port: 8080,
          database: {
            ...baseConfig.database,
            host: 'staging-db.example.com',
            password: '${DB_PASSWORD}'
          },
          redis: {
            host: 'staging-redis.example.com',
            port: 6379
          },
          features: { analytics: true, newUI: true }
        }

      case 'production':
        return {
          ...baseConfig,
          port: 80,
          database: {
            ...baseConfig.database,
            host: 'prod-db.example.com',
            password: '${DB_PASSWORD}'
          },
          redis: {
            host: 'prod-redis.example.com',
            port: 6379
          },
          logging: { level: 'warn', format: 'json' },
          features: { analytics: true, newUI: false }
        }
    }
  }

  /**
   * 生成 .env 文件
   */
  private async generateEnvFile(env: Environment, config: AppConfig): Promise<void> {
    const content = `# Environment: ${env}
# Generated at: ${new Date().toISOString()}

NODE_ENV=${env}
PORT=${config.port}

# Database
DB_HOST=${config.database.host}
DB_PORT=${config.database.port}
DB_NAME=${config.database.database}
DB_USER=${config.database.username}
DB_PASSWORD=${config.database.password}

# Redis
REDIS_HOST=${config.redis.host}
REDIS_PORT=${config.redis.port}

# Logging
LOG_LEVEL=${config.logging.level}
LOG_FORMAT=${config.logging.format}

# Feature Flags
FEATURE_ANALYTICS=${config.features.analytics}
FEATURE_NEW_UI=${config.features.newUI}
`

    const filePath = `.env.${env}`
    fs.writeFileSync(filePath, content)
    console.log(`📄 Created: ${filePath}`)
  }

  /**
   * 生成 config.json
   */
  private async generateConfigJson(env: Environment, config: AppConfig): Promise<void> {
    const filePath = `config.${env}.json`
    fs.writeFileSync(filePath, JSON.stringify(config, null, 2))
    console.log(`📄 Created: ${filePath}`)
  }

  /**
   * 生成 TypeScript 配置类型
   */
  private async generateConfigTypes(): Promise<void> {
    const content = `/**
 * Application configuration types
 * Auto-generated - do not edit manually
 */

export type Environment = 'development' | 'staging' | 'production'

export interface DatabaseConfig {
  host: string
  port: number
  database: string
  username: string
  password: string
}

export interface RedisConfig {
  host: string
  port: number
}

export interface LoggingConfig {
  level: 'debug' | 'info' | 'warn' | 'error'
  format: 'json' | 'pretty'
}

export interface FeatureFlags {
  analytics: boolean
  newUI: boolean
}

export interface AppConfig {
  env: Environment
  port: number
  database: DatabaseConfig
  redis: RedisConfig
  logging: LoggingConfig
  features: FeatureFlags
}

/**
 * Load configuration for current environment
 */
export function loadConfig(): AppConfig {
  const env = (process.env.NODE_ENV || 'development') as Environment
  const config = require(\`./config.\${env}.json\`)
  return config
}
`

    const filePath = 'src/config.ts'
    fs.writeFileSync(filePath, content)
    console.log(`📄 Created: ${filePath}`)
  }
}

// 使用示例
async function main() {
  const generator = new ConfigGenerator()

  // 生成所有环境的配置
  for (const env of ['development', 'staging', 'production'] as Environment[]) {
    await generator.generateEnvConfig(env)
  }
}

main()
```

---

## 场景 3：代码模板生成器

**任务：** 根据模板生成代码文件

```typescript
// code-template-generator.ts
import fs from 'fs'

interface ComponentTemplate {
  name: string
  type: 'class' | 'function'
  props?: string[]
  methods?: string[]
}

class CodeTemplateGenerator {
  /**
   * 生成 React 组件
   */
  async generateReactComponent(template: ComponentTemplate): Promise<void> {
    console.log(`\n🎨 Generating React component: ${template.name}\n`)

    const componentCode = this.generateComponentCode(template)
    const testCode = this.generateTestCode(template)
    const storyCode = this.generateStoryCode(template)

    // 写入文件
    fs.writeFileSync(`src/components/${template.name}.tsx`, componentCode)
    console.log(`📄 Created: src/components/${template.name}.tsx`)

    fs.writeFileSync(`src/components/${template.name}.test.tsx`, testCode)
    console.log(`📄 Created: src/components/${template.name}.test.tsx`)

    fs.writeFileSync(`src/components/${template.name}.stories.tsx`, storyCode)
    console.log(`📄 Created: src/components/${template.name}.stories.tsx`)

    console.log(`\n✅ Component generated successfully!`)
  }

  /**
   * 生成组件代码
   */
  private generateComponentCode(template: ComponentTemplate): string {
    if (template.type === 'function') {
      return this.generateFunctionComponent(template)
    } else {
      return this.generateClassComponent(template)
    }
  }

  /**
   * 生成函数组件
   */
  private generateFunctionComponent(template: ComponentTemplate): string {
    const props = template.props || []
    const propsInterface = props.length > 0
      ? `interface ${template.name}Props {\n${props.map(p => `  ${p}: string`).join('\n')}\n}\n\n`
      : ''

    const propsParam = props.length > 0 ? `props: ${template.name}Props` : ''

    return `import React from 'react'

${propsInterface}/**
 * ${template.name} component
 */
export const ${template.name}: React.FC${props.length > 0 ? `<${template.name}Props>` : ''} = (${propsParam}) => {
  return (
    <div className="${template.name.toLowerCase()}">
      <h2>${template.name}</h2>
      ${props.map(p => `<p>{props.${p}}</p>`).join('\n      ')}
    </div>
  )
}
`
  }

  /**
   * 生成类组件
   */
  private generateClassComponent(template: ComponentTemplate): string {
    const props = template.props || []
    const methods = template.methods || []

    const propsInterface = props.length > 0
      ? `interface ${template.name}Props {\n${props.map(p => `  ${p}: string`).join('\n')}\n}\n\n`
      : ''

    const stateInterface = `interface ${template.name}State {\n  // Add state properties here\n}\n\n`

    const methodsCode = methods.map(m => `
  ${m}() {
    // Implementation here
  }`).join('\n')

    return `import React, { Component } from 'react'

${propsInterface}${stateInterface}/**
 * ${template.name} component
 */
export class ${template.name} extends Component<${template.name}Props, ${template.name}State> {
  constructor(props: ${template.name}Props) {
    super(props)
    this.state = {}
  }
${methodsCode}

  render() {
    return (
      <div className="${template.name.toLowerCase()}">
        <h2>${template.name}</h2>
        ${props.map(p => `<p>{this.props.${p}}</p>`).join('\n        ')}
      </div>
    )
  }
}
`
  }

  /**
   * 生成测试代码
   */
  private generateTestCode(template: ComponentTemplate): string {
    return `import React from 'react'
import { render, screen } from '@testing-library/react'
import { ${template.name} } from './${template.name}'

describe('${template.name}', () => {
  it('should render successfully', () => {
    render(<${template.name} />)
    expect(screen.getByText('${template.name}')).toBeInTheDocument()
  })

  ${(template.props || []).map(prop => `
  it('should display ${prop} prop', () => {
    render(<${template.name} ${prop}="test value" />)
    expect(screen.getByText('test value')).toBeInTheDocument()
  })`).join('\n')}
})
`
  }

  /**
   * 生成 Storybook 故事
   */
  private generateStoryCode(template: ComponentTemplate): string {
    const props = template.props || []
    const defaultProps = props.map(p => `${p}: 'Example ${p}'`).join(',\n    ')

    return `import type { Meta, StoryObj } from '@storybook/react'
import { ${template.name} } from './${template.name}'

const meta: Meta<typeof ${template.name}> = {
  title: 'Components/${template.name}',
  component: ${template.name},
  tags: ['autodocs']
}

export default meta
type Story = StoryObj<typeof ${template.name}>

export const Default: Story = {
  args: {
    ${defaultProps}
  }
}

${props.length > 0 ? `
export const WithCustomProps: Story = {
  args: {
    ${props.map(p => `${p}: 'Custom ${p}'`).join(',\n    ')}
  }
}
` : ''}
`
  }
}

// 使用示例
async function main() {
  const generator = new CodeTemplateGenerator()

  // 生成函数组件
  await generator.generateReactComponent({
    name: 'UserProfile',
    type: 'function',
    props: ['username', 'email', 'avatar']
  })

  // 生成类组件
  await generator.generateReactComponent({
    name: 'DataTable',
    type: 'class',
    props: ['data', 'columns'],
    methods: ['handleSort', 'handleFilter']
  })
}

main()
```

---

## 最佳实践总结

### 1. 原子写入

```typescript
// ✅ 好的做法：使用临时文件
const tempPath = `${filePath}.tmp`
fs.writeFileSync(tempPath, content)
fs.renameSync(tempPath, filePath)
```

### 2. 目录检查

```typescript
// ✅ 好的做法：确保目录存在
const dir = path.dirname(filePath)
if (!fs.existsSync(dir)) {
  fs.mkdirSync(dir, { recursive: true })
}
fs.writeFileSync(filePath, content)
```

### 3. 错误处理

```typescript
// ✅ 好的做法
try {
  fs.writeFileSync(filePath, content)
  console.log(`✅ Created: ${filePath}`)
} catch (error) {
  console.error(`❌ Failed to create ${filePath}:`, error.message)
}
```

---

**版本：** v1.0
**最后更新：** 2026-02-19
**维护者：** Claude Code
