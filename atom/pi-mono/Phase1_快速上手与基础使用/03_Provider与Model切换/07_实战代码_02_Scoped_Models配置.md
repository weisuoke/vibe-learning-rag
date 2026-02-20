# 实战代码 02：Scoped Models 配置

> **快速切换配置与自动化测试**

---

## 基础配置

### 最小配置

```json
// ~/.pi/agent/settings.json
{
  "scopedModels": [
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "claude-opus-4-20250514"
  ]
}
```

### 推荐配置

```json
// ~/.pi/agent/settings.json
{
  "defaultModel": "claude-3-5-haiku-20241022",
  "scopedModels": [
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "claude-opus-4-20250514"
  ],
  "maxTokens": 4096,
  "temperature": 0.7
}
```

---

## 场景化配置

### 前端项目

```json
// project-frontend/.pi/settings.json
{
  "defaultModel": "claude-3-5-haiku-20241022",
  "scopedModels": [
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "gpt-4o"
  ]
}
```

### 后端项目

```json
// project-backend/.pi/settings.json
{
  "defaultModel": "claude-3-5-haiku-20241022",
  "scopedModels": [
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "claude-opus-4-20250514"
  ]
}
```

### 本地开发

```json
// project-local/.pi/settings.json
{
  "defaultModel": "llama3.1:8b",
  "scopedModels": [
    "llama3.1:8b",
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022"
  ]
}
```

---

## 配置脚本

### 交互式配置

```bash
#!/bin/bash
# setup-scoped-models.sh

echo "🚀 Scoped Models Configuration"
echo ""

# 选择配置类型
echo "Select configuration type:"
echo "1) Frontend development"
echo "2) Backend development"
echo "3) Local development"
echo "4) Custom"
read -p "Enter choice [1-4]: " choice

case $choice in
  1)
    MODELS='["claude-3-5-haiku-20241022","claude-3-5-sonnet-20241022","gpt-4o"]'
    ;;
  2)
    MODELS='["claude-3-5-haiku-20241022","claude-3-5-sonnet-20241022","claude-opus-4-20250514"]'
    ;;
  3)
    MODELS='["llama3.1:8b","claude-3-5-haiku-20241022","claude-3-5-sonnet-20241022"]'
    ;;
  4)
    echo "Enter models (comma-separated):"
    read -p "> " custom_models
    MODELS="[\"$(echo $custom_models | sed 's/,/","/g')\"]"
    ;;
esac

# 创建配置
mkdir -p .pi

cat > .pi/settings.json <<EOF
{
  "defaultModel": "claude-3-5-haiku-20241022",
  "scopedModels": $MODELS
}
EOF

echo "✅ Scoped Models configured in .pi/settings.json"
echo ""
echo "Test with:"
echo "  pi"
echo "  Press Ctrl+P to cycle through models"
```

### 批量配置

```bash
#!/bin/bash
# batch-setup.sh

# 为多个项目配置 Scoped Models

PROJECTS=(
  "project-a:frontend"
  "project-b:backend"
  "project-c:local"
)

for project in "${PROJECTS[@]}"; do
  IFS=':' read -r dir type <<< "$project"

  echo "Configuring $dir ($type)..."

  mkdir -p "$dir/.pi"

  case $type in
    frontend)
      MODELS='["claude-3-5-haiku-20241022","claude-3-5-sonnet-20241022","gpt-4o"]'
      ;;
    backend)
      MODELS='["claude-3-5-haiku-20241022","claude-3-5-sonnet-20241022","claude-opus-4-20250514"]'
      ;;
    local)
      MODELS='["llama3.1:8b","claude-3-5-haiku-20241022","claude-3-5-sonnet-20241022"]'
      ;;
  esac

  cat > "$dir/.pi/settings.json" <<EOF
{
  "defaultModel": "claude-3-5-haiku-20241022",
  "scopedModels": $MODELS
}
EOF

  echo "✅ $dir configured"
done

echo ""
echo "All projects configured!"
```

---

## 测试脚本

### 基础测试

```bash
#!/bin/bash
# test-scoped-models.sh

echo "Testing Scoped Models..."

# 检查配置文件
if [ ! -f .pi/settings.json ]; then
  echo "❌ .pi/settings.json not found"
  exit 1
fi

# 验证 JSON 语法
if ! jq . .pi/settings.json > /dev/null 2>&1; then
  echo "❌ Invalid JSON syntax"
  exit 1
fi

# 读取配置
SCOPED_MODELS=$(jq -r '.scopedModels[]' .pi/settings.json)
MODEL_COUNT=$(echo "$SCOPED_MODELS" | wc -l | tr -d ' ')

echo "✅ Found $MODEL_COUNT scoped models:"
echo "$SCOPED_MODELS" | sed 's/^/  - /'

# 验证模型数量
if [ "$MODEL_COUNT" -lt 2 ]; then
  echo "⚠️  Warning: Less than 2 models configured"
elif [ "$MODEL_COUNT" -gt 5 ]; then
  echo "⚠️  Warning: More than 5 models (recommended: 3-5)"
fi

echo ""
echo "Test manually:"
echo "  pi"
echo "  Press Ctrl+P to cycle through models"
```

### 自动化测试

```typescript
// test-scoped-models.ts
import { readFileSync } from 'fs';
import { join } from 'path';

interface Settings {
  defaultModel?: string;
  scopedModels?: string[];
}

function testScopedModels() {
  console.log('Testing Scoped Models configuration...\n');

  // 读取配置
  const settingsPath = join(process.cwd(), '.pi/settings.json');
  let settings: Settings;

  try {
    settings = JSON.parse(readFileSync(settingsPath, 'utf-8'));
  } catch (error) {
    console.error('❌ Failed to read .pi/settings.json');
    process.exit(1);
  }

  // 验证 scopedModels
  if (!settings.scopedModels || !Array.isArray(settings.scopedModels)) {
    console.error('❌ scopedModels not found or invalid');
    process.exit(1);
  }

  const modelCount = settings.scopedModels.length;
  console.log(`✅ Found ${modelCount} scoped models:`);
  settings.scopedModels.forEach((model, i) => {
    console.log(`  ${i + 1}. ${model}`);
  });

  // 验证数量
  if (modelCount < 2) {
    console.warn('\n⚠️  Warning: Less than 2 models configured');
  } else if (modelCount > 5) {
    console.warn('\n⚠️  Warning: More than 5 models (recommended: 3-5)');
  }

  // 验证默认模型
  if (settings.defaultModel) {
    if (settings.scopedModels.includes(settings.defaultModel)) {
      console.log(`\n✅ Default model is in scoped models: ${settings.defaultModel}`);
    } else {
      console.warn(`\n⚠️  Default model not in scoped models: ${settings.defaultModel}`);
    }
  }

  console.log('\n✅ Configuration is valid');
}

testScopedModels();
```

---

## 项目模板

### 创建模板

```bash
#!/bin/bash
# create-template.sh

TEMPLATE_DIR="pi-project-template"

echo "Creating Pi project template..."

mkdir -p "$TEMPLATE_DIR/.pi"

# 配置文件
cat > "$TEMPLATE_DIR/.pi/settings.json" <<'EOF'
{
  "defaultModel": "claude-3-5-haiku-20241022",
  "scopedModels": [
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "claude-opus-4-20250514"
  ],
  "maxTokens": 4096
}
EOF

# README
cat > "$TEMPLATE_DIR/.pi/README.md" <<'EOF'
# Pi Configuration

## Scoped Models

- **Haiku**: Fast, cheap ($0.8/MTok)
- **Sonnet**: Balanced ($3/MTok)
- **Opus**: Powerful ($15/MTok)

## Usage

```bash
# Start Pi
pi

# Cycle through models
Ctrl+P

# View current model
/session
```
EOF

# .gitignore
cat > "$TEMPLATE_DIR/.gitignore" <<'EOF'
.pi/auth.json
EOF

echo "✅ Template created at $TEMPLATE_DIR"
echo ""
echo "Use template:"
echo "  cp -r $TEMPLATE_DIR/.pi new-project/"
```

### 使用模板

```bash
#!/bin/bash
# use-template.sh

PROJECT_NAME=$1

if [ -z "$PROJECT_NAME" ]; then
  echo "Usage: $0 <project-name>"
  exit 1
fi

echo "Creating project: $PROJECT_NAME"

# 复制模板
cp -r pi-project-template/.pi "$PROJECT_NAME/"

echo "✅ Project created: $PROJECT_NAME"
echo ""
echo "Next steps:"
echo "  cd $PROJECT_NAME"
echo "  pi"
```

---

## 动态配置

### 基于环境

```typescript
// config-by-env.ts
import { writeFileSync } from 'fs';
import { join } from 'path';

const env = process.env.NODE_ENV || 'development';

const configs = {
  development: {
    defaultModel: 'llama3.1:8b',
    scopedModels: [
      'llama3.1:8b',
      'claude-3-5-haiku-20241022',
      'claude-3-5-sonnet-20241022'
    ]
  },
  staging: {
    defaultModel: 'claude-3-5-haiku-20241022',
    scopedModels: [
      'claude-3-5-haiku-20241022',
      'claude-3-5-sonnet-20241022'
    ]
  },
  production: {
    defaultModel: 'claude-3-5-sonnet-20241022',
    scopedModels: [
      'claude-3-5-sonnet-20241022',
      'claude-opus-4-20250514'
    ]
  }
};

const config = configs[env];
const settingsPath = join(process.cwd(), '.pi/settings.json');

writeFileSync(settingsPath, JSON.stringify(config, null, 2));

console.log(`✅ Configured for ${env} environment`);
```

### 基于项目类型

```typescript
// config-by-type.ts
import { writeFileSync } from 'fs';
import { join } from 'path';

const projectType = process.argv[2] || 'general';

const configs = {
  frontend: {
    scopedModels: [
      'claude-3-5-haiku-20241022',
      'claude-3-5-sonnet-20241022',
      'gpt-4o'
    ]
  },
  backend: {
    scopedModels: [
      'claude-3-5-haiku-20241022',
      'claude-3-5-sonnet-20241022',
      'claude-opus-4-20250514'
    ]
  },
  fullstack: {
    scopedModels: [
      'claude-3-5-haiku-20241022',
      'claude-3-5-sonnet-20241022',
      'gpt-4o',
      'claude-opus-4-20250514'
    ]
  },
  general: {
    scopedModels: [
      'claude-3-5-haiku-20241022',
      'claude-3-5-sonnet-20241022',
      'claude-opus-4-20250514'
    ]
  }
};

const config = configs[projectType];
const settingsPath = join(process.cwd(), '.pi/settings.json');

writeFileSync(settingsPath, JSON.stringify(config, null, 2));

console.log(`✅ Configured for ${projectType} project`);
```

---

## 验证工具

### 配置验证器

```typescript
// validate-config.ts
import { readFileSync } from 'fs';
import { join } from 'path';

interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

function validateScopedModels(): ValidationResult {
  const result: ValidationResult = {
    valid: true,
    errors: [],
    warnings: []
  };

  // 读取配置
  const settingsPath = join(process.cwd(), '.pi/settings.json');
  let settings: any;

  try {
    settings = JSON.parse(readFileSync(settingsPath, 'utf-8'));
  } catch (error) {
    result.valid = false;
    result.errors.push('Failed to read or parse .pi/settings.json');
    return result;
  }

  // 验证 scopedModels 存在
  if (!settings.scopedModels) {
    result.valid = false;
    result.errors.push('scopedModels not found');
    return result;
  }

  // 验证是数组
  if (!Array.isArray(settings.scopedModels)) {
    result.valid = false;
    result.errors.push('scopedModels must be an array');
    return result;
  }

  // 验证数量
  const count = settings.scopedModels.length;
  if (count < 2) {
    result.warnings.push('Less than 2 models (recommended: 3-5)');
  } else if (count > 5) {
    result.warnings.push('More than 5 models (recommended: 3-5)');
  }

  // 验证模型 ID 格式
  settings.scopedModels.forEach((model: string, i: number) => {
    if (typeof model !== 'string' || model.trim() === '') {
      result.errors.push(`Model at index ${i} is invalid`);
      result.valid = false;
    }
  });

  // 验证默认模型
  if (settings.defaultModel) {
    if (!settings.scopedModels.includes(settings.defaultModel)) {
      result.warnings.push('Default model not in scoped models');
    }
  }

  return result;
}

// 运行验证
const result = validateScopedModels();

console.log('Validation Results:\n');

if (result.errors.length > 0) {
  console.log('❌ Errors:');
  result.errors.forEach(err => console.log(`  - ${err}`));
}

if (result.warnings.length > 0) {
  console.log('\n⚠️  Warnings:');
  result.warnings.forEach(warn => console.log(`  - ${warn}`));
}

if (result.valid && result.errors.length === 0) {
  console.log('✅ Configuration is valid');
}

process.exit(result.valid ? 0 : 1);
```

---

## 完整示例

```bash
#!/bin/bash
# complete-setup.sh

set -e

echo "🚀 Complete Scoped Models Setup"
echo ""

# 1. 创建配置
echo "1. Creating configuration..."
mkdir -p .pi

cat > .pi/settings.json <<'EOF'
{
  "defaultModel": "claude-3-5-haiku-20241022",
  "scopedModels": [
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "claude-opus-4-20250514"
  ]
}
EOF

# 2. 验证配置
echo "2. Validating configuration..."
if ! jq . .pi/settings.json > /dev/null 2>&1; then
  echo "❌ Invalid JSON"
  exit 1
fi

# 3. 测试
echo "3. Testing..."
MODEL_COUNT=$(jq -r '.scopedModels | length' .pi/settings.json)
echo "   Found $MODEL_COUNT models"

# 4. 完成
echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  pi                 # Start Pi"
echo "  Ctrl+P             # Cycle through models"
echo "  /scoped-models     # Manage models"
```

---

**记住**：Scoped Models 是效率工具，3-5 个模型 + Ctrl+P = 零中断切换。
