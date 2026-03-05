# 实战代码：场景7 - CI/CD 集成

> GitHub Actions 自动化构建与部署实战

---

## 场景描述

**目标**：配置 GitHub Actions 实现自动化构建、测试和部署。

**时间**：约 40 分钟

**前置条件**：
- GitHub 仓库
- 已完成场景1-6

---

## 步骤 1：基础 CI 配置

### 1.1 创建工作流文件

**文件**：`.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint-and-typecheck:
    name: Lint and Type Check
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '22'
          cache: 'pnpm'

      - name: Install pnpm
        run: npm install -g pnpm@10.23.0

      - name: Install dependencies
        run: pnpm install --frozen-lockfile

      - name: Run format check
        run: pnpm format:check

      - name: Run type check
        run: pnpm tsgo

      - name: Run lint
        run: pnpm lint

  test:
    name: Test
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '22'
          cache: 'pnpm'

      - name: Install pnpm
        run: npm install -g pnpm@10.23.0

      - name: Install dependencies
        run: pnpm install --frozen-lockfile

      - name: Run tests
        run: pnpm test:fast

      - name: Generate coverage
        run: pnpm test:coverage

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage/coverage-final.json
          flags: unittests
          name: codecov-umbrella

  build:
    name: Build
    runs-on: ubuntu-latest
    needs: [lint-and-typecheck, test]

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '22'
          cache: 'pnpm'

      - name: Install pnpm
        run: npm install -g pnpm@10.23.0

      - name: Install dependencies
        run: pnpm install --frozen-lockfile

      - name: Build project
        run: NODE_ENV=production pnpm build

      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/
          retention-days: 7
```

---

### 1.2 配置缓存优化

```yaml
jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '22'
          cache: 'pnpm'

      - name: Cache pnpm store
        uses: actions/cache@v4
        with:
          path: ~/.pnpm-store
          key: ${{ runner.os }}-pnpm-${{ hashFiles('**/pnpm-lock.yaml') }}
          restore-keys: |
            ${{ runner.os }}-pnpm-

      - name: Cache tsdown
        uses: actions/cache@v4
        with:
          path: .tsdown-cache
          key: ${{ runner.os }}-tsdown-${{ hashFiles('src/**/*.ts') }}
          restore-keys: |
            ${{ runner.os }}-tsdown-

      - name: Install dependencies
        run: pnpm install --frozen-lockfile

      - name: Build
        run: pnpm build
```

---

## 步骤 2：多平台测试

### 2.1 跨平台矩阵

```yaml
name: Cross-Platform CI

on: [push, pull_request]

jobs:
  test:
    name: Test on ${{ matrix.os }} with Node ${{ matrix.node }}
    runs-on: ${{ matrix.os }}

    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        node: ['22']

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node }}

      - name: Install pnpm
        run: npm install -g pnpm@10.23.0

      - name: Install dependencies
        run: pnpm install --frozen-lockfile

      - name: Run tests
        run: pnpm test:fast

      - name: Build
        run: pnpm build
```

---

### 2.2 原生依赖编译

```yaml
jobs:
  build-native:
    name: Build Native Dependencies
    runs-on: ${{ matrix.os }}

    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '22'

      - name: Install build tools (Ubuntu)
        if: runner.os == 'Linux'
        run: |
          sudo apt-get update
          sudo apt-get install -y build-essential python3

      - name: Install build tools (macOS)
        if: runner.os == 'macOS'
        run: |
          xcode-select --install || true

      - name: Install build tools (Windows)
        if: runner.os == 'Windows'
        run: |
          npm install --global windows-build-tools

      - name: Install pnpm
        run: npm install -g pnpm@10.23.0

      - name: Install dependencies
        run: pnpm install --frozen-lockfile

      - name: Build
        run: pnpm build
```

---

## 步骤 3：Docker 构建与发布

### 3.1 Docker 构建工作流

**文件**：`.github/workflows/docker.yml`

```yaml
name: Docker Build and Push

on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    name: Build and Push Docker Image
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

### 3.2 多架构构建

```yaml
jobs:
  build-multi-arch:
    name: Build Multi-Architecture Image
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push multi-arch
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

## 步骤 4：自动化部署

### 4.1 部署到生产环境

**文件**：`.github/workflows/deploy.yml`

```yaml
name: Deploy to Production

on:
  push:
    tags: ['v*']

jobs:
  deploy:
    name: Deploy
    runs-on: ubuntu-latest
    environment: production

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '22'

      - name: Install pnpm
        run: npm install -g pnpm@10.23.0

      - name: Install dependencies
        run: pnpm install --frozen-lockfile

      - name: Build
        run: NODE_ENV=production pnpm build

      - name: Deploy to server
        uses: appleboy/ssh-action@v1.0.0
        with:
          host: ${{ secrets.DEPLOY_HOST }}
          username: ${{ secrets.DEPLOY_USER }}
          key: ${{ secrets.DEPLOY_KEY }}
          script: |
            cd /opt/openclaw
            git pull origin main
            pnpm install --frozen-lockfile
            pnpm build
            pm2 restart openclaw

      - name: Health check
        run: |
          sleep 10
          curl -f ${{ secrets.DEPLOY_URL }}/health || exit 1
```

---

### 4.2 蓝绿部署

```yaml
jobs:
  blue-green-deploy:
    name: Blue-Green Deployment
    runs-on: ubuntu-latest

    steps:
      - name: Deploy to blue environment
        run: |
          docker-compose -f docker-compose.blue.yml up -d

      - name: Health check blue
        run: |
          sleep 10
          curl -f http://blue.openclaw.dev/health || exit 1

      - name: Switch traffic to blue
        run: |
          # 更新负载均衡器配置
          kubectl set image deployment/openclaw openclaw=openclaw:blue

      - name: Stop green environment
        run: |
          docker-compose -f docker-compose.green.yml down
```

---

## 步骤 5：发布管理

### 5.1 自动创建 Release

**文件**：`.github/workflows/release.yml`

```yaml
name: Release

on:
  push:
    tags: ['v*']

jobs:
  create-release:
    name: Create Release
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Generate changelog
        id: changelog
        uses: metcalfc/changelog-generator@v4.1.0
        with:
          myToken: ${{ secrets.GITHUB_TOKEN }}

      - name: Create Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
          body: ${{ steps.changelog.outputs.changelog }}
          draft: false
          prerelease: false

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '22'

      - name: Install pnpm
        run: npm install -g pnpm@10.23.0

      - name: Install dependencies
        run: pnpm install --frozen-lockfile

      - name: Build
        run: NODE_ENV=production pnpm build

      - name: Package artifacts
        run: |
          tar -czf openclaw-${{ github.ref_name }}.tar.gz dist/

      - name: Upload Release Asset
        uses: actions/upload-release-asset@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          upload_url: ${{ steps.create_release.outputs.upload_url }}
          asset_path: ./openclaw-${{ github.ref_name }}.tar.gz
          asset_name: openclaw-${{ github.ref_name }}.tar.gz
          asset_content_type: application/gzip
```

---

### 5.2 自动更新版本号

```yaml
jobs:
  bump-version:
    name: Bump Version
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.PAT }}

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '22'

      - name: Install pnpm
        run: npm install -g pnpm@10.23.0

      - name: Bump version
        run: |
          pnpm version patch -m "chore: bump version to %s"

      - name: Push changes
        run: |
          git push origin main --tags
```

---

## 步骤 6：性能监控

### 6.1 构建时间监控

```yaml
jobs:
  build-with-metrics:
    name: Build with Metrics
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '22'

      - name: Install pnpm
        run: npm install -g pnpm@10.23.0

      - name: Install dependencies
        id: install
        run: |
          START_TIME=$(date +%s)
          pnpm install --frozen-lockfile
          END_TIME=$(date +%s)
          echo "duration=$((END_TIME - START_TIME))" >> $GITHUB_OUTPUT

      - name: Build
        id: build
        run: |
          START_TIME=$(date +%s)
          pnpm build
          END_TIME=$(date +%s)
          echo "duration=$((END_TIME - START_TIME))" >> $GITHUB_OUTPUT

      - name: Report metrics
        run: |
          echo "Install duration: ${{ steps.install.outputs.duration }}s"
          echo "Build duration: ${{ steps.build.outputs.duration }}s"

      - name: Send metrics to monitoring
        run: |
          curl -X POST ${{ secrets.METRICS_URL }} \
            -H "Content-Type: application/json" \
            -d '{
              "install_duration": ${{ steps.install.outputs.duration }},
              "build_duration": ${{ steps.build.outputs.duration }},
              "commit": "${{ github.sha }}"
            }'
```

---

### 6.2 测试覆盖率趋势

```yaml
jobs:
  coverage-trend:
    name: Coverage Trend
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '22'

      - name: Install pnpm
        run: npm install -g pnpm@10.23.0

      - name: Install dependencies
        run: pnpm install --frozen-lockfile

      - name: Generate coverage
        run: pnpm test:coverage

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage/coverage-final.json

      - name: Coverage Report
        uses: romeovs/lcov-reporter-action@v0.3.1
        with:
          lcov-file: ./coverage/lcov.info
          github-token: ${{ secrets.GITHUB_TOKEN }}
```

---

## 步骤 7：安全扫描

### 7.1 依赖安全扫描

```yaml
name: Security Scan

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # 每周日运行

jobs:
  dependency-scan:
    name: Dependency Security Scan
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Run Snyk
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high

      - name: Run npm audit
        run: npm audit --audit-level=high

      - name: Run pnpm audit
        run: pnpm audit --audit-level=high
```

---

### 7.2 代码安全扫描

```yaml
jobs:
  code-scan:
    name: Code Security Scan
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Run CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: javascript, typescript

      - name: Autobuild
        uses: github/codeql-action/autobuild@v3

      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v3
```

---

## 步骤 8：通知集成

### 8.1 Slack 通知

```yaml
jobs:
  notify:
    name: Notify
    runs-on: ubuntu-latest
    if: always()
    needs: [build, test, deploy]

    steps:
      - name: Send Slack notification
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: |
            Build: ${{ needs.build.result }}
            Test: ${{ needs.test.result }}
            Deploy: ${{ needs.deploy.result }}
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
          fields: repo,message,commit,author,action,eventName,ref,workflow
```

---

### 8.2 Email 通知

```yaml
jobs:
  email-notify:
    name: Email Notification
    runs-on: ubuntu-latest
    if: failure()

    steps:
      - name: Send email
        uses: dawidd6/action-send-mail@v3
        with:
          server_address: smtp.gmail.com
          server_port: 465
          username: ${{ secrets.EMAIL_USERNAME }}
          password: ${{ secrets.EMAIL_PASSWORD }}
          subject: CI/CD Failed - ${{ github.repository }}
          body: |
            Build failed for commit ${{ github.sha }}

            Workflow: ${{ github.workflow }}
            Branch: ${{ github.ref }}
            Author: ${{ github.actor }}

            View details: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}
          to: team@openclaw.dev
          from: ci@openclaw.dev
```

---

## 完整 CI/CD 流程示例

### 综合工作流

**文件**：`.github/workflows/main.yml`

```yaml
name: Main CI/CD

on:
  push:
    branches: [main, develop]
    tags: ['v*']
  pull_request:
    branches: [main]

env:
  NODE_VERSION: '22'
  PNPM_VERSION: '10.23.0'

jobs:
  # 阶段 1: 代码质量检查
  quality:
    name: Code Quality
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'pnpm'

      - name: Install pnpm
        run: npm install -g pnpm@${{ env.PNPM_VERSION }}

      - name: Install dependencies
        run: pnpm install --frozen-lockfile

      - name: Format check
        run: pnpm format:check

      - name: Type check
        run: pnpm tsgo

      - name: Lint
        run: pnpm lint

  # 阶段 2: 测试
  test:
    name: Test
    runs-on: ${{ matrix.os }}
    needs: quality

    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}

      - name: Install pnpm
        run: npm install -g pnpm@${{ env.PNPM_VERSION }}

      - name: Install dependencies
        run: pnpm install --frozen-lockfile

      - name: Run tests
        run: pnpm test:fast

      - name: Generate coverage
        if: matrix.os == 'ubuntu-latest'
        run: pnpm test:coverage

      - name: Upload coverage
        if: matrix.os == 'ubuntu-latest'
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage/coverage-final.json

  # 阶段 3: 构建
  build:
    name: Build
    runs-on: ubuntu-latest
    needs: test

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'pnpm'

      - name: Install pnpm
        run: npm install -g pnpm@${{ env.PNPM_VERSION }}

      - name: Install dependencies
        run: pnpm install --frozen-lockfile

      - name: Build
        run: NODE_ENV=production pnpm build

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  # 阶段 4: Docker 构建
  docker:
    name: Docker Build
    runs-on: ubuntu-latest
    needs: build
    if: github.event_name == 'push' && (github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/tags/v'))

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # 阶段 5: 部署
  deploy:
    name: Deploy
    runs-on: ubuntu-latest
    needs: docker
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    environment: production

    steps:
      - uses: actions/checkout@v4

      - name: Deploy to production
        uses: appleboy/ssh-action@v1.0.0
        with:
          host: ${{ secrets.DEPLOY_HOST }}
          username: ${{ secrets.DEPLOY_USER }}
          key: ${{ secrets.DEPLOY_KEY }}
          script: |
            cd /opt/openclaw
            docker-compose pull
            docker-compose up -d
            docker-compose ps

      - name: Health check
        run: |
          sleep 10
          curl -f ${{ secrets.DEPLOY_URL }}/health || exit 1

  # 阶段 6: 通知
  notify:
    name: Notify
    runs-on: ubuntu-latest
    if: always()
    needs: [quality, test, build, docker, deploy]

    steps:
      - name: Send notification
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

---

## 故障排查

### 问题 1：缓存失效

**症状**：
```
Cache not found for input keys: ubuntu-latest-pnpm-...
```

**解决方案**：
```yaml
- name: Cache pnpm store
  uses: actions/cache@v4
  with:
    path: ~/.pnpm-store
    key: ${{ runner.os }}-pnpm-${{ hashFiles('**/pnpm-lock.yaml') }}
    restore-keys: |
      ${{ runner.os }}-pnpm-
```

---

### 问题 2：权限错误

**症状**：
```
Error: Resource not accessible by integration
```

**解决方案**：
```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
```

---

### 问题 3：超时

**症状**：
```
Error: The operation was canceled.
```

**解决方案**：
```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30  # 增加超时时间
```

---

## 核心洞察

### 1. CI/CD 不是简单的自动化

**表面**：配置 GitHub Actions

**实际**：
- 代码质量门禁
- 自动化测试
- 多平台验证
- 安全扫描
- 自动部署
- 监控告警

---

### 2. 缓存是性能关键

**表面**：每次都重新安装

**实际**：
- pnpm store 缓存
- tsdown 缓存
- Docker layer 缓存
- 减少构建时间 50-80%

---

### 3. 安全扫描很重要

**表面**：代码能跑就行

**实际**：
- 依赖漏洞扫描
- 代码安全分析
- 定期安全审计
- 自动修复建议

---

## 一句话总结

**OpenClaw CI/CD 集成的关键是：配置 GitHub Actions 实现代码质量检查、自动化测试、Docker 构建、安全扫描和自动部署。**

---

[来源: reference/13_github_actions_optimization.md, reference/22_cross_platform_paths.md, 基于 CI/CD 实践经验]
