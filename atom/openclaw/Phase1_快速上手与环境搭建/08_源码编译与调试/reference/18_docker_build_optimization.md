# Docker 多阶段构建优化

> 来源: Grok Web Search
> 查询时间: 2026-02-24

## 搜索结果

### 1. Docker多阶段构建优化Node.js镜像
**URL**: https://oneuptime.com/blog/post/2026-02-20-docker-multi-stage-builds/view
**描述**: 2026最新教程,使用多阶段构建将Node.js TypeScript应用镜像从1GB以上缩小至100MB以下,加快部署速度并降低安全风险。

### 2. Docker官方Node.js容器化指南
**URL**: https://docs.docker.com/guides/nodejs/containerize/
**描述**: 官方文档讲解TypeScript全栈Node.js应用的多阶段Dockerfile,分离依赖构建与生产环境,优化镜像大小和运行安全性。

### 3. TypeScript项目多阶段Docker构建优化
**URL**: https://arnab-k.medium.com/creating-multi-stage-docker-builds-for-optimization-36968b5dfc48
**描述**: 针对TypeScript Node.js项目的多阶段构建指南,通过分离构建和运行阶段显著减小镜像体积,提升部署效率。

## 最佳实践

### Dockerfile 示例
```dockerfile
# 阶段1: 构建
FROM node:22-alpine AS builder
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN npm install -g pnpm && pnpm install --frozen-lockfile
COPY . .
RUN pnpm build

# 阶段2: 生产
FROM node:22-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY package.json ./
EXPOSE 3000
CMD ["node", "dist/index.js"]
```

**文档版本**: 基于 2026-02-24 搜索结果
