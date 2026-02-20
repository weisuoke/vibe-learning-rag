# Session 管理与分支 - 核心概念 04：Session 命名与信息管理

> 掌握 /name、/session 命令，实现会话的组织和元数据管理

---

## 概述

有效的 Session 命名和信息管理能够：
- 快速识别会话内容
- 方便跨项目组织
- 提升团队协作效率
- 支持长期项目管理

---

## 1. /name - 会话命名

### 1.1 基本用法

```bash
# 命名当前会话
/name my-project

# 命名时使用描述性名称
/name feature-user-authentication
/name bugfix-login-error
/name exp-new-algorithm
```

### 1.2 工作原理

```typescript
class SessionNaming {
  async nameSession(sessionId: string, name: string): Promise<void> {
    // 1. 验证名称
    const validName = this.validateName(name);

    // 2. 追加命名元数据
    await this.storage.append({
      id: this.generateId(),
      type: 'metadata',
      action: 'rename',
      name: validName,
      timestamp: Date.now()
    });

    // 3. 可选：重命名文件
    const oldPath = path.join(this.sessionsDir, `${sessionId}.jsonl`);
    const newPath = path.join(this.sessionsDir, `${validName}.jsonl`);
    await fs.rename(oldPath, newPath);
  }

  private validateName(name: string): string {
    // 移除非法字符
    return name
      .replace(/[^a-zA-Z0-9-_]/g, '-')
      .toLowerCase()
      .substring(0, 50);
  }
}
```

### 1.3 命名规范

**推荐的命名模式：**

```typescript
const namingPatterns = {
  // 功能开发
  feature: 'feature-{name}',
  // 示例: feature-user-login, feature-payment-api

  // Bug 修复
  bugfix: 'bugfix-{issue}',
  // 示例: bugfix-auth-error, bugfix-memory-leak

  // 重构
  refactor: 'refactor-{component}',
  // 示例: refactor-database-layer, refactor-api-routes

  // 实验
  experiment: 'exp-{idea}',
  // 示例: exp-new-caching, exp-graphql-migration

  // 检查点
  checkpoint: 'checkpoint-{milestone}',
  // 示例: checkpoint-v1.0, checkpoint-beta

  // 团队协作
  team: 'team-{project}-{member}',
  // 示例: team-backend-alice, team-frontend-bob
};
```

---

## 2. /session - 查看会话信息

### 2.1 基本用法

```bash
# 查看当前会话信息
/session

# 输出示例：
Session Information:
  ID: abc123
  Name: feature-user-login
  Created: 2026-02-18 10:00:00
  Last Modified: 2026-02-18 14:30:00
  Entries: 45
  Size: 128 KB
  Current Node: 45
```

### 2.2 工作原理

```typescript
interface SessionInfo {
  id: string;
  name?: string;
  created: number;
  lastModified: number;
  entryCount: number;
  size: number;
  currentNodeId: string;
  branches: number;
  metadata: Record<string, any>;
}

class SessionInfoManager {
  async getSessionInfo(sessionId: string): Promise<SessionInfo> {
    // 1. 读取文件统计信息
    const filePath = path.join(this.sessionsDir, `${sessionId}.jsonl`);
    const stats = await fs.stat(filePath);

    // 2. 读取所有条目
    const entries = await this.storage.readAll();

    // 3. 提取元数据
    const metadata = this.extractMetadata(entries);

    // 4. 计算分支数
    const branches = this.countBranches(entries);

    return {
      id: sessionId,
      name: metadata.name,
      created: stats.birthtime.getTime(),
      lastModified: stats.mtime.getTime(),
      entryCount: entries.length,
      size: stats.size,
      currentNodeId: entries[entries.length - 1]?.id || '',
      branches,
      metadata
    };
  }

  private extractMetadata(entries: SessionEntry[]): Record<string, any> {
    const metadataEntries = entries.filter(e => e.type === 'metadata');
    const metadata: Record<string, any> = {};

    for (const entry of metadataEntries) {
      Object.assign(metadata, entry);
    }

    return metadata;
  }

  private countBranches(entries: SessionEntry[]): number {
    const childrenCount = new Map<string, number>();

    for (const entry of entries) {
      if (entry.parentId) {
        const count = childrenCount.get(entry.parentId) || 0;
        childrenCount.set(entry.parentId, count + 1);
      }
    }

    // 统计有多个子节点的节点数（分支点）
    return Array.from(childrenCount.values()).filter(count => count > 1).length;
  }
}
```

---

## 3. Session 元数据

### 3.1 元数据类型

```typescript
interface SessionMetadata {
  // 基础信息
  name?: string;
  description?: string;
  tags?: string[];

  // 项目信息
  project?: string;
  repository?: string;
  branch?: string;

  // 团队信息
  owner?: string;
  collaborators?: string[];

  // 状态信息
  status?: 'active' | 'archived' | 'completed';
  priority?: 'low' | 'medium' | 'high';

  // 自定义字段
  [key: string]: any;
}
```

### 3.2 添加元数据

```typescript
class MetadataManager {
  async addMetadata(
    sessionId: string,
    metadata: Partial<SessionMetadata>
  ): Promise<void> {
    await this.storage.append({
      id: this.generateId(),
      type: 'metadata',
      ...metadata,
      timestamp: Date.now()
    });
  }

  // 使用示例
  async setupProjectSession() {
    await this.addMetadata('abc123', {
      name: 'feature-user-auth',
      description: '实现用户认证系统',
      tags: ['authentication', 'security', 'backend'],
      project: 'my-app',
      repository: 'github.com/user/my-app',
      owner: 'alice',
      status: 'active',
      priority: 'high'
    });
  }
}
```

---

## 4. 会话组织策略

### 4.1 按项目组织

```typescript
class ProjectOrganizer {
  async listByProject(): Promise<Map<string, SessionInfo[]>> {
    const sessions = await this.sessionManager.listSessions();
    const grouped = new Map<string, SessionInfo[]>();

    for (const session of sessions) {
      const project = session.metadata.project || 'uncategorized';

      if (!grouped.has(project)) {
        grouped.set(project, []);
      }

      grouped.get(project)!.push(session);
    }

    return grouped;
  }

  // 使用示例
  async showProjectSessions() {
    const projects = await this.listByProject();

    for (const [project, sessions] of projects) {
      console.log(`\n项目: ${project}`);
      console.log(`会话数: ${sessions.length}`);

      for (const session of sessions) {
        console.log(`  - ${session.name} (${session.entryCount} entries)`);
      }
    }
  }
}
```

### 4.2 按标签组织

```typescript
class TagOrganizer {
  async listByTag(): Promise<Map<string, SessionInfo[]>> {
    const sessions = await this.sessionManager.listSessions();
    const grouped = new Map<string, SessionInfo[]>();

    for (const session of sessions) {
      const tags = session.metadata.tags || [];

      for (const tag of tags) {
        if (!grouped.has(tag)) {
          grouped.set(tag, []);
        }

        grouped.get(tag)!.push(session);
      }
    }

    return grouped;
  }

  // 搜索标签
  async searchByTag(tag: string): Promise<SessionInfo[]> {
    const sessions = await this.sessionManager.listSessions();

    return sessions.filter(session =>
      session.metadata.tags?.includes(tag)
    );
  }
}
```

### 4.3 按状态组织

```typescript
class StatusOrganizer {
  async listByStatus(): Promise<Map<string, SessionInfo[]>> {
    const sessions = await this.sessionManager.listSessions();
    const grouped = new Map<string, SessionInfo[]>();

    for (const session of sessions) {
      const status = session.metadata.status || 'active';

      if (!grouped.has(status)) {
        grouped.set(status, []);
      }

      grouped.get(status)!.push(session);
    }

    return grouped;
  }

  // 归档旧会话
  async archiveOldSessions(olderThanDays: number): Promise<number> {
    const sessions = await this.sessionManager.listSessions();
    const cutoffTime = Date.now() - olderThanDays * 24 * 60 * 60 * 1000;

    let archivedCount = 0;

    for (const session of sessions) {
      if (session.lastModified < cutoffTime && session.metadata.status !== 'archived') {
        await this.metadataManager.addMetadata(session.id, {
          status: 'archived',
          archivedAt: Date.now()
        });

        archivedCount++;
      }
    }

    return archivedCount;
  }
}
```

---

## 5. 跨项目会话管理

### 5.1 会话链接

```typescript
class SessionLinker {
  async linkSessions(
    sessionId1: string,
    sessionId2: string,
    relationship: string
  ): Promise<void> {
    // 在两个会话中都添加链接元数据
    await this.metadataManager.addMetadata(sessionId1, {
      linkedSessions: {
        [sessionId2]: relationship
      }
    });

    await this.metadataManager.addMetadata(sessionId2, {
      linkedSessions: {
        [sessionId1]: relationship
      }
    });
  }

  // 使用示例
  async setupRelatedSessions() {
    // 链接前端和后端会话
    await this.linkSessions(
      'frontend-session',
      'backend-session',
      'related-implementation'
    );

    // 链接主会话和实验会话
    await this.linkSessions(
      'main-session',
      'experiment-session',
      'experiment'
    );
  }

  // 获取相关会话
  async getRelatedSessions(sessionId: string): Promise<SessionInfo[]> {
    const session = await this.sessionManager.getSessionInfo(sessionId);
    const linkedIds = Object.keys(session.metadata.linkedSessions || {});

    const related: SessionInfo[] = [];

    for (const linkedId of linkedIds) {
      try {
        const linkedSession = await this.sessionManager.getSessionInfo(linkedId);
        related.push(linkedSession);
      } catch (err) {
        // 会话不存在，跳过
      }
    }

    return related;
  }
}
```

### 5.2 会话依赖

```typescript
class SessionDependency {
  async addDependency(
    sessionId: string,
    dependsOn: string
  ): Promise<void> {
    await this.metadataManager.addMetadata(sessionId, {
      dependencies: [dependsOn]
    });
  }

  // 检查依赖是否满足
  async checkDependencies(sessionId: string): Promise<boolean> {
    const session = await this.sessionManager.getSessionInfo(sessionId);
    const dependencies = session.metadata.dependencies || [];

    for (const depId of dependencies) {
      const depSession = await this.sessionManager.getSessionInfo(depId);

      if (depSession.metadata.status !== 'completed') {
        return false;
      }
    }

    return true;
  }

  // 获取依赖图
  async getDependencyGraph(): Promise<Map<string, string[]>> {
    const sessions = await this.sessionManager.listSessions();
    const graph = new Map<string, string[]>();

    for (const session of sessions) {
      const dependencies = session.metadata.dependencies || [];
      graph.set(session.id, dependencies);
    }

    return graph;
  }
}
```

---

## 6. 实战示例

### 示例 1：完整的项目会话设置

```typescript
async function setupProjectSessions() {
  const manager = new SessionManager();
  const metadata = new MetadataManager();

  // 1. 创建主会话
  const mainSession = await manager.createNewSession('project-main');
  await metadata.addMetadata(mainSession.id, {
    name: 'project-main',
    description: '项目主会话',
    project: 'my-app',
    tags: ['main', 'planning'],
    status: 'active',
    priority: 'high'
  });

  // 2. 创建后端会话
  const backendSession = await manager.createNewSession('project-backend');
  await metadata.addMetadata(backendSession.id, {
    name: 'project-backend',
    description: '后端开发',
    project: 'my-app',
    tags: ['backend', 'api'],
    status: 'active',
    owner: 'alice'
  });

  // 3. 创建前端会话
  const frontendSession = await manager.createNewSession('project-frontend');
  await metadata.addMetadata(frontendSession.id, {
    name: 'project-frontend',
    description: '前端开发',
    project: 'my-app',
    tags: ['frontend', 'ui'],
    status: 'active',
    owner: 'bob'
  });

  // 4. 建立链接
  const linker = new SessionLinker();
  await linker.linkSessions(mainSession.id, backendSession.id, 'implementation');
  await linker.linkSessions(mainSession.id, frontendSession.id, 'implementation');

  console.log('项目会话设置完成');
}
```

### 示例 2：会话搜索和过滤

```typescript
class SessionSearch {
  async search(criteria: {
    name?: string;
    tags?: string[];
    project?: string;
    status?: string;
    owner?: string;
  }): Promise<SessionInfo[]> {
    const sessions = await this.sessionManager.listSessions();

    return sessions.filter(session => {
      // 按名称过滤
      if (criteria.name && !session.name?.includes(criteria.name)) {
        return false;
      }

      // 按标签过滤
      if (criteria.tags) {
        const sessionTags = session.metadata.tags || [];
        if (!criteria.tags.some(tag => sessionTags.includes(tag))) {
          return false;
        }
      }

      // 按项目过滤
      if (criteria.project && session.metadata.project !== criteria.project) {
        return false;
      }

      // 按状态过滤
      if (criteria.status && session.metadata.status !== criteria.status) {
        return false;
      }

      // 按所有者过滤
      if (criteria.owner && session.metadata.owner !== criteria.owner) {
        return false;
      }

      return true;
    });
  }

  // 使用示例
  async findActiveFrontendSessions() {
    return await this.search({
      tags: ['frontend'],
      status: 'active'
    });
  }

  async findAliceSessions() {
    return await this.search({
      owner: 'alice'
    });
  }
}
```

### 示例 3：会话报告生成

```typescript
class SessionReporter {
  async generateReport(projectName: string): Promise<string> {
    const sessions = await this.sessionSearch.search({
      project: projectName
    });

    let report = `# 项目报告: ${projectName}\n\n`;
    report += `生成时间: ${new Date().toISOString()}\n\n`;

    // 统计信息
    report += `## 统计信息\n\n`;
    report += `- 总会话数: ${sessions.length}\n`;
    report += `- 活跃会话: ${sessions.filter(s => s.metadata.status === 'active').length}\n`;
    report += `- 已完成: ${sessions.filter(s => s.metadata.status === 'completed').length}\n`;
    report += `- 已归档: ${sessions.filter(s => s.metadata.status === 'archived').length}\n\n`;

    // 按标签分组
    const tagGroups = new Map<string, SessionInfo[]>();
    for (const session of sessions) {
      const tags = session.metadata.tags || [];
      for (const tag of tags) {
        if (!tagGroups.has(tag)) {
          tagGroups.set(tag, []);
        }
        tagGroups.get(tag)!.push(session);
      }
    }

    report += `## 按标签分组\n\n`;
    for (const [tag, tagSessions] of tagGroups) {
      report += `### ${tag} (${tagSessions.length})\n\n`;
      for (const session of tagSessions) {
        report += `- ${session.name} (${session.entryCount} entries)\n`;
      }
      report += `\n`;
    }

    return report;
  }
}
```

---

## 7. 最佳实践

### 7.1 命名最佳实践

```typescript
const namingBestPractices = {
  // ✅ 好的命名
  good: [
    'feature-user-authentication',
    'bugfix-login-timeout',
    'refactor-api-layer',
    'exp-caching-strategy',
    'checkpoint-v1.0-release'
  ],

  // ❌ 不好的命名
  bad: [
    'session1',           // 不描述性
    'test',               // 太通用
    'my-session',         // 不具体
    'abc123',             // 随机ID
    'untitled'            // 无意义
  ],

  // 命名规则
  rules: [
    '使用描述性名称',
    '包含类型前缀（feature/bugfix/refactor）',
    '使用连字符分隔单词',
    '保持简洁（50字符以内）',
    '避免特殊字符'
  ]
};
```

### 7.2 元数据最佳实践

```typescript
const metadataBestPractices = {
  // 必需字段
  required: ['name', 'description', 'project'],

  // 推荐字段
  recommended: ['tags', 'owner', 'status', 'priority'],

  // 可选字段
  optional: ['repository', 'branch', 'collaborators', 'dependencies'],

  // 示例
  example: {
    name: 'feature-user-auth',
    description: '实现用户认证系统，包括登录、注册、密码重置',
    project: 'my-app',
    tags: ['authentication', 'security', 'backend'],
    owner: 'alice',
    status: 'active',
    priority: 'high',
    repository: 'github.com/user/my-app',
    branch: 'feature/user-auth'
  }
};
```

### 7.3 组织最佳实践

```typescript
const organizationBestPractices = {
  // 按项目组织
  byProject: {
    structure: 'project-{name}-{component}',
    example: 'myapp-backend-api'
  },

  // 按功能组织
  byFeature: {
    structure: 'feature-{name}',
    example: 'feature-user-login'
  },

  // 按时间组织
  byTime: {
    structure: '{type}-{date}-{name}',
    example: 'sprint-2026-02-{name}'
  },

  // 定期清理
  cleanup: {
    archiveAfterDays: 30,
    deleteAfterDays: 90,
    keepCheckpoints: true
  }
};
```

---

## 学习检查清单

完成本节学习后，你应该能够：

- [ ] 使用 /name 命名会话
- [ ] 使用 /session 查看会话信息
- [ ] 添加和管理会话元数据
- [ ] 按项目、标签、状态组织会话
- [ ] 建立会话之间的链接和依赖
- [ ] 实现会话搜索和过滤
- [ ] 生成会话报告
- [ ] 应用命名和组织最佳实践

---

**版本：** v1.0
**最后更新：** 2026-02-18
**维护者：** Claude Code
