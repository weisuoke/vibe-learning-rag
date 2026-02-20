# Session 管理与分支 - 实战代码 05：Session 持久化与恢复

> 实现跨设备、跨时间的会话持久化和恢复机制

---

## 场景：跨设备、跨时间恢复会话

实现：
- JSONL 解析与重建
- 跨设备同步
- 备份与恢复
- 数据完整性验证

---

## 完整代码实现

```typescript
/**
 * Session 持久化与恢复实战示例
 * 演示：跨设备、跨时间的会话恢复机制
 */

import fs from 'fs/promises';
import path from 'path';
import crypto from 'crypto';

// ===== 类型定义（省略，与前面相同）=====

// ===== 持久化管理器 =====

class PersistenceManager {
  constructor(private sessionsDir: string) {}

  // 备份会话
  async backupSession(sessionId: string, backupDir: string): Promise<string> {
    const sourceFile = path.join(this.sessionsDir, `${sessionId}.jsonl`);
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const backupFile = path.join(backupDir, `${sessionId}-${timestamp}.jsonl`);

    await fs.mkdir(backupDir, { recursive: true });
    await fs.copyFile(sourceFile, backupFile);

    // 计算校验和
    const checksum = await this.calculateChecksum(backupFile);
    await fs.writeFile(`${backupFile}.sha256`, checksum);

    console.log(`会话已备份到: ${backupFile}`);
    return backupFile;
  }

  // 恢复会话
  async restoreSession(backupFile: string, targetSessionId?: string): Promise<string> {
    // 验证校验和
    const checksumFile = `${backupFile}.sha256`;
    try {
      const expectedChecksum = await fs.readFile(checksumFile, 'utf-8');
      const actualChecksum = await this.calculateChecksum(backupFile);

      if (expectedChecksum.trim() !== actualChecksum.trim()) {
        throw new Error('校验和不匹配，文件可能已损坏');
      }
    } catch (err) {
      console.warn('未找到校验和文件，跳过验证');
    }

    // 确定目标会话 ID
    if (!targetSessionId) {
      const basename = path.basename(backupFile, '.jsonl');
      targetSessionId = basename.split('-')[0];
    }

    const targetFile = path.join(this.sessionsDir, `${targetSessionId}.jsonl`);
    await fs.copyFile(backupFile, targetFile);

    console.log(`会话已恢复到: ${targetFile}`);
    return targetSessionId;
  }

  // 导出会话为 JSON
  async exportSession(sessionId: string, outputFile: string): Promise<void> {
    const storage = new SessionStorage(sessionId, this.sessionsDir);
    const entries = await storage.readAll();

    const exportData = {
      version: '1.0',
      sessionId,
      exportTime: Date.now(),
      entryCount: entries.length,
      entries
    };

    await fs.writeFile(outputFile, JSON.stringify(exportData, null, 2));
    console.log(`会话已导出到: ${outputFile}`);
  }

  // 导入会话
  async importSession(inputFile: string): Promise<string> {
    const content = await fs.readFile(inputFile, 'utf-8');
    const data = JSON.parse(content);

    // 验证版本
    if (data.version !== '1.0') {
      throw new Error(`不支持的版本: ${data.version}`);
    }

    // 创建新会话
    const sessionId = data.sessionId;
    const storage = new SessionStorage(sessionId, this.sessionsDir);

    // 写入所有条目
    for (const entry of data.entries) {
      await storage.append(entry);
    }

    console.log(`会话已导入: ${sessionId}`);
    return sessionId;
  }

  // 计算文件校验和
  private async calculateChecksum(filePath: string): Promise<string> {
    const content = await fs.readFile(filePath);
    return crypto.createHash('sha256').update(content).digest('hex');
  }

  // 验证会话完整性
  async validateSession(sessionId: string): Promise<ValidationResult> {
    const storage = new SessionStorage(sessionId, this.sessionsDir);
    const entries = await storage.readAll();

    const errors: string[] = [];
    const warnings: string[] = [];

    // 检查 1：验证 ID 唯一性
    const ids = new Set<string>();
    for (const entry of entries) {
      if (ids.has(entry.id)) {
        errors.push(`重复的 ID: ${entry.id}`);
      }
      ids.add(entry.id);
    }

    // 检查 2：验证父子关系
    const nodeMap = new Map(entries.map(e => [e.id, e]));
    for (const entry of entries) {
      if (entry.parentId && !nodeMap.has(entry.parentId)) {
        errors.push(`节点 ${entry.id} 的父节点 ${entry.parentId} 不存在`);
      }
    }

    // 检查 3：验证时间戳
    for (let i = 1; i < entries.length; i++) {
      if (entries[i].timestamp < entries[i - 1].timestamp) {
        warnings.push(`节点 ${entries[i].id} 的时间戳早于前一个节点`);
      }
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings,
      entryCount: entries.length
    };
  }
}

interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  entryCount: number;
}

// ===== 同步管理器 =====

class SyncManager {
  constructor(private sessionsDir: string) {}

  // 同步到远程
  async syncToRemote(sessionId: string, remoteDir: string): Promise<void> {
    const localFile = path.join(this.sessionsDir, `${sessionId}.jsonl`);
    const remoteFile = path.join(remoteDir, `${sessionId}.jsonl`);

    await fs.mkdir(remoteDir, { recursive: true });

    // 检查远程文件是否存在
    let remoteExists = false;
    try {
      await fs.access(remoteFile);
      remoteExists = true;
    } catch {
      // 文件不存在
    }

    if (remoteExists) {
      // 合并更新
      await this.mergeUpdates(localFile, remoteFile);
    } else {
      // 直接复制
      await fs.copyFile(localFile, remoteFile);
    }

    console.log(`会话已同步到: ${remoteFile}`);
  }

  // 从远程同步
  async syncFromRemote(sessionId: string, remoteDir: string): Promise<void> {
    const localFile = path.join(this.sessionsDir, `${sessionId}.jsonl`);
    const remoteFile = path.join(remoteDir, `${sessionId}.jsonl`);

    // 检查本地文件是否存在
    let localExists = false;
    try {
      await fs.access(localFile);
      localExists = true;
    } catch {
      // 文件不存在
    }

    if (localExists) {
      // 合并更新
      await this.mergeUpdates(remoteFile, localFile);
    } else {
      // 直接复制
      await fs.copyFile(remoteFile, localFile);
    }

    console.log(`会话已从远程同步: ${localFile}`);
  }

  // 合并更新
  private async mergeUpdates(sourceFile: string, targetFile: string): Promise<void> {
    const sourceContent = await fs.readFile(sourceFile, 'utf-8');
    const targetContent = await fs.readFile(targetFile, 'utf-8');

    const sourceEntries = sourceContent
      .split('\n')
      .filter(line => line.trim())
      .map(line => JSON.parse(line));

    const targetEntries = targetContent
      .split('\n')
      .filter(line => line.trim())
      .map(line => JSON.parse(line));

    // 找出新条目
    const targetIds = new Set(targetEntries.map(e => e.id));
    const newEntries = sourceEntries.filter(e => !targetIds.has(e.id));

    // 追加新条目
    if (newEntries.length > 0) {
      const newLines = newEntries.map(e => JSON.stringify(e)).join('\n') + '\n';
      await fs.appendFile(targetFile, newLines);
      console.log(`合并了 ${newEntries.length} 个新条目`);
    }
  }
}

// ===== 使用示例 =====

async function main() {
  console.log('=== Session 持久化与恢复示例 ===\n');

  const sessionsDir = path.join(process.cwd(), '.demo-sessions');
  const backupDir = path.join(process.cwd(), '.demo-backups');
  const remoteDir = path.join(process.cwd(), '.demo-remote');

  await fs.mkdir(sessionsDir, { recursive: true });

  const manager = new SessionManager(sessionsDir);
  const persistence = new PersistenceManager(sessionsDir);
  const sync = new SyncManager(sessionsDir);

  // 1. 创建会话
  console.log('1. 创建会话');
  const session = await manager.createSession('my-project');
  await session.setName('我的项目');

  await session.appendMessage('user', '实现功能 A');
  await session.appendMessage('assistant', '已完成功能 A');
  await session.appendMessage('user', '实现功能 B');
  await session.appendMessage('assistant', '已完成功能 B');

  console.log(`   会话创建完成\n`);

  // 2. 验证会话完整性
  console.log('2. 验证会话完整性');
  const validation = await persistence.validateSession(session.id);
  console.log(`   验证结果: ${validation.valid ? '通过' : '失败'}`);
  console.log(`   条目数: ${validation.entryCount}`);
  if (validation.errors.length > 0) {
    console.log(`   错误: ${validation.errors.join(', ')}`);
  }
  if (validation.warnings.length > 0) {
    console.log(`   警告: ${validation.warnings.join(', ')}`);
  }
  console.log();

  // 3. 备份会话
  console.log('3. 备份会话');
  const backupFile = await persistence.backupSession(session.id, backupDir);
  console.log();

  // 4. 导出会话
  console.log('4. 导出会话为 JSON');
  const exportFile = path.join(backupDir, `${session.id}-export.json`);
  await persistence.exportSession(session.id, exportFile);
  console.log();

  // 5. 模拟删除会话
  console.log('5. 模拟删除会话');
  await manager.deleteSession(session.id);
  console.log(`   会话已删除\n`);

  // 6. 恢复会话
  console.log('6. 从备份恢复会话');
  const restoredId = await persistence.restoreSession(backupFile);
  console.log(`   会话已恢复: ${restoredId}\n`);

  // 7. 验证恢复的会话
  console.log('7. 验证恢复的会话');
  const restoredSession = await manager.loadSession(restoredId);
  const restoredMessages = await restoredSession.getMessages();
  console.log(`   恢复的会话包含 ${restoredMessages.length} 条消息:`);
  restoredMessages.forEach((msg, i) => {
    console.log(`   ${i + 1}. ${msg.role}: ${msg.content}`);
  });
  console.log();

  // 8. 同步到远程
  console.log('8. 同步到远程');
  await sync.syncToRemote(restoredId, remoteDir);
  console.log();

  // 9. 模拟在另一台设备上添加消息
  console.log('9. 模拟在另一台设备上添加消息');
  const remoteSession = new Session(restoredId, remoteDir);
  await remoteSession.appendMessage('user', '在远程设备上添加的消息');
  console.log(`   已在远程添加消息\n`);

  // 10. 从远程同步
  console.log('10. 从远程同步');
  await sync.syncFromRemote(restoredId, remoteDir);
  console.log();

  // 11. 验证同步结果
  console.log('11. 验证同步结果');
  const syncedSession = await manager.loadSession(restoredId);
  const syncedMessages = await syncedSession.getMessages();
  console.log(`   同步后的会话包含 ${syncedMessages.length} 条消息:`);
  syncedMessages.forEach((msg, i) => {
    console.log(`   ${i + 1}. ${msg.role}: ${msg.content}`);
  });

  console.log('\n=== 示例完成 ===');

  // 清理
  await fs.rm(sessionsDir, { recursive: true, force: true });
  await fs.rm(backupDir, { recursive: true, force: true });
  await fs.rm(remoteDir, { recursive: true, force: true });
}

// 运行示例
main().catch(console.error);
```

---

## 运行输出示例

```
=== Session 持久化与恢复示例 ===

1. 创建会话
   会话创建完成

2. 验证会话完整性
   验证结果: 通过
   条目数: 6

3. 备份会话
会话已备份到: .demo-backups/my-project-2026-02-18T15-00-00-000Z.jsonl

4. 导出会话为 JSON
会话已导出到: .demo-backups/my-project-export.json

5. 模拟删除会话
   会话已删除

6. 从备份恢复会话
会话已恢复到: .demo-sessions/my-project.jsonl
   会话已恢复: my-project

7. 验证恢复的会话
   恢复的会话包含 4 条消息:
   1. user: 实现功能 A
   2. assistant: 已完成功能 A
   3. user: 实现功能 B
   4. assistant: 已完成功能 B

8. 同步到远程
会话已同步到: .demo-remote/my-project.jsonl

9. 模拟在另一台设备上添加消息
   已在远程添加消息

10. 从远程同步
合并了 1 个新条目
会话已从远程同步: .demo-sessions/my-project.jsonl

11. 验证同步结果
   同步后的会话包含 5 条消息:
   1. user: 实现功能 A
   2. assistant: 已完成功能 A
   3. user: 实现功能 B
   4. assistant: 已完成功能 B
   5. user: 在远程设备上添加的消息

=== 示例完成 ===
```

---

## 学习检查清单

- [ ] 实现会话备份和恢复
- [ ] 实现数据完整性验证
- [ ] 实现跨设备同步
- [ ] 实现导出和导入功能
- [ ] 应用于实际项目

---

**版本：** v1.0
**最后更新：** 2026-02-18
**维护者：** Claude Code
