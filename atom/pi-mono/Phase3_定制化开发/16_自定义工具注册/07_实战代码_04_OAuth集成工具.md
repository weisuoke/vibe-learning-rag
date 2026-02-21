# 实战代码 04：OAuth 集成工具

> **基于 pi-mono antigravity-image-gen.ts 示例 + 2025-2026 OAuth 最佳实践**

---

## 场景描述

实现一个集成 OAuth 认证的图片生成工具，展示如何在 AI Agent 中集成外部 API 服务。核心特性：

- **OAuth 认证**：使用 OAuth 2.0 获取访问令牌
- **流式更新**：使用 onUpdate 回调显示进度
- **SSE 解析**：处理 Server-Sent Events 流式响应
- **配置管理**：支持环境变量、配置文件、工具参数
- **文件保存**：支持多种保存模式（project/global/custom）

**2025-2026 行业趋势：**
- **AI Agent API 集成**：5 种集成模式，OAuth 2.0 为主流（来源：[Composio Integration Patterns](https://composio.dev/blog/apis-ai-agents-integration-patterns)）
- **Next.js + MCP**：企业级 AI Agent 使用 OAuth2.0 安全认证（来源：[MintMCP Enterprise AI](https://www.mintmcp.com/blog/mcp-build-enterprise-ai-agents)）
- **TypeScript SDK**：Day AI SDK 提供 OAuth 集成支持（来源：[Day AI SDK](https://github.com/day-ai/day-ai-sdk)）
- **VoltAgent 示例**：Google Drive OAuth 认证实践（来源：[VoltAgent Examples](https://github.com/VoltAgent/ai-agent-examples)）

---

## 完整代码实现

```typescript
/**
 * OAuth 集成图片生成工具 - 基于 pi-mono antigravity-image-gen.ts
 *
 * 核心特性：
 * 1. OAuth 2.0 认证流程
 * 2. 流式更新（onUpdate 回调）
 * 3. SSE 响应解析
 * 4. 多级配置管理
 * 5. 文件保存策略
 *
 * 2025-2026 最佳实践：
 * - 使用 modelRegistry 管理 OAuth 凭证
 * - 支持自定义 API 端点
 * - 流式进度反馈
 * - 灵活的配置系统
 *
 * 参考：
 * - pi-mono: packages/coding-agent/examples/extensions/antigravity-image-gen.ts
 * - Composio: https://composio.dev/blog/apis-ai-agents-integration-patterns
 * - MintMCP: https://www.mintmcp.com/blog/mcp-build-enterprise-ai-agents
 */

import { randomUUID } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { StringEnum } from "@mariozechner/pi-ai";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { type Static, Type } from "@sinclair/typebox";

// ============================================================================
// 常量定义
// ============================================================================

const PROVIDER = "google-antigravity";
const ANTIGRAVITY_ENDPOINT = "https://daily-cloudcode-pa.sandbox.googleapis.com";
const DEFAULT_MODEL = "gemini-3-pro-image";
const DEFAULT_ASPECT_RATIO = "1:1";
const DEFAULT_SAVE_MODE = "none";

const ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"] as const;
const SAVE_MODES = ["none", "project", "global", "custom"] as const;

type AspectRatio = (typeof ASPECT_RATIOS)[number];
type SaveMode = (typeof SAVE_MODES)[number];

// ============================================================================
// 类型定义
// ============================================================================

interface ParsedCredentials {
  accessToken: string;
  projectId: string;
}

interface ExtensionConfig {
  save?: SaveMode;
  saveDir?: string;
}

interface SaveConfig {
  mode: SaveMode;
  outputDir?: string;
}

// ============================================================================
// Schema 定义
// ============================================================================

const TOOL_PARAMS = Type.Object({
  prompt: Type.String({ description: "Image description." }),
  model: Type.Optional(Type.String({
    description: "Image model id (e.g., gemini-3-pro-image, imagen-3). Default: gemini-3-pro-image.",
  })),
  aspectRatio: Type.Optional(StringEnum(ASPECT_RATIOS)),
  save: Type.Optional(StringEnum(SAVE_MODES)),
  saveDir: Type.Optional(Type.String({
    description: "Directory to save image when save=custom. Defaults to PI_IMAGE_SAVE_DIR if set.",
  })),
});

type ToolParams = Static<typeof TOOL_PARAMS>;

// ============================================================================
// OAuth 凭证管理
// ============================================================================

/**
 * 解析 OAuth 凭证
 *
 * 2025-2026 最佳实践：
 * - 凭证存储在 modelRegistry 中
 * - 使用 /login 命令获取凭证
 * - 支持多个 provider
 */
function parseOAuthCredentials(raw: string): ParsedCredentials {
  let parsed: { token?: string; projectId?: string };
  try {
    parsed = JSON.parse(raw) as { token?: string; projectId?: string };
  } catch {
    throw new Error("Invalid Google OAuth credentials. Run /login to re-authenticate.");
  }
  if (!parsed.token || !parsed.projectId) {
    throw new Error("Missing token or projectId in Google OAuth credentials. Run /login.");
  }
  return { accessToken: parsed.token, projectId: parsed.projectId };
}

async function getCredentials(ctx: {
  modelRegistry: { getApiKeyForProvider: (provider: string) => Promise<string | undefined> };
}): Promise<ParsedCredentials> {
  const apiKey = await ctx.modelRegistry.getApiKeyForProvider(PROVIDER);
  if (!apiKey) {
    throw new Error("Missing Google Antigravity OAuth credentials. Run /login for google-antigravity.");
  }
  return parseOAuthCredentials(apiKey);
}

// ============================================================================
// 配置管理
// ============================================================================

/**
 * 读取配置文件
 *
 * 支持两级配置：
 * 1. 全局配置：~/.pi/agent/extensions/antigravity-image-gen.json
 * 2. 项目配置：<repo>/.pi/extensions/antigravity-image-gen.json
 *
 * 项目配置覆盖全局配置
 */
function readConfigFile(path: string): ExtensionConfig {
  if (!existsSync(path)) return {};
  try {
    const content = readFileSync(path, "utf-8");
    return JSON.parse(content) as ExtensionConfig;
  } catch {
    return {};
  }
}

function loadConfig(cwd: string): ExtensionConfig {
  const globalConfig = readConfigFile(
    join(homedir(), ".pi", "agent", "extensions", "antigravity-image-gen.json")
  );
  const projectConfig = readConfigFile(
    join(cwd, ".pi", "extensions", "antigravity-image-gen.json")
  );
  return { ...globalConfig, ...projectConfig };
}

/**
 * 解析保存配置
 *
 * 优先级：工具参数 > 环境变量 > 配置文件 > 默认值
 *
 * 2025-2026 最佳实践：
 * - 多级配置系统
 * - 灵活的保存策略
 * - 环境变量支持
 */
function resolveSaveConfig(params: ToolParams, cwd: string): SaveConfig {
  const config = loadConfig(cwd);
  const envMode = (process.env.PI_IMAGE_SAVE_MODE || "").toLowerCase();
  const paramMode = params.save;
  const mode = (paramMode || envMode || config.save || DEFAULT_SAVE_MODE) as SaveMode;

  if (!SAVE_MODES.includes(mode)) {
    return { mode: DEFAULT_SAVE_MODE as SaveMode };
  }

  if (mode === "project") {
    return { mode, outputDir: join(cwd, ".pi", "generated-images") };
  }

  if (mode === "global") {
    return { mode, outputDir: join(homedir(), ".pi", "agent", "generated-images") };
  }

  if (mode === "custom") {
    const dir = params.saveDir || process.env.PI_IMAGE_SAVE_DIR || config.saveDir;
    if (!dir || !dir.trim()) {
      throw new Error("save=custom requires saveDir or PI_IMAGE_SAVE_DIR.");
    }
    return { mode, outputDir: dir };
  }

  return { mode };
}

// ============================================================================
// 文件保存
// ============================================================================

function imageExtension(mimeType: string): string {
  const lower = mimeType.toLowerCase();
  if (lower.includes("jpeg") || lower.includes("jpg")) return "jpg";
  if (lower.includes("gif")) return "gif";
  if (lower.includes("webp")) return "webp";
  return "png";
}

async function saveImage(base64Data: string, mimeType: string, outputDir: string): Promise<string> {
  await mkdir(outputDir, { recursive: true });
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const ext = imageExtension(mimeType);
  const filename = `image-${timestamp}-${randomUUID().slice(0, 8)}.${ext}`;
  const filePath = join(outputDir, filename);
  await writeFile(filePath, Buffer.from(base64Data, "base64"));
  return filePath;
}

// ============================================================================
// SSE 解析
// ============================================================================

/**
 * 解析 Server-Sent Events 流式响应
 *
 * 2025-2026 最佳实践：
 * - 使用 ReadableStream API
 * - 支持 AbortSignal 取消
 * - 逐行解析 SSE 格式
 * - 提取图片数据和文本
 */
async function parseSseForImage(
  response: Response,
  signal?: AbortSignal,
): Promise<{ image: { data: string; mimeType: string }; text: string[] }> {
  if (!response.body) {
    throw new Error("No response body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const textParts: string[] = [];

  try {
    while (true) {
      if (signal?.aborted) {
        throw new Error("Request was aborted");
      }

      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const jsonStr = line.slice(5).trim();
        if (!jsonStr) continue;

        let chunk: any;
        try {
          chunk = JSON.parse(jsonStr);
        } catch {
          continue;
        }

        const responseData = chunk.response;
        if (!responseData?.candidates) continue;

        for (const candidate of responseData.candidates) {
          const parts = candidate.content?.parts;
          if (!parts) continue;
          for (const part of parts) {
            if (part.text) {
              textParts.push(part.text);
            }
            if (part.inlineData?.data) {
              await reader.cancel();
              return {
                image: {
                  data: part.inlineData.data,
                  mimeType: part.inlineData.mimeType || "image/png",
                },
                text: textParts,
              };
            }
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  throw new Error("No image data returned by the model");
}

// ============================================================================
// 工具注册
// ============================================================================

export default function antigravityImageGen(pi: ExtensionAPI) {
  pi.registerTool({
    name: "generate_image",
    label: "Generate image",
    description:
      "Generate an image via Google Antigravity image models. Returns the image as a tool result attachment. Optional saving via save=project|global|custom|none, or PI_IMAGE_SAVE_MODE/PI_IMAGE_SAVE_DIR.",
    parameters: TOOL_PARAMS,

    async execute(_toolCallId, params: ToolParams, signal, onUpdate, ctx) {
      // ----------------------------------------------------------------------
      // 1. 获取 OAuth 凭证
      // ----------------------------------------------------------------------
      const { accessToken, projectId } = await getCredentials(ctx);
      const model = params.model || DEFAULT_MODEL;
      const aspectRatio = params.aspectRatio || DEFAULT_ASPECT_RATIO;

      // ----------------------------------------------------------------------
      // 2. 构建请求
      // ----------------------------------------------------------------------
      const requestBody = {
        project: projectId,
        model,
        request: {
          contents: [{
            role: "user",
            parts: [{ text: params.prompt }],
          }],
          systemInstruction: {
            parts: [{ text: "You are an AI image generator. Generate images based on user descriptions." }],
          },
          generationConfig: {
            imageConfig: { aspectRatio },
            candidateCount: 1,
          },
        },
        requestType: "agent",
        requestId: `agent-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`,
      };

      // ----------------------------------------------------------------------
      // 3. 流式更新 - 通知 LLM 正在请求
      // ----------------------------------------------------------------------
      /**
       * onUpdate 回调 - 2025-2026 最佳实践
       *
       * 用途：
       * - 显示进度信息
       * - 提升用户体验
       * - 让 LLM 知道工具正在执行
       */
      onUpdate?.({
        content: [{ type: "text", text: `Requesting image from ${PROVIDER}/${model}...` }],
        details: { provider: PROVIDER, model, aspectRatio },
      });

      // ----------------------------------------------------------------------
      // 4. 发送请求
      // ----------------------------------------------------------------------
      const response = await fetch(`${ANTIGRAVITY_ENDPOINT}/v1internal:streamGenerateContent?alt=sse`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify(requestBody),
        signal,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Image request failed (${response.status}): ${errorText}`);
      }

      // ----------------------------------------------------------------------
      // 5. 解析 SSE 响应
      // ----------------------------------------------------------------------
      const parsed = await parseSseForImage(response, signal);

      // ----------------------------------------------------------------------
      // 6. 保存图片（如果配置）
      // ----------------------------------------------------------------------
      const saveConfig = resolveSaveConfig(params, ctx.cwd);
      let savedPath: string | undefined;
      let saveError: string | undefined;

      if (saveConfig.mode !== "none" && saveConfig.outputDir) {
        try {
          savedPath = await saveImage(parsed.image.data, parsed.image.mimeType, saveConfig.outputDir);
        } catch (error) {
          saveError = error instanceof Error ? error.message : String(error);
        }
      }

      // ----------------------------------------------------------------------
      // 7. 构建返回结果
      // ----------------------------------------------------------------------
      const summaryParts = [
        `Generated image via ${PROVIDER}/${model}.`,
        `Aspect ratio: ${aspectRatio}.`
      ];

      if (savedPath) {
        summaryParts.push(`Saved image to: ${savedPath}`);
      } else if (saveError) {
        summaryParts.push(`Failed to save image: ${saveError}`);
      }

      if (parsed.text.length > 0) {
        summaryParts.push(`Model notes: ${parsed.text.join(" ")}`);
      }

      return {
        content: [
          { type: "text", text: summaryParts.join(" ") },
          { type: "image", data: parsed.image.data, mimeType: parsed.image.mimeType },
        ],
        details: {
          provider: PROVIDER,
          model,
          aspectRatio,
          savedPath,
          saveMode: saveConfig.mode
        },
      };
    },
  });
}
```

---

## 预期输出

### 场景 1：生成图片（不保存）

**LLM 调用：**
```json
{
  "tool": "generate_image",
  "params": {
    "prompt": "A sunset over mountains with vibrant colors",
    "aspectRatio": "16:9"
  }
}
```

**流式更新：**
```
Requesting image from google-antigravity/gemini-3-pro-image...
```

**最终结果：**
```
Generated image via google-antigravity/gemini-3-pro-image. Aspect ratio: 16:9.
[图片显示在终端]
```

### 场景 2：生成并保存图片

**LLM 调用：**
```json
{
  "tool": "generate_image",
  "params": {
    "prompt": "Cyberpunk city at night",
    "save": "project"
  }
}
```

**最终结果：**
```
Generated image via google-antigravity/gemini-3-pro-image. Aspect ratio: 1:1.
Saved image to: /path/to/project/.pi/generated-images/image-2026-02-21T10-30-45-abc12345.png
[图片显示在终端]
```

---

## 2025-2026 最佳实践

### 1. OAuth 认证流程

**凭证管理：**
```typescript
// 使用 modelRegistry 管理凭证
const apiKey = await ctx.modelRegistry.getApiKeyForProvider(PROVIDER);

// 解析 OAuth 凭证
const { accessToken, projectId } = parseOAuthCredentials(apiKey);
```

**为什么使用 modelRegistry？**
- 统一的凭证管理
- 支持多个 provider
- 与 /login 命令集成
- 安全存储

**2025-2026 行业标准：**
- **OAuth 2.0**：主流认证方式（来源：[Composio](https://composio.dev/blog/apis-ai-agents-integration-patterns)）
- **Token 刷新**：自动刷新过期 token
- **多租户支持**：企业级应用需求（来源：[MintMCP](https://www.mintmcp.com/blog/mcp-build-enterprise-ai-agents)）

### 2. 流式更新

**onUpdate 回调：**
```typescript
onUpdate?.({
  content: [{ type: "text", text: "Requesting image..." }],
  details: { provider, model, aspectRatio },
});
```

**用途：**
- 显示进度信息
- 提升用户体验
- 让 LLM 知道工具正在执行

**最佳实践：**
- 在长时间操作前调用
- 提供有意义的进度信息
- 包含关键参数在 details 中

### 3. SSE 解析

**Server-Sent Events 处理：**
```typescript
const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  buffer += decoder.decode(value, { stream: true });
  const lines = buffer.split("\n");
  buffer = lines.pop() || "";

  for (const line of lines) {
    if (line.startsWith("data:")) {
      const data = JSON.parse(line.slice(5));
      // 处理数据...
    }
  }
}
```

**关键点：**
- 使用 ReadableStream API
- 支持 AbortSignal 取消
- 正确处理缓冲区
- 逐行解析 SSE 格式

### 4. 配置管理

**多级配置系统：**
```
优先级：工具参数 > 环境变量 > 配置文件 > 默认值

1. 工具参数：params.save, params.saveDir
2. 环境变量：PI_IMAGE_SAVE_MODE, PI_IMAGE_SAVE_DIR
3. 配置文件：
   - 项目：<repo>/.pi/extensions/antigravity-image-gen.json
   - 全局：~/.pi/agent/extensions/antigravity-image-gen.json
4. 默认值：DEFAULT_SAVE_MODE = "none"
```

**配置文件示例：**
```json
{
  "save": "global",
  "saveDir": "/custom/path"
}
```

### 5. 错误处理

**OAuth 错误：**
```typescript
if (!apiKey) {
  throw new Error("Missing OAuth credentials. Run /login for google-antigravity.");
}
```

**API 错误：**
```typescript
if (!response.ok) {
  const errorText = await response.text();
  throw new Error(`Request failed (${response.status}): ${errorText}`);
}
```

**保存错误：**
```typescript
try {
  savedPath = await saveImage(data, mimeType, outputDir);
} catch (error) {
  saveError = error instanceof Error ? error.message : String(error);
}
```

---

## 扩展应用场景

### 1. 其他 OAuth API 集成

```typescript
// Google Drive
const driveApi = await getCredentials("google-drive");

// GitHub
const githubApi = await getCredentials("github");

// Slack
const slackApi = await getCredentials("slack");
```

### 2. 流式文本生成

```typescript
onUpdate?.({
  content: [{ type: "text", text: partialText }],
  details: { isPartial: true },
});
```

### 3. 文件上传

```typescript
const formData = new FormData();
formData.append("file", fileBuffer);

const response = await fetch(uploadUrl, {
  method: "POST",
  headers: { Authorization: `Bearer ${accessToken}` },
  body: formData,
  signal,
});
```

---

## 参考资料

### Pi-mono 源码
- **antigravity-image-gen.ts**：`sourcecode/pi-mono/packages/coding-agent/examples/extensions/antigravity-image-gen.ts`
- **Model Registry**：`sourcecode/pi-mono/packages/coding-agent/src/core/model-registry/`

### 2025-2026 行业标准
- **Composio 集成模式**：https://composio.dev/blog/apis-ai-agents-integration-patterns
- **MintMCP 企业 AI**：https://www.mintmcp.com/blog/mcp-build-enterprise-ai-agents
- **Day AI SDK**：https://github.com/day-ai/day-ai-sdk
- **VoltAgent 示例**：https://github.com/VoltAgent/ai-agent-examples

### 相关文档
- **03_核心概念_03_Execute执行函数.md**：execute 函数详解
- **03_核心概念_07_错误处理.md**：错误处理最佳实践

---

**版本：** v1.0
**最后更新：** 2026-02-21
**作者：** Claude Code
**基于：** pi-mono antigravity-image-gen.ts + 2025-2026 OAuth 标准
