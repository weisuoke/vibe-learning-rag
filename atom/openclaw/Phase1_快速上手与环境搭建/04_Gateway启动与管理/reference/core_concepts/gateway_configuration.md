# Configuration - OpenClaw Gateway

**Source:** https://docs.openclaw.ai/gateway/configuration
**Fetched:** 2026-02-21

## Overview

OpenClaw reads an optional **JSON5** config from `~/.openclaw/openclaw.json`. If the file is missing, OpenClaw uses safe defaults.

## Minimal config

```json5
// ~/.openclaw/openclaw.json
{
  agents: { defaults: { workspace: "~/.openclaw/workspace" } },
  channels: { whatsapp: { allowFrom: ["+15555550123"] } },
}
```

## Editing config

```bash
openclaw onboard       # full setup wizard
openclaw configure     # config wizard
```

```bash
openclaw config get agents.defaults.workspace
openclaw config set agents.defaults.heartbeat.every "2h"
openclaw config unset tools.web.search.apiKey
```

Open http://127.0.0.1:18789 and use the **Config** tab.

Edit `~/.openclaw/openclaw.json` directly. The Gateway watches the file and applies changes automatically (see hot reload).

## Strict validation

OpenClaw only accepts configurations that fully match the schema. Unknown keys, malformed types, or invalid values cause the Gateway to **refuse to start**.

When validation fails:
- The Gateway does not boot
- Only diagnostic commands work (`openclaw doctor`, `openclaw logs`, `openclaw health`, `openclaw status`)
- Run `openclaw doctor` to see exact issues
- Run `openclaw doctor --fix` (or `--yes`) to apply repairs

## Config hot reload

The Gateway watches `~/.openclaw/openclaw.json` and applies changes automatically — no manual restart needed for most settings.

### Reload modes

| Mode      | Behavior                                                                 |
|-----------|--------------------------------------------------------------------------|
| **hybrid** (default) | Hot-applies safe changes instantly. Automatically restarts for critical ones. |
| **hot**   | Hot-applies safe changes only. Logs a warning when a restart is needed. |
| **restart** | Restarts the Gateway on any config change, safe or not.                |
| **off**   | Disables file watching. Changes take effect on the next manual restart. |

```json5
{
  gateway: {
    reload: { mode: "hybrid", debounceMs: 300 },
  },
}
```

### What hot-applies vs what needs a restart

| Category          | Fields                                      | Restart needed? |
|-------------------|---------------------------------------------|-----------------|
| Channels          | channels.*, web (WhatsApp) — all built-in and extension channels | No              |
| Agent & models    | agent, agents, models, routing              | No              |
| Automation        | hooks, cron, agent.heartbeat                | No              |
| Sessions & messages | session, messages                         | No              |
| Tools & media     | tools, browser, skills, audio, talk         | No              |
| UI & misc         | ui, logging, identity, bindings             | No              |
| Gateway server    | gateway.* (port, bind, auth, tailscale, TLS, HTTP) | **Yes**         |
| Infrastructure    | discovery, canvasHost, plugins              | **Yes**         |

## Config RPC (programmatic updates)

Control-plane write RPCs (`config.apply`, `config.patch`, `update.run`) are rate-limited to **3 requests per 60 seconds** per `deviceId+clientIp`.

### config.apply (full replace)

Validates + writes the full config and restarts the Gateway in one step.

Params:
- raw (string) — JSON5 payload for the entire config
- baseHash (optional) — config hash from config.get (required when config exists)
- sessionKey (optional) — session key for the post-restart wake-up ping
- note (optional) — note for the restart sentinel
- restartDelayMs (optional) — delay before restart (default 2000)

```bash
openclaw gateway call config.apply --params '{
  "raw": "{ agents: { defaults: { workspace: \"~/.openclaw/workspace\" } } }",
  "baseHash": "<hash>",
  "sessionKey": "agent:main:whatsapp:dm:+15555550123"
}'
```

### config.patch (partial update)

Merges a partial update into the existing config (JSON merge patch semantics):
- Objects merge recursively
- null deletes a key
- Arrays replace

```bash
openclaw gateway call config.patch --params '{
  "raw": "{ channels: { telegram: { groups: { \"*\": { requireMention: false } } } } }",
  "baseHash": "<hash>"
}'
```

## Environment variables

OpenClaw reads env vars from:
- The parent process
- .env from the current working directory (if present)
- ~/.openclaw/.env (global fallback)

You can also set inline env vars in config:

```json5
{
  env: {
    OPENROUTER_API_KEY: "sk-or-...",
    vars: { GROQ_API_KEY: "gsk-..." },
  },
}
```

### Env var substitution in config values

Reference env vars in any config string value with `${VAR_NAME}`:

```json5
{
  gateway: { auth: { token: "${OPENCLAW_GATEWAY_TOKEN}" } },
  models: { providers: { custom: { apiKey: "${CUSTOM_API_KEY}" } } },
}
```

Rules:
- Only uppercase names matched: `[A-Z_][A-Z0-9_]*`
- Missing/empty vars throw an error at load time
- Escape with `$${VAR}` for literal output
- Works inside `$include` files
- Inline substitution: `"${BASE}/v1"` → `"https://api.example.com/v1"`
