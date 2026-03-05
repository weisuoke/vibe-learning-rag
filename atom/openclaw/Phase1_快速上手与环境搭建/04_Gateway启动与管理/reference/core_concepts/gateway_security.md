# Security - OpenClaw Gateway

**Source:** https://docs.openclaw.ai/gateway/security
**Fetched:** 2026-02-21

## Quick check: `openclaw security audit`

Run this regularly (especially after changing config or exposing network surfaces):

```bash
openclaw security audit
openclaw security audit --deep
openclaw security audit --fix
openclaw security audit --json
```

It flags common footguns (Gateway auth exposure, browser control exposure, elevated allowlists, filesystem permissions).

## Deployment assumption (important)

OpenClaw assumes the host and config boundary are trusted:
- If someone can modify Gateway host state/config (`~/.openclaw`, including openclaw.json), treat them as a trusted operator
- Running one Gateway for multiple mutually untrusted/adversarial operators is **not a recommended setup**
- For mixed-trust teams, split trust boundaries with separate gateways (or at minimum separate OS users/hosts)

## Hardened baseline in 60 seconds

Use this baseline first, then selectively re-enable tools per trusted agent:

```json
{
  "gateway": {
    "mode": "local",
    "bind": "loopback",
    "auth": { "mode": "token", "token": "replace-with-long-random-token" }
  },
  "session": {
    "dmScope": "per-channel-peer"
  },
  "tools": {
    "profile": "messaging",
    "deny": ["group:automation", "group:runtime", "group:fs", "sessions_spawn", "sessions_send"],
    "fs": { "workspaceOnly": true },
    "exec": { "security": "deny", "ask": "always" },
    "elevated": { "enabled": false }
  },
  "channels": {
    "whatsapp": { "dmPolicy": "pairing", "groups": { "*": { "requireMention": true } } }
  }
}
```

This keeps the Gateway local-only, isolates DMs, and disables control-plane/runtime tools by default.

## Shared inbox quick rule

If more than one person can DM your bot:
- Set `session.dmScope: "per-channel-peer"` (or `"per-account-channel-peer"` for multi-account channels)
- Keep `dmPolicy: "pairing"` or strict allowlists
- Never combine shared DMs with broad tool access

## What the audit checks (high level)

- **Inbound access** (DM policies, group policies, allowlists): can strangers trigger the bot?
- **Tool blast radius** (elevated tools + open rooms): could prompt injection turn into shell/file/network actions?
- **Network exposure** (Gateway bind/auth, Tailscale Serve/Funnel, weak/short auth tokens)
- **Browser control exposure** (remote nodes, relay ports, remote CDP endpoints)
- **Local disk hygiene** (permissions, symlinks, config includes, "synced folder" paths)
- **Plugins** (extensions exist without an explicit allowlist)
- **Policy drift/misconfig** (sandbox docker settings configured but sandbox mode off)

## Credential storage map

- **WhatsApp**: `~/.openclaw/credentials/whatsapp/<accountId>/creds.json`
- **Telegram bot token**: config/env or `channels.telegram.tokenFile`
- **Discord bot token**: config/env (token file not yet supported)
- **Slack tokens**: config/env (`channels.slack.*`)
- **Pairing allowlists**: `~/.openclaw/credentials/<channel>-allowFrom.json`
- **Model auth profiles**: `~/.openclaw/agents/<agentId>/agent/auth-profiles.json`
- **Legacy OAuth import**: `~/.openclaw/credentials/oauth.json`

## Security Audit Checklist

Priority order:

1. **Anything "open" + tools enabled**: lock down DMs/groups first (pairing/allowlists), then tighten tool policy/sandboxing
2. **Public network exposure** (LAN bind, Funnel, missing auth): fix immediately
3. **Browser control remote exposure**: treat it like operator access (tailnet-only, pair nodes deliberately)
4. **Filesystem permissions**: make sure state/config/credentials/auth are not group/world-readable
5. **Plugins/extensions**: only load what you explicitly trust
6. **Model choice**: prefer modern, instruction-hardened models for any bot with tools

## Key Security Findings

| checkId | Severity | Why it matters | Primary fix |
|---------|----------|----------------|-------------|
| fs.state_dir.perms_world_writable | critical | Other users can modify full OpenClaw state | filesystem perms on ~/.openclaw |
| fs.config.perms_writable | critical | Others can change auth/tool policy/config | filesystem perms on openclaw.json |
| gateway.bind_no_auth | critical | Remote bind without shared secret | gateway.bind, gateway.auth.* |
| gateway.tailscale_funnel | critical | Public internet exposure | gateway.tailscale.mode |
| hooks.token_too_short | warn | Easier brute force on hook ingress | hooks.token |
| logging.redact_off | warn | Sensitive values leak to logs/status | logging.redactSensitive |
| models.small_params | critical/info | Small models + unsafe tool surfaces raise injection risk | model choice + sandbox/tool policy |

## Network exposure (bind + port + firewall)

Gateway bind modes:
- **loopback** (default): only localhost can connect
- **lan**: binds to all interfaces (0.0.0.0)
- **tailnet**: binds only to Tailscale interface
- **auto**: chooses based on Tailscale availability
- **custom**: user-specified bind address

## Lock down the Gateway WebSocket (local auth)

```json5
{
  gateway: {
    auth: {
      mode: "token",  // or "password"
      token: "long-random-token-here"
    }
  }
}
```

Auth modes:
- **token**: shared secret in connect.params.auth.token
- **password**: password-based auth
- **trusted-proxy**: trust X-Forwarded-* headers (reverse proxy only)
- **none**: no auth (dangerous for non-loopback)

## Tailscale Serve identity headers

When using Tailscale Serve, OpenClaw can trust identity headers:

```json5
{
  gateway: {
    auth: { mode: "trusted-proxy" },
    tailscale: { mode: "serve" }
  }
}
```

## Sandboxing (recommended)

Run agent sessions in isolated Docker containers:

```json5
{
  agents: {
    defaults: {
      sandbox: {
        mode: "non-main",  // off | non-main | all
        scope: "agent",    // session | agent | shared
      },
    },
  },
}
```

Build the image first: `scripts/sandbox-setup.sh`

## Incident Response

### Contain
- Stop the Gateway: `openclaw gateway stop`
- Revoke channel access (WhatsApp: unpair device, Telegram: revoke bot token)

### Rotate (assume compromise if secrets leaked)
- Regenerate Gateway auth token
- Rotate model API keys
- Regenerate channel credentials

### Audit
- Check `~/.openclaw/logs/` for suspicious activity
- Review session transcripts in `~/.openclaw/agents/<agentId>/sessions/`
- Check config audit: `~/.openclaw/logs/config-audit.jsonl`

### Collect for a report
- Gateway logs
- Session transcripts
- Config snapshots
- System logs (launchd/systemd)
