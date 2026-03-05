# Official Documentation Summary

## Onboarding Wizard Overview

**Source**: https://docs.openclaw.ai/start/wizard

### Recommended Setup Method

```bash
openclaw onboard
```

Fastest first chat alternative:
```bash
openclaw dashboard  # No channel setup needed
```

### Two Modes

**QuickStart (defaults)**:
- Local gateway (loopback)
- Workspace default or existing
- Gateway port 18789
- Gateway auth Token (auto-generated)
- Tailscale exposure Off
- Telegram + WhatsApp DMs default to allowlist

**Advanced (full control)**:
- Exposes every configuration step
- Mode, workspace, gateway, channels, daemon, skills

### Configuration Steps (Local Mode)

1. **Model/Auth** - Anthropic API key (recommended), OpenAI, or Custom Provider
2. **Workspace** - Location for agent files (default `~/.openclaw/workspace`)
3. **Gateway** - Port, bind address, auth mode, Tailscale exposure
4. **Channels** - WhatsApp, Telegram, Discord, Google Chat, etc.
5. **Daemon** - LaunchAgent (macOS) or systemd user unit (Linux/WSL2)
6. **Health check** - Starts Gateway and verifies running
7. **Skills** - Installs recommended skills and dependencies

### Important Notes

- Re-running wizard does NOT wipe anything unless you choose Reset
- Invalid config requires running `openclaw doctor` first
- Remote mode only configures local client, doesn't modify remote host

### Multi-Agent Setup

```bash
openclaw agents add <name>
```

Creates separate agent with own workspace, sessions, and auth profiles.

## CLI Reference

**Source**: https://docs.openclaw.ai/cli/onboard

### Basic Usage

```bash
openclaw onboard
openclaw onboard --flow quickstart
openclaw onboard --flow manual
openclaw onboard --mode remote --remote-url ws://gateway-host:18789
```

### Non-Interactive Custom Provider

```bash
openclaw onboard --non-interactive \
  --auth-choice custom-api-key \
  --custom-base-url "https://llm.example.com/v1" \
  --custom-model-id "foo-large" \
  --custom-api-key "$CUSTOM_API_KEY" \
  --custom-compatibility openai
```

### Flow Notes

- `quickstart`: minimal prompts, auto-generates gateway token
- `manual`: full prompts for port/bind/auth (alias of `advanced`)
- Custom Provider: connect any OpenAI or Anthropic compatible endpoint
- Use **Unknown** to auto-detect compatibility

### Common Follow-up Commands

```bash
openclaw configure
openclaw agents add <name>
```

**Important**: `--json` does not imply non-interactive mode. Use `--non-interactive` for scripts.

## Configuration Reference

**Source**: https://docs.openclaw.ai/gateway/configuration

### Config File Location

`~/.openclaw/openclaw.json` (JSON5 format)

### Minimal Config

```json5
{
  agents: { defaults: { workspace: "~/.openclaw/workspace" } },
  channels: { whatsapp: { allowFrom: ["+15555550123"] } },
}
```

### Editing Methods

1. **Wizard**: `openclaw onboard` or `openclaw configure`
2. **CLI**: `openclaw config get/set/unset`
3. **Web UI**: http://127.0.0.1:18789 → Config tab
4. **Direct edit**: Edit `~/.openclaw/openclaw.json` (hot reload enabled)

### Strict Validation

- Unknown keys, malformed types, or invalid values cause Gateway to refuse to start
- Run `openclaw doctor` to see exact issues
- Run `openclaw doctor --fix` to apply repairs

### Hot Reload

Gateway watches config file and applies changes automatically:
- **hybrid** (default): Hot-applies safe changes, auto-restarts for critical ones
- **hot**: Hot-applies safe changes only, logs warning for restart-needed
- **restart**: Restarts Gateway on any config change
- **off**: Disables file watching

Most fields hot-apply without downtime. Gateway server settings (port, bind, auth, TLS) require restart.
