# OpenClaw Multiple Gateways Configuration (2026)

**Source:** Web search results
**Query:** "OpenClaw multiple gateways profile configuration 2026"
**Fetched:** 2026-02-22

## Official Documentation

### 1. Multiple Gateways - OpenClaw Official Documentation
**URL:** https://docs.openclaw.ai/gateway/multiple-gateways
**Key Points:**
- OpenClaw supports running multiple Gateways on the same host
- Uses independent profiles, state directories, workspaces, and ports for isolation
- Recommended to use `--profile` parameter for automatic configuration isolation
- Common use cases: rescue bot, development/production separation

### 2. Configuration - OpenClaw Official Documentation
**URL:** https://docs.openclaw.ai/gateway/configuration
**Key Points:**
- Gateway configuration guide including multi-agent routing
- Multi-channel settings and profile isolation
- Supports interactive setup via `openclaw onboard` or direct config file editing
- Configuration management for multiple gateways

### 3. gateway - OpenClaw CLI Documentation
**URL:** https://docs.openclaw.ai/cli/gateway
**Key Points:**
- Gateway command detailed explanation
- Supports `--profile` for isolating multiple instances
- `discover` can find multiple gateways
- Suitable for startup, query, and isolation management in multi-gateway environments

### 4. OpenClaw Gateway Commands 2026: Operator Q&A Guide
**URL:** https://vpn07.com/en/blog/2026-openclaw-gateway-commands-operator-qa.html
**Key Points:**
- 2026 OpenClaw Gateway command operation guide
- Running multiple gateway instances on the same host requires independent ports, config paths, state directories, and workspaces
- Ensures isolated configuration to avoid conflicts

## Profile-Based Isolation

### What is a Profile?

A **profile** is an isolated Gateway instance with its own:
- Configuration file (`~/.openclaw-<profile>/openclaw.json`)
- State directory (`~/.openclaw-<profile>/`)
- Workspace (`~/.openclaw-<profile>/workspace/`)
- Port (must be unique)
- Credentials
- Sessions
- Logs

### Use Cases

**1. Development vs Production**
- `default` profile: production Gateway
- `dev` profile: development Gateway with test channels

**2. Rescue Bot**
- `default` profile: main bot
- `rescue` profile: backup bot with minimal config

**3. Multi-User Scenarios**
- `alice` profile: Alice's personal Gateway
- `bob` profile: Bob's personal Gateway

**4. Testing**
- `default` profile: stable Gateway
- `test` profile: experimental features

## CLI Usage

### Starting Multiple Gateways

**Basic Profile Usage:**
```bash
# Start default profile (port 18789)
openclaw gateway

# Start dev profile (port 18790)
openclaw gateway --profile dev --port 18790

# Start rescue profile (port 18791)
openclaw gateway --profile rescue --port 18791
```

**With Daemon Installation:**
```bash
# Install default profile daemon
openclaw onboard --install-daemon

# Install dev profile daemon
openclaw onboard --install-daemon --profile dev

# Install rescue profile daemon
openclaw onboard --install-daemon --profile rescue
```

### Managing Multiple Gateways

**Status Checking:**
```bash
# Check default profile
openclaw gateway status

# Check dev profile
openclaw gateway status --profile dev

# Check rescue profile
openclaw gateway status --profile rescue
```

**Lifecycle Management:**
```bash
# Start/stop/restart default profile
openclaw gateway start
openclaw gateway stop
openclaw gateway restart

# Start/stop/restart dev profile
openclaw gateway start --profile dev
openclaw gateway stop --profile dev
openclaw gateway restart --profile dev
```

**Discovery:**
```bash
# Discover all gateways on the network
openclaw gateway discover

# Probe all gateways (local + remote)
openclaw gateway probe
```

## Configuration

### Profile-Specific Config Files

Each profile has its own config file:

```
~/.openclaw/openclaw.json           # default profile
~/.openclaw-dev/openclaw.json       # dev profile
~/.openclaw-rescue/openclaw.json    # rescue profile
```

### Example: Default Profile Config

```json5
// ~/.openclaw/openclaw.json
{
  gateway: {
    mode: "local",
    port: 18789,
    bind: "loopback",
    auth: {
      mode: "token",
      token: "${OPENCLAW_GATEWAY_TOKEN}"
    }
  },
  agents: {
    defaults: {
      workspace: "~/.openclaw/workspace"
    }
  },
  channels: {
    whatsapp: {
      dmPolicy: "pairing",
      allowFrom: ["+15555550123"]
    }
  }
}
```

### Example: Dev Profile Config

```json5
// ~/.openclaw-dev/openclaw.json
{
  gateway: {
    mode: "local",
    port: 18790,  // Different port!
    bind: "loopback",
    auth: {
      mode: "token",
      token: "${OPENCLAW_DEV_GATEWAY_TOKEN}"
    }
  },
  agents: {
    defaults: {
      workspace: "~/.openclaw-dev/workspace"  // Different workspace!
    }
  },
  channels: {
    telegram: {
      botToken: "${TELEGRAM_DEV_BOT_TOKEN}",  // Dev bot token
      dmPolicy: "open",
      allowFrom: ["*"]
    }
  }
}
```

### Example: Rescue Profile Config

```json5
// ~/.openclaw-rescue/openclaw.json
{
  gateway: {
    mode: "local",
    port: 18791,  // Different port!
    bind: "loopback",
    auth: {
      mode: "token",
      token: "${OPENCLAW_RESCUE_GATEWAY_TOKEN}"
    }
  },
  agents: {
    defaults: {
      workspace: "~/.openclaw-rescue/workspace",
      model: {
        primary: "anthropic/claude-haiku-4"  // Faster, cheaper model
      }
    }
  },
  channels: {
    whatsapp: {
      dmPolicy: "allowlist",
      allowFrom: ["+15555550123"]  // Only admin
    }
  }
}
```

## Directory Structure

### Default Profile
```
~/.openclaw/
├── openclaw.json
├── credentials/
├── agents/
│   └── main/
│       ├── sessions/
│       └── workspace/
└── logs/
```

### Dev Profile
```
~/.openclaw-dev/
├── openclaw.json
├── credentials/
├── agents/
│   └── main/
│       ├── sessions/
│       └── workspace/
└── logs/
```

### Rescue Profile
```
~/.openclaw-rescue/
├── openclaw.json
├── credentials/
├── agents/
│   └── main/
│       ├── sessions/
│       └── workspace/
└── logs/
```

## Port Management

### Port Allocation Strategy

**Recommended Port Ranges:**
- Default profile: 18789
- Dev profile: 18790
- Rescue profile: 18791
- Test profiles: 18792-18799

**Checking Port Usage:**
```bash
# Check if port is in use
lsof -i :18789
lsof -i :18790
lsof -i :18791

# Probe all gateways
openclaw gateway probe
```

## Environment Variables

### Profile-Specific Environment Variables

**Default Profile:**
```bash
export OPENCLAW_GATEWAY_TOKEN="default-token"
export ANTHROPIC_API_KEY="sk-ant-default"
```

**Dev Profile:**
```bash
export OPENCLAW_DEV_GATEWAY_TOKEN="dev-token"
export ANTHROPIC_API_KEY="sk-ant-dev"
export TELEGRAM_DEV_BOT_TOKEN="123:abc"
```

**Rescue Profile:**
```bash
export OPENCLAW_RESCUE_GATEWAY_TOKEN="rescue-token"
export ANTHROPIC_API_KEY="sk-ant-rescue"
```

### Using .env Files

**Default Profile:**
```bash
# ~/.openclaw/.env
OPENCLAW_GATEWAY_TOKEN=default-token
ANTHROPIC_API_KEY=sk-ant-default
```

**Dev Profile:**
```bash
# ~/.openclaw-dev/.env
OPENCLAW_GATEWAY_TOKEN=dev-token
ANTHROPIC_API_KEY=sk-ant-dev
TELEGRAM_DEV_BOT_TOKEN=123:abc
```

## Daemon Management

### Installing Multiple Daemons

**macOS (launchd):**
```bash
# Install default profile daemon
openclaw onboard --install-daemon

# Install dev profile daemon
openclaw onboard --install-daemon --profile dev

# Install rescue profile daemon
openclaw onboard --install-daemon --profile rescue
```

**Service Files:**
```
~/Library/LaunchAgents/ai.openclaw.gateway.plist           # default
~/Library/LaunchAgents/ai.openclaw.gateway.dev.plist       # dev
~/Library/LaunchAgents/ai.openclaw.gateway.rescue.plist    # rescue
```

**Linux (systemd):**
```bash
# Install default profile daemon
openclaw onboard --install-daemon

# Install dev profile daemon
openclaw onboard --install-daemon --profile dev

# Install rescue profile daemon
openclaw onboard --install-daemon --profile rescue
```

**Service Files:**
```
~/.config/systemd/user/openclaw-gateway.service           # default
~/.config/systemd/user/openclaw-gateway-dev.service       # dev
~/.config/systemd/user/openclaw-gateway-rescue.service    # rescue
```

### Managing Multiple Daemons

**macOS:**
```bash
# Start/stop default profile
launchctl start ai.openclaw.gateway
launchctl stop ai.openclaw.gateway

# Start/stop dev profile
launchctl start ai.openclaw.gateway.dev
launchctl stop ai.openclaw.gateway.dev

# Start/stop rescue profile
launchctl start ai.openclaw.gateway.rescue
launchctl stop ai.openclaw.gateway.rescue
```

**Linux:**
```bash
# Start/stop default profile
systemctl --user start openclaw-gateway
systemctl --user stop openclaw-gateway

# Start/stop dev profile
systemctl --user start openclaw-gateway-dev
systemctl --user stop openclaw-gateway-dev

# Start/stop rescue profile
systemctl --user start openclaw-gateway-rescue
systemctl --user stop openclaw-gateway-rescue
```

## Best Practices

### 1. Use Descriptive Profile Names
```bash
# Good
openclaw gateway --profile production
openclaw gateway --profile staging
openclaw gateway --profile development

# Avoid
openclaw gateway --profile p1
openclaw gateway --profile test123
```

### 2. Isolate Credentials
- Each profile should have its own credentials
- Never share credentials between profiles
- Use separate API keys for dev/prod

### 3. Different Ports for Each Profile
- Always use unique ports
- Document port assignments
- Check for conflicts before starting

### 4. Separate Workspaces
- Each profile should have its own workspace
- Avoid sharing workspaces between profiles
- Use profile-specific paths

### 5. Profile-Specific Logging
- Each profile logs to its own directory
- Use profile name in log file names
- Monitor logs separately

## Common Scenarios

### Scenario 1: Development + Production

**Setup:**
```bash
# Production Gateway (default profile)
openclaw onboard --install-daemon

# Development Gateway (dev profile)
openclaw onboard --install-daemon --profile dev
```

**Usage:**
```bash
# Production: stable, production channels
openclaw gateway status

# Development: test channels, experimental features
openclaw gateway status --profile dev
```

### Scenario 2: Rescue Bot

**Setup:**
```bash
# Main Gateway (default profile)
openclaw onboard --install-daemon

# Rescue Gateway (rescue profile)
openclaw onboard --install-daemon --profile rescue
```

**Usage:**
```bash
# Main Gateway: full features
openclaw gateway status

# Rescue Gateway: minimal config, backup
openclaw gateway status --profile rescue
```

### Scenario 3: Multi-User

**Setup:**
```bash
# Alice's Gateway
openclaw onboard --install-daemon --profile alice

# Bob's Gateway
openclaw onboard --install-daemon --profile bob
```

**Usage:**
```bash
# Alice's Gateway
openclaw gateway status --profile alice

# Bob's Gateway
openclaw gateway status --profile bob
```

## Troubleshooting

### Issue 1: Port Conflicts
```bash
# Check which profile is using which port
openclaw gateway probe

# Change port in config
# ~/.openclaw-dev/openclaw.json
{
  gateway: { port: 18790 }
}
```

### Issue 2: Profile Not Found
```bash
# List all profiles
ls -la ~/.openclaw*

# Create profile directory
mkdir -p ~/.openclaw-dev

# Initialize profile config
openclaw onboard --profile dev
```

### Issue 3: Daemon Conflicts
```bash
# Check all running daemons
# macOS
launchctl list | grep openclaw

# Linux
systemctl --user list-units | grep openclaw
```

### Issue 4: Credential Isolation
```bash
# Verify credentials are separate
ls -la ~/.openclaw/credentials/
ls -la ~/.openclaw-dev/credentials/
ls -la ~/.openclaw-rescue/credentials/
```

## 2026 Updates

### New Features
- Enhanced profile isolation
- Automatic port conflict detection
- Profile-specific discovery
- Improved daemon management
- Better error messages for profile conflicts

### Profile Discovery
```bash
# Discover all local profiles
openclaw gateway probe

# Output shows all profiles:
# - default (port 18789)
# - dev (port 18790)
# - rescue (port 18791)
```
