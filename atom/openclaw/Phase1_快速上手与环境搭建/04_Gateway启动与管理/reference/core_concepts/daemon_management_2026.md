# OpenClaw Gateway Daemon Management (2026)

**Source:** Web search results
**Query:** "OpenClaw gateway daemon launchd systemd service management 2026"
**Fetched:** 2026-02-22

## Overview

OpenClaw Gateway runs as a user-level daemon service on macOS (launchd) and Linux (systemd), ensuring the Gateway persists across terminal sessions and system reboots.

## Official Documentation

### 1. OpenClaw GitHub Repository
**URL:** https://github.com/openclaw/openclaw
**Key Points:**
- Gateway runs as launchd/systemd user service
- `openclaw onboard --install-daemon` installs daemon
- Automatic startup on system boot
- User-level service (no root required)

### 2. OpenClaw Getting Started Guide
**URL:** https://docs.openclaw.ai/start/getting-started
**Key Points:**
- Use `openclaw onboard --install-daemon` to install Gateway daemon
- Check status with `openclaw gateway status`
- Access dashboard after daemon starts

### 3. Gateway Runbook
**URL:** https://docs.openclaw.ai/gateway
**Key Points:**
- Covers startup, restart, stop daemon operations
- Supports systemd user service management
- Example: `systemctl --user enable --now openclaw-gateway`

## Platform-Specific Implementation

### macOS (launchd)

**Service File Location:**
```
~/Library/LaunchAgents/ai.openclaw.gateway.plist
```

**Installation:**
```bash
# Install daemon via onboard wizard
openclaw onboard --install-daemon

# Manual launchd commands
launchctl load ~/Library/LaunchAgents/ai.openclaw.gateway.plist
launchctl start ai.openclaw.gateway
launchctl stop ai.openclaw.gateway
launchctl unload ~/Library/LaunchAgents/ai.openclaw.gateway.plist
```

**Key Features:**
- Runs in user context (no sudo required)
- Starts automatically on login
- Persists across terminal sessions
- Logs to `~/.openclaw/logs/`

**Reference:**
- Medium article: "How to Install and Run OpenClaw on Mac"
- URL: https://medium.com/@zilliz_learn/how-to-install-and-run-openclaw-previously-clawdbot-moltbot-on-mac-9cb6adb64eef

### Linux (systemd)

**Service File Location:**
```
~/.config/systemd/user/openclaw-gateway.service
```

**Installation:**
```bash
# Install daemon via onboard wizard
openclaw onboard --install-daemon

# Enable lingering (persist after logout)
loginctl enable-linger $USER

# Manual systemd commands
systemctl --user enable openclaw-gateway
systemctl --user start openclaw-gateway
systemctl --user stop openclaw-gateway
systemctl --user restart openclaw-gateway
systemctl --user status openclaw-gateway
```

**Key Features:**
- User service (no root required)
- Requires lingering enabled for persistence after logout
- Wizard generates unit file automatically
- Logs via journalctl: `journalctl --user -u openclaw-gateway -f`

**Reference:**
- RepoVive guide: "The Gateway Daemon: Linux"
- URL: https://repovive.com/roadmaps/openclaw/setting-up-your-ai-assistant/the-gateway-daemon-linux

## CLI Commands

### Installation
```bash
# Install daemon (interactive wizard)
openclaw onboard --install-daemon

# Install daemon (programmatic)
openclaw gateway install
openclaw gateway install --port 18789
openclaw gateway install --runtime node
openclaw gateway install --token <token>
openclaw gateway install --force
```

### Lifecycle Management
```bash
# Start daemon
openclaw gateway start

# Stop daemon
openclaw gateway stop

# Restart daemon
openclaw gateway restart

# Check status
openclaw gateway status
openclaw gateway status --json
openclaw gateway status --deep

# Uninstall daemon
openclaw gateway uninstall
```

### Status Checking
```bash
# Basic status (service + RPC probe)
openclaw gateway status

# Service-only status (no RPC probe)
openclaw gateway status --no-probe

# Deep scan (includes system-level services)
openclaw gateway status --deep

# JSON output (for scripting)
openclaw gateway status --json
```

## Service Lifecycle

### Startup Sequence
1. System boot / user login
2. launchd/systemd starts openclaw-gateway service
3. Gateway reads config from `~/.openclaw/openclaw.json`
4. Gateway binds to configured port (default 18789)
5. Gateway advertises via Bonjour (if enabled)
6. Gateway ready to accept connections

### Restart Behavior
- Config changes trigger automatic restart (in `hybrid` mode)
- Manual restart via `openclaw gateway restart`
- Restart coalescing (30-second cooldown between restarts)
- Restart sentinel for tracking restart reasons

### Shutdown Sequence
1. Receive SIGTERM/SIGINT signal
2. Stop accepting new connections
3. Gracefully close active sessions
4. Clean up resources (Tailscale, Bonjour)
5. Exit process

## Troubleshooting

### Common Issues

**1. Daemon not starting**
```bash
# Check service status
openclaw gateway status

# Check logs
openclaw logs
openclaw logs --follow

# Reinstall daemon
openclaw gateway uninstall
openclaw gateway install
```

**2. Port already in use**
```bash
# Check port usage
lsof -i :18789

# Force kill existing process
openclaw gateway run --force
```

**3. Daemon not persisting after logout (Linux)**
```bash
# Enable lingering
loginctl enable-linger $USER

# Verify lingering
loginctl show-user $USER | grep Linger
```

**4. Config validation errors**
```bash
# Run diagnostics
openclaw doctor

# Auto-fix config issues
openclaw doctor --fix
```

## Best Practices

### 1. Use Daemon for Production
- Always install daemon for persistent operation
- Avoid running Gateway in foreground for production
- Use foreground mode only for development/debugging

### 2. Monitor Service Health
```bash
# Regular health checks
openclaw gateway health

# Status monitoring
openclaw gateway status --deep

# Log monitoring
openclaw logs --follow
```

### 3. Graceful Restarts
```bash
# Use restart command (not stop + start)
openclaw gateway restart

# Config changes auto-restart in hybrid mode
# No manual restart needed for most config changes
```

### 4. Security Hardening
```bash
# Install with authentication
openclaw gateway install --token <long-random-token>

# Verify auth is enabled
openclaw gateway status
```

## 2026 Updates

### New Features
- Improved restart coalescing (30-second cooldown)
- Restart sentinel tracking
- Deep service scanning (`--deep` flag)
- Enhanced status reporting
- Better error messages for service failures

### Breaking Changes
- None (backward compatible with previous versions)

### Deprecations
- Legacy environment variables (use config file instead)
- Direct launchd/systemd manipulation (use CLI commands)
