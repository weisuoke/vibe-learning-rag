# OpenClaw Gateway Foreground Run (2026)

**Source:** Web search results
**Query:** "OpenClaw gateway foreground run development debugging 2026"
**Fetched:** 2026-02-22

## Official Documentation

### 1. OpenClaw CLI Gateway Documentation
**URL:** https://docs.openclaw.ai/cli/gateway
**Key Points:**
- OpenClaw Gateway CLI reference
- Running local Gateway process
- Supports `--port` to specify port
- `--verbose` for debug output
- Used for development and foreground running

### 2. Gateway Runbook - OpenClaw
**URL:** https://docs.openclaw.ai/gateway
**Key Points:**
- OpenClaw Gateway operation manual
- Foreground startup commands like `openclaw gateway --port 18789 --verbose`
- Suitable for development debugging and daily operations

### 3. Getting Started - OpenClaw
**URL:** https://docs.openclaw.ai/start/getting-started
**Key Points:**
- OpenClaw getting started guide
- Recommends foreground running Gateway for quick testing and troubleshooting
- `openclaw gateway --port 18789`
- Supports direct access to Control UI for debugging

### 4. OpenClaw GitHub Repository
**URL:** https://github.com/openclaw/openclaw
**Key Points:**
- OpenClaw official open source repository
- Contains Gateway core implementation
- Foreground running development mode (`pnpm dev`)
- Debugging-related code
- Suitable for 2026 latest development debugging

### 5. OpenClaw Architecture, Explained
**URL:** https://ppaolo.substack.com/p/openclaw-system-architecture-overview
**Key Points:**
- OpenClaw architecture detailed explanation
- Development mode foreground startup Gateway (`pnpm dev`) enables hot reload
- Suitable for development debugging scenarios
- Default binds to 127.0.0.1:18789

## Foreground vs Background (Daemon)

### Foreground Mode

**Characteristics:**
- Runs in current terminal session
- Logs output to stdout/stderr
- Stops when terminal closes
- Interactive debugging
- Immediate feedback

**Use Cases:**
- Development and debugging
- Quick testing
- Troubleshooting issues
- Learning and experimentation
- One-time operations

**Advantages:**
- ✅ Immediate log visibility
- ✅ Easy to stop (Ctrl+C)
- ✅ No daemon installation needed
- ✅ Quick iteration
- ✅ Interactive debugging

**Disadvantages:**
- ❌ Stops when terminal closes
- ❌ No automatic restart
- ❌ Not suitable for production
- ❌ Requires active terminal session

### Background Mode (Daemon)

**Characteristics:**
- Runs as system service
- Persists across terminal sessions
- Automatic startup on boot
- Logs to files
- Production-ready

**Use Cases:**
- Production deployment
- Long-running operations
- Persistent bot operation
- Server environments
- Unattended operation

**Advantages:**
- ✅ Persists across sessions
- ✅ Automatic restart
- ✅ Production-ready
- ✅ No active terminal needed
- ✅ System integration

**Disadvantages:**
- ❌ Less immediate feedback
- ❌ Requires daemon installation
- ❌ Logs in separate files
- ❌ More complex debugging

## Foreground Startup Commands

### Basic Foreground Startup

```bash
# Simplest form (default port 18789)
openclaw gateway

# Explicit foreground run
openclaw gateway run

# With port specification
openclaw gateway --port 18789

# With bind mode
openclaw gateway --bind loopback
```

### Development Mode Startup

```bash
# Verbose logging (debug level)
openclaw gateway --verbose

# Development mode (creates dev config if missing)
openclaw gateway --dev

# Development mode with reset
openclaw gateway --dev --reset

# Allow unconfigured (skip gateway.mode=local check)
openclaw gateway --allow-unconfigured
```

### Debugging Options

```bash
# Full WebSocket logs
openclaw gateway --ws-log full

# Compact WebSocket logs
openclaw gateway --ws-log compact
openclaw gateway --compact  # alias

# Claude CLI logs only
openclaw gateway --claude-cli-logs

# Raw stream logging
openclaw gateway --raw-stream
openclaw gateway --raw-stream-path ~/debug/stream.jsonl
```

### Port and Binding Options

```bash
# Custom port
openclaw gateway --port 8080

# LAN binding (requires auth)
openclaw gateway --bind lan --token <token>

# Tailnet binding
openclaw gateway --bind tailnet --tailscale serve

# Force kill existing process on port
openclaw gateway --force
```

### Authentication Options

```bash
# Token authentication
openclaw gateway --auth token --token <token>

# Password authentication
openclaw gateway --auth password --password <password>

# No authentication (loopback only)
openclaw gateway --auth none
```

### Profile-Specific Foreground Run

```bash
# Default profile
openclaw gateway

# Dev profile
openclaw gateway --profile dev --port 18790

# Rescue profile
openclaw gateway --profile rescue --port 18791
```

## Development Workflow

### Typical Development Session

**1. Start Gateway in Foreground:**
```bash
# Terminal 1: Gateway with verbose logging
openclaw gateway --verbose --dev
```

**2. Open Control UI:**
```bash
# Browser: http://127.0.0.1:18789
```

**3. Test Changes:**
```bash
# Terminal 2: Send test messages
openclaw message send --to +1234567890 --message "Test"

# Terminal 2: Check agent status
openclaw agent --message "Hello"
```

**4. Monitor Logs:**
```bash
# Terminal 3: Follow logs
openclaw logs --follow
```

**5. Stop Gateway:**
```bash
# Terminal 1: Press Ctrl+C
```

### Hot Reload Development

**From Source (pnpm dev):**
```bash
# Clone repository
git clone https://github.com/openclaw/openclaw.git
cd openclaw

# Install dependencies
pnpm install

# Build packages
pnpm build

# Start Gateway with hot reload
pnpm dev

# Or run specific package
cd packages/openclaw
pnpm dev
```

**Characteristics:**
- Automatic restart on code changes
- TypeScript compilation on save
- Fast iteration cycle
- Development-optimized

### Quick Testing Workflow

**1. Quick Start:**
```bash
openclaw gateway --dev --verbose
```

**2. Test Feature:**
```bash
# In another terminal
openclaw agent --message "Test feature"
```

**3. Check Logs:**
```bash
# Logs are in the foreground terminal
# Or use:
openclaw logs --tail 50
```

**4. Stop and Restart:**
```bash
# Ctrl+C to stop
# Up arrow + Enter to restart
```

## Debugging Techniques

### 1. Verbose Logging

```bash
# Enable verbose logging
openclaw gateway --verbose

# Output includes:
# - Debug-level messages
# - WebSocket traffic
# - Tool calls
# - Model requests/responses
```

### 2. WebSocket Debugging

```bash
# Full WebSocket logs (all messages)
openclaw gateway --ws-log full

# Compact WebSocket logs (summary)
openclaw gateway --ws-log compact

# Auto mode (smart formatting)
openclaw gateway --ws-log auto
```

### 3. Raw Stream Logging

```bash
# Log raw model stream events
openclaw gateway --raw-stream

# Custom path
openclaw gateway --raw-stream-path ~/debug/stream-$(date +%Y%m%d-%H%M%S).jsonl
```

### 4. Claude CLI Logs Only

```bash
# Filter to show only claude-cli subsystem
openclaw gateway --claude-cli-logs

# Useful for debugging agent behavior
```

### 5. Development Mode

```bash
# Create dev config if missing
openclaw gateway --dev

# Reset dev config + credentials + sessions
openclaw gateway --dev --reset

# Useful for clean slate testing
```

## Common Development Scenarios

### Scenario 1: Testing New Channel Integration

```bash
# Start Gateway with verbose logging
openclaw gateway --verbose --dev

# In another terminal: check channel status
openclaw channels status

# Send test message
openclaw message send --to <channel-id> --message "Test"

# Monitor logs for errors
# (logs are in the foreground terminal)
```

### Scenario 2: Debugging Agent Behavior

```bash
# Start Gateway with full WebSocket logs
openclaw gateway --ws-log full --verbose

# In another terminal: interact with agent
openclaw agent --message "Debug test"

# Watch tool calls and model responses in real-time
```

### Scenario 3: Testing Configuration Changes

```bash
# Start Gateway in foreground
openclaw gateway --verbose

# In another terminal: edit config
vim ~/.openclaw/openclaw.json

# Gateway auto-reloads (in hybrid mode)
# Watch logs for reload messages
```

### Scenario 4: Port Conflict Debugging

```bash
# Check port usage
lsof -i :18789

# Force kill existing process
openclaw gateway --force --verbose

# Or use different port
openclaw gateway --port 18790 --verbose
```

### Scenario 5: Authentication Testing

```bash
# Start Gateway with token auth
openclaw gateway --auth token --token test-token-123 --verbose

# In another terminal: test connection
openclaw gateway health --token test-token-123

# Watch auth logs in foreground terminal
```

## Stopping Foreground Gateway

### Graceful Shutdown

```bash
# Press Ctrl+C in the terminal
# Gateway will:
# 1. Stop accepting new connections
# 2. Gracefully close active sessions
# 3. Clean up resources (Tailscale, Bonjour)
# 4. Exit process
```

### Force Shutdown

```bash
# Press Ctrl+C twice
# Or send SIGKILL:
kill -9 $(pgrep -f "openclaw gateway")
```

### Shutdown Signals

**SIGINT (Ctrl+C):**
- Graceful shutdown
- Closes connections
- Cleans up resources

**SIGTERM:**
- Graceful shutdown
- Same as SIGINT

**SIGUSR1:**
- Triggers in-process restart
- Used by config hot reload

## Best Practices

### 1. Use Foreground for Development

```bash
# Development: foreground with verbose logging
openclaw gateway --verbose --dev

# Production: daemon mode
openclaw gateway start
```

### 2. Enable Verbose Logging

```bash
# Always use --verbose during development
openclaw gateway --verbose

# Helps catch issues early
```

### 3. Use Development Mode

```bash
# Development mode creates clean config
openclaw gateway --dev

# Reset when needed
openclaw gateway --dev --reset
```

### 4. Monitor Logs in Real-Time

```bash
# Foreground mode shows logs immediately
# No need to tail log files
```

### 5. Use Separate Terminals

```bash
# Terminal 1: Gateway
openclaw gateway --verbose

# Terminal 2: Testing commands
openclaw agent --message "Test"

# Terminal 3: Log monitoring (if needed)
openclaw logs --follow
```

### 6. Quick Iteration Cycle

```bash
# 1. Start Gateway
openclaw gateway --dev --verbose

# 2. Test feature
openclaw agent --message "Test"

# 3. Stop Gateway (Ctrl+C)

# 4. Make changes

# 5. Restart Gateway (Up arrow + Enter)
```

## Troubleshooting

### Issue 1: Gateway Won't Start

```bash
# Check logs in foreground
openclaw gateway --verbose

# Common issues:
# - Port already in use
# - Config validation errors
# - Missing dependencies
```

### Issue 2: Port Already in Use

```bash
# Check what's using the port
lsof -i :18789

# Force kill and start
openclaw gateway --force --verbose
```

### Issue 3: Config Validation Errors

```bash
# Run diagnostics
openclaw doctor

# Start with minimal config
openclaw gateway --dev --verbose
```

### Issue 4: No Logs Appearing

```bash
# Ensure verbose mode is enabled
openclaw gateway --verbose

# Check log level in config
# ~/.openclaw/openclaw.json
{
  logging: { level: "debug" }
}
```

### Issue 5: WebSocket Connection Issues

```bash
# Enable full WebSocket logs
openclaw gateway --ws-log full --verbose

# Check bind mode
openclaw gateway --bind loopback --verbose
```

## 2026 Updates

### New Features
- Enhanced verbose logging
- Improved WebSocket log modes
- Raw stream logging
- Better error messages
- Development mode improvements

### Performance Improvements
- Faster startup time
- Reduced memory usage in dev mode
- Better log buffering
- Optimized hot reload

### Developer Experience
- Clearer log formatting
- Better error messages
- Improved debugging tools
- Enhanced development mode
