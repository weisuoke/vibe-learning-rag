# OpenClaw Gateway Logging and Debugging (2026)

**Source:** Web search results
**Query:** "OpenClaw gateway logging debugging best practices 2026"
**Fetched:** 2026-02-22

## Official Documentation

### 1. OpenClaw Logging Official Documentation
**URL:** https://docs.openclaw.ai/gateway/logging
**Key Points:**
- Gateway logging guide
- Console output and file logs (JSON lines)
- Log levels, --verbose option
- debug/trace settings
- Message body log control

### 2. OpenClaw Security Documentation - Logging Section
**URL:** https://docs.openclaw.ai/gateway/security
**Key Points:**
- Security best practices for logging
- Sensitive data redaction
- Retention policies
- Session transcript management
- Avoid log leakage of secrets
- Recommended configuration

### 3. Technical Best Practices to Securely Deploy OpenClaw
**URL:** https://repello.ai/blog/technical-best-practices-to-securely-deploy-openclaw
**Key Points:**
- Comprehensive logging configuration examples
- Using debug level logging
- syscalls monitoring
- SIEM integration
- Debugging and logging practices

### 4. OpenClaw Debugging and Fixing Issues Video Tutorial
**URL:** https://www.youtube.com/watch?v=dMiirEFH7J8
**Key Points:**
- Multiple OpenClaw debugging methods
- Using `openclaw logs --follow` for real-time log viewing
- Identifying problems
- Verbose mode startup for detailed debugging

### 5. OpenClaw Security Hardening Guide - Log Audit and Monitoring
**URL:** https://aimaker.substack.com/p/openclaw-security-hardening-guide
**Key Points:**
- Three-layer security hardening guide
- Regular session log review
- Tool call auditing
- Anomaly detection
- Enhanced monitoring for effective debugging

## Log Types

### 1. Console Logs (stdout/stderr)
**Description:** Real-time logs printed to terminal

**Characteristics:**
- Immediate feedback
- Colored output (in TTY)
- Timestamp prefixes
- Subsystem filtering

**Usage:**
```bash
# Basic startup with console logs
openclaw gateway

# Verbose console logs
openclaw gateway --verbose

# Filter specific subsystem
openclaw gateway --claude-cli-logs
```

**Log Format:**
```
[2026-02-22T06:45:58.288Z] [gateway] Gateway starting on port 18789
[2026-02-22T06:45:58.289Z] [gateway] Bind mode: loopback
[2026-02-22T06:45:58.290Z] [gateway] Auth mode: token
[2026-02-22T06:45:58.291Z] [gateway] Gateway ready
```

### 2. File Logs (JSON Lines)
**Description:** Structured logs written to disk

**Location:**
```
~/.openclaw/logs/gateway.log
~/.openclaw/logs/gateway-YYYY-MM-DD.log
```

**Characteristics:**
- JSON format (one object per line)
- Persistent across restarts
- Structured data for parsing
- Automatic rotation

**Log Entry Structure:**
```json
{
  "timestamp": "2026-02-22T06:45:58.288Z",
  "level": "info",
  "subsystem": "gateway",
  "message": "Gateway starting on port 18789",
  "context": {
    "port": 18789,
    "bind": "loopback",
    "auth": "token"
  }
}
```

### 3. Session Transcripts
**Description:** Complete conversation history

**Location:**
```
~/.openclaw/agents/<agentId>/sessions/<sessionKey>/transcript.jsonl
```

**Characteristics:**
- Full message history
- Tool calls and results
- Model responses
- Timestamps

**Use Cases:**
- Debugging agent behavior
- Auditing tool usage
- Analyzing conversation flow
- Reproducing issues

### 4. Config Audit Log
**Description:** Configuration change history

**Location:**
```
~/.openclaw/logs/config-audit.jsonl
```

**Characteristics:**
- Tracks all config changes
- Before/after snapshots
- Timestamp and source
- Validation errors

**Use Cases:**
- Tracking config changes
- Debugging config issues
- Security auditing
- Rollback reference

## Log Levels

### 1. error
**Description:** Error conditions that need attention

**Examples:**
- Gateway failed to start
- Port already in use
- Config validation errors
- Channel connection failures

**Configuration:**
```json5
{
  logging: {
    level: "error"
  }
}
```

### 2. warn
**Description:** Warning conditions that may need attention

**Examples:**
- Security audit warnings
- Deprecated config options
- Performance issues
- Non-critical failures

### 3. info (Default)
**Description:** Informational messages

**Examples:**
- Gateway startup
- Channel connections
- Session creation
- Config reloads

### 4. debug
**Description:** Detailed debugging information

**Examples:**
- WebSocket messages
- Tool calls
- Model requests/responses
- Internal state changes

**Configuration:**
```json5
{
  logging: {
    level: "debug"
  }
}
```

**CLI:**
```bash
openclaw gateway --verbose
```

### 5. trace
**Description:** Very detailed tracing information

**Examples:**
- Low-level protocol details
- Raw message payloads
- Internal function calls
- Performance metrics

**Configuration:**
```json5
{
  logging: {
    level: "trace"
  }
}
```

## CLI Commands

### View Logs
```bash
# View recent logs
openclaw logs

# Follow logs in real-time
openclaw logs --follow

# View logs with tail
openclaw logs --tail 100

# View logs for specific date
openclaw logs --date 2026-02-22

# View logs in JSON format
openclaw logs --json
```

### Gateway Startup with Logging
```bash
# Basic startup (info level)
openclaw gateway

# Verbose logging (debug level)
openclaw gateway --verbose

# WebSocket log styles
openclaw gateway --ws-log full
openclaw gateway --ws-log compact
openclaw gateway --ws-log auto

# Compact mode (alias)
openclaw gateway --compact

# Claude CLI logs only
openclaw gateway --claude-cli-logs

# Raw stream logging
openclaw gateway --raw-stream
openclaw gateway --raw-stream-path /path/to/stream.jsonl
```

## Configuration

### Basic Logging Configuration
```json5
{
  logging: {
    level: "info",              // error | warn | info | debug | trace
    redactSensitive: true,      // Redact sensitive values
    console: {
      enabled: true,
      timestamp: true,
      subsystem: true,
      color: true
    },
    file: {
      enabled: true,
      path: "~/.openclaw/logs/gateway.log",
      maxSize: "100MB",
      maxFiles: 10,
      rotation: "daily"
    }
  }
}
```

### Advanced Logging Configuration
```json5
{
  logging: {
    level: "debug",
    redactSensitive: true,
    subsystems: {
      gateway: "info",
      channels: "debug",
      agent: "debug",
      tools: "trace"
    },
    console: {
      enabled: true,
      timestamp: true,
      subsystem: true,
      color: true,
      filter: ["gateway", "channels"]  // Only show these subsystems
    },
    file: {
      enabled: true,
      path: "~/.openclaw/logs/gateway.log",
      format: "json",              // json | text
      maxSize: "100MB",
      maxFiles: 10,
      rotation: "daily",           // daily | size | none
      compress: true
    },
    audit: {
      enabled: true,
      path: "~/.openclaw/logs/audit.jsonl",
      events: ["config", "auth", "tools"]
    }
  }
}
```

### Sensitive Data Redaction
```json5
{
  logging: {
    redactSensitive: true,
    redactPatterns: [
      "password",
      "token",
      "apiKey",
      "secret"
    ]
  }
}
```

**Effect:**
```json
// Before redaction
{"auth": {"token": "sk-ant-1234567890"}}

// After redaction
{"auth": {"token": "[REDACTED]"}}
```

## Debugging Techniques

### 1. Real-Time Log Monitoring
```bash
# Follow logs in real-time
openclaw logs --follow

# Follow with grep filter
openclaw logs --follow | grep "error"

# Follow specific subsystem
openclaw gateway --verbose --claude-cli-logs
```

### 2. WebSocket Message Debugging
```bash
# Full WebSocket logs
openclaw gateway --ws-log full

# Compact WebSocket logs
openclaw gateway --ws-log compact

# Auto mode (smart formatting)
openclaw gateway --ws-log auto
```

### 3. Raw Stream Logging
```bash
# Log raw model stream events
openclaw gateway --raw-stream

# Custom path for raw stream
openclaw gateway --raw-stream-path ~/debug/stream.jsonl
```

### 4. Session Transcript Analysis
```bash
# View session transcript
cat ~/.openclaw/agents/main/sessions/<sessionKey>/transcript.jsonl | jq

# Extract tool calls
cat transcript.jsonl | jq 'select(.type == "tool_use")'

# Extract model responses
cat transcript.jsonl | jq 'select(.type == "text")'
```

### 5. Config Audit Analysis
```bash
# View config changes
cat ~/.openclaw/logs/config-audit.jsonl | jq

# Find recent changes
cat config-audit.jsonl | jq 'select(.timestamp > "2026-02-22")'

# Find validation errors
cat config-audit.jsonl | jq 'select(.error != null)'
```

## Common Debugging Scenarios

### Scenario 1: Gateway Won't Start
```bash
# Check logs for errors
openclaw logs --tail 50

# Run diagnostics
openclaw doctor

# Check port usage
lsof -i :18789

# Try verbose startup
openclaw gateway --verbose
```

### Scenario 2: Channel Connection Issues
```bash
# Check channel status
openclaw channels status

# View channel logs
openclaw logs --follow | grep "channel"

# Verbose startup with channel debugging
openclaw gateway --verbose
```

### Scenario 3: Agent Not Responding
```bash
# Check session transcript
cat ~/.openclaw/agents/main/sessions/<sessionKey>/transcript.jsonl | jq

# Check tool calls
openclaw logs --follow | grep "tool"

# Check model requests
openclaw gateway --verbose --raw-stream
```

### Scenario 4: Performance Issues
```bash
# Enable trace logging
openclaw gateway --verbose

# Monitor resource usage
top -pid $(pgrep -f "openclaw gateway")

# Check session size
du -sh ~/.openclaw/agents/main/sessions/*
```

### Scenario 5: Security Audit Failures
```bash
# Run security audit
openclaw security audit

# Check audit log
cat ~/.openclaw/logs/audit.jsonl | jq

# View sensitive data leaks
openclaw logs | grep -i "redacted"
```

## Best Practices

### 1. Always Enable Sensitive Data Redaction
```json5
{
  logging: {
    redactSensitive: true
  }
}
```

### 2. Use Appropriate Log Levels
- **Production**: `info` or `warn`
- **Development**: `debug`
- **Troubleshooting**: `trace`

### 3. Monitor Logs Regularly
```bash
# Set up log monitoring
openclaw logs --follow | tee -a ~/openclaw-monitor.log
```

### 4. Rotate Logs to Prevent Disk Filling
```json5
{
  logging: {
    file: {
      maxSize: "100MB",
      maxFiles: 10,
      rotation: "daily"
    }
  }
}
```

### 5. Use Structured Logging for Parsing
```bash
# Parse JSON logs with jq
openclaw logs --json | jq 'select(.level == "error")'
```

### 6. Archive Session Transcripts
```bash
# Archive old sessions
tar -czf sessions-backup-$(date +%Y%m%d).tar.gz ~/.openclaw/agents/*/sessions/
```

## 2026 Updates

### New Features
- Enhanced WebSocket logging (`--ws-log` modes)
- Raw stream logging (`--raw-stream`)
- Improved subsystem filtering
- Better sensitive data redaction
- Config audit logging
- Structured JSON logging

### Performance Improvements
- Async log writing
- Log buffering
- Compression support
- Efficient rotation

### Security Enhancements
- Automatic sensitive data redaction
- Audit logging for security events
- Log integrity verification
- Secure log storage permissions
