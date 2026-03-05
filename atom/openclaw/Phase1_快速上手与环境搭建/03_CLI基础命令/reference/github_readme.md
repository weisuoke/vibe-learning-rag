# OpenClaw GitHub README - CLI Commands Reference

**Source:** https://github.com/openclaw/openclaw/blob/main/README.md
**Fetched:** 2026-02-21

## Quick Start Commands

```bash
# Install OpenClaw
npm install -g openclaw@latest
# or: pnpm add -g openclaw@latest

# Run onboarding wizard
openclaw onboard --install-daemon

# Start Gateway
openclaw gateway --port 18789 --verbose

# Send a message
openclaw message send --to +1234567890 --message "Hello from OpenClaw"

# Talk to the assistant
openclaw agent --message "Ship checklist" --thinking high
```

## Core CLI Commands

### 1. openclaw gateway
- Start the Gateway control plane
- Options:
  - `--port 18789`: Specify port (default 18789)
  - `--verbose`: Enable verbose logging
  - Subcommands: `status`, `stop`, `restart`, `run`

### 2. openclaw message send
- Send messages to connected channels
- Options:
  - `--to <target>`: Recipient (phone number, username, etc.)
  - `--message <text>`: Message content
- Supports: WhatsApp, Telegram, Slack, Discord, Google Chat, Signal, iMessage, etc.

### 3. openclaw agent
- Interact with the AI assistant
- Options:
  - `--message <text>`: Query or command
  - `--thinking <level>`: Thinking level (low, medium, high)
  - `--deliver=false`: Don't deliver response back to channels
- Can deliver responses to any connected channel

### 4. openclaw channels
- Manage messaging channels
- Subcommands:
  - `list`: Show all configured channels
  - `status`: Check channel connection status
  - `add`: Add new channel
  - `remove`: Remove channel
  - `logout`: Logout from channel

### 5. openclaw config
- Configuration management
- Subcommands:
  - `get`: Get config value
  - `set`: Set config value
  - `unset`: Remove config value

## Additional Important Commands

### openclaw onboard
- Interactive setup wizard
- Options:
  - `--install-daemon`: Install Gateway as system service
  - `--workspace <dir>`: Set workspace directory
  - `--reset`: Reset config before wizard
  - `--non-interactive`: Run without prompts

### openclaw doctor
- Diagnostic tool for troubleshooting
- Checks:
  - Gateway status
  - Channel connections
  - Configuration issues
  - Security settings

### openclaw status
- Quick system status check
- Shows:
  - Gateway status
  - Active channels
  - Model configuration
  - Session info

## Development Commands

```bash
# From source
git clone https://github.com/openclaw/openclaw.git
cd openclaw

pnpm install
pnpm ui:build
pnpm build

pnpm openclaw onboard --install-daemon

# Dev loop (auto-reload)
pnpm gateway:watch
```

## Security Notes

- Default DM policy: `pairing` (requires approval)
- Approve pairing: `openclaw pairing approve <channel> <code>`
- Run security audit: `openclaw doctor`

## References

- Official Docs: https://docs.openclaw.ai
- CLI Reference: https://docs.openclaw.ai/cli
- Getting Started: https://docs.openclaw.ai/start/getting-started
- Discord: https://discord.gg/clawd
