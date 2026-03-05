# OpenClaw CLI Reference Documentation

**Source:** https://docs.openclaw.ai/cli
**Fetched:** 2026-02-21

## CLI Command Tree

```
openclaw [--dev] [--profile <name>] <command>
  setup
  onboard
  configure
  config
    get
    set
    unset
  doctor
  security
    audit
  reset
  uninstall
  update
  channels
    list
    status
    logs
    add
    remove
    logout
  skills
    list
    info
    check
  plugins
    list
    info
    install
    enable
    disable
    doctor
  memory
    status
    index
    search
  message
  agent
  agents
    list
    add
    delete
  acp
  status
  health
  sessions
  gateway
    call
    health
    status
    probe
    discover
    install
    uninstall
    start
    stop
    restart
    run
  logs
  system
    event
    heartbeat last|enable|disable
    presence
  models
    list
    status
    set
    set-image
    aliases list|add|remove
    fallbacks list|add|remove|clear
    image-fallbacks list|add|remove|clear
    scan
    auth add|setup-token|paste-token
    auth order get|set|clear
  sandbox
    list
    recreate
    explain
  cron
    status
    list
    add
    edit
    rm
    enable
    disable
    runs
    run
  nodes
  devices
  node
    run
    status
    install
    uninstall
    start
    stop
    restart
  approvals
    get
    set
    allowlist add|remove
  browser
    status
    start
    stop
    reset-profile
    tabs
    open
    focus
    profiles
    create-profile
    delete-profile
    screenshot
    snapshot
    navigate
    resize
    click
    type
    press
    hover
    drag
    select
    upload
    fill
    dialog
    wait
    evaluate
    console
    pdf
  hooks
    list
    info
    check
    enable
    disable
    install
    update
  webhooks
    gmail setup|run
  pairing
    list
    approve
  docs
  dns
    setup
  tui
```

## Global Flags

- `--dev`: Isolate state under `~/.openclaw-dev` and shift default ports
- `--profile <name>`: Isolate state under `~/.openclaw-<name>`
- `--no-color`: Disable ANSI colors
- `--update`: Shorthand for `openclaw update` (source installs only)
- `-V`, `--version`, `-v`: Print version and exit

## Output Styling

- ANSI colors and progress indicators only render in TTY sessions
- OSC-8 hyperlinks render as clickable links in supported terminals
- `--json` (and `--plain` where supported) disables styling for clean output
- `--no-color` disables ANSI styling; `NO_COLOR=1` is also respected
- Long-running commands show a progress indicator (OSC 9;4 when supported)

## Color Palette

OpenClaw uses a lobster palette for CLI output:

- accent (#FF5A2D): headings, labels, primary highlights
- accentBright (#FF7A3D): command names, emphasis
- accentDim (#D14A22): secondary highlight text
- info (#FF8A5B): informational values
- success (#2FBF71): success states
- warn (#FFB020): warnings, fallbacks, attention
- error (#E23D2D): errors, failures
- muted (#8B7F77): de-emphasis, metadata

## Setup + Onboarding

### setup
Initialize config + workspace. Options:
- `--workspace <dir>`: Agent workspace path (default `~/.openclaw/workspace`)
- `--wizard`: Run the onboarding wizard
- `--non-interactive`: Run wizard without prompts
- `--mode <local|remote>`: Wizard mode
- `--remote-url <url>`: Remote Gateway URL
- `--remote-token <token>`: Remote Gateway token

### onboard
Interactive wizard to set up gateway, workspace, and skills. Options:
- `--workspace <dir>`
- `--reset`: Reset config + credentials + sessions + workspace before wizard
- `--non-interactive`
- `--mode <local|remote>`
- `--flow <quickstart|advanced|manual>`
- `--auth-choice <setup-token|token|...>`
- `--token-provider <id>`
- `--token <token>`
- `--token-profile-id <id>`
- `--token-expires-in <duration>`

### configure
Alias for `config set`.

### config
Manage configuration:
- `config get <key>`: Get config value
- `config set <key> <value>`: Set config value
- `config unset <key>`: Remove config value

### doctor
Diagnostic tool for troubleshooting. Checks:
- Gateway status
- Channel connections
- Configuration issues
- Security settings

## Channel Helpers

### channels
Manage messaging channels:
- `channels list`: Show all configured channels
- `channels status`: Check channel connection status
- `channels logs`: View channel logs
- `channels add`: Add new channel
- `channels remove`: Remove channel
- `channels logout`: Logout from channel

### skills
Manage skills:
- `skills list`: Discover skills
- `skills info <id>`: Show details for a skill
- `skills check`: Check skill status

### pairing
Manage channel pairing:
- `pairing list`: List pending pairing requests
- `pairing approve <channel> <code>`: Approve pairing request

## Messaging + Agent

### message
Send messages to channels. Usage:
```bash
openclaw message send --to <target> --message <text>
```

### agent
Interact with the AI assistant. Usage:
```bash
openclaw agent --message <text> --thinking <level>
```
Options:
- `--message <text>`: Query or command
- `--thinking <level>`: Thinking level (low, medium, high)
- `--deliver=false`: Don't deliver response back to channels

### agents
Manage multiple agents:
- `agents list`: List all agents
- `agents add [name]`: Add new agent
- `agents delete <id>`: Delete agent

### status
Quick system status check. Shows:
- Gateway status
- Active channels
- Model configuration
- Session info

### health
Health check for Gateway and channels.

### sessions
Manage chat sessions.

## Gateway

### gateway
Start the Gateway control plane. Usage:
```bash
openclaw gateway --port 18789 --verbose
```

Subcommands:
- `gateway call`: Call Gateway RPC method
- `gateway health`: Health check
- `gateway status`: Check Gateway status
- `gateway probe`: Probe Gateway
- `gateway discover`: Discover Gateways on network
- `gateway install`: Install Gateway daemon
- `gateway uninstall`: Uninstall Gateway daemon
- `gateway start`: Start Gateway daemon
- `gateway stop`: Stop Gateway daemon
- `gateway restart`: Restart Gateway daemon
- `gateway run`: Run Gateway in foreground

### logs
View Gateway logs.

## Models

### models
Manage AI models:
- `models list`: List available models
- `models status`: Show current model status
- `models set <model>`: Set default model
- `models set-image <model>`: Set image model
- `models aliases list|add|remove`: Manage model aliases
- `models fallbacks list|add|remove|clear`: Manage model fallbacks
- `models scan`: Scan for available models
- `models auth add|setup-token|paste-token`: Manage model authentication
- `models auth order get|set|clear`: Manage auth order

## Security

### security
Security management:
- `security audit`: Audit config + local state for security issues
- `security audit --deep`: Best-effort live Gateway probe
- `security audit --fix`: Tighten safe defaults and chmod state/config

## Plugins

### plugins
Manage extensions:
- `plugins list`: Discover plugins
- `plugins info <id>`: Show details for a plugin
- `plugins install <path|.tgz|npm-spec>`: Install a plugin
- `plugins enable <id>` / `disable <id>`: Toggle plugin
- `plugins doctor`: Report plugin load errors

## Memory

### memory
Vector search over memory files:
- `memory status`: Show index stats
- `memory index`: Reindex memory files
- `memory search "<query>"`: Semantic search over memory

## Chat Slash Commands

Chat messages support `/...` commands:
- `/status`: Quick diagnostics
- `/config`: Persisted config changes
- `/debug`: Runtime-only config overrides (requires `commands.debug: true`)

## Reset / Uninstall

### reset
Reset OpenClaw state (config, sessions, workspace).

### uninstall
Uninstall OpenClaw completely.

## References

- Full CLI docs: https://docs.openclaw.ai/cli
- Individual command pages: https://docs.openclaw.ai/cli/<command>
- Getting Started: https://docs.openclaw.ai/start/getting-started
