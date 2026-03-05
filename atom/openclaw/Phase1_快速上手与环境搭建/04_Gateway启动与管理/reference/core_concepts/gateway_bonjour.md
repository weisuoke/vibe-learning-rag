# Bonjour Discovery - OpenClaw Gateway

**Source:** https://docs.openclaw.ai/gateway/bonjour
**Fetched:** 2026-02-21

## Overview

OpenClaw uses Bonjour (mDNS / DNS‑SD) as a **LAN‑only convenience** to discover an active Gateway (WebSocket endpoint). It is best‑effort and does **not** replace SSH or Tailnet-based connectivity.

## Wide‑area Bonjour (Unicast DNS‑SD) over Tailscale

If the node and gateway are on different networks, multicast mDNS won't cross the boundary. You can keep the same discovery UX by switching to **unicast DNS‑SD** ("Wide‑Area Bonjour") over Tailscale.

### Gateway config (recommended)

```json
{
  "gateway": { "bind": "tailnet" },
  "discovery": { "wideArea": { "enabled": true } }
}
```

### One‑time DNS server setup (gateway host)

```bash
openclaw dns setup --apply
```

This installs CoreDNS and configures it to:
- listen on port 53 only on the gateway's Tailscale interfaces
- serve your chosen domain (example: openclaw.internal.) from ~/.openclaw/dns/<domain>.db

### Tailscale DNS settings

In the Tailscale admin console:
- Add a nameserver pointing at the gateway's tailnet IP (UDP/TCP 53)
- Add split DNS so your discovery domain uses that nameserver

## What advertises

Only the Gateway advertises `_openclaw-gw._tcp`.

## Service types

- `_openclaw-gw._tcp` — gateway transport beacon (used by macOS/iOS/Android nodes)

## TXT keys (non‑secret hints)

The Gateway advertises small non‑secret hints:

- `role=gateway`
- `displayName=<friendly name>`
- `lanHost=<hostname>.local`
- `gatewayPort=<port>` (Gateway WS + HTTP)
- `gatewayTls=1` (only when TLS is enabled)
- `gatewayTlsSha256=<sha256>` (only when TLS is enabled)
- `canvasPort=<port>` (only when canvas host is enabled)
- `sshPort=<port>` (defaults to 22)
- `transport=gateway`
- `cliPath=<path>` (optional; absolute path to openclaw entrypoint)
- `tailnetDns=<magicdns>` (optional hint when Tailnet is available)

**Security notes:**
- Bonjour/mDNS TXT records are **unauthenticated**
- Clients must not treat TXT as authoritative routing
- TLS pinning must never allow an advertised `gatewayTlsSha256` to override a previously stored pin

## Debugging on macOS

```bash
# Browse instances
dns-sd -B _openclaw-gw._tcp local.

# Resolve one instance
dns-sd -L "<instance>" _openclaw-gw._tcp local.
```

## Common failure modes

- **Bonjour doesn't cross networks**: use Tailnet or SSH
- **Multicast blocked**: some Wi‑Fi networks disable mDNS
- **Sleep / interface churn**: macOS may temporarily drop mDNS results; retry
- **Browse works but resolve fails**: keep machine names simple (avoid emojis or punctuation)

## Disabling / configuration

- `OPENCLAW_DISABLE_BONJOUR=1` disables advertising
- `gateway.bind` in `~/.openclaw/openclaw.json` controls the Gateway bind mode
- `OPENCLAW_SSH_PORT` overrides the SSH port advertised in TXT
- `OPENCLAW_TAILNET_DNS` publishes a MagicDNS hint in TXT
- `OPENCLAW_CLI_PATH` overrides the advertised CLI path
