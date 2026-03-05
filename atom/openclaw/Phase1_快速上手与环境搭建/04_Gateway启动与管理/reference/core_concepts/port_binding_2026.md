# OpenClaw Gateway Port Binding Configuration (2026)

**Source:** Web search results
**Query:** "OpenClaw gateway port binding configuration network security 2026"
**Fetched:** 2026-02-22

## Overview

OpenClaw Gateway port binding configuration is the core of network security. Default port is 18789, and bind mode directly affects exposure risk. **Always use loopback (127.0.0.1) to avoid external access**, combined with token authentication or Tailscale/SSH tunnel for secure remote access. **Avoid binding to 0.0.0.0 exposing to public internet** - many security incidents originate from this.

## Official Documentation

### 1. OpenClaw Gateway Configuration
**URL:** https://docs.openclaw.ai/gateway/configuration
**Key Points:**
- Complete Gateway configuration guide
- `gateway.port` and bind modes (loopback/lan/tailnet)
- Auth settings and minimal configuration examples
- Ensures secure binding and startup verification

### 2. OpenClaw Gateway Security
**URL:** https://docs.openclaw.ai/gateway/security
**Key Points:**
- Gateway security best practices
- Port binding (recommend loopback to prevent LAN exposure)
- Authentication requirements
- Firewall rules and Tailscale integration
- Avoid public internet exposure risks

### 3. OpenClaw CLI - gateway Command Reference
**URL:** https://docs.openclaw.ai/cli/gateway
**Key Points:**
- Gateway CLI options detailed explanation
- `--port`, `--bind` (loopback|lan|etc modes)
- `--auth` override
- Default refuses non-local mode startup for security

## Bind Modes

### 1. loopback (Default - Recommended)
**Description:** Binds to 127.0.0.1 (localhost only)

**Security Level:** ✅ Highest

**Use Cases:**
- Local development
- Single-user desktop usage
- Remote access via SSH tunnel or Tailscale

**Configuration:**
```json5
{
  gateway: {
    port: 18789,
    bind: "loopback"
  }
}
```

**CLI:**
```bash
openclaw gateway --bind loopback
```

**Characteristics:**
- Only accessible from localhost
- No network exposure
- Safest option
- Requires SSH tunnel or Tailscale for remote access

### 2. lan (LAN Access)
**Description:** Binds to 0.0.0.0 (all interfaces)

**Security Level:** ⚠️ Medium (requires authentication)

**Use Cases:**
- LAN-only access (home network, office network)
- Multiple devices on same network
- iOS/Android app access

**Configuration:**
```json5
{
  gateway: {
    port: 18789,
    bind: "lan",
    auth: {
      mode: "token",
      token: "long-random-token-here"
    }
  }
}
```

**CLI:**
```bash
openclaw gateway --bind lan --token <token>
```

**Security Requirements:**
- **MUST** enable authentication (token or password)
- Gateway refuses to start without auth
- Firewall rules recommended
- Avoid exposing to public internet

### 3. tailnet (Tailscale Only)
**Description:** Binds only to Tailscale interface

**Security Level:** ✅ High

**Use Cases:**
- Secure remote access across networks
- Team collaboration
- Multi-device access without VPN

**Configuration:**
```json5
{
  gateway: {
    port: 18789,
    bind: "tailnet",
    tailscale: {
      mode: "serve"  // or "funnel" for public
    }
  }
}
```

**CLI:**
```bash
openclaw gateway --bind tailnet --tailscale serve
```

**Characteristics:**
- Only accessible via Tailscale network
- End-to-end encrypted
- No public internet exposure
- Supports identity-based auth (trusted-proxy mode)

### 4. auto (Automatic Selection)
**Description:** Chooses bind mode based on Tailscale availability

**Security Level:** ✅ High (if Tailscale available)

**Logic:**
- If Tailscale is running: binds to tailnet
- If Tailscale is not running: binds to loopback

**Configuration:**
```json5
{
  gateway: {
    port: 18789,
    bind: "auto"
  }
}
```

### 5. custom (Custom Address)
**Description:** Binds to user-specified address

**Security Level:** ⚠️ Varies (depends on address)

**Use Cases:**
- Specific interface binding
- Advanced networking scenarios

**Configuration:**
```json5
{
  gateway: {
    port: 18789,
    bind: "custom",
    customBind: "192.168.1.100"
  }
}
```

## Port Configuration

### Default Port
```
18789
```

### Changing Port

**Via Config:**
```json5
{
  gateway: {
    port: 8080
  }
}
```

**Via CLI:**
```bash
openclaw gateway --port 8080
```

**Via Environment Variable:**
```bash
export OPENCLAW_GATEWAY_PORT=8080
openclaw gateway
```

### Port Conflicts

**Check Port Usage:**
```bash
# macOS/Linux
lsof -i :18789

# Check diagnostics
openclaw gateway probe
```

**Force Kill Existing Process:**
```bash
openclaw gateway --force
```

This will:
1. Kill any process listening on the target port
2. Wait for port to be freed
3. Start Gateway on the port

## Authentication Modes

### 1. token (Recommended)
**Description:** Shared secret token

**Configuration:**
```json5
{
  gateway: {
    auth: {
      mode: "token",
      token: "long-random-token-here"
    }
  }
}
```

**CLI:**
```bash
openclaw gateway --auth token --token <token>
```

**Environment Variable:**
```bash
export OPENCLAW_GATEWAY_TOKEN="long-random-token"
openclaw gateway
```

**Client Connection:**
```typescript
const client = new GatewayClient({
  url: "ws://127.0.0.1:18789",
  auth: { token: "long-random-token-here" }
});
```

### 2. password
**Description:** Password-based authentication

**Configuration:**
```json5
{
  gateway: {
    auth: {
      mode: "password",
      password: "secure-password"
    }
  }
}
```

**CLI:**
```bash
openclaw gateway --auth password --password <password>
```

### 3. trusted-proxy
**Description:** Trust reverse proxy headers (Tailscale Serve)

**Configuration:**
```json5
{
  gateway: {
    auth: { mode: "trusted-proxy" },
    tailscale: { mode: "serve" }
  }
}
```

**Use Cases:**
- Tailscale Serve with identity headers
- Reverse proxy with X-Forwarded-* headers

**Security Warning:**
- Only use with trusted reverse proxies
- Never expose trusted-proxy mode to public internet

### 4. none (Dangerous)
**Description:** No authentication

**Security Level:** ❌ Dangerous

**Configuration:**
```json5
{
  gateway: {
    auth: { mode: "none" }
  }
}
```

**Restrictions:**
- Only allowed with `bind: "loopback"`
- Gateway refuses to start with `bind: "lan"` and `auth: "none"`
- Logs warning message

## Security Best Practices (2026)

### 1. Default to Loopback
```json5
{
  gateway: {
    bind: "loopback",
    auth: { mode: "token", token: "${OPENCLAW_GATEWAY_TOKEN}" }
  }
}
```

### 2. Use Tailscale for Remote Access
```json5
{
  gateway: {
    bind: "tailnet",
    tailscale: { mode: "serve" },
    auth: { mode: "trusted-proxy" }
  }
}
```

### 3. LAN Access with Strong Auth
```json5
{
  gateway: {
    bind: "lan",
    auth: {
      mode: "token",
      token: "at-least-32-characters-long-random-token"
    }
  }
}
```

### 4. Firewall Rules
```bash
# macOS (pf)
# Allow only LAN access to port 18789
sudo pfctl -e
echo "pass in proto tcp from 192.168.1.0/24 to any port 18789" | sudo pfctl -f -

# Linux (ufw)
sudo ufw allow from 192.168.1.0/24 to any port 18789
sudo ufw enable
```

### 5. Regular Security Audits
```bash
# Run security audit
openclaw security audit

# Deep audit with live probe
openclaw security audit --deep

# Auto-fix issues
openclaw security audit --fix
```

## Common Security Mistakes

### ❌ Mistake 1: Binding to 0.0.0.0 without Auth
```json5
{
  gateway: {
    bind: "lan",
    auth: { mode: "none" }  // DANGEROUS!
  }
}
```

**Result:** Gateway refuses to start

### ❌ Mistake 2: Weak Token
```json5
{
  gateway: {
    bind: "lan",
    auth: {
      mode: "token",
      token: "123456"  // TOO SHORT!
    }
  }
}
```

**Result:** Security audit warning

### ❌ Mistake 3: Exposing to Public Internet
```bash
# Port forwarding on router
# 0.0.0.0:18789 -> public IP
```

**Result:** High security risk, potential unauthorized access

### ❌ Mistake 4: Using Funnel without Understanding
```json5
{
  gateway: {
    tailscale: { mode: "funnel" }  // PUBLIC INTERNET!
  }
}
```

**Result:** Gateway exposed to entire internet via Tailscale Funnel

## Troubleshooting

### Issue 1: "Gateway failed to start: port already in use"
```bash
# Check what's using the port
lsof -i :18789

# Force kill and start
openclaw gateway --force
```

### Issue 2: "Refusing to bind gateway to lan without auth"
```bash
# Add authentication
openclaw gateway --bind lan --token <long-random-token>
```

### Issue 3: Cannot connect from other devices
```bash
# Check bind mode
openclaw gateway status

# Verify firewall rules
# macOS
sudo pfctl -s rules | grep 18789

# Linux
sudo ufw status | grep 18789
```

### Issue 4: Tailscale not working
```bash
# Check Tailscale status
tailscale status

# Verify Gateway is using tailnet
openclaw gateway status --deep
```

## 2026 Security Updates

### New Security Features
- Enhanced bind mode validation
- Automatic security audit on startup
- Improved error messages for misconfigurations
- Rate limiting for auth attempts
- Better Tailscale integration

### Security Audit Checks
- `gateway.bind_no_auth`: Critical if LAN bind without auth
- `gateway.loopback_no_auth`: Warning if loopback without auth (reverse proxy scenario)
- `gateway.tailscale_funnel`: Critical if Funnel mode enabled
- `hooks.token_too_short`: Warning if webhook token < 32 chars

### Recommended Configuration (2026)
```json5
{
  gateway: {
    mode: "local",
    port: 18789,
    bind: "loopback",
    auth: {
      mode: "token",
      token: "${OPENCLAW_GATEWAY_TOKEN}"
    },
    reload: { mode: "hybrid" }
  }
}
```
