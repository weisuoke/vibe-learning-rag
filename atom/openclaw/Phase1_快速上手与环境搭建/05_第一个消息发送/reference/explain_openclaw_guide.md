---
source: https://github.com/centminmod/explain-openclaw/blob/master/README.md
title: Explain OpenClaw - Integrated Beginner + Technical Guide
fetched_at: 2026-02-22
---

# Explain OpenClaw (formerly Moltbot/Clawdbot) - Integrated Beginner + Technical Guide

## What is OpenClaw? (30-second version)

OpenClaw is a **self-hosted AI assistant platform**. You run an always-on process called the **Gateway** on a machine you control (a Mac mini at home or an isolated VPS). The Gateway connects to messaging apps (WhatsApp/Telegram/Discord/iMessage/… via built-in channels + plugins), receives messages, runs an agent turn (the "brain"), optionally invokes tools/devices, and sends responses back.

**Key idea:** your **Gateway host** is the trust boundary. If it's compromised (or configured too openly), your assistant can be turned into a data-exfil / automation engine.

## Quick start (safe-ish defaults)

The repo strongly recommends using the onboarding wizard; it sets up:

- a working Gateway service (launchd/systemd)
- auth/provider credentials
- safe access defaults (pairing, token)

### Install

Recommended installer:

```bash
curl -fsSL https://openclaw.ai/install.sh | bash
```

Alternative:

```bash
npm install -g openclaw@latest
```

### Onboard + install background service

```bash
openclaw onboard --install-daemon
```

### Verify

```bash
openclaw gateway status
openclaw status
openclaw health
```

### Security audit

Three levels of security auditing:

```bash
# Read-only scan of config + filesystem permissions (no network calls)
openclaw security audit

# Everything above + live WebSocket probe of the running gateway
openclaw security audit --deep

# Apply safe auto-fixes first, then run full audit
openclaw security audit --fix --deep
```
