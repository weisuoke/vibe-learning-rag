# macOS Gateway Readiness Detection Issue

**Source**: GitHub Issue #6156
**Status**: Open (Fix in PR #8260)
**Impact**: Blocks macOS app onboarding

## Problem Summary

Setup Wizard shows "Gateway did not become ready. Check that it is running." and clicking Retry loops forever, even though gateway process is running.

## Symptoms

1. Gateway process confirmed running via `ps`
2. Wizard never detects readiness
3. No errors shown in terminal even with `--log-level debug`
4. "Configure later" option is irreversible - no way to re-run setup

## Root Cause

When gateway can't start (e.g., `openclaw` CLI not found in PATH), `enableLaunchdGateway()` fails immediately and sets `status = .failed(reason)`. But `waitForGatewayReady()` never checked for this - it kept polling the WebSocket health endpoint for the full 12-second timeout, then showed a generic "Gateway did not become ready" error that hid the actual cause.

## Fix (PR #8260)

Two lines in `waitForGatewayReady()` to:
1. Bail early when status is already `.failed`
2. Surface the actual failure reason in wizard error dialog
3. Add "Run Setup..." button in menu bar so "Configure later" is reversible

## Workarounds

### Workaround 1: Ensure CLI in PATH
```bash
# Verify openclaw CLI is accessible
which openclaw
openclaw --version
```

### Workaround 2: Manual gateway start
```bash
openclaw gateway run
# Or in screen session for persistence
screen -R openclaw
openclaw gateway run
```

### Workaround 3: Check Accessibility permissions
- System Settings → Privacy & Security → Accessibility
- Enable for OpenClaw app

## Related Issues

- #9390: Configuration regression causing startup dead-lock
- #5573: MacOS app Health Check continually pending
