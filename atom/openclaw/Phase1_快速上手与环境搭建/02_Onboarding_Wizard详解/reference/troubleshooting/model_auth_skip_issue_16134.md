# Model/Auth Setup Skipped in Onboarding

**Source**: GitHub Issue #16134
**Status**: Open
**Impact**: Critical for new users - agent completely non-functional after setup

## Problem Summary

After fresh install, `openclaw onboard` completes without prompting for model or API key configuration. The wizard jumps directly to "How do you want to hatch your bot?" and launches TUI/WebChat.

## Symptoms

1. Wizard skips Model/Auth step entirely
2. Defaults to `anthropic/claude-opus-4-6` (requires paid API key)
3. No error or warning about missing credentials
4. TUI shows "conjuring... • 2m 10s | connected" then hangs
5. WebChat shows loading dots (...) forever
6. Re-running `openclaw onboard` does not help

## Affected Versions

- 2026.2.12 (confirmed broken)
- 2026.2.13 (still broken)

## Root Cause

Onboarding regression where the wizard can skip the Model/Auth step and default to a paid model without configuring provider credentials.

## Workarounds

### Workaround 1: Manual configuration after onboarding
```bash
# For Anthropic
openclaw config set env.ANTHROPIC_API_KEY "sk-ant-..."
openclaw gateway restart

# For OpenAI
openclaw config set env.OPENAI_API_KEY "sk-..."
openclaw gateway restart
```

### Workaround 2: Use configure command
```bash
openclaw configure
openclaw gateway restart
```

### Workaround 3: Downgrade to working version
```bash
npm uninstall -g openclaw
npm install -g openclaw@2026.2.9
openclaw onboard
```

## Verification

After applying workaround, verify:
```bash
openclaw --version
openclaw models status --probe
```

Check if gateway is running as service and inheriting shell env properly.
