# Chat Channels Overview

**Source**: https://docs.openclaw.ai/channels

## Supported Channels

### Core Channels (Built-in)

1. **WhatsApp** - Most popular; uses Baileys, requires QR pairing
2. **Telegram** - Bot API via grammY; supports groups
3. **Discord** - Discord Bot API + Gateway; servers, channels, DMs
4. **IRC** - Classic IRC servers; channels + DMs with pairing/allowlist
5. **Slack** - Bolt SDK; workspace apps
6. **Signal** - signal-cli; privacy-focused
7. **WebChat** - Gateway WebChat UI over WebSocket

### iMessage Options

- **BlueBubbles** (Recommended) - Uses BlueBubbles macOS server REST API
  - Full feature support: edit, unsend, effects, reactions, group management
  - Note: Edit currently broken on macOS 26 Tahoe
- **iMessage (legacy)** - Legacy macOS integration via imsg CLI (deprecated)

### Plugin Channels (Installed Separately)

- **Feishu** - Feishu/Lark bot via WebSocket
- **Google Chat** - Google Chat API app via HTTP webhook
- **Mattermost** - Bot API + WebSocket; channels, groups, DMs
- **Microsoft Teams** - Bot Framework; enterprise support
- **LINE** - LINE Messaging API bot
- **Nextcloud Talk** - Self-hosted chat via Nextcloud Talk
- **Matrix** - Matrix protocol
- **Nostr** - Decentralized DMs via NIP-04
- **Tlon** - Urbit-based messenger
- **Twitch** - Twitch chat via IRC connection
- **Zalo** - Zalo Bot API; Vietnam's popular messenger
- **Zalo Personal** - Zalo personal account via QR login

## Setup Recommendations

### Fastest Setup

**Telegram** - Simple bot token, no QR pairing needed

### Most Popular

**WhatsApp** - Requires QR pairing, stores more state on disk

### Setup Complexity Ranking

1. **Easiest**: Telegram (bot token only)
2. **Medium**: Discord (bot token + OAuth)
3. **Complex**: WhatsApp (QR pairing + session management)
4. **Advanced**: Signal (signal-cli setup), BlueBubbles (macOS server required)

## Common Configuration Pattern

All channels share the same DM policy pattern:

```json5
{
  channels: {
    telegram: {
      enabled: true,
      botToken: "123:abc",
      dmPolicy: "pairing",   // pairing | allowlist | open | disabled
      allowFrom: ["tg:123"], // only for allowlist/open
    },
  },
}
```

## DM Policy Options

- **pairing** (default): Unknown senders get one-time pairing code to approve
- **allowlist**: Only senders in `allowFrom` (or paired allow store)
- **open**: Allow all inbound DMs (requires `allowFrom: ["*"]`)
- **disabled**: Ignore all DMs

## Important Notes

- Channels can run simultaneously
- OpenClaw routes messages per chat automatically
- Group behavior varies by channel
- DM pairing and allowlists enforced for safety
- Text supported everywhere; media and reactions vary by channel

## Related Documentation

- Groups: /channels/groups
- Security: /channels/security
- Telegram internals: /channels/telegram
- Troubleshooting: /channels/troubleshooting
- Model providers: /providers
