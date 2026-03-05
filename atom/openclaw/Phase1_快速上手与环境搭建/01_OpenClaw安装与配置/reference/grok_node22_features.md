# Node.js 22 Features - Official Release

**Source:** https://nodejs.org/en/blog/announcements/v22-release-announce
**Fetched:** 2026-02-21

## Node.js 22 Key Features for OpenClaw

### Critical Requirements

**OpenClaw requires Node.js 22+** for the following features:

### 1. V8 Engine Update to 12.4

**New JavaScript Features:**
- WebAssembly Garbage Collection
- `Array.fromAsync()` - async iteration support
- Set methods (union, intersection, difference)
- Iterator helpers

**Performance:**
- Maglev Compiler enabled by default
- Improves performance for short-lived CLI programs (like OpenClaw)

### 2. ESM Support Improvements

**Synchronous require() for ESM:**
```javascript
// Now possible with --experimental-require-module flag
const esmModule = require('./esm-module.mjs');
```

**Requirements:**
- Module must be explicitly marked as ESM (`"type": "module"` or `.mjs`)
- Must be fully synchronous (no top-level await)
- Returns module namespace object directly

**Future:** Will be enabled by default (no flag needed)

### 3. Native TypeScript Support

**From Node.js 22.6.0+:**
```bash
# Run TypeScript directly (experimental)
node --experimental-strip-types script.ts
```

**From Node.js 22.18.0+:**
- Native TypeScript support (erasable syntax only)
- No transpilation needed for simple types

### 4. Built-in WebSocket Client

**Now stable (no flag needed):**
```javascript
const ws = new WebSocket('ws://localhost:8080');
ws.onmessage = (event) => console.log(event.data);
```

**Benefits for OpenClaw:**
- No external dependencies for WebSocket communication
- Browser-compatible API
- Stable and production-ready

### 5. Package.json Scripts Runner

**New `--run` flag:**
```bash
# Instead of npm run test
node --run test

# Executes scripts from package.json
```

**Benefits:**
- Faster than npm/pnpm script runners
- No package manager needed
- Direct Node.js execution

### 6. Watch Mode (Stable)

**Auto-restart on file changes:**
```bash
node --watch server.js
```

**Status:** Now stable (was experimental in Node.js 18-20)

### 7. Stream Performance Improvements

**Default High Water Mark increased:**
- Old: 16 KiB
- New: 64 KiB
- Performance boost across the board
- Slightly higher memory usage

### 8. File System Enhancements

**New `glob` and `globSync` functions:**
```javascript
import { glob } from 'node:fs';

// Pattern matching for file paths
const files = await glob('**/*.js');
```

### 9. AbortSignal Performance

**Significantly improved:**
- Faster `AbortSignal` creation
- Better performance for `fetch()` and test runner
- Critical for CLI tools making many API calls

## Version Compatibility

**Node.js 22 LTS Timeline:**
- Released: April 2024
- Current: Until October 2024
- LTS: October 2024 - April 2027
- Maintenance: April 2027 - April 2029

**OpenClaw Requirement:**
- Minimum: Node.js 22.0.0
- Recommended: Node.js 22.18.0+ (native TypeScript support)

## Migration from Node.js 20

**Breaking Changes:**
- V8 engine updated (some native modules may need recompilation)
- Default stream High Water Mark changed
- Some deprecated APIs removed

**Upgrade Path:**
```bash
# Using nvm
nvm install 22
nvm use 22

# Using asdf
asdf install nodejs 22.18.0
asdf global nodejs 22.18.0

# Verify
node --version  # Should show v22.x.x
```

## Why OpenClaw Needs Node.js 22

1. **ESM Support:** OpenClaw uses modern ES modules
2. **WebSocket:** Built-in WebSocket for gateway communication
3. **Performance:** Maglev compiler improves CLI startup time
4. **TypeScript:** Native support for TypeScript configuration files
5. **Stability:** LTS support until 2029

---

*For complete release notes, see: https://nodejs.org/blog/release/v22.0.0*
