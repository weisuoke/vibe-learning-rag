# Node.js CLI Apps Best Practices

**Source:** https://github.com/lirantal/nodejs-cli-apps-best-practices
**Fetched:** 2026-02-21

## Key Best Practices for OpenClaw Installation & Configuration

### 1. Command Line Experience

#### 1.1 Respect POSIX Args
- Use POSIX-compliant argument syntax
- Short form: `-h` (single letter)
- Long form: `--help` (full word)
- Optional args: `[arg]`
- Required args: `<arg>`

**OpenClaw Example:**
```bash
openclaw onboard              # Command
openclaw gateway start        # Subcommand
openclaw gateway start -d     # With flag
openclaw --version            # Global flag
```

#### 1.2 Build Empathic CLIs
- Provide interactive prompts when data is missing
- Don't just show error messages
- Guide users to successful interactions

**OpenClaw Onboarding Wizard:**
- Interactive prompts for API keys
- Validates input before proceeding
- Provides helpful error messages

#### 1.3 Stateful Data
- Store configuration in standard locations
- Unix/Linux: `~/.config/app-name/`
- macOS: `~/.config/app-name/` or `~/Library/Application Support/`
- Windows: `%APPDATA%\app-name\`

**OpenClaw Configuration:**
- Location: `~/.openclaw/`
- Files: `settings.json`, `gateway.log`

#### 1.7 Zero Configuration
- Provide sensible defaults
- Allow configuration override
- Work out-of-box when possible

**OpenClaw Approach:**
- Onboarding wizard sets up defaults
- Can run with minimal configuration
- Advanced users can customize

#### 1.8 Respect POSIX Signals
- Handle `SIGINT` (Ctrl+C) gracefully
- Clean up resources on exit
- Save state before terminating

### 2. Distribution

#### 2.1 Small Dependency Footprint
- Minimize dependencies
- Reduce install time and disk usage
- Improve security surface

#### 2.2 Use Shrinkwrap
- Lock dependency versions
- Ensure reproducible installs
- Use `package-lock.json` (npm) or `pnpm-lock.yaml` (pnpm)

#### 2.3 Cleanup Configuration Files
- Remove config files on uninstall
- Provide uninstall instructions
- Don't leave orphaned files

### 3. Interoperability

#### 3.1 Accept Input as STDIN
```bash
# Allow piping
echo "data" | openclaw process
cat file.txt | openclaw process
```

#### 3.2 Enable Structured Output
```bash
# JSON output for scripting
openclaw status --json
openclaw list --format=json
```

#### 3.3 Cross-Platform Etiquette
- Use `path.join()` for file paths
- Handle line endings (CRLF vs LF)
- Test on Windows, macOS, Linux

#### 3.4 Configuration Precedence
**Priority order (highest to lowest):**
1. Command-line flags: `--config=path`
2. Environment variables: `OPENCLAW_CONFIG`
3. Project config: `./.openclawrc`
4. User config: `~/.openclaw/settings.json`
5. Default values

### 4. Accessibility

#### 4.1 Containerize the CLI
- Provide Docker image
- Simplify deployment
- Ensure consistent environment

#### 4.2 Graceful Degradation
- Work without color support
- Handle missing features
- Provide fallbacks

#### 4.3 Node.js Version Compatibility
- Specify minimum version in `package.json`
- Test on supported versions
- Document version requirements

**OpenClaw:**
```json
{
  "engines": {
    "node": ">=22.0.0"
  }
}
```

#### 4.4 Shebang Autodetect
```javascript
#!/usr/bin/env node
// Automatically finds node in PATH
```

### 5. Errors

#### 6.1 Trackable Errors
- Include error codes
- Make errors searchable
- Provide context

```typescript
throw new Error('[OPENCLAW_001] Failed to connect to gateway');
```

#### 6.2 Actionable Errors
- Tell users what went wrong
- Suggest how to fix it
- Provide next steps

**Bad:**
```
Error: Connection failed
```

**Good:**
```
Error: Failed to connect to OpenClaw Gateway at http://localhost:3000

Possible causes:
1. Gateway is not running (run: openclaw gateway start)
2. Wrong port configured (check: ~/.openclaw/settings.json)
3. Firewall blocking connection

For more help: openclaw gateway --help
```

#### 6.3 Provide Debug Mode
```bash
# Enable verbose logging
openclaw --debug gateway start
DEBUG=openclaw:* openclaw gateway start
```

#### 6.4 Proper Exit Codes
- `0`: Success
- `1`: General error
- `2`: Misuse of command
- `126`: Command cannot execute
- `127`: Command not found
- `130`: Terminated by Ctrl+C

#### 6.5 Effortless Bug Reports
- Collect system information
- Include relevant logs
- Provide report template

```bash
openclaw report-bug
# Generates bug report with:
# - Node.js version
# - OpenClaw version
# - OS information
# - Recent logs
```

### 7. Development

#### 7.1 Use bin Object
```json
{
  "bin": {
    "openclaw": "./dist/cli.js"
  }
}
```

#### 7.2 Use Relative Paths
- Import with relative paths
- Don't assume global installation
- Work in any directory

#### 7.3 Use files Field
```json
{
  "files": [
    "dist/",
    "README.md",
    "LICENSE"
  ]
}
```

### 9. Versioning

#### 9.1 Include --version Flag
```bash
openclaw --version
# Output: 1.2.3
```

#### 9.2 Use Semantic Versioning
- MAJOR.MINOR.PATCH
- Breaking changes: increment MAJOR
- New features: increment MINOR
- Bug fixes: increment PATCH

#### 9.5 Backward Compatibility
- Don't break existing workflows
- Deprecate before removing
- Provide migration guides

### 10. Security

#### 10.1 Minimize Argument Injection
- Validate user input
- Sanitize file paths
- Escape shell commands
- Use parameterized queries

**Vulnerable:**
```typescript
exec(`openclaw process ${userInput}`);
```

**Safe:**
```typescript
execFile('openclaw', ['process', userInput]);
```

## OpenClaw-Specific Applications

### Installation Best Practices
1. **Global installation:** Use npm/pnpm global install
2. **PATH setup:** Automatically added by package managers
3. **Verification:** Provide `openclaw --version` check
4. **Onboarding:** Interactive wizard for first-time setup

### Configuration Best Practices
1. **Location:** `~/.openclaw/` (cross-platform)
2. **Format:** JSON for human readability
3. **Validation:** Check config on startup
4. **Defaults:** Sensible defaults for all settings

### Gateway Management
1. **Start/stop commands:** Clear subcommands
2. **Status checking:** `openclaw gateway status`
3. **Logging:** Structured logs in `~/.openclaw/gateway.log`
4. **Process management:** Daemon mode with PID file

---

*Full guide contains 37 best practices across 12 categories. See repository for complete details.*
