# Systemd Installation Issue on Ubuntu

**Source**: GitHub Issue #1818
**Status**: Closed
**Impact**: Critical for Linux VPS deployments

## Problem Summary

The onboarding wizard fails to install systemd service on Ubuntu 22.04/24.04 due to missing DBUS environment variables.

## Root Cause

The `isSystemdUserServiceAvailable` function throws error:
```
Failed to connect to bus: $DBUS_SESSION_BUS_ADDRESS and $XDG_RUNTIME_DIR not defined
```

This happens when:
- User switches with `su` or `sudo -iu` (doesn't create full user session)
- systemd-logind not properly configured
- dbus-user-session not installed

## Solutions

### Solution 1: Install dbus-user-session
```bash
sudo apt install -y dbus-user-session
sudo loginctl enable-linger $USER
# Log out and log back in
systemctl --user status
```

### Solution 2: Use machinectl
```bash
sudo machinectl shell username@.host
systemctl --user status
```

### Solution 3: Manual environment setup
```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export DBUS_SESSION_BUS_ADDRESS=unix:path=$XDG_RUNTIME_DIR/bus
systemctl --user status
```

### Solution 4: SSH PAM configuration
```bash
# Edit /etc/ssh/sshd_config
UsePAM yes

# Verify /etc/pam.d/sshd contains:
session optional pam_systemd.so

# Reboot and enable lingering
sudo loginctl enable-linger $(whoami)
openclaw gateway install
```

## Related PRs

- #3527: Detect and manage systemd system services
- #7730: Handle missing DBUS env vars gracefully
- #8326: Detect missing systemd user session env vars on headless Ubuntu
