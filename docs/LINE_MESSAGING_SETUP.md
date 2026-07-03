# LINE Messaging API Setup

LINE Notify is no longer available, so this project sends alerts through the LINE Messaging API from a LINE Official Account.

## Required Values

- `LINE_CHANNEL_ACCESS_TOKEN`: Messaging API channel access token.
- `LINE_USER_ID`: Your LINE user ID, or a group ID if sending to a group.

Do not commit these values to the repository. Use environment variables.

## One-Time Setup

1. Create or open a LINE Official Account.
2. Enable Messaging API in LINE Developers.
3. Issue a long-lived channel access token.
4. Add the Official Account as a friend from your LINE app.
5. Confirm your user ID in the LINE Developers Console, or receive it through a webhook.

## Configure This PC

Temporary, current PowerShell only:

```powershell
$env:LINE_CHANNEL_ACCESS_TOKEN = "YOUR_CHANNEL_ACCESS_TOKEN"
$env:LINE_USER_ID = "Uxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Persistent for your Windows user:

```powershell
[Environment]::SetEnvironmentVariable("LINE_CHANNEL_ACCESS_TOKEN", "YOUR_CHANNEL_ACCESS_TOKEN", "User")
[Environment]::SetEnvironmentVariable("LINE_USER_ID", "Uxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "User")
```

After persistent setup, restart Codex/PowerShell so the new variables are loaded.

## Dry Run

This builds the message and writes a preview, but does not send it.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\send_current_strongest_line_alert.ps1
```

## Send

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\send_current_strongest_line_alert.ps1 -Send
```

Safe mode: send only when the environment variables are configured, otherwise dry-run.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\send_current_strongest_line_alert.ps1 -SendIfConfigured
```

## Dashboard URL

If a public tunnel exists at `outputs\runtime\public_dashboard_tunnel.json`, the LINE alert uses that URL automatically. Otherwise it falls back to a local file URL, which is useful on this PC but not from your phone outside the network.
