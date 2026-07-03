# LINE Automation

This project can automatically refresh the current strongest tickets and send a LINE alert when the final ticket set changes.

## One-Shot Update

Use this as an update button. It refreshes JRA official odds, rebuilds the current strongest tickets, rebuilds the dashboard, and sends LINE only when the ticket set changed since the last successful notification.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_current_strongest_line_update.ps1 -SendIfConfigured
```

Force a notification even if the tickets did not change:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_current_strongest_line_update.ps1 -SendIfConfigured -ForceNotify
```

Dry-run without LINE:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_current_strongest_line_update.ps1 -SkipOddsFetch
```

## Watch Mode

Run every 60 seconds. Stop with `Ctrl+C`.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\watch_current_strongest_line_alerts.ps1 -SendIfConfigured -IntervalSeconds 60
```

Run only three times for a smoke test:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\watch_current_strongest_line_alerts.ps1 -SkipOddsFetch -MaxRuns 3
```

## Race-Day Timed Notices

This is the recommended race-day operation.

It refreshes odds and tickets, then sends:

- Around 08:00: the full day race list with `購入候補`, `購入可能性あり`, `要監視`, or `見送り濃厚`.
- Around 5 minutes before each race: recommended tickets, stake, expected payout, and whether it is `購入候補` or `参考・見送り`.
- Around 3 minutes before each race: final version with `購入` or `参考・見送り`.

Start the timed watcher:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\watch_race_day_timed_line_alerts.ps1 -SendIfConfigured -IntervalSeconds 60
```

Smoke test without LINE and without odds fetch:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\watch_race_day_timed_line_alerts.ps1 -SkipOddsFetch -MaxRuns 1 -IntervalSeconds 1
```

Preview the morning message:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& 'C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\send_race_day_line_notifications.py --date 20260620 --event morning --now '2026-06-20 08:00:00'"
```

Preview a 3-minute final message:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& 'C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\send_race_day_line_notifications.py --date 20260620 --event final3 --race-key 2026062002010312 --now '2026-06-20 16:02:00'"
```

Register the timed watcher to start every day at 08:00:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\register_race_day_timed_line_task.ps1 -Register
```

## Windows Scheduled Task

Preview the task settings:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\register_current_strongest_line_watcher_task.ps1
```

Register a daily task. By default it starts at 09:00, refreshes every 60 seconds, and stops after about 9 hours.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\register_current_strongest_line_watcher_task.ps1 -Register
```

Remove the task:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\register_current_strongest_line_watcher_task.ps1 -Unregister
```

## Behavior

- Fetches JRA official win/place, umaren, and wide odds unless `-SkipOddsFetch` is passed.
- Rebuilds `outputs\analysis\current_strongest_runtime_v1\selected_after_live_safety.csv`.
- Rebuilds `outputs\ui\live_odds_dashboard.html`.
- Stores state in `data\processed\notifications\current_strongest_line_state.json`.
- Sends LINE only when the ticket hash changed, unless `-ForceNotify` is passed.

## Credentials

The scripts read Windows user environment variables:

- `LINE_CHANNEL_ACCESS_TOKEN`
- `LINE_USER_ID`

Use `scripts\setup_line_credentials.ps1` if they need to be reconfigured.
