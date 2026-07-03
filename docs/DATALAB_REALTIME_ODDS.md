# Data Lab Realtime Odds

This project can call JRA-VAN Data Lab realtime odds through the locally installed 32-bit JV-Link COM component.

## Important

JV-Link is registered as a 32-bit COM component on this PC. Run the fetch script with 32-bit PowerShell:

```powershell
C:\Windows\SysWOW64\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\fetch_jv_realtime_odds.ps1 -RaceKey 202606140212 -BetType all
```

64-bit PowerShell can see the ProgID but cannot instantiate the COM class.

## SID / Service Key

JV-Link uses the JRA-VAN Data Lab service key as the SID passed to `JVInit`.

The project resolves it in this order:

1. `-Sid` command argument
2. `JV_SID` environment variable
3. Windows registry `servicekey` saved by JRA-VAN Data Lab

On this PC, the saved key is under:

```text
HKLM\SOFTWARE\WOW6432Node\JRA-VAN Data Lab.\uid_pass
value: servicekey
```

Do not write the actual key into project files or command examples. The realtime scripts read it automatically and only write `sid_present` / `sid_source` to metadata.

## Race Key

Use the 16-digit JRA-VAN/TARGET race id whenever it is available:

```text
YYYYMMDDJJKKHHRR
```

Where:

```text
YYYYMMDD + JRA venue code + meeting number + day number + race number
```

Example:

```text
2026062002010312
```

means `2026-06-20`, venue code `02`, 1st meeting, 3rd day, race `12`.

The older 12-digit form is not reliable enough for production live odds because it
can omit meeting/day information. The race-day scripts now prefer a 16-digit
`race_id` from TARGET/JRA-VAN data when present.

## Bet Type Mapping

The script maps `-BetType` to realtime dataspec IDs:

```text
all             -> 0B30
win_place_frame -> 0B31
umaren          -> 0B32
wide            -> 0B33
umatan          -> 0B34
trio            -> 0B35
trifecta        -> 0B36
```

## Wide Odds Snapshot

ワイドだけ取得する場合は、32-bit PowerShell を直接指定せず、専用ラッパーを使える。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\fetch_jv_wide_odds_snapshot.ps1 -RaceKey 202606140212
```

発売中のレースであれば、`data\raw\jv_realtime_odds\<RaceKey>\` に `0B33_wide.txt` とメタ情報JSONが保存される。
発売時間外・対象外レースでは `JVRTOpen failed: -413` などが返り、record_count は 0 になる。

## Output

Raw JV records and metadata are written under:

```text
data\raw\jv_realtime_odds\<RaceKey>\
```

The metadata JSON records the dataspec, return codes, raw path, record count, and any error message.

## Current State

The wrapper can instantiate JV-Link from 32-bit PowerShell and call `JVRTOpen`.

Pair-odds normalization has been added for the Priority-S betting gate:

```powershell
& 'C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\normalize_jv_realtime_pair_odds.py
```

Output:

```text
data\processed\live_odds\realtime_pair_odds_latest.csv
```

Expected normalized columns:

```text
race_id,ticket_type,a_no,b_no,live_pay_per100,live_odds,popularity,snapshot_at,parser_mode
```

Important: the JV raw parser is heuristic until a successful live sample is captured and audited. If a separate TARGET/PC-KEIBA/Data Lab export is available, pass it via `--manual-csv` with columns equivalent to:

```text
race_id,ticket_type,a_no,b_no,live_pay_per100
```

or:

```text
race_id,ticket_type,a_no,b_no,live_odds
```

## Snapshot Loop For Live Operation

Multiple races and bet types can be polled repeatedly with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_jv_realtime_odds_snapshot_loop.ps1 -RaceKeys 202606140212,202606140211 -BetTypes win_place_frame,wide -IntervalSeconds 60 -Count 10
```

Recommended live polling:

- Start around 20 minutes before post time.
- Poll `umaren` and `wide` every 30-60 seconds for pair-ticket EV checks.
- Poll `win_place_frame` separately if market-probability and late-steam/drift features need to be refreshed.
- Poll more frequently in the final 5 minutes if the machine/API load is acceptable.
- Use `wide` and `umaren` for ticket EV checks and `win_place_frame` for market-probability and late-steam/drift features.

## Priority-S Live Odds Gate

After Priority-S tickets have been generated, apply the live odds gate:

```powershell
& 'C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\apply_live_odds_gate_to_priority_s_tickets.py `
  --tickets-csv outputs\analysis\priority_s_betting_policy_ticket_choice_v1\walkforward_selected_tickets.csv `
  --live-odds-csv data\processed\live_odds\realtime_pair_odds_latest.csv `
  --output-dir outputs\analysis\priority_s_live_odds_gate_latest
```

Without `--allow-missing-live`, tickets without live odds are skipped. This is the intended production behavior.

One-command live pipeline:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_priority_s_live_odds_pipeline.ps1 `
  -RaceKeys 202606140212,202606140211 `
  -Sid UNKNOWN `
  -IntervalSeconds 45 `
  -Count 3
```

This performs:

1. JV realtime odds polling for `umaren` and `wide`.
2. Normalization to `data\processed\live_odds\realtime_pair_odds_latest.csv`.
3. Priority-S live odds gate application.

After polling, summarize snapshot status:

```powershell
& 'C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\summarize_jv_realtime_odds_snapshots.py
```

Outputs:

```text
outputs\analysis\jv_realtime_odds_snapshots\snapshot_metadata_summary.csv
outputs\analysis\jv_realtime_odds_snapshots\snapshot_status_summary.csv
```

If `open_return = -413`, JV-Link did not return data for that realtime dataspec/key. It can mean the race is outside the JV realtime delivery window, the dataspec is unavailable, the race key is wrong, or JRA-VAN has not published that realtime feed yet. It is not proof that JRA online betting is closed; IPAT/JRA online sales and JV-Link realtime odds delivery can differ.

## JRA Official Odds Fallback

If JV-Link returns no realtime records, the project can use the official JRA odds
pages as a non-netkeiba fallback:

```powershell
& 'C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\fetch_jra_official_odds.py `
  --date 20260620 `
  --race-keys 2026062002010312 `
  --bet-types win_place_frame umaren wide
```

Outputs:

```text
data\processed\live_odds\jra_official_pair_odds_latest.csv
data\processed\live_odds\jra_official_single_odds_latest.csv
```

The race-day operation command runs this fallback automatically unless
`-SkipJraOfficialFallback` is supplied. The fallback is still live/public odds;
if both JV-Link and JRA official odds are unavailable, strict mode keeps tickets
at `WAIT` and does not use proxy odds for real betting.
