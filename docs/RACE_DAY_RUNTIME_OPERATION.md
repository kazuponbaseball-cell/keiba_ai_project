# Race-Day Runtime Operation

This is the production-oriented entry point for race-day updates.

It connects the current strongest runtime stack:

- JV/Data Lab realtime odds fetch
- JRA official odds fallback when JV-Link returns no usable odds
- realtime odds normalization
- pair probability calibration
- strict buy / reduce / wait / skip decisions
- live safety overlay
- MCS/PBO survivor overlay
- optional netkeiba handoff file export
- dashboard rebuild
- odds timeline append for late-odds analysis
- fixed-time pair-edge validation for `T-5` / `T-3` snapshots

## One-shot update

Use this when you want the equivalent of an update button.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_race_day_runtime_operation.ps1 -RaceKeys 2026062002010312 -DecisionLabel final_check -OfficialOnly
```

For multiple races:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_race_day_runtime_operation.ps1 -RaceKeys 2026062002010310,2026062002010311,2026062002010312 -DecisionLabel T-3 -OfficialOnly
```

The JV SID is resolved automatically from the local JRA-VAN Data Lab `servicekey`.
You can still override it with `-Sid` or `JV_SID`.

```powershell
$env:JV_SID = "your-sid"
```

## Scheduled race-day automation

This builds a schedule from the TARGET entry snapshot and runs updates at
`T-10`, `T-5`, and `T-3` minutes before post time by default.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\start_race_day_runtime_automation.ps1 -Date 2026-06-20
```

## Safety behavior

Strict live mode is the default. If live odds are absent, tickets remain `WAIT`
instead of falling back to historical/proxy odds.

Use `-ProxyWhenMissing` only for historical debugging, never for real betting.

If JV-Link does not return odds, the script tries `scripts\fetch_jra_official_odds.py`
against JRA official odds pages. This provides official single, umaren, and wide
odds without netkeiba. To disable that fallback, pass `-SkipJraOfficialFallback`.

`-OfficialOnly` also skips the netkeiba handoff export. It does not weaken the
live-odds rule: if official live odds are still missing, tickets stay `WAIT`.

The race-day operation script applies `mcs_full_margin095_s0304_skip03119` by
default. This uses the full MCS survivor family: `umaren` tickets with
`min_odds_margin_ratio` of at least `0.95`, plus a pre-race skip-risk gate
(`skip_risk_score <= 0.3119`). It kept enough races while improving ROI,
hit rate, drawdown, and top-10-payout-removed robustness versus the previous
`mcs_full_margin095_s0304` default.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_race_day_runtime_operation.ps1 -RaceKeys 202606200512
```

For a broader coverage-first mode that keeps all races while reducing wide/win
exposure, use:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_race_day_runtime_operation.ps1 -RaceKeys 202606200512 -McsPboPolicy reduce_wide_win_50
```

For a tighter but lower-coverage ROI mode:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_race_day_runtime_operation.ps1 -RaceKeys 202606200512 -McsPboPolicy mcs_full_margin095_s0304_danger020_skip03119
```

To disable the overlay:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_race_day_runtime_operation.ps1 -RaceKeys 202606200512 -McsPboPolicy ""
```

## Main outputs

- `outputs\analysis\race_day_runtime_operation_latest\summary.json`
- `outputs\analysis\race_day_runtime_operation_latest\jra_official_pair_odds_latest.csv`
- `outputs\analysis\race_day_runtime_operation_latest\jra_official_single_odds_latest.csv`
- `outputs\analysis\race_day_runtime_operation_latest\netkeiba_bet_plan\netkeiba_bet_plan.csv` when netkeiba export is enabled
- `outputs\ui\keiba_dashboard_runtime.html`
- `data\processed\live_odds\realtime_pair_odds_timeline.csv`
- `data\processed\live_odds\realtime_single_odds_timeline.csv`
- `outputs\analysis\race_day_runtime_operation_latest\fixed_time_pair_edge\summary.json`

## Fixed-Time Pair Edge

When `-DecisionLabel T-5`, `-DecisionLabel T-3`, or `-DecisionLabel final_check`
is used, the latest realtime odds snapshot is appended to the odds timeline.
The operation script then runs fixed-time pair-edge validation automatically.

The validation checks whether the `umaren` pair still had value at that exact
captured odds point:

```text
market implied probability = 0.775 / captured umaren odds
model-market ratio = model pair probability / market implied probability
fixed-time expected ROI = model pair probability * captured umaren odds
```

If the timeline has no matching rows yet, the output will correctly show
`matched_rows: 0`. That means the race-day T-5/T-3 snapshots have not been
accumulated for those tickets yet.
