# TARGET Connection

This project reads `TARGET frontier JV` data through CSV exports.

Detected TARGET paths on this PC:

- App/data folder: `C:\Users\kazup\Data Lab`
- TARGET text output folder: `C:\Users\kazup\Data Lab\TXT`
- Project entry inbox: `data\inbox\target\entries`
- Project race inbox: `data\inbox\target\races`

## Sync Latest Export

After exporting a CSV from TARGET, run:

```powershell
C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m src.pipelines.sync_target_exports --mode entry --run-import
```

This scans the configured TARGET output folders, copies the newest recent CSV that has the expected entry columns into `data\inbox\target\entries`, then imports it into:

```text
data\datasets\inference\weekly\entry_snapshot.csv
```

For race-detail CSVs:

```powershell
C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m src.pipelines.sync_target_exports --mode race --run-import
```

## Notes

- By default, only files modified within the last 14 days are considered.
- Use `--dry-run` to check which file would be selected.
- Use `--allow-old` only when intentionally importing an older export.
- The current `C:\Users\kazup\Data Lab\TXT\re-su.csv` appears to be a race-ID list, not an entry table.

## Data Availability Preflight

Before running lap or horse-history analyses, check whether TARGET's local `SE_DATA` has the
race-lap (`SR*.DAT` RA records) and starter/result (`SU*.DAT` SE records) data required by the
current scripts:

```powershell
C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts\diagnose_target_data_availability.py --target-date 20260607
```

To stop an update flow when the target date is not ready:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_target_data_preflight.ps1 -TargetDate 20260607 -RequireReady
```

The weekly TARGET update wrapper runs this preflight by default:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_target_weekly_update.ps1 -TargetDate 20260607
```
