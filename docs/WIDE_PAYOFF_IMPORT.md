# Wide Payoff Import

## Current Finding

The current historical race CSV contains these payoff columns:

- `単勝配当`
- `複勝配当`
- `枠連`
- `馬連`
- `馬単`
- `３連複`
- `３連単`

It does not contain wide payoff columns.

However, `docs/PC-KEIBAテーブル定義書.xlsx` shows that wide payoff data exists in the payoff table:

- table: `jvd_hr`
- columns:
  - `haraimodoshi_wide_1a`, `haraimodoshi_wide_1b`, `haraimodoshi_wide_1c`
  - ...
  - `haraimodoshi_wide_7a`, `haraimodoshi_wide_7b`, `haraimodoshi_wide_7c`

The meaning is:

- `a`: horse-number pair, such as `0105`
- `b`: payoff
- `c`: popularity

Wide odds are separately defined in:

- historical odds table: `jvd_o3`
- realtime odds table: `apd_sokuho_o3`

## Files Added

- `scripts/document_wide_payoff_sources.py`
  - Extracts the relevant PC-KEIBA/TARGET table definitions.
- `scripts/import_wide_payoff_csv.py`
  - Normalizes a `jvd_hr`-style CSV into `race_id, horse_a, horse_b, wide_pay`.
- `scripts/sql/export_wide_payoffs_from_jvd_hr.sql`
  - SQL template for exporting wide payoffs directly from PostgreSQL.

## Direct DB Route

PC-KEIBA stores the PostgreSQL connection settings in:

```text
%APPDATA%\PC-KEIBA Database\AppConfig.xml
```

The project can read that config and export wide payoffs directly:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\export_pc_keiba_wide_payoffs.ps1
```

This creates:

```text
data/processed/target/wide_payoffs.csv
```

The exported key is the 16-digit TARGET race id:

```text
開催年 + 開催月日 + 競馬場コード + 開催回 + 開催日目 + レース番号
```

Manual psql route:

When the database name and password are available, the SQL template can also be run directly:

```powershell
$env:PGPASSWORD='<password>'
& 'C:\Program Files\PostgreSQL\18\bin\psql.exe' -h localhost -p 3517 -U postgres -d <database_name> -f scripts\sql\export_wide_payoffs_from_jvd_hr.sql
```

## CSV Route

If TARGET/PC-KEIBA can export the payoff table `jvd_hr`, include the wide columns listed above.

Recommended inbox:

```text
data/inbox/target/payoffs/
```

Then run the latest-file importer:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\import_latest_wide_payoffs.ps1
```

Or pass the CSV explicitly:

```powershell
& 'C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\import_wide_payoff_csv.py --input-csv <jvd_hr_export.csv> --encoding cp932 --output-csv data\processed\target\wide_payoffs.csv
```

## Next Integration

Once `data/processed/target/wide_payoffs.csv` exists, the market-edge pair strategy can report actual wide payoff ROI:

```powershell
& 'C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\evaluate_market_edge_pair_strategy.py --output-dir outputs\analysis\market_edge_pair_strategy_with_wide
```

The output summary includes:

- wide hit rate
- break-even average wide payoff
- actual wide ROI
- actual wide profit per flat 100 yen
