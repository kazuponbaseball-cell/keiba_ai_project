param(
    [string]$AppConfig = "$env:APPDATA\PC-KEIBA Database\AppConfig.xml",
    [string]$OutputCsv = "data\processed\target\wide_payoffs.csv"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$resolvedOutput = if ([System.IO.Path]::IsPathRooted($OutputCsv)) {
    $OutputCsv
} else {
    Join-Path $projectRoot $OutputCsv
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $resolvedOutput) | Out-Null

if (-not (Test-Path $AppConfig)) {
    throw "PC-KEIBA AppConfig.xml was not found: $AppConfig"
}

[xml]$config = Get-Content $AppConfig
$app = $config.DocumentElement.AppConfig
$env:PGPASSWORD = [string]$app.DbPassword

$psql = "C:\Program Files\PostgreSQL\18\bin\psql.exe"
if (-not (Test-Path $psql)) {
    $psql = "psql"
}

$copySql = @"
\copy (select kaisai_nen || kaisai_tsukihi || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id, substring(pair_code from 1 for 2)::int as horse_a, substring(pair_code from 3 for 2)::int as horse_b, trim(wide_pay)::int as wide_pay, nullif(trim(wide_popularity), '')::int as wide_popularity from (select kaisai_nen, kaisai_tsukihi, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, haraimodoshi_wide_1a as pair_code, haraimodoshi_wide_1b as wide_pay, haraimodoshi_wide_1c as wide_popularity from jvd_hr union all select kaisai_nen, kaisai_tsukihi, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, haraimodoshi_wide_2a, haraimodoshi_wide_2b, haraimodoshi_wide_2c from jvd_hr union all select kaisai_nen, kaisai_tsukihi, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, haraimodoshi_wide_3a, haraimodoshi_wide_3b, haraimodoshi_wide_3c from jvd_hr union all select kaisai_nen, kaisai_tsukihi, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, haraimodoshi_wide_4a, haraimodoshi_wide_4b, haraimodoshi_wide_4c from jvd_hr union all select kaisai_nen, kaisai_tsukihi, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, haraimodoshi_wide_5a, haraimodoshi_wide_5b, haraimodoshi_wide_5c from jvd_hr union all select kaisai_nen, kaisai_tsukihi, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, haraimodoshi_wide_6a, haraimodoshi_wide_6b, haraimodoshi_wide_6c from jvd_hr union all select kaisai_nen, kaisai_tsukihi, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, haraimodoshi_wide_7a, haraimodoshi_wide_7b, haraimodoshi_wide_7c from jvd_hr) wide where nullif(trim(pair_code), '') is not null and nullif(trim(wide_pay), '') is not null and pair_code <> '0000' and wide_pay <> '000000000') to '$($resolvedOutput.Replace('\', '/'))' with csv header encoding 'UTF8'
"@

& $psql -h ([string]$app.DbServer) -p ([string]$app.DbPort) -U ([string]$app.DbUserId) -d ([string]$app.DbDatabase) -c $copySql
exit $LASTEXITCODE
