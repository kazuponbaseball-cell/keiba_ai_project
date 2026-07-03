param(
    [string]$InputDir = "data\inbox\target\payoffs",
    [string]$OutputCsv = "data\processed\target\wide_payoffs.csv",
    [string]$Encoding = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$inputPath = Join-Path $projectRoot $InputDir
if (-not (Test-Path $inputPath)) {
    New-Item -ItemType Directory -Force -Path $inputPath | Out-Null
}

$latest = Get-ChildItem -Path (Join-Path $inputPath "*") -File -Include *.csv,*.txt |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $latest) {
    throw "No payoff CSV found in $inputPath. Export jvd_hr/payoff CSV from TARGET or PC-KEIBA into this folder."
}

$python = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$args = @(
    (Join-Path $projectRoot "scripts\import_wide_payoff_csv.py"),
    "--input-csv", $latest.FullName,
    "--output-csv", (Join-Path $projectRoot $OutputCsv)
)
if ($Encoding) {
    $args += @("--encoding", $Encoding)
}

& $python @args
exit $LASTEXITCODE
