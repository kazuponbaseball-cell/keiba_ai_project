param(
    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [string]$ProjectRoot = "",
    [string]$TargetRoot = "C:\Users\kazup\Data Lab",
    [string]$TargetDate = "",
    [string]$EntryCsv = "data\datasets\inference\weekly\entry_snapshot.csv",
    [string]$HistoryEnrichedEntryCsv = "data\datasets\inference\weekly\entry_snapshot_with_recent_history.csv",
    [string]$BiasedEntryCsv = "data\datasets\inference\weekly\entry_snapshot_with_same_day_bias.csv",
    [string]$FeatureConfig = "config\baseline_features_workout_optimized_core_same_day_bias.json",
    [string]$ModelPath = "models\workout_optimized_core_same_day_bias_v3_retro\baseline_ranker.pkl",
    [string]$OutputDir = "outputs\predictions\realtime_same_day_bias",
    [int]$IntervalSeconds = 180,
    [string]$StopAt = "",
    [switch]$Once,
    [switch]$RunPreflight,
    [switch]$SkipPrediction,
    [switch]$RequirePrediction
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}

if ([string]::IsNullOrWhiteSpace($TargetDate)) {
    $TargetDate = (Get-Date).ToString("yyyyMMdd")
}

Set-Location -LiteralPath $ProjectRoot

$logDir = Join-Path $ProjectRoot "outputs\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logPath = Join-Path $logDir ("realtime_same_day_bias_{0}.log" -f $TargetDate)
$statePath = Join-Path $ProjectRoot "data\state\realtime_same_day_bias_state.json"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $statePath) | Out-Null

function Write-RunLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -Path $logPath -Value $line -Encoding UTF8
}

function Get-LatestWriteTicks {
    param([string[]]$Paths)
    $latest = 0L
    foreach ($path in $Paths) {
        if (Test-Path -LiteralPath $path) {
            $item = Get-Item -LiteralPath $path
            if ($item.PSIsContainer) {
                $child = Get-ChildItem -LiteralPath $path -Recurse -File -ErrorAction SilentlyContinue |
                    Sort-Object LastWriteTimeUtc -Descending |
                    Select-Object -First 1
                if ($null -ne $child) {
                    $latest = [Math]::Max($latest, $child.LastWriteTimeUtc.Ticks)
                }
            } else {
                $latest = [Math]::Max($latest, $item.LastWriteTimeUtc.Ticks)
            }
        }
    }
    return $latest
}

function Invoke-Refresh {
    Write-RunLog "refresh start"

    if ($RunPreflight) {
        & powershell -ExecutionPolicy Bypass -File "scripts\run_target_data_preflight.ps1" `
            -PythonExe $PythonExe `
            -ProjectRoot $ProjectRoot `
            -TargetRoot $TargetRoot `
            -TargetDate $TargetDate
        if ($LASTEXITCODE -ne 0) {
            throw "preflight failed: exit code $LASTEXITCODE"
        }
    }

    if (-not (Test-Path -LiteralPath $EntryCsv)) {
        Write-RunLog "skip: entry csv not found: $EntryCsv"
        return
    }

    & $PythonExe "scripts\enrich_entry_with_recent_history.py" `
        --input-csv $EntryCsv `
        --output-csv $HistoryEnrichedEntryCsv `
        --config $FeatureConfig
    if ($LASTEXITCODE -ne 0) {
        throw "recent-history enrichment failed: exit code $LASTEXITCODE"
    }

    & $PythonExe "scripts\build_same_day_bias_features.py" `
        --entry-csv $HistoryEnrichedEntryCsv `
        --output-csv $BiasedEntryCsv
    if ($LASTEXITCODE -ne 0) {
        throw "same-day bias build failed: exit code $LASTEXITCODE"
    }

    if (-not $SkipPrediction) {
        & $PythonExe -m src.predict.predict_baseline `
            --config $FeatureConfig `
            --model $ModelPath `
            --input-csv $BiasedEntryCsv `
            --output-dir $OutputDir
        if ($LASTEXITCODE -ne 0) {
            $message = "prediction failed: exit code $LASTEXITCODE"
            if ($RequirePrediction) {
                throw $message
            }
            Write-RunLog "$message; biased CSV was still updated"
        }
    }

    Write-RunLog "refresh done: biased_entry=$BiasedEntryCsv output_dir=$OutputDir"
}

$watchPaths = @(
    (Join-Path $TargetRoot "SE_DATA"),
    (Join-Path $TargetRoot "DE_DATA"),
    (Join-Path $ProjectRoot "data\processed\normalized\results.csv"),
    (Join-Path $ProjectRoot "data\processed\normalized\runners.csv"),
    (Join-Path $ProjectRoot "data\processed\normalized\races.csv"),
    (Join-Path $ProjectRoot $EntryCsv)
)

$lastSignature = -1L
if (Test-Path -LiteralPath $statePath) {
    try {
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        if ($state.TargetDate -eq $TargetDate) {
            $lastSignature = [int64]$state.Signature
        }
    } catch {
        $lastSignature = -1L
    }
}

Write-RunLog "monitor start: target_date=$TargetDate interval=${IntervalSeconds}s once=$Once"

while ($true) {
    try {
        $signature = Get-LatestWriteTicks -Paths $watchPaths
        if ($signature -ne $lastSignature) {
            Write-RunLog "change detected: signature=$signature previous=$lastSignature"
            Invoke-Refresh
            $lastSignature = $signature
            @{
                TargetDate = $TargetDate
                Signature = "$lastSignature"
                UpdatedAt = (Get-Date).ToString("s")
                HistoryEnrichedEntryCsv = $HistoryEnrichedEntryCsv
                BiasedEntryCsv = $BiasedEntryCsv
                OutputDir = $OutputDir
            } | ConvertTo-Json | Set-Content -Path $statePath -Encoding UTF8
        } else {
            Write-RunLog "no change"
        }
    } catch {
        Write-RunLog ("error: {0}" -f $_.Exception.Message)
    }

    if ($Once) {
        break
    }
    if (-not [string]::IsNullOrWhiteSpace($StopAt)) {
        $nowText = (Get-Date).ToString("HH:mm")
        if ($nowText -ge $StopAt) {
            Write-RunLog "monitor stop: StopAt reached ($StopAt)"
            break
        }
    }
    Start-Sleep -Seconds $IntervalSeconds
}
