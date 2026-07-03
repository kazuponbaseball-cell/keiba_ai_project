$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$InfoJson = Join-Path $ProjectRoot "outputs\runtime\current_strongest_timed_snapshots_background.json"

if (-not (Test-Path -LiteralPath $InfoJson)) {
    Write-Host "No timed snapshot watcher info found: $InfoJson"
    exit 0
}

$info = Get-Content -LiteralPath $InfoJson -Raw -Encoding UTF8 | ConvertFrom-Json
$proc = Get-Process -Id $info.pid -ErrorAction SilentlyContinue
if ($proc) {
    Stop-Process -Id $info.pid -Force
    Write-Host "Stopped current strongest timed snapshot watcher. pid=$($info.pid)"
} else {
    Write-Host "Timed snapshot watcher was not running. pid=$($info.pid)"
}

Remove-Item -LiteralPath $InfoJson -Force -ErrorAction SilentlyContinue
