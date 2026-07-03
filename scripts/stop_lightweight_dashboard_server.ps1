param(
    [string]$RuntimeRoot = "C:\keiba_ai_runtime",
    [int]$Port = 8766
)

$ErrorActionPreference = "Stop"

$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
$infoPath = Join-Path $RuntimeRoot "state\dashboard_server.json"
$stopped = @()

if (Test-Path -LiteralPath $infoPath) {
    try {
        $info = Get-Content -LiteralPath $infoPath -Raw | ConvertFrom-Json
        if ($info.pid) {
            $p = Get-Process -Id ([int]$info.pid) -ErrorAction SilentlyContinue
            if ($p) {
                Stop-Process -Id $p.Id -Force
                $stopped += $p.Id
            }
        }
    } catch {
        Write-Warning "Could not read $infoPath: $_"
    }
}

$listeners = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
foreach ($listener in $listeners) {
    $pidValue = [int]$listener.OwningProcess
    if ($stopped -contains $pidValue) { continue }
    $p = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
    if ($p -and $p.ProcessName -match "python") {
        Stop-Process -Id $pidValue -Force
        $stopped += $pidValue
    }
}

if ($stopped.Count -eq 0) {
    Write-Host "No lightweight dashboard server process was stopped."
} else {
    Write-Host "Stopped lightweight dashboard server PID(s): $($stopped -join ', ')"
}
