param(
    [ValidateSet("preday", "body", "final", "skip")]
    [string]$Stage = "preday",

    [string]$Today = "",

    [string]$DashboardUrl = "",

    [string]$BodyWeightCsv = "data\processed\live_body_weight\body_weight_latest.csv",

    [string]$TicketsCsv = "",

    [int]$MaxRaces = 8,

    [switch]$Send,

    [switch]$SendIfConfigured,

    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",

    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}

Set-Location -LiteralPath $ProjectRoot

$BuildPreday = Join-Path $ProjectRoot "scripts\build_preday_dashboard_html.py"
$LineAlert = Join-Path $ProjectRoot "scripts\send_line_keiba_alert.ps1"
$PredayHtml = "outputs\ui\keiba_preday_dashboard.html"

function Resolve-ProjectPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $ProjectRoot $Path)
}

function Resolve-PublicDashboardUrl {
    param([string]$DefaultPath)
    if (-not [string]::IsNullOrWhiteSpace($DashboardUrl)) {
        return $DashboardUrl
    }
    $publicInfoPath = Join-Path $ProjectRoot "outputs\runtime\public_dashboard_tunnel.json"
    if (Test-Path $publicInfoPath) {
        try {
            $publicInfo = Get-Content -Path $publicInfoPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($publicInfo.public_url) {
                return "$($publicInfo.public_url)/$($DefaultPath.Replace('\', '/'))"
            }
        } catch {
            Write-Warning "Could not read public tunnel info: $($_.Exception.Message)"
        }
    }
    return ""
}

function Resolve-FinalTicketsCsv {
    if (-not [string]::IsNullOrWhiteSpace($TicketsCsv)) {
        return $TicketsCsv
    }

    $summaryPath = Join-Path $ProjectRoot "outputs\analysis\race_day_runtime_operation_latest\summary.json"
    if (Test-Path $summaryPath) {
        try {
            $summary = Get-Content -Path $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($summary.selected_csv -and (Test-Path (Resolve-ProjectPath $summary.selected_csv))) {
                return $summary.selected_csv
            }
            if ($summary.final_tickets_csv -and (Test-Path (Resolve-ProjectPath $summary.final_tickets_csv))) {
                return $summary.final_tickets_csv
            }
        } catch {
            Write-Warning "Could not read runtime summary: $($_.Exception.Message)"
        }
    }

    $latest = Get-ChildItem -Path (Join-Path $ProjectRoot "outputs\analysis") -Recurse -Filter "recommended_runtime_tickets.csv" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($latest) {
        return $latest.FullName
    }
    return ""
}

function Invoke-LineAlert {
    param(
        [string]$Mode,
        [string]$Url,
        [string]$Tickets = "",
        [switch]$RequireBody
    )

    $lineArgs = @(
        "-File", $LineAlert,
        "-Mode", $Mode,
        "-DashboardUrl", $Url,
        "-MaxRaces", "$MaxRaces",
        "-PythonExe", $PythonExe
    )
    if (-not [string]::IsNullOrWhiteSpace($BodyWeightCsv)) {
        $lineArgs += @("-BodyWeightCsv", $BodyWeightCsv)
    }
    if (-not [string]::IsNullOrWhiteSpace($Tickets)) {
        $lineArgs += @("-TicketsCsv", $Tickets)
    }
    if ($RequireBody) {
        $lineArgs += "-RequireBodyWeight"
    }
    if ($Send) {
        $lineArgs += "-Send"
    }
    if ($SendIfConfigured) {
        $lineArgs += "-SendIfConfigured"
    }
    powershell.exe -NoProfile -ExecutionPolicy Bypass @lineArgs
}

if ($Stage -in @("preday", "body", "skip")) {
    $buildArgs = @($BuildPreday, "--output-html", $PredayHtml, "--max-races", "72")
    if (-not [string]::IsNullOrWhiteSpace($Today)) {
        $buildArgs += @("--today", $Today)
    }
    & $PythonExe @buildArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Preday dashboard build failed."
    }
}

switch ($Stage) {
    "preday" {
        $url = Resolve-PublicDashboardUrl -DefaultPath $PredayHtml
        Invoke-LineAlert -Mode "preday" -Url $url
    }
    "body" {
        $url = Resolve-PublicDashboardUrl -DefaultPath $PredayHtml
        Invoke-LineAlert -Mode "buyable" -Url $url -RequireBody
    }
    "skip" {
        $url = Resolve-PublicDashboardUrl -DefaultPath $PredayHtml
        Invoke-LineAlert -Mode "skip" -Url $url
    }
    "final" {
        $tickets = Resolve-FinalTicketsCsv
        if ([string]::IsNullOrWhiteSpace($tickets)) {
            throw "No final ticket CSV was found. Pass -TicketsCsv explicitly."
        }
        $url = Resolve-PublicDashboardUrl -DefaultPath "outputs\ui\keiba_dashboard_runtime.html"
        Invoke-LineAlert -Mode "final" -Url $url -Tickets $tickets
    }
}
