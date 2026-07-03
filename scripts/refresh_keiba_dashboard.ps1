param(
    [string]$ScoredCsv = "outputs\analysis\risk_models_v1\investment_features_with_risk_models.csv",
    [string]$TicketsCsv = "outputs\analysis\operational_ticket_profiles_v1\ticket_profiles.csv",
    [string]$LivePairOddsCsv = "data\processed\live_odds\realtime_pair_odds_latest.csv",
    [string]$LiveSingleOddsCsv = "data\processed\live_odds\realtime_single_odds_latest.csv",
    [string]$BodyWeightCsv = "",
    [string]$OutputHtml = "outputs\ui\keiba_dashboard.html",
    [string]$LiveOddsOutputHtml = "outputs\ui\live_odds_dashboard.html",
    [string]$EntryCsvForLiveOddsDashboard = "",
    [int]$MaxRaces = 120,
    [int]$IntervalSeconds = 60,
    [switch]$Loop
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$liveOddsDashboard = Join-Path $projectRoot "scripts\build_live_odds_dashboard_html.py"

function Invoke-DashboardBuild {
    Set-Location $projectRoot
    $args = @(
        "scripts\build_keiba_dashboard_html.py",
        "--scored-csv", $ScoredCsv,
        "--tickets-csv", $TicketsCsv,
        "--live-pair-odds-csv", $LivePairOddsCsv,
        "--live-single-odds-csv", $LiveSingleOddsCsv,
        "--output-html", $OutputHtml,
        "--max-races", "$MaxRaces"
    )
    if ($BodyWeightCsv -ne "") {
        $args += @("--body-weight-csv", $BodyWeightCsv)
    }
    & $python @args

    if (Test-Path $liveOddsDashboard) {
        $liveArgs = @(
            $liveOddsDashboard,
            "--single-odds-csv", $LiveSingleOddsCsv,
            "--pair-odds-csv", $LivePairOddsCsv,
            "--output-html", $LiveOddsOutputHtml
        )
        if ($EntryCsvForLiveOddsDashboard -ne "") {
            $liveArgs += @("--entry-csv", $EntryCsvForLiveOddsDashboard)
        }
        & $python @liveArgs
    }
}

do {
    Invoke-DashboardBuild
    if ($Loop) {
        Start-Sleep -Seconds $IntervalSeconds
    }
} while ($Loop)
