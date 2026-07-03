param(
    [string]$Date = "",

    [string[]]$Venues = @(),

    [int[]]$Races = @(),

    [string]$Sid = "",

    [int[]]$OffsetsMinutes = @(10, 5, 3),

    [int]$PollSeconds = 15,

    [int]$GroupWindowSeconds = 45,

    [string]$EntryCsv = "data\datasets\inference\weekly\entry_snapshot.csv",

    [string]$ScheduleCsv = "data\processed\live_odds\live_odds_race_schedule.csv",

    [string]$StateJson = "data\processed\live_odds\race_day_runtime_automation_state.json",

    [string]$LogPath = "outputs\analysis\race_day_runtime_operation_latest\automation_log.txt",

    [string]$DashboardHtml = "outputs\ui\keiba_dashboard_runtime.html",

    [string]$OutputRoot = "outputs\analysis\race_day_runtime_operation_latest",

    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$BuildSchedule = Join-Path $ProjectRoot "scripts\build_live_odds_race_schedule.py"
$Runner = Join-Path $ProjectRoot "scripts\run_race_day_runtime_operation.ps1"
$SchedulePath = if ([System.IO.Path]::IsPathRooted($ScheduleCsv)) { $ScheduleCsv } else { Join-Path $ProjectRoot $ScheduleCsv }
$StatePath = if ([System.IO.Path]::IsPathRooted($StateJson)) { $StateJson } else { Join-Path $ProjectRoot $StateJson }
$ResolvedLogPath = if ([System.IO.Path]::IsPathRooted($LogPath)) { $LogPath } else { Join-Path $ProjectRoot $LogPath }

. (Join-Path $ProjectRoot "scripts\Resolve-JvSid.ps1")
$sidResolution = Resolve-JvSid -Sid $Sid
$Sid = $sidResolution.Sid
if ([string]::IsNullOrWhiteSpace($Sid)) {
    throw "JV SID is required for automation. Pass -Sid, set environment variable JV_SID, or configure JRA-VAN Data Lab servicekey."
}
if (-not (Test-Path $PythonExe)) {
    throw "Python executable was not found: $PythonExe"
}
if (-not (Test-Path $Runner)) {
    throw "Runtime operation runner was not found: $Runner"
}

Set-Location $ProjectRoot

$buildArgs = @(
    $BuildSchedule,
    "--entry-csv", $EntryCsv,
    "--output-csv", $ScheduleCsv,
    "--output-json", ($ScheduleCsv -replace "\.csv$", ".json")
)
if ($Date) {
    $buildArgs += @("--date", $Date)
}
if ($Venues.Count -gt 0) {
    $buildArgs += "--venues"
    $buildArgs += $Venues
}
if ($Races.Count -gt 0) {
    $buildArgs += "--races"
    $buildArgs += ($Races | ForEach-Object { [string]$_ })
}

Write-Host "Building live odds schedule..."
& $PythonExe @buildArgs
if ($LASTEXITCODE -ne 0) {
    throw "Schedule build failed with exit code $LASTEXITCODE."
}

if (-not (Test-Path $SchedulePath)) {
    throw "Schedule CSV was not found: $SchedulePath"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $StatePath) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ResolvedLogPath) | Out-Null

function Write-AutoLog {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Host $line
    Add-Content -Path $ResolvedLogPath -Value $line -Encoding UTF8
}

function Read-State {
    if (-not (Test-Path $StatePath)) {
        return @{}
    }
    $text = Get-Content -Path $StatePath -Raw -Encoding UTF8
    if (-not $text.Trim()) {
        return @{}
    }
    $obj = $text | ConvertFrom-Json
    $state = @{}
    foreach ($prop in $obj.PSObject.Properties) {
        $state[$prop.Name] = [bool]$prop.Value
    }
    return $state
}

function Save-State {
    param([hashtable]$State)
    $State | ConvertTo-Json -Depth 3 | Set-Content -Path $StatePath -Encoding UTF8
}

$state = Read-State
$schedule = Import-Csv -Path $SchedulePath
if (-not $schedule -or $schedule.Count -eq 0) {
    throw "Schedule CSV has no races: $SchedulePath"
}

$jobs = @()
foreach ($row in $schedule) {
    $postTime = [datetime]::Parse($row.post_time)
    foreach ($offset in $OffsetsMinutes) {
        $label = "T-$offset"
        $due = $postTime.AddMinutes(-1 * $offset)
        $jobs += [pscustomobject]@{
            race_key = [string]$row.race_key
            venue = [string]$row.venue
            race_no = [int]$row.race_no
            race_name = [string]$row.race_name
            post_time = $postTime
            due_time = $due
            decision_label = $label
            state_key = "$($row.race_key)|$label"
        }
    }
}

$jobs = $jobs | Sort-Object due_time, race_key
$lastDue = ($jobs | Sort-Object due_time -Descending | Select-Object -First 1).due_time
Write-AutoLog "Started race-day runtime automation. jobs=$($jobs.Count) schedule=$SchedulePath"

while ($true) {
    $now = Get-Date
    $dueJobs = @(
        $jobs | Where-Object {
            (-not $state.ContainsKey($_.state_key)) -and
            ($_.due_time -le $now) -and
            ($_.due_time -ge $now.AddMinutes(-20))
        }
    )

    if ($dueJobs.Count -gt 0) {
        $groups = $dueJobs | Group-Object decision_label
        foreach ($group in $groups) {
            $label = $group.Name
            $raceKeys = @($group.Group | Select-Object -ExpandProperty race_key -Unique)
            Write-AutoLog "Running runtime update label=$label races=$($raceKeys -join ',')"
            try {
                & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Runner `
                    -RaceKeys $raceKeys `
                    -DecisionLabel $label `
                    -Sid $Sid `
                    -OutputRoot $OutputRoot `
                    -DashboardHtml $DashboardHtml
                foreach ($job in $group.Group) {
                    $state[$job.state_key] = $true
                }
                Save-State -State $state
                Write-AutoLog "Completed runtime update label=$label races=$($raceKeys -join ',')"
            } catch {
                Write-AutoLog "ERROR runtime update label=$label races=$($raceKeys -join ',') message=$($_.Exception.Message)"
            }
            Start-Sleep -Seconds $GroupWindowSeconds
        }
    }

    $remaining = @($jobs | Where-Object { -not $state.ContainsKey($_.state_key) })
    if ($remaining.Count -eq 0 -or $now -gt $lastDue.AddMinutes(20)) {
        Write-AutoLog "Finished race-day runtime automation. remaining=$($remaining.Count)"
        break
    }
    Start-Sleep -Seconds $PollSeconds
}
