param(
    [string]$Date = (Get-Date -Format "yyyyMMdd"),
    [string[]]$RaceKeys = @(),
    [switch]$SkipOddsFetch,
    [switch]$SkipTrackFetch,
    [switch]$SkipBodyWeight,
    [switch]$SkipResultFetch,
    [switch]$SkipWin5,
    [switch]$SkipDashboard,
    [switch]$Notify,
    [switch]$SendIfConfigured,
    [switch]$ForceNotify,

    [ValidateSet("T-10", "T-5", "T-3", "final_check", "manual")]
    [string]$DecisionLabel = "manual",

    [switch]$SkipTimeline,
    [switch]$SkipExternalAudit,

    [int]$OddsFetchWorkers = 4,
    [double]$OddsFetchSleepSeconds = 0.05,
    [string]$EntryCsv = "",
    [string]$PredictionCsv = "",
    [string]$CurrentInputsJson = "outputs\runtime\current_dashboard_inputs.json",
    [string]$BodyWeightCsv = "data\processed\live_body_weight\body_weight_latest.csv",
    [string]$Win5Json = "outputs\analysis\win5_runtime\win5_plan.json",
    [string[]]$Win5RaceIds = @(),
    [string]$StateJson = "data\processed\notifications\current_strongest_line_state.json",
    [string]$TrackChangeStateJson = "data\processed\notifications\track_condition_change_state.json",
    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}

Set-Location -LiteralPath $ProjectRoot

$FetchJraOdds = Join-Path $ProjectRoot "scripts\fetch_jra_official_odds.py"
$ExtractBodyWeights = Join-Path $ProjectRoot "scripts\extract_jra_official_body_weights.py"
$FetchTrackConditions = Join-Path $ProjectRoot "scripts\fetch_jra_current_track_conditions.py"
$EnrichBasicAbility = Join-Path $ProjectRoot "scripts\enrich_prediction_basic_ability_features.py"
$BuildTickets = Join-Path $ProjectRoot "scripts\build_current_strongest_tickets.py"
$ApplyLiveSafety = Join-Path $ProjectRoot "scripts\apply_live_runtime_safety_overlay.py"
$ApplyPairJointV2Guard = Join-Path $ProjectRoot "scripts\apply_pair_joint_probability_v2_guard_to_runtime.py"
$BuildWin5 = Join-Path $ProjectRoot "scripts\build_win5_candidates.py"
$BuildDashboard = Join-Path $ProjectRoot "scripts\build_live_odds_dashboard_html.py"
$AppendTimeline = Join-Path $ProjectRoot "scripts\append_live_odds_timeline.py"
$BuildTimelineFeatures = Join-Path $ProjectRoot "scripts\build_odds_timeline_features.py"
$BuildTargetRaLapHistory = Join-Path $ProjectRoot "scripts\build_official_lap_history_features.py"
$EvaluateFixedTimePairEdge = Join-Path $ProjectRoot "scripts\evaluate_fixed_time_pair_edge.py"
$EvaluateCurrentPnl = Join-Path $ProjectRoot "scripts\evaluate_current_live_pnl.py"
$DetectTrackChanges = Join-Path $ProjectRoot "scripts\detect_track_condition_changes.py"
$FreezeDecisionSnapshot = Join-Path $ProjectRoot "scripts\freeze_current_strongest_decision_snapshot.py"
$FreezeChampionManifest = Join-Path $ProjectRoot "scripts\freeze_champion_strategy_manifest.py"
$BuildFinalOddsSurvival = Join-Path $ProjectRoot "scripts\build_final_odds_survival_dataset.py"
$BuildCandidateRejectionLedger = Join-Path $ProjectRoot "scripts\build_candidate_rejection_ledger.py"
$LineAlert = Join-Path $ProjectRoot "scripts\send_current_strongest_line_alert.ps1"
$PairOddsCsv = "data\processed\live_odds\realtime_pair_odds_latest.csv"
$SingleOddsCsv = "data\processed\live_odds\realtime_single_odds_latest.csv"
$TicketsCsv = "outputs\analysis\current_strongest_runtime_v1\selected_after_live_safety.csv"
$DashboardTicketsCsv = "outputs\analysis\current_strongest_runtime_v1\current_strongest_dashboard_tickets.csv"
$WideShadowCsv = "outputs\analysis\current_strongest_runtime_v1\pair_joint_v2_runtime_guard\wide_shadow_guard_ok_candidates.csv"
$DashboardHtml = "outputs\ui\live_odds_dashboard.html"
$TrackConditionCsv = "data\processed\live_track_conditions\current_track_conditions.csv"
$TargetRaRaceLapsCsv = "data\processed\target_ra_race_laps\race_laps.csv"
$CurrentTargetRaLapHistoryCsv = "outputs\analysis\current_strongest_runtime_v1\current_target_ra_lap_history_features.csv"
$CurrentTargetRaLapHistorySummaryJson = "outputs\analysis\current_strongest_runtime_v1\current_target_ra_lap_history_summary.json"

function Resolve-ProjectPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $ProjectRoot $Path)
}

function Invoke-Checked {
    param(
        [string]$Label,
        [scriptblock]$Command
    )
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Label"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

function Get-TicketHash {
    param([string]$Path)
    $resolved = Resolve-ProjectPath $Path
    if (-not (Test-Path -LiteralPath $resolved)) {
        return @{
            hash = ""
            tickets = 0
            races = 0
            stake_yen = 0
        }
    }

    $rows = @(Import-Csv -LiteralPath $resolved)
    if ($rows.Count -eq 0) {
        return @{
            hash = "EMPTY"
            tickets = 0
            races = 0
            stake_yen = 0
        }
    }

    $normalized = @(
        $rows |
            Sort-Object race_id, ticket_type, anchor_no, partner_no |
            Select-Object `
                race_id,
                ticket_type,
                anchor_no,
                partner_no,
                anchor_name,
                partner_name,
                runtime_odds,
                runtime_stake_yen,
                runtime_action,
                runtime_ticket_status,
                buy_reason_summary
    )
    $json = $normalized | ConvertTo-Json -Depth 6 -Compress
    $bytes = [Text.Encoding]::UTF8.GetBytes($json)
    $sha = [Security.Cryptography.SHA256]::Create()
    $hash = [BitConverter]::ToString($sha.ComputeHash($bytes)).Replace("-", "")
    $raceCount = @($rows | Select-Object -ExpandProperty race_id -Unique).Count
    $stake = 0
    foreach ($row in $rows) {
        $stake += [double]($row.runtime_stake_yen -as [double])
    }
    return @{
        hash = $hash
        tickets = $rows.Count
        races = $raceCount
        stake_yen = [int][Math]::Round($stake)
    }
}

function Read-State {
    param([string]$Path)
    $resolved = Resolve-ProjectPath $Path
    if (-not (Test-Path -LiteralPath $resolved)) {
        return [pscustomobject]@{
            last_observed_hash = ""
            last_notified_hash = ""
        }
    }
    try {
        return Get-Content -LiteralPath $resolved -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        return [pscustomobject]@{
            last_observed_hash = ""
            last_notified_hash = ""
        }
    }
}

function Get-StateValue {
    param(
        [object]$State,
        [string]$Name,
        [object]$Default = ""
    )
    if ($null -ne $State -and $State.PSObject.Properties[$Name]) {
        return $State.$Name
    }
    return $Default
}

function Find-LatestProjectFile {
    param([string]$Pattern)
    $resolvedPattern = Resolve-ProjectPath $Pattern
    $parent = Split-Path -Parent $resolvedPattern
    $leaf = Split-Path -Leaf $resolvedPattern
    if (-not (Test-Path -LiteralPath $parent)) {
        return ""
    }
    $file = Get-ChildItem -Path $parent -Filter $leaf -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $file) {
        return ""
    }
    return $file.FullName
}

function Resolve-DateSpecificInputs {
    if (-not [string]::IsNullOrWhiteSpace($EntryCsv) -and $EntryCsv -notmatch [regex]::Escape($Date)) {
        Write-Warning "Entry CSV does not match refresh date $Date; ignoring stale input: $EntryCsv"
        $script:EntryCsv = ""
    }
    if (-not [string]::IsNullOrWhiteSpace($PredictionCsv) -and $PredictionCsv -notmatch [regex]::Escape($Date)) {
        Write-Warning "Prediction CSV does not match refresh date $Date; ignoring stale input: $PredictionCsv"
        $script:PredictionCsv = ""
    }

    if ([string]::IsNullOrWhiteSpace($EntryCsv)) {
        $entryCandidates = @(
            "data\datasets\inference\weekly\entry_snapshot_netkeiba_${Date}_target_de_overlay_enriched_workout_knowledge.csv",
            "data\datasets\inference\weekly\entry_snapshot_netkeiba_${Date}_target_de_overlay_enriched_workout.csv",
            "data\datasets\inference\weekly\entry_snapshot_netkeiba_${Date}_target_de_overlay_enriched.csv",
            "data\datasets\inference\weekly\entry_snapshot_netkeiba_${Date}_target_de_overlay.csv",
            "data\datasets\inference\weekly\entry_snapshot_netkeiba_${Date}.csv"
        )
        foreach ($candidate in $entryCandidates) {
            $resolvedCandidate = Resolve-ProjectPath $candidate
            if (Test-Path -LiteralPath $resolvedCandidate) {
                $script:EntryCsv = $candidate
                Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Auto-selected entry CSV for ${Date}: $candidate"
                break
            }
        }
    }

    if ([string]::IsNullOrWhiteSpace($PredictionCsv)) {
        $predictionCandidate = Find-LatestProjectFile "outputs\predictions\preday_target_de_overlay_${Date}\baseline_predictions_*.csv"
        if (-not [string]::IsNullOrWhiteSpace($predictionCandidate)) {
            $script:PredictionCsv = $predictionCandidate
            Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Auto-selected prediction CSV for ${Date}: $predictionCandidate"
        }
    }
}

function Set-StateValue {
    param(
        [object]$State,
        [string]$Name,
        [object]$Value
    )
    if ($State.PSObject.Properties[$Name]) {
        $State.$Name = $Value
    } else {
        $State | Add-Member -NotePropertyName $Name -NotePropertyValue $Value -Force
    }
}

function Write-State {
    param(
        [string]$Path,
        [object]$State
    )
    $resolved = Resolve-ProjectPath $Path
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $resolved) | Out-Null
    $State | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $resolved -Encoding UTF8
}

if ([string]::IsNullOrWhiteSpace($EntryCsv) -or [string]::IsNullOrWhiteSpace($PredictionCsv)) {
    $resolvedInputsJson = Resolve-ProjectPath $CurrentInputsJson
    if (Test-Path -LiteralPath $resolvedInputsJson) {
        try {
            $currentInputs = Get-Content -LiteralPath $resolvedInputsJson -Raw -Encoding UTF8 | ConvertFrom-Json
            $inputDate = Get-StateValue -State $currentInputs -Name "date" -Default ""
            if ([string]::IsNullOrWhiteSpace($inputDate) -or $inputDate -eq $Date) {
                if ([string]::IsNullOrWhiteSpace($EntryCsv)) {
                    $EntryCsv = Get-StateValue -State $currentInputs -Name "entry_csv" -Default ""
                }
                if ([string]::IsNullOrWhiteSpace($PredictionCsv)) {
                    $PredictionCsv = Get-StateValue -State $currentInputs -Name "prediction_csv" -Default ""
                }
            }
        } catch {
            Write-Warning "Could not read current dashboard input overrides: $resolvedInputsJson"
            Write-Warning $_.Exception.Message
        }
    }
}

Resolve-DateSpecificInputs

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable was not found: $PythonExe"
}

if (
    (Test-Path -LiteralPath $BuildTargetRaLapHistory) -and
    (-not [string]::IsNullOrWhiteSpace($EntryCsv)) -and
    (Test-Path -LiteralPath (Resolve-ProjectPath $EntryCsv)) -and
    (Test-Path -LiteralPath (Resolve-ProjectPath $TargetRaRaceLapsCsv))
) {
    $currentTargetRaLapHistoryResolved = Resolve-ProjectPath $CurrentTargetRaLapHistoryCsv
    $rebuiltTargetRaLapInputs = @(
        "data\datasets\cache\workout_lap_pedigree_interactions_confirmed_opponent_2023plus_rebuilt_20260623\train_features.csv",
        "data\datasets\cache\workout_lap_pedigree_interactions_confirmed_opponent_2023plus_rebuilt_20260623\test_features.csv"
    ) | ForEach-Object { Resolve-ProjectPath $_ }
    $baseTargetRaLapInputs = @(
        "data\datasets\cache\workout_lap_pedigree_interactions_confirmed_opponent_2023plus\train_features.csv",
        "data\datasets\cache\workout_lap_pedigree_interactions_confirmed_opponent_2023plus\test_features.csv"
    ) | ForEach-Object { Resolve-ProjectPath $_ }
    if (($rebuiltTargetRaLapInputs | Where-Object { Test-Path -LiteralPath $_ }).Count -eq 2) {
        $targetRaLapInputs = @($rebuiltTargetRaLapInputs)
    } else {
        $targetRaLapInputs = @($baseTargetRaLapInputs | Where-Object { Test-Path -LiteralPath $_ })
    }
    if ($targetRaLapInputs.Count -ge 2) {
        $entryResolvedForLapHistory = Resolve-ProjectPath $EntryCsv
        $targetRaRaceLapsResolved = Resolve-ProjectPath $TargetRaRaceLapsCsv
        $allTargetRaInputs = @($entryResolvedForLapHistory, $targetRaRaceLapsResolved) + $targetRaLapInputs
        $needsTargetRaLapHistory = -not (Test-Path -LiteralPath $currentTargetRaLapHistoryResolved)
        if (-not $needsTargetRaLapHistory) {
            $outTime = (Get-Item -LiteralPath $currentTargetRaLapHistoryResolved).LastWriteTime
            foreach ($inputPath in $allTargetRaInputs) {
                if ((Get-Item -LiteralPath $inputPath).LastWriteTime -gt $outTime) {
                    $needsTargetRaLapHistory = $true
                    break
                }
            }
        }
        if ($needsTargetRaLapHistory) {
            try {
                $lapHistoryArgs = @(
                    $BuildTargetRaLapHistory,
                    "--laps-csv", $TargetRaRaceLapsCsv,
                    "--output-csv", $CurrentTargetRaLapHistoryCsv,
                    "--summary-json", $CurrentTargetRaLapHistorySummaryJson
                )
                foreach ($runnerCsv in $targetRaLapInputs) {
                    $lapHistoryArgs += "--runner-csv"
                    $lapHistoryArgs += $runnerCsv
                }
                $lapHistoryArgs += "--runner-csv"
                $lapHistoryArgs += $entryResolvedForLapHistory
                Invoke-Checked -Label "Build current TARGET RA official-lap history features" -Command {
                    & $PythonExe @lapHistoryArgs
                }
            } catch {
                Write-Warning "TARGET RA official-lap history build failed; continuing without this shadow feature."
                Write-Warning $_.Exception.Message
            }
        }
    } else {
        Write-Warning "TARGET RA official-lap history skipped because historical runner CSVs were not found."
    }
}

$ExplicitRaceKeysProvided = $RaceKeys.Count -gt 0
$EffectiveRaceKeys = @($RaceKeys)
$ListEntryRaceKeys = Join-Path $ProjectRoot "scripts\list_entry_race_keys.py"
if ($EffectiveRaceKeys.Count -eq 0 -and -not [string]::IsNullOrWhiteSpace($EntryCsv) -and (Test-Path -LiteralPath $ListEntryRaceKeys)) {
    $resolvedEntryCsvForRaceKeys = Resolve-ProjectPath $EntryCsv
    if (Test-Path -LiteralPath $resolvedEntryCsvForRaceKeys) {
        try {
            $EffectiveRaceKeys = @(
                & $PythonExe $ListEntryRaceKeys --entry-csv $resolvedEntryCsvForRaceKeys --date $Date |
                    Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
            )
            if ($EffectiveRaceKeys.Count -gt 0) {
                Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Auto-detected $($EffectiveRaceKeys.Count) race keys from entry CSV for odds fetch."
            }
        } catch {
            Write-Warning "Could not auto-detect race keys from entry CSV: $resolvedEntryCsvForRaceKeys"
            Write-Warning $_.Exception.Message
            $EffectiveRaceKeys = @()
        }
    }
}

if (-not $SkipOddsFetch) {
    $oddsFetchSleepText = $OddsFetchSleepSeconds.ToString([Globalization.CultureInfo]::InvariantCulture)
    $fetchArgs = @(
        $FetchJraOdds,
        "--date", $Date,
        "--bet-types", "win_place_frame", "umaren", "wide",
        "--max-workers", "$OddsFetchWorkers",
        "--sleep-seconds", $oddsFetchSleepText,
        "--pair-output-csv", $PairOddsCsv,
        "--single-output-csv", $SingleOddsCsv,
        "--summary-json", "outputs\analysis\current_strongest_line_update\jra_official_odds_summary.json"
    )
    if ($EffectiveRaceKeys.Count -gt 0) {
        $fetchArgs += "--race-keys"
        $fetchArgs += $EffectiveRaceKeys
    }
    if ($ExplicitRaceKeysProvided) {
        $fetchArgs += "--merge-existing"
    }
    Invoke-Checked -Label "Fetch JRA official odds" -Command { & $PythonExe @fetchArgs }
}

if (-not $SkipTimeline -and (Test-Path -LiteralPath $AppendTimeline)) {
    $timelineArgs = @(
        $AppendTimeline,
        "--pair-latest-csv", $PairOddsCsv,
        "--single-latest-csv", $SingleOddsCsv,
        "--decision-label", $DecisionLabel,
        "--summary-json", "outputs\analysis\current_strongest_line_update\odds_timeline_append_summary.json"
    )
    if ($EffectiveRaceKeys.Count -gt 0) {
        $timelineArgs += "--race-ids"
        $timelineArgs += $EffectiveRaceKeys
    }
    try {
        Invoke-Checked -Label "Append live odds timeline ($DecisionLabel)" -Command { & $PythonExe @timelineArgs }
    } catch {
        Write-Warning "Append live odds timeline failed; continuing."
        Write-Warning $_.Exception.Message
    }
}

if (-not $SkipTimeline -and (Test-Path -LiteralPath $BuildTimelineFeatures)) {
    try {
        Invoke-Checked -Label "Build live odds timeline features" -Command {
            & $PythonExe $BuildTimelineFeatures `
                --timeline-csv "data\processed\live_odds\realtime_single_odds_timeline.csv" `
                --output-csv "data\processed\live_odds\realtime_single_odds_timeline_features.csv"
        }
    } catch {
        Write-Warning "Build live odds timeline features failed; continuing."
        Write-Warning $_.Exception.Message
    }
}

if ((-not $SkipBodyWeight) -and (Test-Path -LiteralPath $ExtractBodyWeights)) {
    Invoke-Checked -Label "Extract JRA official body weights" -Command {
        & $PythonExe $ExtractBodyWeights `
            --date $Date `
            --output-csv $BodyWeightCsv `
            --summary-json "outputs\analysis\live_body_weight\jra_official_body_weight_summary.json"
    }
}

if (-not $SkipTrackFetch) {
    $trackArgs = @(
        $FetchTrackConditions,
        "--output-csv", $TrackConditionCsv,
        "--summary-json", "outputs\analysis\live_track_conditions\current_track_conditions_summary.json"
    )
    try {
        Invoke-Checked -Label "Fetch JRA current track conditions" -Command { & $PythonExe @trackArgs }
    } catch {
        $resolvedTrackConditionCsv = Resolve-ProjectPath $TrackConditionCsv
        if (Test-Path -LiteralPath $resolvedTrackConditionCsv) {
            Write-Warning "Fetch JRA current track conditions failed; continuing with existing file: $resolvedTrackConditionCsv"
            Write-Warning $_.Exception.Message
        } else {
            throw
        }
    }
}

if ((Test-Path -LiteralPath $EnrichBasicAbility) -and (-not [string]::IsNullOrWhiteSpace($PredictionCsv))) {
    try {
        $predictionBaseName = [System.IO.Path]::GetFileNameWithoutExtension((Resolve-ProjectPath $PredictionCsv))
        $basicAbilityOut = "outputs\predictions\runtime_basic_ability_enriched_${Date}\${predictionBaseName}_basic_ability.csv"
        $enrichArgs = @(
            $EnrichBasicAbility,
            "--prediction-csv", $PredictionCsv,
            "--output-csv", $basicAbilityOut
        )
        if (-not [string]::IsNullOrWhiteSpace($EntryCsv)) {
            $enrichArgs += "--entry-csv"
            $enrichArgs += $EntryCsv
        }
        Invoke-Checked -Label "Enrich prediction with basic ability transforms" -Command {
            & $PythonExe @enrichArgs
        }
        $PredictionCsv = $basicAbilityOut
    } catch {
        Write-Warning "Basic ability enrichment failed; continuing with original prediction CSV."
        Write-Warning $_.Exception.Message
    }
}

Invoke-Checked -Label "Build current strongest tickets" -Command {
    $buildTicketArgs = @(
        $BuildTickets,
        "--track-condition-csv", $TrackConditionCsv,
        "--target-ra-lap-history-csv", $CurrentTargetRaLapHistoryCsv,
        "--update-latest-summary"
    )
    if (-not [string]::IsNullOrWhiteSpace($EntryCsv)) {
        $buildTicketArgs += "--entry-csv"
        $buildTicketArgs += $EntryCsv
    }
    if (-not [string]::IsNullOrWhiteSpace($PredictionCsv)) {
        $buildTicketArgs += "--prediction-csv"
        $buildTicketArgs += $PredictionCsv
    }
    & $PythonExe @buildTicketArgs
}

if ((Test-Path -LiteralPath $ApplyLiveSafety) -and (Test-Path -LiteralPath (Resolve-ProjectPath $BodyWeightCsv))) {
    $liveSafetyOutDir = "outputs\analysis\current_strongest_runtime_v1\live_safety_overlay"
    Invoke-Checked -Label "Apply live body-weight safety overlay" -Command {
        & $PythonExe $ApplyLiveSafety --tickets-csv $TicketsCsv --body-weight-csv $BodyWeightCsv --output-dir $liveSafetyOutDir
    }
    $overlaidTickets = Resolve-ProjectPath (Join-Path $liveSafetyOutDir "live_safety_overlaid_tickets.csv")
    if (Test-Path -LiteralPath $overlaidTickets) {
        Copy-Item -LiteralPath $overlaidTickets -Destination (Resolve-ProjectPath $TicketsCsv) -Force
        Copy-Item -LiteralPath $overlaidTickets -Destination (Resolve-ProjectPath $DashboardTicketsCsv) -Force
    }
}

if (-not $SkipTimeline -and (Test-Path -LiteralPath $EvaluateFixedTimePairEdge) -and (Test-Path -LiteralPath (Resolve-ProjectPath $TicketsCsv))) {
    try {
        Invoke-Checked -Label "Evaluate fixed-time pair edge" -Command {
            & $PythonExe $EvaluateFixedTimePairEdge `
                --tickets-csv $TicketsCsv `
                --pair-timeline-csv "data\processed\live_odds\realtime_pair_odds_timeline.csv" `
                --output-dir "outputs\analysis\current_strongest_runtime_v1\fixed_time_pair_edge"
        }
    } catch {
        Write-Warning "Evaluate fixed-time pair edge failed; continuing."
        Write-Warning $_.Exception.Message
    }
}

if (Test-Path -LiteralPath $ApplyPairJointV2Guard) {
    try {
        Invoke-Checked -Label "Apply pair joint V2 runtime guard" -Command {
            & $PythonExe $ApplyPairJointV2Guard `
                --candidates-csv "outputs\analysis\current_strongest_runtime_v1\current_strongest_all_candidates.csv" `
                --tickets-csv $TicketsCsv `
                --pair-odds-csv $PairOddsCsv `
                --output-dir "outputs\analysis\current_strongest_runtime_v1\pair_joint_v2_runtime_guard"
        }
    } catch {
        Write-Warning "Apply pair joint V2 runtime guard failed; continuing without V2 wide shadow candidates."
        Write-Warning $_.Exception.Message
    }
}

if ((-not $SkipWin5) -and (Test-Path -LiteralPath $BuildWin5) -and (-not [string]::IsNullOrWhiteSpace($PredictionCsv))) {
    try {
        Invoke-Checked -Label "Build WIN5 candidates" -Command {
            $win5Args = @(
                $BuildWin5,
                "--prediction-csv", $PredictionCsv,
                "--single-odds-csv", $SingleOddsCsv,
                "--date", $Date,
                "--output-json", $Win5Json,
                "--output-csv", "outputs\analysis\win5_runtime\win5_candidates.csv"
            )
            if ($Win5RaceIds.Count -gt 0) {
                $win5Args += "--race-ids"
                $win5Args += $Win5RaceIds
            }
            & $PythonExe @win5Args
        }
    } catch {
        Write-Warning "Build WIN5 candidates failed; continuing without WIN5 panel."
        Write-Warning $_.Exception.Message
    }
}

if (-not $SkipDashboard) {
    Invoke-Checked -Label "Build live dashboard" -Command {
        $dashboardArgs = @(
            $BuildDashboard,
            "--body-weight-csv", $BodyWeightCsv,
            "--win5-json", $Win5Json,
            "--wide-shadow-csv", $WideShadowCsv,
            "--default-date", $Date,
            "--output-html", $DashboardHtml
        )
        if (-not [string]::IsNullOrWhiteSpace($EntryCsv)) {
            $dashboardArgs += "--entry-csv"
            $dashboardArgs += $EntryCsv
        }
        if (-not [string]::IsNullOrWhiteSpace($PredictionCsv)) {
            $dashboardArgs += "--prediction-csv"
            $dashboardArgs += $PredictionCsv
        }
        & $PythonExe @dashboardArgs
    }
}

if (Test-Path -LiteralPath $DetectTrackChanges) {
    try {
        $trackChangeArgs = @(
            $DetectTrackChanges,
            "--track-csv", $TrackConditionCsv,
            "--state-json", $TrackChangeStateJson,
            "--output-json", "outputs\analysis\live_track_conditions\track_condition_change_summary.json",
            "--message-text", "outputs\notifications\track_condition_change_latest.txt"
        )
        if ($Notify) {
            $trackChangeArgs += "--send"
        } elseif ($SendIfConfigured) {
            $trackChangeArgs += "--send-if-configured"
        }
        Invoke-Checked -Label "Detect JRA track condition changes" -Command {
            & $PythonExe @trackChangeArgs
        }
    } catch {
        Write-Warning "Track condition change detection failed; continuing."
        Write-Warning $_.Exception.Message
    }
}

if (Test-Path -LiteralPath $FreezeDecisionSnapshot) {
    try {
        $snapshotArgs = @(
            $FreezeDecisionSnapshot,
            "--dashboard-html", $DashboardHtml,
            "--candidates-csv", "outputs\analysis\current_strongest_runtime_v1\current_strongest_all_candidates.csv",
            "--tickets-csv", $TicketsCsv,
            "--track-condition-csv", $TrackConditionCsv,
            "--track-change-summary-json", "outputs\analysis\live_track_conditions\track_condition_change_summary.json",
            "--decision-label", $DecisionLabel,
            "--output-csv", "data\processed\live_decision_snapshots\current_strongest_decision_snapshots.csv",
            "--latest-csv", "outputs\analysis\current_strongest_runtime_v1\decision_snapshot_latest.csv",
            "--summary-json", "outputs\analysis\current_strongest_runtime_v1\decision_snapshot_summary.json"
        )
        if ($EffectiveRaceKeys.Count -gt 0) {
            $snapshotArgs += "--race-ids"
            $snapshotArgs += $EffectiveRaceKeys
        }
        Invoke-Checked -Label "Freeze current strongest decision snapshot ($DecisionLabel)" -Command {
            & $PythonExe @snapshotArgs
        }
    } catch {
        Write-Warning "Freeze current strongest decision snapshot failed; continuing."
        Write-Warning $_.Exception.Message
    }
}

if (-not $SkipExternalAudit) {
    if (Test-Path -LiteralPath $BuildFinalOddsSurvival) {
        try {
            Invoke-Checked -Label "Build final odds survival dataset" -Command {
                & $PythonExe $BuildFinalOddsSurvival `
                    --candidates-csv "outputs\analysis\current_strongest_runtime_v1\current_strongest_all_candidates.csv" `
                    --pair-timeline-csv "data\processed\live_odds\realtime_pair_odds_timeline.csv" `
                    --output-dir "outputs\analysis\final_odds_survival_model_v1"
            }
        } catch {
            Write-Warning "Build final odds survival dataset failed; continuing."
            Write-Warning $_.Exception.Message
        }
    }
    if (Test-Path -LiteralPath $BuildCandidateRejectionLedger) {
        try {
            Invoke-Checked -Label "Build candidate rejection ledger" -Command {
                & $PythonExe $BuildCandidateRejectionLedger `
                    --candidates-csv "outputs\analysis\current_strongest_runtime_v1\current_strongest_all_candidates.csv" `
                    --selected-csv $TicketsCsv `
                    --decision-label $DecisionLabel `
                    --history-csv "data\processed\live_decision_snapshots\current_strongest_candidate_rejection_ledger_history.csv" `
                    --output-dir "outputs\analysis\candidate_rejection_ledger_v1"
            }
        } catch {
            Write-Warning "Build candidate rejection ledger failed; continuing."
            Write-Warning $_.Exception.Message
        }
    }
    if (Test-Path -LiteralPath $FreezeChampionManifest) {
        try {
            Invoke-Checked -Label "Freeze Champion manifest" -Command {
                & $PythonExe $FreezeChampionManifest --output-dir "outputs\analysis\champion_strategy_freeze_v1"
            }
        } catch {
            Write-Warning "Freeze Champion manifest failed; continuing."
            Write-Warning $_.Exception.Message
        }
    }
}

if ((-not $SkipResultFetch) -and (Test-Path -LiteralPath $EvaluateCurrentPnl)) {
    try {
        Invoke-Checked -Label "Fetch settled race results / actual going" -Command {
            & $PythonExe $EvaluateCurrentPnl --fetch
        }
        if ($SkipDashboard) {
            Write-Warning "Skipped live dashboard rebuild with result going because -SkipDashboard was specified."
        } else {
            Invoke-Checked -Label "Rebuild live dashboard with result going" -Command {
            $dashboardArgs = @(
                $BuildDashboard,
                "--body-weight-csv", $BodyWeightCsv,
                "--win5-json", $Win5Json,
                "--wide-shadow-csv", $WideShadowCsv,
                "--default-date", $Date,
                "--output-html", $DashboardHtml
            )
            if (-not [string]::IsNullOrWhiteSpace($EntryCsv)) {
                $dashboardArgs += "--entry-csv"
                $dashboardArgs += $EntryCsv
            }
            if (-not [string]::IsNullOrWhiteSpace($PredictionCsv)) {
                $dashboardArgs += "--prediction-csv"
                $dashboardArgs += $PredictionCsv
            }
            & $PythonExe @dashboardArgs
            }
        }
    } catch {
        Write-Warning "Fetch settled race results / actual going failed; continuing with current going only."
        Write-Warning $_.Exception.Message
    }
}

$currentDashboardInputs = [ordered]@{
    date = $Date
    entry_csv = $EntryCsv
    prediction_csv = $PredictionCsv
    note = "Auto-selected by run_current_strongest_line_update.ps1 for the active dashboard refresh date."
}
Write-State -Path $CurrentInputsJson -State $currentDashboardInputs

$ticketState = Get-TicketHash -Path $TicketsCsv
$state = Read-State -Path $StateJson
$lastObservedHash = Get-StateValue -State $state -Name "last_observed_hash" -Default ""
$lastNotifiedHash = Get-StateValue -State $state -Name "last_notified_hash" -Default ""
$changedSinceObserved = $ForceNotify -or ($ticketState.hash -ne $lastObservedHash)
$changedSinceNotified = $ForceNotify -or ($ticketState.hash -ne $lastNotifiedHash)

$result = [ordered]@{
    generated_at = (Get-Date).ToString("s")
    skip_odds_fetch = [bool]$SkipOddsFetch
    skip_track_fetch = [bool]$SkipTrackFetch
    skip_body_weight = [bool]$SkipBodyWeight
    skip_result_fetch = [bool]$SkipResultFetch
    skip_win5 = [bool]$SkipWin5
    skip_dashboard = [bool]$SkipDashboard
    odds_fetch_workers = [int]$OddsFetchWorkers
    odds_fetch_sleep_seconds = [double]$OddsFetchSleepSeconds
    entry_csv = $EntryCsv
    prediction_csv = $PredictionCsv
    body_weight_csv = $BodyWeightCsv
    win5_json = $Win5Json
    notify_requested = [bool]($Notify -or $SendIfConfigured)
    force_notify = [bool]$ForceNotify
    changed_since_observed = [bool]$changedSinceObserved
    changed_since_notified = [bool]$changedSinceNotified
    tickets = $ticketState.tickets
    races = $ticketState.races
    stake_yen = $ticketState.stake_yen
    hash = $ticketState.hash
    notification = "not_requested"
}

if (($Notify -or $SendIfConfigured) -and $changedSinceNotified -and $ticketState.tickets -gt 0) {
    $lineArgs = @(
        "-File", $LineAlert,
        "-TicketsCsv", $TicketsCsv,
        "-MaxRaces", "8"
    )
    if ($Notify) {
        $lineArgs += "-Send"
    }
    if ($SendIfConfigured) {
        $lineArgs += "-SendIfConfigured"
    }

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Send LINE alert"
    $lineOutput = @(powershell.exe -NoProfile -ExecutionPolicy Bypass @lineArgs)
    $lineText = $lineOutput -join "`n"
    Write-Host $lineText
    if ($lineText -match '"ok"\s*:\s*true') {
        $result.notification = "sent"
        Set-StateValue -State $state -Name "last_notified_hash" -Value $ticketState.hash
    } elseif ($lineText -match '"dry_run"\s*:\s*true') {
        $result.notification = "dry_run"
    } else {
        $result.notification = "attempted_but_not_confirmed"
    }
} elseif (($Notify -or $SendIfConfigured) -and -not $changedSinceNotified) {
    $result.notification = "skipped_duplicate"
} elseif (($Notify -or $SendIfConfigured) -and $ticketState.tickets -eq 0) {
    $result.notification = "skipped_no_tickets"
}

Set-StateValue -State $state -Name "last_observed_hash" -Value $ticketState.hash
Set-StateValue -State $state -Name "last_observed_at" -Value (Get-Date).ToString("s")
Set-StateValue -State $state -Name "last_observed_tickets" -Value $ticketState.tickets
Set-StateValue -State $state -Name "last_observed_races" -Value $ticketState.races
Set-StateValue -State $state -Name "last_observed_stake_yen" -Value $ticketState.stake_yen
Write-State -Path $StateJson -State $state

$resultPath = Resolve-ProjectPath "outputs\analysis\current_strongest_line_update\summary.json"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $resultPath) | Out-Null
$result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $resultPath -Encoding UTF8
$result | ConvertTo-Json -Depth 6
