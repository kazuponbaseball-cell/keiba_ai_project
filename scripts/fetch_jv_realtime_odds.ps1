param(
    [Parameter(Mandatory = $true)]
    [string]$RaceKey,

    [ValidateSet("all", "win_place_frame", "umaren", "wide", "umatan", "trio", "trifecta")]
    [string]$BetType = "all",

    [string]$Sid = "",

    [string]$OutputDir = "data\raw\jv_realtime_odds",

    [int]$BufferSize = 100000
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
. (Join-Path $ProjectRoot "scripts\Resolve-JvSid.ps1")
$sidResolution = Resolve-JvSid -Sid $Sid
$Sid = $sidResolution.Sid
if ([string]::IsNullOrWhiteSpace($Sid)) {
    throw "JV SID was not found. Set JV_SID, pass -Sid, or configure JRA-VAN Data Lab servicekey."
}

$DataSpecByBetType = @{
    all = "0B30"
    win_place_frame = "0B31"
    umaren = "0B32"
    wide = "0B33"
    umatan = "0B34"
    trio = "0B35"
    trifecta = "0B36"
}

$ResolvedOutputDir = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir
} else {
    Join-Path $ProjectRoot $OutputDir
}
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$dataspec = $DataSpecByBetType[$BetType]
$raceDir = Join-Path $ResolvedOutputDir $RaceKey
New-Item -ItemType Directory -Force -Path $raceDir | Out-Null
$rawPath = Join-Path $raceDir "${stamp}_${dataspec}_${BetType}.txt"
$metaPath = Join-Path $raceDir "${stamp}_${dataspec}_${BetType}.json"

$jv = $null
$records = New-Object System.Collections.Generic.List[string]
$readLog = New-Object System.Collections.Generic.List[object]
$openReturn = $null
$initReturn = $null
$closeReturn = $null
$errorMessage = $null

try {
    $jv = New-Object -ComObject "JVDTLab.JVLink"
    $initReturn = $jv.JVInit($Sid)
    if ($initReturn -lt 0) {
        throw "JVInit failed: $initReturn"
    }

    $openReturn = $jv.JVRTOpen($dataspec, $RaceKey)
    if ($openReturn -lt 0) {
        throw "JVRTOpen failed: $openReturn dataspec=$dataspec key=$RaceKey"
    }

    while ($true) {
        $buff = ""
        $fileName = ""
        $rc = $jv.JVRead([ref]$buff, $BufferSize, [ref]$fileName)
        $readLog.Add([pscustomobject]@{
            return_code = $rc
            file_name = $fileName
            bytes = if ($buff -ne $null) { $buff.Length } else { 0 }
        })
        if ($rc -gt 0) {
            $records.Add($buff)
            continue
        }
        if ($rc -eq -1) {
            continue
        }
        if ($rc -eq 0) {
            break
        }
        throw "JVRead failed: $rc"
    }
} catch {
    $errorMessage = $_.Exception.Message
} finally {
    if ($jv -ne $null) {
        try {
            $closeReturn = $jv.JVClose()
        } catch {
            $closeReturn = "JVClose failed: $($_.Exception.Message)"
        }
    }
}

[System.IO.File]::WriteAllLines($rawPath, $records, [System.Text.Encoding]::GetEncoding(932))
$meta = [pscustomobject]@{
    race_key = $RaceKey
    bet_type = $BetType
    dataspec = $dataspec
    sid_present = -not [string]::IsNullOrWhiteSpace($Sid)
    sid_source = $sidResolution.Source
    snapshot_at = $stamp
    init_return = $initReturn
    open_return = $openReturn
    close_return = $closeReturn
    error = $errorMessage
    raw_path = $rawPath
    record_count = $records.Count
    read_log = $readLog
}
$meta | ConvertTo-Json -Depth 5 | Set-Content -Path $metaPath -Encoding UTF8
$meta | ConvertTo-Json -Depth 5
if ($errorMessage) {
    exit 1
}
