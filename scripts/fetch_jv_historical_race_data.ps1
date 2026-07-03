param(
    [string]$DataSpec = "RACE",

    [Parameter(Mandatory = $true)]
    [string]$FromTime,

    [ValidateSet(1, 2, 3, 4)]
    [int]$Option = 4,

    [string]$Sid = "",

    [string]$OutputDir = "data\raw\jv_historical",

    [int]$BufferSize = 110000
)

$ErrorActionPreference = "Stop"

$PowerShell32 = "$env:WINDIR\SysWOW64\WindowsPowerShell\v1.0\powershell.exe"
if ([Environment]::Is64BitProcess -and (Test-Path -LiteralPath $PowerShell32)) {
    $relayArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $PSCommandPath,
        "-DataSpec", $DataSpec,
        "-FromTime", $FromTime,
        "-Option", $Option,
        "-OutputDir", $OutputDir,
        "-BufferSize", $BufferSize
    )
    if (-not [string]::IsNullOrWhiteSpace($Sid)) {
        $relayArgs += @("-Sid", $Sid)
    }
    & $PowerShell32 @relayArgs
    exit $LASTEXITCODE
}

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
. (Join-Path $ProjectRoot "scripts\Resolve-JvSid.ps1")
$sidResolution = Resolve-JvSid -Sid $Sid
$Sid = $sidResolution.Sid
if ([string]::IsNullOrWhiteSpace($Sid)) {
    throw "JV SID was not found. Set JV_SID, pass -Sid, or configure JRA-VAN Data Lab servicekey."
}

$ResolvedOutputDir = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir
} else {
    Join-Path $ProjectRoot $OutputDir
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $ResolvedOutputDir "${stamp}_${DataSpec}_${FromTime}_opt${Option}"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$rawPath = Join-Path $runDir "records.txt"
$metaPath = Join-Path $runDir "metadata.json"

$jv = $null
$records = New-Object System.Collections.Generic.List[string]
$readLog = New-Object System.Collections.Generic.List[object]
$openReturn = $null
$initReturn = $null
$closeReturn = $null
$readCount = 0
$downloadCount = 0
$lastFileTimestamp = ""
$errorMessage = $null

try {
    $jv = New-Object -ComObject "JVDTLab.JVLink"
    $initReturn = $jv.JVInit($Sid)
    if ($initReturn -lt 0) {
        throw "JVInit failed: $initReturn"
    }

    $openReturn = $jv.JVOpen($DataSpec, $FromTime, $Option, [ref]$readCount, [ref]$downloadCount, [ref]$lastFileTimestamp)
    if ($openReturn -lt 0) {
        throw "JVOpen failed: $openReturn dataspec=$DataSpec from=$FromTime option=$Option"
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
    data_spec = $DataSpec
    from_time = $FromTime
    option = $Option
    sid_present = -not [string]::IsNullOrWhiteSpace($Sid)
    sid_source = $sidResolution.Source
    snapshot_at = $stamp
    init_return = $initReturn
    open_return = $openReturn
    close_return = $closeReturn
    read_count = $readCount
    download_count = $downloadCount
    last_file_timestamp = $lastFileTimestamp
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
