param(
    [string[]]$DataSpecs = @("RACE"),
    [string[]]$FromTimes = @("00000000000000", "20240101000000", "20260601000000", "20260628000000"),
    [int[]]$Options = @(1, 2),
    [string]$Sid = "",
    [string]$OutputJson = "outputs\analysis\jvopen_probe_matrix\summary.json"
)

$ErrorActionPreference = "Stop"

$PowerShell32 = "$env:WINDIR\SysWOW64\WindowsPowerShell\v1.0\powershell.exe"
if ([Environment]::Is64BitProcess -and (Test-Path -LiteralPath $PowerShell32)) {
    $relayArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $PSCommandPath,
        "-OutputJson", $OutputJson
    )
    if ($DataSpecs.Count -gt 0) {
        $relayArgs += "-DataSpecs"
        $relayArgs += $DataSpecs
    }
    if ($FromTimes.Count -gt 0) {
        $relayArgs += "-FromTimes"
        $relayArgs += $FromTimes
    }
    if ($Options.Count -gt 0) {
        $relayArgs += "-Options"
        $relayArgs += $Options
    }
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

$results = New-Object System.Collections.Generic.List[object]
foreach ($dataSpec in $DataSpecs) {
    foreach ($fromTime in $FromTimes) {
        foreach ($option in $Options) {
            $jv = $null
            $initReturn = $null
            $openReturn = $null
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
                $openReturn = $jv.JVOpen($dataSpec, $fromTime, $option, [ref]$readCount, [ref]$downloadCount, [ref]$lastFileTimestamp)
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
            $results.Add([pscustomobject]@{
                data_spec = $dataSpec
                from_time = $fromTime
                option = $option
                init_return = $initReturn
                open_return = $openReturn
                read_count = $readCount
                download_count = $downloadCount
                last_file_timestamp = $lastFileTimestamp
                close_return = $closeReturn
                error = $errorMessage
            })
        }
    }
}

$outPath = if ([System.IO.Path]::IsPathRooted($OutputJson)) { $OutputJson } else { Join-Path $ProjectRoot $OutputJson }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outPath) | Out-Null
$payload = [pscustomobject]@{
    generated_at = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    sid_present = -not [string]::IsNullOrWhiteSpace($Sid)
    sid_source = $sidResolution.Source
    rows = $results
}
$payload | ConvertTo-Json -Depth 6 | Set-Content -Path $outPath -Encoding UTF8
$payload | ConvertTo-Json -Depth 6
