param(
    [string]$Sid = "",
    [switch]$SkipInit,
    [switch]$RequireSid,
    [int]$TimeoutSeconds = 20
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
. (Join-Path $ProjectRoot "scripts\Resolve-JvSid.ps1")
$sidResolution = Resolve-JvSid -Sid $Sid
$Sid = $sidResolution.Sid

$PowerShell32 = "$env:WINDIR\SysWOW64\WindowsPowerShell\v1.0\powershell.exe"
$result = [ordered]@{
    powershell32_exists = Test-Path $PowerShell32
    sid_present = -not [string]::IsNullOrWhiteSpace($Sid)
    sid_source = $sidResolution.Source
    com_available = $false
    init_return = $null
    ready = $false
    error = $null
}

if ($RequireSid -and -not $result.sid_present) {
    $result.error = "JV_SID is not set. Set the JRA-VAN Data Lab service key in JV_SID or pass -Sid."
    $result | ConvertTo-Json -Depth 4
    exit 1
}

if (-not $result.powershell32_exists) {
    $result.error = "32-bit PowerShell was not found. JV-Link COM normally requires 32-bit PowerShell."
    $result | ConvertTo-Json -Depth 4
    exit 1
}

$sidLiteral = $Sid.Replace("'", "''")
$skipInitLiteral = if ($SkipInit) { 'true' } else { 'false' }
$probe = @"
`$Sid = '$sidLiteral'
`$SkipInit = [bool]::Parse('$skipInitLiteral')
`$ErrorActionPreference = "Stop"
`$out = [ordered]@{
    com_available = `$false
    init_return = `$null
    error = `$null
}
try {
    `$jv = New-Object -ComObject "JVDTLab.JVLink"
    `$out.com_available = `$true
    if (-not `$SkipInit -and -not [string]::IsNullOrWhiteSpace(`$Sid)) {
        `$out.init_return = `$jv.JVInit(`$Sid)
    }
    try { `$null = `$jv.JVClose() } catch {}
} catch {
    `$out.error = `$_.Exception.Message
}
`$out | ConvertTo-Json -Depth 4
"@

$encodedProbe = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($probe))
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $PowerShell32
$psi.Arguments = "-NoProfile -ExecutionPolicy Bypass -EncodedCommand $encodedProbe"
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.CreateNoWindow = $true
try {
    if ($psi.EnvironmentVariables.ContainsKey("PATH") -and $psi.EnvironmentVariables.ContainsKey("Path")) {
        $psi.EnvironmentVariables.Remove("PATH")
    }
} catch {}

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi
$null = $proc.Start()

if (-not $proc.WaitForExit($TimeoutSeconds * 1000)) {
    try { Stop-Process -Id $proc.Id -Force } catch {}
    $result.error = "JV-Link probe timed out after $TimeoutSeconds seconds. JV-Link may be unregistered, waiting for a dialog, or blocked."
    $result | ConvertTo-Json -Depth 4
    exit 1
}

$stdout = $proc.StandardOutput.ReadToEnd()
$stderr = $proc.StandardError.ReadToEnd()
if ($proc.ExitCode -ne 0) {
    $result.error = "JV-Link probe failed with exit code $($proc.ExitCode). $stderr"
    $result | ConvertTo-Json -Depth 4
    exit 1
}

$probeJson = $stdout

$probeResult = $probeJson | ConvertFrom-Json
$result.com_available = [bool]$probeResult.com_available
$result.init_return = $probeResult.init_return
$result.error = $probeResult.error
$result.ready = $result.powershell32_exists -and $result.com_available -and ($result.sid_present -or $SkipInit) -and (-not $result.error)

$result | ConvertTo-Json -Depth 4
if (-not $result.ready) {
    exit 1
}
