param(
    [switch]$TestAfterSave,
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}

Set-Location -LiteralPath $ProjectRoot

function Read-RequiredSecret {
    param(
        [string]$Name,
        [int]$MinLength,
        [string]$Pattern = ""
    )

    while ($true) {
        $secure = Read-Host "Paste $Name" -AsSecureString
        $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
        try {
            $value = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
        } finally {
            if ($bstr -ne [IntPtr]::Zero) {
                [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
            }
        }

        $value = $value.Trim()
        if ($value.Length -lt $MinLength) {
            Write-Warning "$Name is too short. length=$($value.Length)"
            continue
        }
        if (-not [string]::IsNullOrWhiteSpace($Pattern) -and $value -notmatch $Pattern) {
            Write-Warning "$Name format looks invalid."
            continue
        }
        return $value
    }
}

Write-Host "LINE credential setup"
Write-Host "1. Copy the long-lived channel access token from LINE Developers > Messaging API settings."
Write-Host "2. Copy Your user ID from LINE Developers > Basic settings."
Write-Host "The pasted values are saved as Windows user environment variables."
Write-Host ""

$token = Read-RequiredSecret -Name "LINE_CHANNEL_ACCESS_TOKEN" -MinLength 100
$userId = Read-RequiredSecret -Name "LINE_USER_ID" -MinLength 33 -Pattern "^U[0-9a-fA-F]{32}$"

[Environment]::SetEnvironmentVariable("LINE_CHANNEL_ACCESS_TOKEN", $token, "User")
[Environment]::SetEnvironmentVariable("LINE_USER_ID", $userId, "User")

$env:LINE_CHANNEL_ACCESS_TOKEN = $token
$env:LINE_USER_ID = $userId

Write-Host ""
Write-Host "Saved:"
Write-Host "LINE_CHANNEL_ACCESS_TOKEN length=$($token.Length)"
Write-Host "LINE_USER_ID length=$($userId.Length)"

if ($TestAfterSave) {
    Write-Host ""
    Write-Host "Running dry-run preview..."
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\send_current_strongest_line_alert.ps1
}
