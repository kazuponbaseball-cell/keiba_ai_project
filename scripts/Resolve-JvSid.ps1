function Resolve-JvSid {
    param(
        [string]$Sid = ""
    )

    if (-not [string]::IsNullOrWhiteSpace($Sid) -and $Sid -ne "UNKNOWN") {
        return [pscustomobject]@{
            Sid = $Sid
            Source = "parameter"
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($env:JV_SID) -and $env:JV_SID -ne "UNKNOWN") {
        return [pscustomobject]@{
            Sid = $env:JV_SID
            Source = "env:JV_SID"
        }
    }

    $registryPaths = @(
        "HKCU:\SOFTWARE\WOW6432Node\JRA-VAN Data Lab.\uid_pass",
        "HKCU:\SOFTWARE\JRA-VAN Data Lab.\uid_pass",
        "HKLM:\SOFTWARE\WOW6432Node\JRA-VAN Data Lab.\uid_pass",
        "HKLM:\SOFTWARE\JRA-VAN Data Lab.\uid_pass"
    )

    foreach ($path in $registryPaths) {
        if (-not (Test-Path $path)) {
            continue
        }
        try {
            $serviceKey = (Get-ItemProperty -Path $path -ErrorAction Stop).servicekey
            if (-not [string]::IsNullOrWhiteSpace($serviceKey)) {
                return [pscustomobject]@{
                    Sid = $serviceKey
                    Source = "registry:$path/servicekey"
                }
            }
        } catch {
            continue
        }
    }

    return [pscustomobject]@{
        Sid = ""
        Source = "missing"
    }
}

function Get-MaskedJvSid {
    param(
        [string]$Sid = ""
    )

    if ([string]::IsNullOrWhiteSpace($Sid)) {
        return ""
    }
    if ($Sid.Length -le 4) {
        return "****"
    }
    return ("****" + $Sid.Substring($Sid.Length - 4))
}
