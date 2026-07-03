param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteUrl,
    [string]$RemoteName = "origin",
    [string]$Branch = "main",
    [string]$GitExe = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $ProjectRoot

if ([string]::IsNullOrWhiteSpace($GitExe)) {
    $LocalGit = Join-Path $ProjectRoot "tools\git\mingit\cmd\git.exe"
    if (Test-Path -LiteralPath $LocalGit) {
        $GitExe = $LocalGit
    } else {
        $GitExe = "git"
    }
}

if ($RemoteUrl -notmatch "^https://github\.com/.+/.+\.git$" -and $RemoteUrl -notmatch "^git@github\.com:.+/.+\.git$") {
    throw "RemoteUrl should look like https://github.com/<owner>/<repo>.git or git@github.com:<owner>/<repo>.git"
}

$Status = & $GitExe status --porcelain
if (-not [string]::IsNullOrWhiteSpace(($Status -join ""))) {
    throw "Working tree is not clean. Commit or stash local changes before connecting remote."
}

$CurrentBranch = (& $GitExe branch --show-current).Trim()
if ($CurrentBranch -ne $Branch) {
    & $GitExe branch -M $Branch
}

$Existing = & $GitExe remote
if ($Existing -contains $RemoteName) {
    & $GitExe remote set-url $RemoteName $RemoteUrl
} else {
    & $GitExe remote add $RemoteName $RemoteUrl
}

& $GitExe remote -v
& $GitExe push -u $RemoteName $Branch
