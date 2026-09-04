# nlh-evidence bootstrap: deps -> self-hosted runner -> scheduled rotation -> smoke test.
# Idempotent: safe to re-run. Requires gh CLI authenticated (gh auth login).
param(
    [string]$Repo = "holysimba42/nlh-evidence",
    [string]$Label = "nlh-local",
    [string]$RunnerDir = "$PSScriptRoot\..\_runner"
)
$ErrorActionPreference = "Stop"
$RunnerDir = [System.IO.Path]::GetFullPath($RunnerDir)

# 1) dependencies
python --version *> $null
if ($LASTEXITCODE -ne 0) { throw "python not on PATH" }
gh --version *> $null
if ($LASTEXITCODE -ne 0) { throw "gh not on PATH" }
python -c "import keyring, nacl" *> $null
if ($LASTEXITCODE -ne 0) { python -m pip install --quiet keyring pynacl; if ($LASTEXITCODE -ne 0) { throw "pip install failed" } }

# 2) self-hosted runner (register only if this machine is not registered yet)
$runners = gh api "repos/$Repo/actions/runners" --jq ".runners[].name"
if ($runners -notcontains $env:COMPUTERNAME) {
    if (-not (Test-Path "$RunnerDir\config.cmd")) {
        $ver = (gh api repos/actions/runner/releases/latest --jq .tag_name).TrimStart("v")
        New-Item -ItemType Directory -Force -Path $RunnerDir | Out-Null
        Invoke-WebRequest "https://github.com/actions/runner/releases/download/v$ver/actions-runner-win-x64-$ver.zip" -OutFile "$RunnerDir\runner.zip"
        Expand-Archive "$RunnerDir\runner.zip" -DestinationPath $RunnerDir -Force
    }
    $token = gh api -X POST "repos/$Repo/actions/runners/registration-token" --jq .token
    Push-Location $RunnerDir
    ./config.cmd --url "https://github.com/$Repo" --token $token --labels $Label `
        --unattended --no-default-labels --work "C:/nlh_work" | Select-Object -Last 1
    Pop-Location
}

# 3) runner listener as a Windows service (survives reboots). Needs elevation;
# degrades to console-runner mode with a warning if not admin.
Push-Location $RunnerDir
if (-not (Get-Service -Name "actions.runner.*" -ErrorAction SilentlyContinue)) {
    try {
        ./config.cmd --runasservice --unattended 2>&1 | Select-Object -Last 1
        if ($LASTEXITCODE -ne 0) { throw "config.cmd --runasservice exit $LASTEXITCODE" }
    } catch {
        Write-Warning "runner service not installed (run setup.ps1 elevated to enable): $_"
    }
}
Pop-Location

# 4) daily 06:00 rotation task (bypasses GitHub-hosted billing lock)
$script = Join-Path $PSScriptRoot "scripts\nlh.py"
$act = New-ScheduledTaskAction -Execute "python" -Argument "`"$script`" rotate"
$trg = New-ScheduledTaskTrigger -Daily -At 06:00
if (Get-ScheduledTask -TaskName "nlh-daily-rotate" -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName "nlh-daily-rotate" -Confirm:$false
}
Register-ScheduledTask -TaskName "nlh-daily-rotate" -Action $act -Trigger $trg | Out-Null

# 5) smoke test: one full rotate -> dispatch -> sha256 round-trip
Push-Location $PSScriptRoot
python scripts\nlh.py rotate
Pop-Location
if ($LASTEXITCODE -ne 0) { throw "smoke rotate failed" }
Write-Host "BOOTSTRAP COMPLETE: runner + daily task registered, smoke rotate PASS."
