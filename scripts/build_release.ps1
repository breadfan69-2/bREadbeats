param(
    [switch]$SkipSmokeTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Step {
    param([string]$Message)
    Write-Host "[build_release] $Message"
}

function Stop-BreadbeatsProcesses {
    $procs = Get-Process -Name 'bREadbeats' -ErrorAction SilentlyContinue
    if ($procs) {
        Write-Step "Stopping running bREadbeats process(es): $($procs.Count)"
        $procs | Stop-Process -Force
        Start-Sleep -Milliseconds 400
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$pyInstallerExe = Join-Path $repoRoot '.venv\Scripts\pyinstaller.exe'
$specFile = Join-Path $repoRoot 'bREadbeats.spec'
$distDir = Join-Path $repoRoot 'dist'
$buildDir = Join-Path $repoRoot 'build'
$exePath = Join-Path $distDir 'bREadbeats.exe'

if (-not (Test-Path $pyInstallerExe)) {
    throw "PyInstaller not found at: $pyInstallerExe"
}

if (-not (Test-Path $specFile)) {
    throw "Spec file not found at: $specFile"
}

Push-Location $repoRoot
try {
    Write-Step "Repo root: $repoRoot"

    if (Test-Path $buildDir) {
        Write-Step 'Removing build directory...'
        Remove-Item $buildDir -Recurse -Force
    }

    if (Test-Path $distDir) {
        Stop-BreadbeatsProcesses
        Write-Step 'Removing dist directory...'
        Remove-Item $distDir -Recurse -Force
    }

    Write-Step 'Running PyInstaller...'
    & $pyInstallerExe --clean --noconfirm $specFile
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }

    if (-not (Test-Path $exePath)) {
        throw "Build finished but EXE not found: $exePath"
    }

    $exeInfo = Get-Item $exePath
    Write-Step "Build OK: $($exeInfo.FullName)"
    Write-Step "EXE size: $([math]::Round($exeInfo.Length / 1MB, 2)) MB"

    if ($SkipSmokeTest) {
        Write-Step 'Skipping smoke test (--SkipSmokeTest provided).'
        exit 0
    }

    Write-Step 'Running smoke test (launch for ~8s)...'
    $oldDebug = $env:BREADBEATS_DEBUG_STDIO
    $env:BREADBEATS_DEBUG_STDIO = '1'
    try {
        $proc = Start-Process -FilePath $exePath -PassThru
        Start-Sleep -Seconds 8

        if ($proc.HasExited) {
            if ($proc.ExitCode -ne 0) {
                throw "Smoke test failed: EXE exited with code $($proc.ExitCode)"
            }
            Write-Step 'Smoke test OK: EXE launched and exited cleanly.'
        }
        else {
            Stop-Process -Id $proc.Id -Force
            Write-Step "Smoke test OK: EXE launched (stopped test process $($proc.Id))."
        }
    }
    finally {
        if ($null -eq $oldDebug) {
            Remove-Item Env:BREADBEATS_DEBUG_STDIO -ErrorAction SilentlyContinue
        }
        else {
            $env:BREADBEATS_DEBUG_STDIO = $oldDebug
        }
    }

    Write-Step 'Done.'
}
finally {
    Pop-Location
}
