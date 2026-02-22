# PowerShell script to test version handling locally
# Run this to see how version detection works

Write-Host "bREadbeats Version Test" -ForegroundColor Cyan
Write-Host "=" * 40 -ForegroundColor Cyan

# Activate virtual environment if it exists
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & .venv\Scripts\Activate.ps1
}

Write-Host "`nCurrent Git Status:" -ForegroundColor Green
git status --porcelain

Write-Host "`nGit Tags:" -ForegroundColor Green  
git tag --list --sort=-version:refname | Select-Object -First 5

Write-Host "`nGit Describe:" -ForegroundColor Green
try {
    git describe --tags --always --dirty
} catch {
    Write-Host "No tags found or git describe failed" -ForegroundColor Red
}

Write-Host "`nVersion Detection:" -ForegroundColor Green
python version.py

Write-Host "`nTesting About Dialog Version:" -ForegroundColor Green
python -c "from version import __version__; print(f'About dialog will show: bREadbeats {__version__}')"

Write-Host "`nVersion handling test complete!" -ForegroundColor Cyan