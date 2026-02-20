# PowerShell Quickstart (bREadbeats)

This is a copy/paste guide for building and running the EXE.

## Fastest option (one command)

From the project root, run:

```powershell
.\scripts\build_release.ps1
```

Optional (skip smoke test):

```powershell
.\scripts\build_release.ps1 -SkipSmokeTest
```

## 1) Go to the project root directory

```powershell
cd "C:\Users\andre\Documents\vscodeworkspace\bREadbeats-master"
```

If you want to confirm where you are:

```powershell
Get-Location
```

Expected path:

`C:\Users\andre\Documents\vscodeworkspace\bREadbeats-master`

## 2) Build the EXE (clean build)

```powershell
.\.venv\Scripts\pyinstaller.exe --clean --noconfirm bREadbeats.spec
```

## 3) Run the built EXE

```powershell
& .\dist\bREadbeats.exe
```

## 4) Optional: run EXE with debug output

```powershell
$env:BREADBEATS_DEBUG_STDIO='1'; & .\dist\bREadbeats.exe
```

## 5) Optional: run from VS Code Task instead

In VS Code:

- Terminal -> Run Task...
- Select: Build breadbeats.exe with PyInstaller

## Troubleshooting quick checks

Check EXE exists:

```powershell
Get-Item .\dist\bREadbeats.exe
```

If command says file not found, run Step 2 again from the root directory.
