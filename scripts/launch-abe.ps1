#Requires -Version 5.1
<#
  launch-abe.ps1 - one-click launcher for always-best-estimates.

  What it does:
    1. Preflight: warn (do not fail) if the frontend deps or the price DB are missing.
    2. Starts the backend (FastAPI/uvicorn, single worker) on 127.0.0.1:8140 in its
       own window, unless something is already listening there.
    3. Starts the Vite dev server (frontend) on 127.0.0.1:5174 in its own window,
       unless something is already listening there. Vite proxies /api and /health
       to the backend, so the app is served from :5174.
    4. Waits for the frontend to answer, then opens the app in Google Chrome
       (falls back to the default browser if Chrome is not installed).

  The backend and frontend keep running in their own windows. Close those two
  windows to stop the servers; closing THIS window does not stop them.

  Advisory display only, 127.0.0.1 only. No trading, no auth, no LAN bind.
#>
[CmdletBinding()]
param(
    [switch] $SkipBrowser
)

$ErrorActionPreference = 'Stop'

# Self-locating: this script lives at <project>/scripts/, so the project root is
# its parent dir. Set-Location so the relative paths below resolve regardless of
# the caller's working directory (the dev-observatory button spawns us detached).
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$BackendPort  = 8140
$FrontendPort = 5174
$AppUrl       = "http://127.0.0.1:$FrontendPort"

function Test-PortListening {
    param([int] $Port)
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return [bool] $conn
}

function Resolve-ChromePath {
    # App Paths registry is the authoritative install location for chrome.exe.
    foreach ($hive in 'HKLM:', 'HKCU:') {
        $key = "$hive\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe"
        try {
            $val = (Get-ItemProperty -Path $key -ErrorAction Stop).'(default)'
            if ($val -and (Test-Path $val)) { return $val }
        } catch { }
    }
    # Common install locations as a fallback.
    $candidates = @(
        (Join-Path $env:ProgramFiles 'Google\Chrome\Application\chrome.exe'),
        (Join-Path ${env:ProgramFiles(x86)} 'Google\Chrome\Application\chrome.exe'),
        (Join-Path $env:LOCALAPPDATA 'Google\Chrome\Application\chrome.exe')
    )
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    return $null
}

function Open-Webpage {
    param([string] $Url)
    $chrome = Resolve-ChromePath
    if ($chrome) {
        Start-Process -FilePath $chrome -ArgumentList '--new-window', $Url | Out-Null
        Write-Host "Opened in Chrome: $Url" -ForegroundColor Green
    } else {
        Write-Host "Chrome not found - opening in your default browser instead." -ForegroundColor Yellow
        Start-Process $Url | Out-Null
    }
}

Write-Host ""
Write-Host "=== always-best-estimates launcher ===" -ForegroundColor Cyan
Write-Host "Project: $root"
Write-Host ""

# --- 1. Preflight (warn only) -----------------------------------------
if (-not (Test-Path (Join-Path $root 'frontend\node_modules'))) {
    Write-Host "WARN: frontend\node_modules missing - the Vite window will fail." -ForegroundColor Yellow
    Write-Host "      Fix: npm install --prefix frontend" -ForegroundColor Yellow
}
if (-not (Test-Path (Join-Path $root 'data\abe.db'))) {
    Write-Host "WARN: data\abe.db missing - backend starts but has no history." -ForegroundColor Yellow
    Write-Host "      Fix: uv run python -m abe.ingest.prices --backfill" -ForegroundColor Yellow
}

# --- 2. Backend (uvicorn, single worker) in its own window ------------
if (Test-PortListening -Port $BackendPort) {
    Write-Host "Backend already listening on :$BackendPort - not starting a second one." -ForegroundColor Green
} else {
    Write-Host "Starting backend on 127.0.0.1:$BackendPort ..." -ForegroundColor Cyan
    $backendCmd = "Set-Location '$root'; uv run uvicorn abe.api:app --host 127.0.0.1 --port $BackendPort"
    Start-Process powershell -ArgumentList '-NoExit', '-Command', $backendCmd | Out-Null
}

# --- 3. Frontend (Vite dev server) in its own window -----------------
if (Test-PortListening -Port $FrontendPort) {
    Write-Host "Frontend already listening on :$FrontendPort - not starting a second one." -ForegroundColor Green
} else {
    Write-Host "Starting frontend on 127.0.0.1:$FrontendPort ..." -ForegroundColor Cyan
    $frontendCmd = "Set-Location '$root\frontend'; npm run dev"
    Start-Process powershell -ArgumentList '-NoExit', '-Command', $frontendCmd | Out-Null
}

# --- 4. Wait for the frontend, then open Chrome ----------------------
if ($SkipBrowser) {
    Write-Host "Skipping browser launch (-SkipBrowser). App will be at $AppUrl" -ForegroundColor DarkGray
} else {
    Write-Host "Waiting for the frontend at $AppUrl ..." -ForegroundColor Cyan
    $deadline = (Get-Date).AddSeconds(60)
    $up = $false
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-WebRequest -Uri "$AppUrl/" -UseBasicParsing -TimeoutSec 2 | Out-Null
            $up = $true; break
        } catch {
            Start-Sleep -Milliseconds 800
        }
    }
    if ($up) {
        Open-Webpage -Url $AppUrl
    } else {
        Write-Host "Frontend did not answer within 60s - check its window, then open $AppUrl manually." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  App:      $AppUrl        (Vite dev server)" -ForegroundColor Green
Write-Host "  Backend:  http://127.0.0.1:$BackendPort   (FastAPI, proxied via /api)" -ForegroundColor Green
Write-Host "  Servers run in their own windows - close those to stop them." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to close this launcher window (the servers keep running)"
