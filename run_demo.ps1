# =========================================================================
#  Goa Voice-RAG - one-command demo launcher (Windows PowerShell)
#  Starts the app server + a free Cloudflare tunnel and prints your public URL.
#
#  Usage:  right-click -> "Run with PowerShell",  OR in a terminal:
#      cd "C:\My Data\Personal Data\Others\my work\personal projects\Goa Hackathon"
#      powershell -ExecutionPolicy Bypass -File run_demo.ps1
#
#  Stop everything: close this window, or press Ctrl+C.
#  (ASCII-only on purpose - PowerShell 5.1 reads .ps1 as ANSI.)
# =========================================================================

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

Write-Host ""
Write-Host "[1/3] Starting the RAG server on http://localhost:7860 ..." -ForegroundColor Cyan
$server = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "app.py" `
    -PassThru -WindowStyle Hidden -RedirectStandardOutput "server.log" -RedirectStandardError "server.err.log"

# wait for the server to answer /api/health
$ready = $false
$vectors = 0
for ($i = 0; $i -lt 40; $i++) {
    try {
        $h = Invoke-RestMethod -Uri "http://localhost:7860/api/health" -TimeoutSec 3
        if ($h.status -eq "ok") { $ready = $true; $vectors = $h.indexed_vectors; break }
    } catch { Start-Sleep -Seconds 1 }
}
if (-not $ready) {
    Write-Host "Server did not start. Check server.log / server.err.log" -ForegroundColor Red
    exit 1
}
Write-Host ("      Server is up (" + $vectors + " vectors indexed).") -ForegroundColor Green

Write-Host "[2/3] Opening the public Cloudflare tunnel ..." -ForegroundColor Cyan
if (-not (Test-Path ".\tools\cloudflared.exe")) {
    Write-Host "      downloading cloudflared ..."
    New-Item -ItemType Directory -Force -Path ".\tools" | Out-Null
    Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile ".\tools\cloudflared.exe"
}
if (Test-Path ".\tunnel.log") { Remove-Item ".\tunnel.log" -Force -ErrorAction SilentlyContinue }
if (Test-Path ".\tunnel.err.log") { Remove-Item ".\tunnel.err.log" -Force -ErrorAction SilentlyContinue }
$tunnel = Start-Process -FilePath ".\tools\cloudflared.exe" -ArgumentList "tunnel","--url","http://localhost:7860" `
    -PassThru -WindowStyle Hidden -RedirectStandardOutput "tunnel.log" -RedirectStandardError "tunnel.err.log"

# grab the public URL (cloudflared logs it to stderr)
$publicUrl = $null
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 2
    foreach ($f in @(".\tunnel.err.log", ".\tunnel.log")) {
        if (Test-Path $f) {
            $m = Select-String -Path $f -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($m) { $publicUrl = $m.Matches[0].Value; break }
        }
    }
    if ($publicUrl) { break }
}

Write-Host ""
Write-Host "[3/3] READY" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Yellow
if ($publicUrl) {
    Write-Host ("  PUBLIC LIVE LINK:  " + $publicUrl) -ForegroundColor Green
    Write-Host "  (share this / put it in the submission form)"
} else {
    Write-Host "  Tunnel URL not detected yet - check tunnel.err.log for the trycloudflare.com link." -ForegroundColor Yellow
}
Write-Host "  Local link:        http://localhost:7860"
Write-Host "=====================================================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "Keep this window OPEN during your demo / judging."
Write-Host "Close it (or Ctrl+C) to stop the server and tunnel."
Write-Host ""

# keep running until the user closes the window, then clean up
try {
    Wait-Process -Id $tunnel.Id
} finally {
    if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue }
    if ($tunnel -and -not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id -Force -ErrorAction SilentlyContinue }
}
