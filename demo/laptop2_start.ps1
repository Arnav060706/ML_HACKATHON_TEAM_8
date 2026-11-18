# PathSnap demo - LAPTOP 2 (Evaluator UI).  Launched by LAPTOP2_START_DEMO.bat
# Steps: ask for Laptop 1's address -> test /health -> write ui/config.js -> open the UI in the browser.
param(
    [string]$Api = "",
    [switch]$NoBrowser
)

$Repo = Split-Path -Parent $PSScriptRoot
$cfg = Join-Path $Repo "ui\config.js"
$ui  = Join-Path $Repo "ui\index.html"

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  PathSnap demo - LAPTOP 2 (Evaluator UI)" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

function Normalize($a) {
    $a = $a.Trim()
    if ($a -notmatch '^https?://') { $a = "http://" + $a }
    if ($a -notmatch ':\d+/?$') { $a = $a.TrimEnd('/') + ":8000" }
    return $a.TrimEnd('/')
}

# previous address (if this script ran before)
$prev = ""
if (Test-Path $cfg) {
    $m = Select-String -Path $cfg -Pattern 'PATHSNAP_API\s*=\s*"([^"]+)"' | Select-Object -First 1
    if ($m) { $prev = $m.Matches[0].Groups[1].Value }
}

$interactive = ($Api -eq "")
while ($true) {
    if ($interactive) {
        $hint = "Laptop 1 IP shown in its window, e.g. 192.168.43.152"
        if ($prev) { $hint = "$hint [Enter = $prev]" }
        $in = Read-Host "`n[1] Laptop 1 address ($hint)"
        if (-not $in) { $in = $prev }
        if (-not $in) { continue }
        $Api = Normalize $in
    } else { $Api = Normalize $Api }

    Write-Host "[2] Testing $Api/health ..." -ForegroundColor Cyan
    try {
        $h = Invoke-RestMethod -Uri "$Api/health" -TimeoutSec 6
        Write-Host ("    OK   status={0}  model_loaded={1}  partition={2}  trips={3}" -f $h.status, $h.model_loaded, $h.partition, $h.n_trips) -ForegroundColor Green
        Write-Host ("         model={0}  beta={1} m  gap={2} s" -f $h.model_path, $h.decoder.beta, $h.decoder.gap_s) -ForegroundColor Green
        if (-not $h.model_loaded) { Write-Host "    WARN API still starting - wait for 'API ready' on Laptop 1." -ForegroundColor Yellow }
        break
    } catch {
        Write-Host "    FAIL cannot reach $Api" -ForegroundColor Red
        Write-Host "         - Is Laptop 1 showing 'API ready'?" -ForegroundColor Yellow
        Write-Host "         - Same hotspot / Wi-Fi on both laptops?" -ForegroundColor Yellow
        Write-Host "         - Firewall rule on Laptop 1 (its launcher adds it), right IP (Wi-Fi adapter)?" -ForegroundColor Yellow
        Write-Host "         - Phone isolates clients? Use Windows Mobile Hotspot on Laptop 1 (IP usually 192.168.137.1)." -ForegroundColor Yellow
        if (-not $interactive) { exit 1 }
        $prev = $Api
    }
}

Write-Host "[3] Saving address to ui\config.js" -ForegroundColor Cyan
Set-Content -Path $cfg -Encoding ASCII -Value ("// written by LAPTOP2_START_DEMO.bat`nwindow.PATHSNAP_API = `"$Api`";")
Write-Host "    OK" -ForegroundColor Green

Write-Host "[4] Opening the UI" -ForegroundColor Cyan
# Serve the UI over http from THIS laptop (127.0.0.1:8080). OpenStreetMap blocks map tiles for pages
# opened as local files (no Referer), so http is needed for the OSM map. Without Python the file is
# opened directly and the UI uses the Esri street map instead.
$url = $null
$py = Get-Command python -ErrorAction SilentlyContinue
if ($py) {
    $up = $false
    try { Invoke-WebRequest -Uri "http://127.0.0.1:8080/ui/index.html" -UseBasicParsing -TimeoutSec 2 | Out-Null; $up = $true } catch {}
    if (-not $up) {
        Start-Process -FilePath $py.Source -ArgumentList "-m", "http.server", "8080", "--bind", "127.0.0.1" `
            -WorkingDirectory $Repo -WindowStyle Minimized
        for ($k = 0; $k -lt 20 -and -not $up; $k++) {
            Start-Sleep -Milliseconds 300
            try { Invoke-WebRequest -Uri "http://127.0.0.1:8080/ui/index.html" -UseBasicParsing -TimeoutSec 2 | Out-Null; $up = $true } catch {}
        }
    }
    if ($up) { $url = "http://127.0.0.1:8080/ui/index.html"; Write-Host "    OK   UI served locally at $url (minimised 'python' window - leave it open)" -ForegroundColor Green }
}
if (-not $url) { $url = $ui; Write-Host "    WARN no local web server - opening the file directly (Esri map instead of OpenStreetMap)" -ForegroundColor Yellow }
if ($NoBrowser) { Write-Host "    skipped (-NoBrowser): open $url" -ForegroundColor Yellow }
else { Start-Process $url; Write-Host "    OK   the UI connects automatically and lists the trips" -ForegroundColor Green }

Write-Host ""
Write-Host "Demo: pick a trip -> Reconstruct Route -> show metrics -> show Laptop 1's console." -ForegroundColor White
Write-Host "Untick 'Enable HMM / Viterbi' to compare with the classifier alone." -ForegroundColor White
