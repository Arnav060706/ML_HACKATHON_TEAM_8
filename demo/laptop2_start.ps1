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
if ($NoBrowser) { Write-Host "    skipped (-NoBrowser): open $ui" -ForegroundColor Yellow }
else { Start-Process $ui; Write-Host "    OK   the UI connects automatically and lists the trips" -ForegroundColor Green }

Write-Host ""
Write-Host "Demo: pick a trip -> Reconstruct Route -> show metrics -> show Laptop 1's console." -ForegroundColor White
Write-Host "Untick 'Enable HMM / Viterbi' to compare with the classifier alone." -ForegroundColor White
