# PathSnap demo - LAPTOP 1 (Model + API).  Launched by LAPTOP1_START_DEMO.bat
# Steps: Python + packages -> firewall rule -> trip folder -> pre-flight check -> show IP -> start API.
param(
    [string]$Partition = "evaluator_test",
    [switch]$SkipCheck,
    [switch]$NoFirewall
)

$ErrorActionPreference = "Continue"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

function Step($n, $text) { Write-Host ""; Write-Host "[$n] $text" -ForegroundColor Cyan }
function Ok($text)   { Write-Host "    OK   $text" -ForegroundColor Green }
function Warn($text) { Write-Host "    WARN $text" -ForegroundColor Yellow }
function Fail($text) { Write-Host "    FAIL $text" -ForegroundColor Red }

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  PathSnap demo - LAPTOP 1 (Model + API)   partition: $Partition" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

# ---------------------------------------------------------------- 1. Python + packages
Step 1 "Python and packages"
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { Fail "Python not found on PATH. Install Python 3.10+ and re-run."; exit 1 }
Ok ("Python: " + (& python --version 2>&1))
& python -c "import fastapi, uvicorn, sklearn, scipy, pandas, pyarrow, pyproj, joblib, numpy" 2>$null
if ($LASTEXITCODE -ne 0) {
    Warn "Some packages are missing - installing requirements.txt (one time) ..."
    & python -m pip install -q -r requirements.txt
    & python -c "import fastapi, uvicorn, sklearn, scipy, pandas, pyarrow, pyproj, joblib, numpy" 2>$null
    if ($LASTEXITCODE -ne 0) { Fail "Package install failed. Run: pip install -r requirements.txt"; exit 1 }
}
Ok "all required packages import"
if (-not (Test-Path "models\saved\rf_compact.joblib")) { Fail "models\saved\rf_compact.joblib missing - run 'git pull'."; exit 1 }
Ok "model file present (models\saved\rf_compact.joblib)"

# ---------------------------------------------------------------- 2. Firewall (port 8000 inbound)
Step 2 "Firewall rule for port 8000"
if ($NoFirewall) { Warn "skipped (-NoFirewall)" }
else {
    $rule = Get-NetFirewallRule -DisplayName "PathSnap API 8000" -ErrorAction SilentlyContinue
    if ($rule) { Ok "rule 'PathSnap API 8000' already exists" }
    else {
        Warn "rule missing - asking Windows for admin rights once (click Yes) ..."
        try {
            Start-Process powershell -Verb RunAs -Wait -ArgumentList '-NoProfile -Command "New-NetFirewallRule -DisplayName ''PathSnap API 8000'' -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Any | Out-Null"'
            if (Get-NetFirewallRule -DisplayName "PathSnap API 8000" -ErrorAction SilentlyContinue) { Ok "firewall rule added" }
            else { Warn "rule not added. If Windows shows an 'Allow access' pop-up for Python, tick Private AND Public." }
        } catch { Warn "admin prompt declined. If Windows shows an 'Allow access' pop-up for Python, tick Private AND Public." }
    }
}

# ---------------------------------------------------------------- 3. Trip folder
Step 3 "Trip folder"
$dataDir = (& python -c "from data_processing.load_data import DATA_DIR; print(DATA_DIR)" 2>$null | Select-Object -Last 1)
$tripDir = Join-Path $dataDir $Partition
if (-not (Test-Path $tripDir)) {
    # the evaluator folder as handed out: <Hackathon>\evaluator_test\evaluator_test
    $handed = Join-Path (Split-Path -Parent $Repo) "$Partition\$Partition"
    if (Test-Path $handed) {
        Warn "copying $handed -> $tripDir"
        Copy-Item -Recurse $handed $tripDir
    }
}
if (-not (Test-Path $tripDir)) {
    Fail "trip folder not found: $tripDir"
    Write-Host "    Put the trip folders (<id>\<id>.track/.nodes/.arcs/.route) there, or set PATHSNAP_DATA." -ForegroundColor Yellow
    exit 1
}
$nTrips = (Get-ChildItem $tripDir -Directory).Count
Ok "$tripDir  ($nTrips trips)"

# ---------------------------------------------------------------- 4. Pre-flight check
Step 4 "Pre-flight check (every trip through the full pipeline once)"
if ($SkipCheck) { Warn "skipped (-SkipCheck)" }
else {
    & python -W ignore scripts\check_trips.py $Partition
    if ($LASTEXITCODE -ne 0) {
        Fail "pre-flight reported problems (see above)."
        $ans = Read-Host "    Start the API anyway? (y/N)"
        if ($ans -notmatch '^[yY]') { exit 1 }
    }
}

# ---------------------------------------------------------------- 5. Port + IP
Step 5 "Network"
$busy = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($busy) { Warn "port 8000 is already in use (PID $($busy[0].OwningProcess)) - close the other API window first." }
# real adapters only (hide VMware / VirtualBox / Hyper-V / loopback); mark the one with a gateway
$ips = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" -and
                   $_.InterfaceAlias -notmatch 'VMware|VirtualBox|vEthernet|Loopback|Bluetooth|WSL' }
$gwIf = @(Get-NetIPConfiguration -ErrorAction SilentlyContinue |
    Where-Object { $_.IPv4DefaultGateway } | ForEach-Object { $_.InterfaceAlias })
Write-Host ""
Write-Host "    ***********************************************************" -ForegroundColor Green
Write-Host "    *  On LAPTOP 2 run LAPTOP2_START_DEMO.bat and enter:      *" -ForegroundColor Green
foreach ($ip in $ips) {
    $tag = ""
    if ($gwIf -contains $ip.InterfaceAlias) { $tag = "  <-- USE THIS (has internet / hotspot)" }
    $line = ("    *     {0,-16} ({1}){2}" -f $ip.IPAddress, $ip.InterfaceAlias, $tag)
    if ($tag) { Write-Host $line -ForegroundColor Yellow } else { Write-Host $line -ForegroundColor White }
}
Write-Host "    *  Both laptops must be on the same hotspot / Wi-Fi.     *" -ForegroundColor Green
Write-Host "    *  If you started Windows Mobile Hotspot on this laptop,  *" -ForegroundColor Green
Write-Host "    *  Laptop 2 should use the 192.168.137.1 address.        *" -ForegroundColor Green
Write-Host "    ***********************************************************" -ForegroundColor Green

# ---------------------------------------------------------------- 6. Start the API
Step 6 "Starting the API on port 8000  (leave this window open; Ctrl+C to stop)"
$logDir = Join-Path $Repo "results\demo_logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$env:PATHSNAP_LOG_FILE = Join-Path $logDir ("api_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
$env:PATHSNAP_TRIPS_PARTITION = $Partition
Write-Host "    API log also saved to $($env:PATHSNAP_LOG_FILE)" -ForegroundColor DarkGray
Write-Host "    Wait for 'API ready' (about 15 s), then start Laptop 2." -ForegroundColor DarkGray
Write-Host ""
& python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
