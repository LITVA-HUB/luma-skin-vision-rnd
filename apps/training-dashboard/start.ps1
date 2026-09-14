param([int]$Port = 8766)
$ErrorActionPreference = 'Stop'
$dashboardDirectory = $PSScriptRoot
$dashboardRuntime = Join-Path $dashboardDirectory '.runtime'
$dashboardRepository = Split-Path -Parent (Split-Path -Parent $dashboardDirectory)
$dashboardWorkspace = Split-Path -Parent (Split-Path -Parent $dashboardRepository)
$dashboardPython = Join-Path $dashboardWorkspace 'luma-skin-vision-rnd\.venv\Scripts\python.exe'
$dashboardUrl = "http://127.0.0.1:$Port"
$dashboardHealth = $null
try {
    $dashboardHealth = Invoke-RestMethod -Uri "$dashboardUrl/api/health" -TimeoutSec 2
} catch {
    # Start only this monitor; a bind error is detected by the health check below.
}
if ($dashboardHealth) {
    if ($dashboardHealth.app -eq 'luma-training-monitor') {
        Write-Host "Luma monitor is running: $dashboardUrl"
        exit 0
    }
    throw "Port $Port belongs to another application."
}
if (-not (Test-Path -LiteralPath (Join-Path $dashboardDirectory 'dist/index.html'))) {
    Push-Location -LiteralPath $dashboardDirectory
    try {
        & npm.cmd ci --ignore-scripts --no-audit --no-fund
        if ($LASTEXITCODE -ne 0) { throw 'Dashboard dependencies failed to install.' }
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw 'Dashboard build failed.' }
    } finally { Pop-Location }
}
New-Item -ItemType Directory -Path $dashboardRuntime -Force | Out-Null
$dashboardWorker = Start-Process -FilePath $dashboardPython -ArgumentList @('-u', (Join-Path $dashboardDirectory 'server.py'), '--port', "$Port") -WorkingDirectory $dashboardDirectory -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $dashboardRuntime 'server.log') -RedirectStandardError (Join-Path $dashboardRuntime 'server-errors.log')
$dashboardReady = $false
for ($dashboardAttempt = 0; $dashboardAttempt -lt 15; $dashboardAttempt++) {
    Start-Sleep -Milliseconds 200
    try {
        $dashboardHealth = Invoke-RestMethod -Uri "$dashboardUrl/api/health" -TimeoutSec 2
        if ($dashboardHealth.app -eq 'luma-training-monitor') { $dashboardReady = $true; break }
    } catch {}
    if ($dashboardWorker.HasExited) { break }
}
if (-not $dashboardReady) { throw "The monitor did not start. See $dashboardRuntime\server-errors.log" }
@{ pid = $dashboardWorker.Id; url = $dashboardUrl; started = (Get-Date -Format o) } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $dashboardRuntime 'server.json') -Encoding UTF8
Write-Host "Luma monitor started: $dashboardUrl"
