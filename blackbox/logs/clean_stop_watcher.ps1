$ErrorActionPreference = 'Stop'
$mainPid = 50216
$goal = 14376
$root = 'c:\Users\potte\Desktop\research\emotional LLM'
$cleanDir = Join-Path $root 'blackbox\results\clean'
$logPath = Join-Path $root 'blackbox\logs\clean_stop_watcher.log'
function Get-CleanCount {
    if (-not (Test-Path $cleanDir)) { return 0 }
    return (Get-ChildItem $cleanDir -Recurse -Filter *.json -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne 'summary.json' }).Count
}
"[$(Get-Date -Format s)] watcher started pid=$mainPid goal=$goal" | Add-Content -Path $logPath -Encoding UTF8
while ($true) {
    $p = Get-Process -Id $mainPid -ErrorAction SilentlyContinue
    if (-not $p) {
        "[$(Get-Date -Format s)] main process already exited" | Add-Content -Path $logPath -Encoding UTF8
        break
    }
    $count = Get-CleanCount
    "[$(Get-Date -Format s)] clean_count=$count" | Add-Content -Path $logPath -Encoding UTF8
    if ($count -ge $goal) {
        Stop-Process -Id $mainPid -Force
        "[$(Get-Date -Format s)] stopped main process at clean_count=$count" | Add-Content -Path $logPath -Encoding UTF8
        break
    }
    Start-Sleep -Seconds 30
}
