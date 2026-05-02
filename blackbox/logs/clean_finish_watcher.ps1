$ErrorActionPreference = 'Continue'
$mainPid = 50216
$root = 'c:\Users\potte\Desktop\research\emotional LLM'
$logPath = Join-Path $root 'blackbox\logs\clean_finish_watcher.log'
"[$(Get-Date -Format s)] finisher waiting for pid=$mainPid" | Add-Content -Path $logPath -Encoding UTF8
while (Get-Process -Id $mainPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 20
}
"[$(Get-Date -Format s)] main process exited; running analyze and report" | Add-Content -Path $logPath -Encoding UTF8
Set-Location $root
& 'D:\code\anaconda\python.exe' '.\blackbox\analyze.py' *> (Join-Path $root 'blackbox\logs\post_clean_analyze.log')
"[$(Get-Date -Format s)] analyze done" | Add-Content -Path $logPath -Encoding UTF8
& 'D:\code\anaconda\python.exe' '.\blackbox\generate_report.py' *> (Join-Path $root 'blackbox\logs\post_clean_report.log')
"[$(Get-Date -Format s)] report done" | Add-Content -Path $logPath -Encoding UTF8
