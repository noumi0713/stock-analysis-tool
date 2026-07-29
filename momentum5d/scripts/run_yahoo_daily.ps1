$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$logDirectory = Join-Path $projectRoot "logs"
$logPath = Join-Path $logDirectory "yahoo-daily.log"

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
Set-Location -LiteralPath $projectRoot

# yfinance writes recoverable repair notices to stderr. Windows PowerShell 5.1
# wraps native stderr as ErrorRecord, so do not let those notices terminate the job.
$ErrorActionPreference = "Continue"
& $python -m app yahoo-ingest *>> $logPath
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python -m app yahoo-validate *>> $logPath
if ($LASTEXITCODE -eq 1) {
    exit $LASTEXITCODE
}

& $python -m app yahoo-analyze *>> $logPath
exit $LASTEXITCODE
