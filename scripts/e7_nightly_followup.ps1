Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = 'D:\Python\python.exe'
$codex = 'C:\Users\KianaD\AppData\Local\OpenAI\Codex\bin\247581e40ee272fb\codex.exe'
$log = Join-Path $workspace 'results\e7_nightly_20260924.log'
$sent = Join-Path $workspace 'results\e7_nightly_20260924.sent'
$thread = '01a09164-9243-77b0-9b08-b854692ee235'
$taskName = 'Codex-E7-History-20260924'

Set-Location -LiteralPath $workspace
if (Test-Path -LiteralPath $sent) {
    Disable-ScheduledTask -TaskName $taskName | Out-Null
    exit 0
}
if (-not (Test-Path -LiteralPath $python) -or -not (Test-Path -LiteralPath $codex)) {
    'E7 follow-up could not start: Python or Codex executable is missing.' | Add-Content -LiteralPath $log
    exit 1
}

"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') price check:" | Out-File -FilePath $log -Encoding utf8 -Append
$ErrorActionPreference = 'Continue'
$check = & $python 'scripts/prefetch_e7_behavior_history.py' --verify-prices --prices-only 2>&1
$checkExit = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
$check | ForEach-Object { $_.ToString() } | Out-File -FilePath $log -Encoding utf8 -Append
if ($checkExit -ne 0) {
    'Price cache is incomplete; checking again in 30 minutes.' | Out-File -FilePath $log -Encoding utf8 -Append
    exit 2
}

$message = @'
The E7 historical Tushare daily and adj_factor cache is complete for 2016-05-01 through 2023-08-31. Please continue the user's requested E7 work IN THIS CONVERSATION: inspect actual investor transactions and point-in-time data, redesign E7 without mapping later E5 recommendations onto earlier investors, implement the defensible five-method comparison, run it, and analyze the results in Chinese. Preserve E1-E6 and old E7 outputs. Do not describe observed action matching as adoption or model value as reported satisfaction. If complete portfolio holdings cannot be recovered, explain the precise limitation rather than inventing them. Read AGENTS.md and the existing E5/E6 manifests first.
'@
$ErrorActionPreference = 'Continue'
$queueOutput = & $codex queue --thread $thread --message $message 2>&1
$queueExit = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
$queueOutput | ForEach-Object { $_.ToString() } | Out-File -FilePath $log -Encoding utf8 -Append
if ($queueExit -ne 0) {
    'Could not send the follow-up to the current conversation; retrying in 30 minutes.' | Out-File -FilePath $log -Encoding utf8 -Append
    exit 3
}
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') queued E7 follow-up to the current conversation." |
    Set-Content -LiteralPath $sent
Disable-ScheduledTask -TaskName $taskName | Out-Null
exit 0
