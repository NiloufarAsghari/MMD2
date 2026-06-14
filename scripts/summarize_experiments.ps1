$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$env:TQDM_DISABLE = "1"

Push-Location $Root
try {
    & $Python -m clarity_detection.summarize_experiments `
        --outputs-dir outputs `
        --output-dir outputs\experiment_scoreboard
}
finally {
    Pop-Location
}
