$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$env:TQDM_DISABLE = "1"

Push-Location $Root
try {
    New-Item -ItemType Directory -Force docs | Out-Null

    & $Python -m clarity_detection.summarize_experiments `
        --outputs-dir outputs `
        --output-dir outputs\experiment_scoreboard

    Copy-Item outputs\experiment_scoreboard\scoreboard.csv docs\all_experiments.csv -Force
    Copy-Item outputs\experiment_scoreboard\scoreboard.md docs\scoreboard_snapshot.md -Force

    Write-Host "Updated docs\all_experiments.csv"
    Write-Host "Updated docs\scoreboard_snapshot.md"
}
finally {
    Pop-Location
}
