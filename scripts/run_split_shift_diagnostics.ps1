$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$env:TQDM_DISABLE = "1"

Push-Location $Root
try {
    & $Python -m clarity_detection.analyze_split_robustness `
        --output-dir outputs\split_robustness_analysis `
        --run-baseline

    & $Python -m clarity_detection.adversarial_validation `
        --output-dir outputs\adversarial_validation
}
finally {
    Pop-Location
}
