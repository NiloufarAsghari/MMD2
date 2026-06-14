$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"

Push-Location $Root
try {
    & $Python -m clarity_detection.ensemble `
        --model-dirs outputs\deberta_base_seed13 outputs\deberta_base_seed17_fixedsplit outputs\deberta_base_seed23_fixedsplit `
        --boundary-model-dirs outputs\boundary_deberta_seed13 `
        --output-dir outputs\ensemble_final_frozen `
        --eval-final `
        --batch-size 16 `
        --fp16
}
finally {
    Pop-Location
}
