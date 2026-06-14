$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"

Push-Location $Root
try {
    & $Python -m clarity_detection.build_sample_weights `
        --score-csv outputs\adversarial_validation\adversarial_oof_scores.csv `
        --output-csv outputs\sample_weights\final_like_train_weights.csv `
        --score-column final_like_score_oof `
        --source-split official_train `
        --floor 0.5 `
        --power 1.0 `
        --max-weight 3.0 `
        --conflict-key qa `
        --conflict-downweight 0.75
}
finally {
    Pop-Location
}
