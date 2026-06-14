$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$env:TQDM_DISABLE = "1"

Push-Location $Root
try {
    & $Python -m clarity_detection.blend_predictions `
        --dev frozen=outputs\ensemble_final_frozen\dev_predictions.csv `
        --dev full23=outputs\ensemble_3seeds_plus_full23_fixedbias_diag\dev_predictions.csv `
        --dev clear025=outputs\ensemble_full23_clear_alpha0.25_fixedbias_diag\dev_predictions.csv `
        --dev clear030=outputs\ensemble_full23_clear_alpha0.3_fixedbias_diag\dev_predictions.csv `
        --dev clear035=outputs\ensemble_full23_clear_alpha0.35_fixedbias_diag\dev_predictions.csv `
        --dev clear036=outputs\ensemble_full23_clear_alpha0.36_fixedbias_diag\dev_predictions.csv `
        --dev clear037=outputs\ensemble_full23_clear_alpha0.37_fixedbias_diag\dev_predictions.csv `
        --dev clear050=outputs\ensemble_full23_clear_alpha0.5_fixedbias_diag\dev_predictions.csv `
        --dev noboundary=outputs\ensemble_3seeds_plus_full23_noboundary_fixedbias_diag\dev_predictions.csv `
        --final frozen=outputs\ensemble_final_frozen\final_predictions.csv `
        --final full23=outputs\ensemble_3seeds_plus_full23_fixedbias_diag\final_predictions.csv `
        --final clear025=outputs\ensemble_full23_clear_alpha0.25_fixedbias_diag\final_predictions.csv `
        --final clear030=outputs\ensemble_full23_clear_alpha0.3_fixedbias_diag\final_predictions.csv `
        --final clear035=outputs\ensemble_full23_clear_alpha0.35_fixedbias_diag\final_predictions.csv `
        --final clear036=outputs\ensemble_full23_clear_alpha0.36_fixedbias_diag\final_predictions.csv `
        --final clear037=outputs\ensemble_full23_clear_alpha0.37_fixedbias_diag\final_predictions.csv `
        --final clear050=outputs\ensemble_full23_clear_alpha0.5_fixedbias_diag\final_predictions.csv `
        --final noboundary=outputs\ensemble_3seeds_plus_full23_noboundary_fixedbias_diag\final_predictions.csv `
        --output-dir outputs\blend_existing_predictions_diag `
        --blend-space logit `
        --random-candidates 1500
}
finally {
    Pop-Location
}
