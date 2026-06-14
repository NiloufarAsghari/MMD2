$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$env:TQDM_DISABLE = "1"

Push-Location $Root
try {
    & $Python -m clarity_detection.analyze_errors `
        --dev-predictions outputs\ensemble_full23_clear_alpha0.35_fixedbias_diag\dev_predictions.csv `
        --final-predictions outputs\ensemble_full23_clear_alpha0.35_fixedbias_diag\final_predictions.csv `
        --output-dir outputs\dataset_error_analysis_best_diag

    & $Python -m clarity_detection.compare_prediction_sets `
        --prediction frozen=outputs\ensemble_final_frozen\dev_predictions.csv `
        --prediction full23=outputs\ensemble_3seeds_plus_full23_fixedbias_diag\dev_predictions.csv `
        --prediction clear025=outputs\ensemble_full23_clear_alpha0.25_fixedbias_diag\dev_predictions.csv `
        --prediction clear035=outputs\ensemble_full23_clear_alpha0.35_fixedbias_diag\dev_predictions.csv `
        --prediction clear05=outputs\ensemble_full23_clear_alpha0.5_fixedbias_diag\dev_predictions.csv `
        --prediction noboundary=outputs\ensemble_3seeds_plus_full23_noboundary_fixedbias_diag\dev_predictions.csv `
        --output-dir outputs\prediction_set_comparison_dev_diag

    & $Python -m clarity_detection.tune_conditional_bias `
        --dev-predictions outputs\ensemble_final_frozen\dev_predictions.csv `
        --final-predictions outputs\ensemble_final_frozen\final_predictions.csv `
        --output-dir outputs\conditional_bias_frozen_input
}
finally {
    Pop-Location
}
