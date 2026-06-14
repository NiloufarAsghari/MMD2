param(
    [string[]]$ExtraModelDirs = @("outputs\deberta_large_lora_seed31_r16_lr5e5\best_model"),
    [string]$OutputDir = "outputs\ensemble_architecture_candidate",
    [switch]$EvalFinal
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$env:TQDM_DISABLE = "1"

$BaseModelDirs = @(
    "outputs\deberta_base_seed13",
    "outputs\deberta_base_seed17_fixedsplit",
    "outputs\deberta_base_seed23_fixedsplit",
    "outputs\deberta_base_seed23_fulltrain_e6\final_model"
)

$ExistingExtraModelDirs = @()
foreach ($ModelDir in $ExtraModelDirs) {
    if (Test-Path (Join-Path $Root $ModelDir)) {
        $ExistingExtraModelDirs += $ModelDir
    }
}

if ($ExistingExtraModelDirs.Count -eq 0) {
    throw "No extra architecture candidate model directories exist yet. Train one first or pass -ExtraModelDirs."
}

$ModelDirs = $BaseModelDirs + $ExistingExtraModelDirs
$EvalArgs = @()
if ($EvalFinal) {
    $EvalArgs += "--eval-final"
}

Push-Location $Root
try {
    & $Python -m clarity_detection.ensemble `
        --model-dirs $ModelDirs `
        --boundary-model-dirs outputs\boundary_deberta_seed13 `
        --clear-model-dirs outputs\clear_boundary_deberta_seed13 `
        --clear-alpha 0.35 `
        --bias-json outputs\ensemble_final_frozen\calibration.json `
        --output-dir $OutputDir `
        --batch-size 8 `
        --fp16 `
        $EvalArgs
}
finally {
    Pop-Location
}
