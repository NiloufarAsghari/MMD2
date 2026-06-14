param(
    [double]$ReplyAlpha = 0.35,
    [string]$ReplyModelDir = "outputs\reply_boundary_deberta_seed13_answer_split_robust",
    [string]$OutputDir = "outputs\ensemble_with_reply_boundary",
    [switch]$EvalFinal
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$env:TQDM_DISABLE = "1"

$EvalArgs = @()
if ($EvalFinal) {
    $EvalArgs += "--eval-final"
}

Push-Location $Root
try {
    & $Python -m clarity_detection.ensemble `
        --model-dirs outputs\deberta_base_seed13 outputs\deberta_base_seed17_fixedsplit outputs\deberta_base_seed23_fixedsplit outputs\deberta_base_seed23_fulltrain_e6\final_model `
        --boundary-model-dirs outputs\boundary_deberta_seed13 `
        --clear-model-dirs outputs\clear_boundary_deberta_seed13 `
        --clear-alpha 0.35 `
        --reply-model-dirs $ReplyModelDir `
        --reply-alpha $ReplyAlpha `
        --bias-json outputs\ensemble_final_frozen\calibration.json `
        --output-dir $OutputDir `
        --batch-size 16 `
        --fp16 `
        @EvalArgs
}
finally {
    Pop-Location
}
