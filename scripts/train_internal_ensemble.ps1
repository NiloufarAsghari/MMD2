$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"

Push-Location $Root
try {
    & $Python -m clarity_detection.train_baseline --output-dir outputs\tfidf_baseline

    & $Python -m clarity_detection.train_transformer `
        --model-name microsoft/deberta-v3-base `
        --output-dir outputs\deberta_base_seed13 `
        --seed 13 `
        --split-seed 13 `
        --epochs 8 `
        --batch-size 1 `
        --grad-accum 16 `
        --max-length 512 `
        --truncation head_tail `
        --fp16

    & $Python -m clarity_detection.train_transformer `
        --model-name microsoft/deberta-v3-base `
        --output-dir outputs\deberta_base_seed17_fixedsplit `
        --seed 17 `
        --split-seed 13 `
        --epochs 8 `
        --batch-size 1 `
        --grad-accum 16 `
        --max-length 512 `
        --truncation head_tail `
        --fp16

    & $Python -m clarity_detection.train_transformer `
        --model-name microsoft/deberta-v3-base `
        --output-dir outputs\deberta_base_seed23_fixedsplit `
        --seed 23 `
        --split-seed 13 `
        --epochs 8 `
        --batch-size 1 `
        --grad-accum 16 `
        --max-length 512 `
        --truncation head_tail `
        --fp16

    & $Python -m clarity_detection.train_transformer `
        --task boundary `
        --model-name microsoft/deberta-v3-base `
        --output-dir outputs\boundary_deberta_seed13 `
        --seed 13 `
        --split-seed 13 `
        --epochs 10 `
        --batch-size 1 `
        --grad-accum 16 `
        --max-length 512 `
        --truncation head_tail `
        --fp16

    & $Python -m clarity_detection.ensemble `
        --model-dirs outputs\deberta_base_seed13 outputs\deberta_base_seed17_fixedsplit outputs\deberta_base_seed23_fixedsplit `
        --boundary-model-dirs outputs\boundary_deberta_seed13 `
        --output-dir outputs\ensemble_3seeds_boundary `
        --batch-size 16 `
        --fp16
}
finally {
    Pop-Location
}
