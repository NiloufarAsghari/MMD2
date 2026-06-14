$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$env:TQDM_DISABLE = "1"

Push-Location $Root
try {
    & $Python -m clarity_detection.train_transformer `
        --model-name microsoft/deberta-v3-base `
        --output-dir outputs\deberta_base_rubric_seed29 `
        --seed 29 `
        --split-seed 13 `
        --epochs 8 `
        --batch-size 1 `
        --eval-batch-size 4 `
        --grad-accum 16 `
        --lr 2e-5 `
        --weight-decay 0.01 `
        --warmup-ratio 0.08 `
        --max-length 512 `
        --truncation head_tail `
        --input-format rubric_prompt `
        --max-question-tokens 160 `
        --head-ratio 0.7 `
        --patience 3 `
        --fp16
}
finally {
    Pop-Location
}
