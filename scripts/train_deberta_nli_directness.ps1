$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$env:TQDM_DISABLE = "1"

Push-Location $Root
try {
    & $Python -m clarity_detection.train_transformer `
        --model-name MoritzLaurer/deberta-v3-base-zeroshot-v2.0 `
        --output-dir outputs\deberta_nli_directness_seed41 `
        --seed 41 `
        --split-seed 13 `
        --epochs 8 `
        --batch-size 1 `
        --eval-batch-size 4 `
        --grad-accum 16 `
        --lr 1e-5 `
        --weight-decay 0.01 `
        --warmup-ratio 0.08 `
        --max-length 512 `
        --truncation head_tail `
        --input-format directness_nli `
        --max-question-tokens 128 `
        --head-ratio 0.7 `
        --patience 3 `
        --fp16
}
finally {
    Pop-Location
}
