$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$env:TQDM_DISABLE = "1"

Push-Location $Root
try {
    & $Python -m clarity_detection.train_transformer `
        --model-name microsoft/deberta-v3-large `
        --output-dir outputs\deberta_large_lora_seed31_r16_lr5e5 `
        --seed 31 `
        --split-seed 13 `
        --epochs 8 `
        --batch-size 1 `
        --eval-batch-size 1 `
        --grad-accum 16 `
        --lr 5e-5 `
        --weight-decay 0.01 `
        --warmup-ratio 0.06 `
        --max-length 512 `
        --truncation head_tail `
        --max-question-tokens 96 `
        --head-ratio 0.7 `
        --patience 3 `
        --fp16 `
        --gradient-checkpointing `
        --lora `
        --lora-r 16 `
        --lora-alpha 32 `
        --lora-dropout 0.1
}
finally {
    Pop-Location
}
