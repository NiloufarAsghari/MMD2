$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$env:TQDM_DISABLE = "1"

Push-Location $Root
try {
    & $Python -m clarity_detection.train_transformer `
        --model-name allenai/longformer-base-4096 `
        --output-dir outputs\longformer_base_seed37_len1024 `
        --seed 37 `
        --split-seed 13 `
        --epochs 6 `
        --batch-size 1 `
        --eval-batch-size 1 `
        --grad-accum 16 `
        --lr 1e-5 `
        --weight-decay 0.01 `
        --warmup-ratio 0.08 `
        --max-length 1024 `
        --truncation standard `
        --patience 2 `
        --fp16 `
        --gradient-checkpointing
}
finally {
    Pop-Location
}
