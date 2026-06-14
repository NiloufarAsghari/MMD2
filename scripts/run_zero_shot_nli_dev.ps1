$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$env:TQDM_DISABLE = "1"

Push-Location $Root
try {
    & $Python -m clarity_detection.zero_shot_nli `
        --model-name MoritzLaurer/deberta-v3-base-zeroshot-v2.0 `
        --output-dir outputs\zero_shot_nli_directness_dev `
        --split dev `
        --batch-size 8 `
        --max-length 512 `
        --hypothesis-set directness `
        --score-mode entailment `
        --calibrate-dev-prior `
        --fp16
}
finally {
    Pop-Location
}
