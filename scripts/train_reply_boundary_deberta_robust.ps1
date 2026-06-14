$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$env:TQDM_DISABLE = "1"
$WeightCsv = Join-Path $Root "outputs\sample_weights\final_like_train_weights.csv"

Push-Location $Root
try {
    if (-not (Test-Path $WeightCsv)) {
        & $Python -m clarity_detection.build_sample_weights `
            --score-csv outputs\adversarial_validation\adversarial_oof_scores.csv `
            --output-csv outputs\sample_weights\final_like_train_weights.csv `
            --score-column final_like_score_oof `
            --source-split official_train `
            --floor 0.5 `
            --power 1.0 `
            --max-weight 3.0 `
            --conflict-key qa `
            --conflict-downweight 0.75
    }

    & $Python -m clarity_detection.train_transformer `
        --task reply_boundary `
        --model-name microsoft/deberta-v3-base `
        --output-dir outputs\reply_boundary_deberta_seed13_answer_split_robust `
        --seed 13 `
        --split-seed 13 `
        --group-mode answer_text `
        --epochs 8 `
        --batch-size 1 `
        --eval-batch-size 4 `
        --grad-accum 16 `
        --lr 2e-5 `
        --weight-decay 0.01 `
        --warmup-ratio 0.08 `
        --label-smoothing 0.03 `
        --sample-weight-mode qa_conflict_downweight `
        --conflict-downweight 0.5 `
        --sample-weight-csv outputs\sample_weights\final_like_train_weights.csv `
        --sample-weight-column sample_weight `
        --missing-sample-weight one `
        --max-length 512 `
        --truncation head_tail `
        --max-question-tokens 96 `
        --head-ratio 0.7 `
        --patience 3 `
        --fp16
}
finally {
    Pop-Location
}
