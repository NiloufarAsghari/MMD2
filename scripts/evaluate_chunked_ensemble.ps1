param(
    [ValidateSet("Clean", "DiagnosticBest")]
    [string]$Mode = "Clean",
    [ValidateSet("mean", "max", "mean_max", "noisy_or")]
    [string]$Aggregation = "mean_max",
    [int]$ChunkSize = 384,
    [int]$ChunkStride = 192,
    [int]$MaxChunks = 6,
    [string]$OutputDir = "",
    [switch]$EvalFinal,
    [int]$MaxDevSamples = 0
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$env:TQDM_DISABLE = "1"

if ($OutputDir -eq "") {
    $OutputDir = if ($Mode -eq "Clean") {
        "outputs\ensemble_chunked_clean_$Aggregation"
    }
    else {
        "outputs\ensemble_chunked_diagnostic_best_$Aggregation"
    }
}

$ModelDirs = @(
    "outputs\deberta_base_seed13",
    "outputs\deberta_base_seed17_fixedsplit",
    "outputs\deberta_base_seed23_fixedsplit"
)
$BoundaryDirs = @("outputs\boundary_deberta_seed13")
$ClearArgs = @()
$BiasArgs = @()

if ($Mode -eq "DiagnosticBest") {
    $ModelDirs += "outputs\deberta_base_seed23_fulltrain_e6\final_model"
    $ClearArgs = @("--clear-model-dirs", "outputs\clear_boundary_deberta_seed13", "--clear-alpha", "0.35")
    $BiasArgs = @("--bias-json", "outputs\ensemble_final_frozen\calibration.json")
}

$EvalArgs = @()
if ($EvalFinal) {
    $EvalArgs += "--eval-final"
}
if ($MaxDevSamples -gt 0) {
    $EvalArgs += @("--max-dev-samples", "$MaxDevSamples")
}

Push-Location $Root
try {
    & $Python -m clarity_detection.ensemble `
        --model-dirs $ModelDirs `
        --boundary-model-dirs $BoundaryDirs `
        @ClearArgs `
        @BiasArgs `
        --output-dir $OutputDir `
        --batch-size 8 `
        --fp16 `
        --chunked-inference `
        --chunk-size $ChunkSize `
        --chunk-stride $ChunkStride `
        --max-chunks $MaxChunks `
        --chunk-aggregation $Aggregation `
        @EvalArgs
}
finally {
    Pop-Location
}
