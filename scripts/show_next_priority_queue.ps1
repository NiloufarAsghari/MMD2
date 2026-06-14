$ErrorActionPreference = "Stop"

Write-Host "Next experiment queue. This script prints commands only; it does not run training."
Write-Host ""

$Commands = @(
    @{
        Rank = 1
        Name = "NLI/directness representation run"
        Rationale = "The reply-boundary specialist overfits internal dev; use a different semantic framing next."
        Command = ".\scripts\train_deberta_nli_directness.ps1"
        Evaluate = ".\scripts\evaluate_representation_ensemble.ps1 -ExtraModelDirs outputs\deberta_nli_directness_seed41"
    },
    @{
        Rank = 2
        Name = "Rubric-prompt DeBERTa-base run"
        Rationale = "Keeps proven base architecture but gives the model explicit class definitions."
        Command = ".\scripts\train_deberta_base_rubric_prompt.ps1"
        Evaluate = ".\scripts\evaluate_representation_ensemble.ps1 -ExtraModelDirs outputs\deberta_base_rubric_seed29"
    },
    @{
        Rank = 3
        Name = "DeBERTa-v3-large LoRA"
        Rationale = "Higher capacity architecture; previous partial run collapsed and needs the revised script."
        Command = ".\scripts\train_deberta_large_lora.ps1"
        Evaluate = ".\scripts\evaluate_architecture_ensemble.ps1 -ExtraModelDirs outputs\deberta_large_lora_seed31_r16_lr5e5"
    },
    @{
        Rank = 4
        Name = "Chunked long-answer inference"
        Rationale = "No retraining; lower priority because current errors are more semantic than context-length limited."
        Command = ".\scripts\evaluate_chunked_ensemble.ps1 -Mode DiagnosticBest"
        Evaluate = "Same command writes internal-dev metrics."
    },
    @{
        Rank = 5
        Name = "Reply-boundary gating follow-up"
        Rationale = "Only revisit if using a conservative gate; raw reply alpha improves dev but reduces final diagnostics."
        Command = ".\.venv\Scripts\python.exe -m clarity_detection.sweep_reply_alpha --help"
        Evaluate = "Review outputs\ensemble_reply_fixed_final_diag_alphasummary.csv first."
    }
)

$Commands | ForEach-Object {
    Write-Host "[$($_.Rank)] $($_.Name)"
    Write-Host "    Rationale: $($_.Rationale)"
    Write-Host "    Run:       $($_.Command)"
    Write-Host "    Evaluate:  $($_.Evaluate)"
    Write-Host ""
}
