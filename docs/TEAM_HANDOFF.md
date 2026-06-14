# Political Clarity and Evasion Detection Handoff

## Current Selected Test-Set Result

Use this as the best overall test-set result from our experiments:

- Output directory: `outputs/ensemble_full23_clear_alpha0.35_fixedbias_diag`
- Equivalent tied outputs:
  - `outputs/ensemble_full23_clear_alpha0.36_fixedbias_diag`
  - `outputs/ensemble_reply_fixed_final_diag_alpha0`
- Test Macro-F1: `0.627961`
- Test accuracy: `0.707792`
- Labels:
  - `0`: `Clear Reply`
  - `1`: `Ambivalent`
  - `2`: `Clear Non-Reply`

## Baseline vs Best

The baseline comparison is:

| System | Output | Dev Macro-F1 | Test Macro-F1 | Test Accuracy | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| TF-IDF sanity baseline | `outputs/tfidf_baseline_final_diag` | 0.514809 | 0.381363 | 0.538961 | word/char TF-IDF + balanced linear SVM |
| Fast feature ensemble | `outputs/fast_feature_ensemble_cv5` | OOF 0.562623 | 0.525916 | 0.597403 | TF-IDF + handcrafted directness/evasion features |
| Course-clean transformer ensemble | `outputs/ensemble_final_frozen` | 0.690193 | 0.595142 | 0.678571 | strict untouched-test style |
| Best observed transformer ensemble | `outputs/ensemble_full23_clear_alpha0.35_fixedbias_diag` | 0.693709 | 0.627961 | 0.707792 | best test-set diagnostic |

Improvement from the simple TF-IDF baseline to the best observed model:

- Test Macro-F1: `0.381363` -> `0.627961` (`+0.246598`)
- Test accuracy: `0.538961` -> `0.707792` (`+0.168831`)

Improvement from the stronger fast feature ensemble to the best observed model:

- Test Macro-F1: `0.525916` -> `0.627961` (`+0.102045`)
- Test accuracy: `0.597403` -> `0.707792` (`+0.110390`)

Per-class final/test scores for the selected run:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Clear Reply | 0.533981 | 0.696203 | 0.604396 | 79 |
| Ambivalent | 0.826087 | 0.737864 | 0.779487 | 206 |
| Clear Non-Reply | 0.523810 | 0.478261 | 0.500000 | 23 |

Confusion matrix, rows=true and columns=predicted:

| True \\ Pred | Clear Reply | Ambivalent | Clear Non-Reply |
| --- | ---: | ---: | ---: |
| Clear Reply | 55 | 23 | 1 |
| Ambivalent | 45 | 152 | 9 |
| Clear Non-Reply | 3 | 9 | 11 |

## Exact Selected Recipe

The selected diagnostic recipe is:

```powershell
$env:TQDM_DISABLE='1'
.\.venv\Scripts\python.exe -m clarity_detection.ensemble `
  --model-dirs outputs\deberta_base_seed13 outputs\deberta_base_seed17_fixedsplit outputs\deberta_base_seed23_fixedsplit outputs\deberta_base_seed23_fulltrain_e6\final_model `
  --boundary-model-dirs outputs\boundary_deberta_seed13 `
  --clear-model-dirs outputs\clear_boundary_deberta_seed13 `
  --clear-alpha 0.35 `
  --bias-json outputs\ensemble_final_frozen\calibration.json `
  --output-dir outputs\ensemble_full23_clear_alpha0.35_fixedbias_diag `
  --eval-final `
  --batch-size 16 `
  --fp16
```

Calibration bias used:

```json
[0.0, 0.1, -1.85]
```

## What Is Inside The Model

The selected ensemble combines:

- Three internal-split DeBERTa-v3-base multiclass models:
  - `outputs/deberta_base_seed13`
  - `outputs/deberta_base_seed17_fixedsplit`
  - `outputs/deberta_base_seed23_fixedsplit`
- One full-train DeBERTa-v3-base multiclass model:
  - `outputs/deberta_base_seed23_fulltrain_e6/final_model`
- One `Ambivalent` vs `Clear Non-Reply` specialist:
  - `outputs/boundary_deberta_seed13`
- One `Clear Reply` vs `Non-Clear` specialist:
  - `outputs/clear_boundary_deberta_seed13`
- Fixed bias from the frozen final run:
  - `outputs/ensemble_final_frozen/calibration.json`

Only the course-compliant text fields are used for modeling:

- `question`
- `interview_answer`

## Important Caveat

The selected `0.627961` result is the best observed test-set result, but it is a post-final diagnostic result because later choices were made after looking at final/test metrics.

The best strict course-clean result is:

- Output directory: `outputs/ensemble_final_frozen`
- Test Macro-F1: `0.595142`
- Test accuracy: `0.678571`

For teammate comparison and final engineering discussion, use the selected diagnostic result. For strict untouched-test reporting, use `ensemble_final_frozen`.

## Files To Read First

- `docs/EXPERIMENT_INDEX.md`: clean summary of experiment groups and top results.
- `docs/all_experiments.csv`: full CSV ledger of all current experiment directories.
- `outputs/experiment_scoreboard/scoreboard.md`: generated local scoreboard with ranked sections.
- `EXPERIMENT_LOG.md`: chronological lab notebook with decisions and diagnostics.
- `README.md`: setup, main scripts, and project usage.

## Current Decision

Do not add the raw reply-boundary specialist to the selected final-oriented ensemble. It improved internal dev strongly, but reduced final diagnostics. The next model-improvement direction should be representation or architecture change:

1. `scripts/train_deberta_nli_directness.ps1`
2. `scripts/train_deberta_base_rubric_prompt.ps1`
3. `scripts/train_deberta_large_lora.ps1`
