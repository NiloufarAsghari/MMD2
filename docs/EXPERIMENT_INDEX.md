# Experiment Index

This file is the teammate-facing map of the experiment artifacts. The full local ledger is generated from the `outputs/` directory.

## Full Experiment Ledger

- Full CSV snapshot for sharing: `docs/all_experiments.csv`
- Ranked Markdown snapshot for sharing: `docs/scoreboard_snapshot.md`
- Generated local Markdown scoreboard: `outputs/experiment_scoreboard/scoreboard.md`
- Generated local CSV scoreboard: `outputs/experiment_scoreboard/scoreboard.csv`
- Chronological notes: `EXPERIMENT_LOG.md`

Current ledger size: `110` experiment directories.

Experiment kinds in the current ledger:

| Kind | Count |
| --- | ---: |
| analysis | 6 |
| blend | 3 |
| decision_layer | 6 |
| ensemble | 51 |
| model | 18 |
| other | 5 |
| smoke | 19 |
| zero_shot | 2 |

Regenerate the scoreboard and refresh shareable docs snapshots:

```powershell
.\scripts\export_team_docs.ps1
```

## Best Test-Set Runs

Baseline comparison:

| System | Output | Dev Macro-F1 | Test Macro-F1 | Test Accuracy |
| --- | --- | ---: | ---: | ---: |
| TF-IDF sanity baseline | `outputs/tfidf_baseline_final_diag` | 0.514809 | 0.381363 | 0.538961 |
| Fast feature ensemble | `outputs/fast_feature_ensemble_cv5` | OOF 0.562623 | 0.525916 | 0.597403 |
| Course-clean transformer ensemble | `outputs/ensemble_final_frozen` | 0.690193 | 0.595142 | 0.678571 |
| Best observed transformer ensemble | `outputs/ensemble_full23_clear_alpha0.35_fixedbias_diag` | 0.693709 | 0.627961 | 0.707792 |

Best observed gain over simple TF-IDF baseline: `+0.246598` test Macro-F1 and `+0.168831` test accuracy.

| Rank | Output | Test Macro-F1 | Test Accuracy | Status |
| ---: | --- | ---: | ---: | --- |
| 1 | `outputs/ensemble_full23_clear_alpha0.35_fixedbias_diag` | 0.627961 | 0.707792 | best observed diagnostic |
| 1 | `outputs/ensemble_full23_clear_alpha0.36_fixedbias_diag` | 0.627961 | 0.707792 | tied diagnostic |
| 1 | `outputs/ensemble_reply_fixed_final_diag_alpha0` | 0.627961 | 0.707792 | equivalent tied diagnostic |
| 4 | `outputs/ensemble_full23_clear_alpha0.37_fixedbias_diag` | 0.625814 | 0.704545 | diagnostic |
| 5 | `outputs/ensemble_reply_fixed_final_diag_alpha0p1` | 0.625170 | 0.704545 | diagnostic |

Best strict course-clean run:

| Output | Test Macro-F1 | Test Accuracy | Status |
| --- | ---: | ---: | --- |
| `outputs/ensemble_final_frozen` | 0.595142 | 0.678571 | untouched final/test style |

## Major Experiment Groups

| Group | Purpose | Key Outcome |
| --- | --- | --- |
| TF-IDF and fast feature baselines | sanity checks and cheap Macro-F1 floor | useful for validation, not competitive with transformers |
| DeBERTa-v3-base multiclass seeds | primary model family | strongest reliable base models |
| Boundary specialist | separates `Ambivalent` from `Clear Non-Reply` | improves final-oriented ensemble behavior |
| Clear specialist | separates `Clear Reply` from non-clear | produced the best observed test-set diagnostic result |
| Reply specialist | separates `Clear Reply` from `Ambivalent` | strong internal-dev gain, poor final transfer |
| Full-train seed23 | adds diversity from all official train rows | useful in best diagnostic ensemble |
| Zero-shot NLI diagnostics | tests fixed rubric/directness hypotheses | not competitive alone, possible diversity signal |
| CSV blends and decision layers | tune over saved predictions | mostly overfit internal dev |
| Large LoRA and chunking | architecture/context experiments | smoke-tested, not selected yet |

## Current Selected Ensemble

Selected output:

```text
outputs/ensemble_full23_clear_alpha0.35_fixedbias_diag
```

Selected recipe:

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

## Cleanup Rules

Keep these:

- `outputs/ensemble_full23_clear_alpha0.35_fixedbias_diag`
- `outputs/ensemble_final_frozen`
- all model directories referenced by the selected recipe
- `outputs/experiment_scoreboard`
- `EXPERIMENT_LOG.md`
- `docs/`

Safe to archive later, but do not delete until the team agrees:

- `outputs/smoke_*`
- intermediate alpha sweeps
- abandoned large-LoRA partial outputs
- zero-shot sample diagnostics

Avoid moving checkpoint directories unless scripts are updated, because many scripts reference output paths directly.
