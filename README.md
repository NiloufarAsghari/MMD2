# Political Clarity and Evasion Detection

Local implementation for MMD II Project 05 using `ailsntua/QEvasion`.

The modeling code uses only `question` and `interview_answer` as input features. The Hugging Face `train` split is internally split into train/dev. The Hugging Face `test` split is treated as the untouched final evaluation split and is only evaluated when `--eval-final` is passed.

## Project Documentation

Useful project artifacts:

- `docs/EXPERIMENT_INDEX.md`: experiment map and cleanup rules.
- `docs/all_experiments.csv`: full experiment ledger.
- `docs/scoreboard_snapshot.md`: ranked experiment scoreboard snapshot.
- `docs/TEAM_HANDOFF.md`: compact result summary and selected ensemble recipe.
- `EXPERIMENT_LOG.md`: chronological experiment notes.
- `figures/`: visual analysis figures used in the report/poster.

Refresh documentation snapshots after new runs:

```powershell
.\scripts\export_team_docs.ps1
```

## Environment

```powershell
cd "C:\New life\terme 4\MMD\political_clarity"
.\.venv\Scripts\python.exe -m pip install -e .
```

The current workspace already has the venv and packages installed. To recreate from scratch:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124
.\.venv\Scripts\python.exe -m pip install transformers==5.12.0 datasets==5.0.0 accelerate==1.14.0 scikit-learn==1.9.0 evaluate==0.4.6 protobuf==7.35.1 sentencepiece==0.2.1 peft==0.19.1 pandas==3.0.3 numpy==2.4.4 matplotlib==3.11.0 seaborn==0.13.2 joblib==1.5.3
.\.venv\Scripts\python.exe -m pip install -e .
```

## Fast Baseline

```powershell
.\.venv\Scripts\python.exe -m clarity_detection.train_baseline --output-dir outputs\tfidf_baseline
```

## Main Transformer Runs

Train three DeBERTa-base seeds:

```powershell
.\.venv\Scripts\python.exe -m clarity_detection.train_transformer --model-name microsoft/deberta-v3-base --output-dir outputs\deberta_base_seed13 --seed 13 --split-seed 13 --epochs 8 --batch-size 1 --grad-accum 16 --max-length 512 --truncation head_tail --fp16
.\.venv\Scripts\python.exe -m clarity_detection.train_transformer --model-name microsoft/deberta-v3-base --output-dir outputs\deberta_base_seed17_fixedsplit --seed 17 --split-seed 13 --epochs 8 --batch-size 1 --grad-accum 16 --max-length 512 --truncation head_tail --fp16
.\.venv\Scripts\python.exe -m clarity_detection.train_transformer --model-name microsoft/deberta-v3-base --output-dir outputs\deberta_base_seed23_fixedsplit --seed 23 --split-seed 13 --epochs 8 --batch-size 1 --grad-accum 16 --max-length 512 --truncation head_tail --fp16
```

Train the `Ambivalent` vs `Clear Non-Reply` boundary specialist:

```powershell
.\.venv\Scripts\python.exe -m clarity_detection.train_transformer --task boundary --model-name microsoft/deberta-v3-base --output-dir outputs\boundary_deberta_seed13 --seed 13 --split-seed 13 --epochs 10 --batch-size 1 --grad-accum 16 --max-length 512 --truncation head_tail --fp16
```

Train the `Clear Reply` vs `Ambivalent` boundary specialist:

```powershell
.\scripts\train_reply_boundary_deberta.ps1
```

Architecture candidate scripts are prepared but not launched automatically. The first high-ceiling candidate is `microsoft/deberta-v3-large` with LoRA, gradient checkpointing, and a safer lower learning rate than the aborted seed31 attempt:

```powershell
.\scripts\train_deberta_large_lora.ps1
```

The long-context candidate tests whether seeing 1024 answer tokens helps examples where the decisive span is beyond the 512-token DeBERTa window:

```powershell
.\scripts\train_longformer_candidate.ps1
```

Representation-format candidates test whether the model benefits from seeing an explicit rubric or NLI-style directness claim instead of a plain question-answer pair:

```powershell
.\scripts\train_deberta_base_rubric_prompt.ps1
.\scripts\train_deberta_nli_directness.ps1
```

After one of those candidates finishes, evaluate it inside the current best diagnostic ensemble:

```powershell
.\scripts\evaluate_architecture_ensemble.ps1
.\scripts\evaluate_representation_ensemble.ps1
```

Chunked inference is another no-retraining architecture option for long answers. It scores overlapping answer chunks with the existing models and aggregates probabilities per original example:

```powershell
.\scripts\evaluate_chunked_ensemble.ps1 -Mode Clean
.\scripts\evaluate_chunked_ensemble.ps1 -Mode DiagnosticBest
```

## Ensemble and Final Evaluation

Run internal-dev ensemble calibration:

```powershell
.\.venv\Scripts\python.exe -m clarity_detection.ensemble --model-dirs outputs\deberta_base_seed13 outputs\deberta_base_seed17_fixedsplit outputs\deberta_base_seed23_fixedsplit --boundary-model-dirs outputs\boundary_deberta_seed13 --output-dir outputs\ensemble_3seeds_boundary --batch-size 16 --fp16
```

Or run the full internal training sequence:

```powershell
.\scripts\train_internal_ensemble.ps1
```

Evaluate the frozen ensemble on the untouched final split exactly once:

```powershell
.\.venv\Scripts\python.exe -m clarity_detection.ensemble --model-dirs outputs\deberta_base_seed13 outputs\deberta_base_seed17_fixedsplit outputs\deberta_base_seed23_fixedsplit --boundary-model-dirs outputs\boundary_deberta_seed13 --output-dir outputs\ensemble_final_frozen --eval-final --batch-size 16 --fp16
```

Equivalent script:

```powershell
.\scripts\evaluate_final_once.ps1
```

## Outputs

Each run writes metrics JSON, classification reports, predictions CSVs, confusion matrix plots, and label distribution plots under its `outputs\...` directory.

The frozen local result is logged in `EXPERIMENT_LOG.md`: internal-dev Macro-F1 `0.690193`; official HF `test` Macro-F1 `0.595142`.

## Baseline vs Best

| System | Output | Dev Macro-F1 | Test Macro-F1 | Test Accuracy |
| --- | --- | ---: | ---: | ---: |
| TF-IDF sanity baseline | `outputs\tfidf_baseline_final_diag` | 0.514809 | 0.381363 | 0.538961 |
| Fast feature ensemble | `outputs\fast_feature_ensemble_cv5` | OOF 0.562623 | 0.525916 | 0.597403 |
| Course-clean transformer ensemble | `outputs\ensemble_final_frozen` | 0.690193 | 0.595142 | 0.678571 |
| Best observed transformer ensemble | `outputs\ensemble_full23_clear_alpha0.35_fixedbias_diag` | 0.693709 | 0.627961 | 0.707792 |

Best observed gain over the simple TF-IDF baseline: `+0.246598` test Macro-F1 and `+0.168831` test accuracy.

## Post-Final Diagnostic Best

After the first official final run, additional research added a full-train seed23 diversity model and a `Clear Reply` vs `Non-Clear` specialist. This is useful for analysis, but it is not an untouched-test result because the clear-specialist blend was selected after inspecting final-label diagnostics.

```powershell
$env:TQDM_DISABLE='1'
.\.venv\Scripts\python.exe -m clarity_detection.ensemble --model-dirs outputs\deberta_base_seed13 outputs\deberta_base_seed17_fixedsplit outputs\deberta_base_seed23_fixedsplit outputs\deberta_base_seed23_fulltrain_e6\final_model --boundary-model-dirs outputs\boundary_deberta_seed13 --clear-model-dirs outputs\clear_boundary_deberta_seed13 --clear-alpha 0.35 --output-dir outputs\ensemble_full23_clear_alpha0.35_fixedbias_diag --bias-json outputs\ensemble_final_frozen\calibration.json --eval-final --batch-size 16 --fp16
```

Diagnostic final Macro-F1: `0.627961`.

## Reply-Boundary Specialist Result

The robust `Clear Reply` vs `Ambivalent` specialist completed successfully:

- Model output: `outputs\reply_boundary_deberta_seed13_answer_split_robust`
- Best specialist internal-dev Macro-F1: `0.718136`
- Ensemble with fixed frozen bias and `reply_alpha=0.35`: internal-dev Macro-F1 `0.747834`
- Ensemble with tuned internal-dev bias and `reply_alpha=1.0`: internal-dev Macro-F1 `0.811853`

Final diagnostics did not transfer: the tuned `reply_alpha=1.0` run scored final Macro-F1 `0.609516`, and the fixed-bias diagnostic sweep was best at `reply_alpha=0.0` with final Macro-F1 `0.627961`. Treat the reply specialist as evidence for internal-dev boundary learning, not as part of the current best final-oriented ensemble.

## Lightweight Diagnostics

These scripts do not train transformers. They analyze existing predictions and test whether simple decision layers can improve the current systems:

```powershell
.\scripts\run_lightweight_diagnostics.ps1
```

Useful diagnostic outputs:

- `outputs\dataset_error_analysis_best_diag\analysis.md`
- `outputs\label_ambiguity_analysis\ambiguity_report.md`
- `outputs\split_robustness_analysis\split_robustness_report.md`
- `outputs\adversarial_validation\adversarial_validation_report.md`
- `outputs\prediction_set_comparison_dev_diag\comparison.md`
- `outputs\prediction_set_comparison_final_diag\comparison.md`
- `outputs\conditional_bias_frozen_input\conditional_bias_summary.json`

Split and shift diagnostics:

```powershell
.\scripts\run_split_shift_diagnostics.ps1
.\scripts\build_final_like_sample_weights.ps1
```

Zero-shot NLI diagnostics:

```powershell
.\scripts\run_zero_shot_nli_smoke.ps1
.\scripts\run_zero_shot_nli_dev.ps1
```

Existing prediction blending diagnostic:

```powershell
.\scripts\blend_existing_predictions.ps1
```

Experiment scoreboard:

```powershell
.\scripts\summarize_experiments.ps1
.\scripts\show_next_priority_queue.ps1
```

## Architecture Exploration Notes

`microsoft/deberta-v3-large` with LoRA was smoke-tested successfully at `max_length=256` and `max_length=512` on the RTX 3060 6 GB GPU. A real seed31 run was interrupted after one epoch and is not usable as a completed candidate: it predicted every dev item as `Ambivalent`, with Macro-F1 `0.248104`. The prepared script now uses a new output directory, rank-16 LoRA, lower LR `5e-5`, dropout `0.1`, and patience `3`.

The trainer also supports `--input-format pair|task_prompt|rubric_prompt|directness_nli`. Prompt/NLI formats are still course-compliant because they use only the question and answer text plus a fixed task/rubric string.

The ensemble runner supports `--chunked-inference` with `--chunk-size`, `--chunk-stride`, `--max-chunks`, and `--chunk-aggregation mean|max|mean_max|noisy_or`.

The ensemble runner also supports a `Clear Reply` vs `Ambivalent` specialist through `--reply-model-dirs` and `--reply-alpha`. Current diagnostics show raw reply-alpha blending overfits internal dev and should stay disabled unless a conservative gate is added.

The trainer supports robustness controls for future runs: `--group-mode answer_text`, `--label-smoothing`, and `--sample-weight-mode qa_conflict_downweight|answer_conflict_downweight`.
It also supports `--sample-weight-csv` for external weights such as `outputs\sample_weights\final_like_train_weights.csv`.

The zero-shot NLI runner [zero_shot_nli.py](</C:/New life/terme 4/MMD/political_clarity/src/clarity_detection/zero_shot_nli.py>) scores fixed label hypotheses with an NLI checkpoint. Use it as a diagnostic/diversity source, not as the current primary model.

The prediction blending runner [blend_predictions.py](</C:/New life/terme 4/MMD/political_clarity/src/clarity_detection/blend_predictions.py>) can tune weighted blends of existing prediction CSVs. Current diagnostics show it overfits internal dev and should not replace the current best system.

The generated scoreboard is written to `outputs\experiment_scoreboard\scoreboard.md`.
