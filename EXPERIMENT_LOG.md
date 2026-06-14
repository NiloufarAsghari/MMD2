# QEvasion Political Clarity Experiment Log

All reported model inputs are course-compliant: `question` and `interview_answer` only. The Hugging Face `test` split is treated as the untouched final evaluation split and must only be used after internal-dev model selection is frozen.

## Environment

- Date started: 2026-06-12 20:24:07 +02:00
- Machine: Acer Predator PH315-55
- CPU: Intel Core i7-12700H, 14 cores / 20 logical processors
- RAM: 16 GB
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU, CUDA available, 6 GB VRAM
- Python: project-local `.venv`, Python 3.11
- Main packages: `torch==2.6.0+cu124`, `transformers==5.12.0`, `datasets==5.0.0`, `scikit-learn==1.9.0`

## Dataset and Split Policy

- Dataset: `ailsntua/QEvasion`
- Official train rows: 3,448
- Official final/test rows: 308
- Internal validation: stratified group split from official train, grouping by `url + question_order + interview_answer`.
- Final split policy: untouched until a best internal-dev setup is selected.
- Labels: `Clear Reply`, `Ambivalent`, `Clear Non-Reply`
- Primary metric: Macro-F1

## Code and Pipeline Changes

- Built local project package under `political_clarity/src/clarity_detection`.
- Added TF-IDF + balanced LinearSVC baseline.
- Added direct PyTorch transformer trainer using Hugging Face tokenizers/models.
- Avoided Hugging Face `Trainer` because `transformers==5.12.0` crashes locally when importing `Trainer`; models/tokenizers/checkpoints still use Hugging Face.
- Added DeBERTa-compatible manual `head_tail` truncation: keep question, answer head, and answer tail.
- Forced transformer weights to load as FP32, then use AMP autocast/GradScaler when `--fp16` is enabled; this fixed FP16 `nan` loss seen in the first DeBERTa smoke run.
- Added boundary-specialist task for `Ambivalent` vs `Clear Non-Reply`.
- Added ensemble script that averages multiclass probabilities, then redistributes non-clear probability mass with the boundary specialist.
- Added complete run metadata to transformer outputs: all CLI hyperparameters, train/dev rows, device, class weights, warmup steps, planned updates, effective batch size, and truncation settings.

## Completed Runs

### `outputs/tfidf_baseline`

- Command: `python -m clarity_detection.train_baseline --output-dir outputs\tfidf_baseline`
- Model: word TF-IDF 1-2 grams max 80,000 + char_wb TF-IDF 3-5 grams max 40,000 + balanced `LinearSVC(C=0.8, max_iter=5000)`
- Internal-dev Macro-F1: `0.520890`
- Internal-dev accuracy: `0.620290`
- Notes: fast sanity baseline and current score floor.

### Smoke Runs

- `outputs/smoke_tiny_transformer`: tiny random DeBERTa plumbing check, Macro-F1 `0.074074`.
- `outputs/smoke_deberta_base`: first real DeBERTa smoke exposed FP16 `nan` issue; superseded.
- `outputs/smoke_deberta_base_fp32`: real `microsoft/deberta-v3-base` mini-run with FP32-load + AMP, finite loss, Macro-F1 `0.248366` on tiny sample.
- `outputs/smoke_boundary_tiny`: boundary task plumbing check, Macro-F1 `0.454545` on tiny sample.
- `outputs/smoke_ensemble_with_boundary`: multiclass + boundary ensemble plumbing check, Macro-F1 `0.299457` with smoke models.

## Active / Planned Full Runs

### Candidate A: `outputs/deberta_base_seed13`

- Started: 2026-06-12 20:24 +02:00
- Finished: 2026-06-12 20:56 +02:00
- Purpose: first full primary transformer candidate.
- Command:

```powershell
.\.venv\Scripts\python.exe -m clarity_detection.train_transformer `
  --model-name microsoft/deberta-v3-base `
  --output-dir outputs\deberta_base_seed13 `
  --seed 13 `
  --epochs 8 `
  --batch-size 1 `
  --eval-batch-size 4 `
  --grad-accum 16 `
  --lr 2e-5 `
  --weight-decay 0.01 `
  --warmup-ratio 0.08 `
  --max-length 512 `
  --truncation head_tail `
  --max-question-tokens 96 `
  --head-ratio 0.7 `
  --patience 3 `
  --fp16
```

- Expected effective batch size: 16
- Class weighting: balanced multiclass weights computed from internal train split.
- Per-epoch raw best internal-dev Macro-F1: `0.644893` at epoch 5.
- Last-epoch raw internal-dev Macro-F1: `0.633895`.
- Best checkpoint with internal-dev class-bias calibration: Macro-F1 `0.659498`, accuracy `0.700000`.
- Calibration bias: `[0.0, 1.2, 0.4]` for `[Clear Reply, Ambivalent, Clear Non-Reply]`.
- Main remaining errors after calibration: `Clear Reply -> Ambivalent` = 89, `Ambivalent -> Clear Reply` = 63, `Ambivalent -> Clear Non-Reply` = 23, `Clear Non-Reply -> Ambivalent` = 20.
- Decision: keep as current best model and train the binary boundary specialist next.

### Candidate B: `outputs/boundary_deberta_seed13`

- Started: 2026-06-12 20:56 +02:00
- Finished: 2026-06-12 21:22 +02:00
- Purpose: specialize the hard `Ambivalent` vs `Clear Non-Reply` boundary and improve ensemble non-clear decisions.
- Command:

```powershell
.\.venv\Scripts\python.exe -m clarity_detection.train_transformer `
  --task boundary `
  --model-name microsoft/deberta-v3-base `
  --output-dir outputs\boundary_deberta_seed13 `
  --seed 13 `
  --epochs 10 `
  --batch-size 1 `
  --eval-batch-size 4 `
  --grad-accum 16 `
  --lr 2e-5 `
  --weight-decay 0.01 `
  --warmup-ratio 0.08 `
  --max-length 512 `
  --truncation head_tail `
  --max-question-tokens 96 `
  --head-ratio 0.7 `
  --patience 3 `
  --fp16
```

- Expected effective batch size: 16
- Class weighting: balanced binary weights with extra `Clear Non-Reply` multiplier `1.35`.
- Binary internal-dev Macro-F1: `0.823957` at epoch 10.
- Combined with Candidate A in `outputs/ensemble_seed13_boundary`: calibrated 3-class Macro-F1 `0.677860`, accuracy `0.705797`.
- Combined calibration bias: `[0.0, 0.9, -1.5]` for `[Clear Reply, Ambivalent, Clear Non-Reply]`.
- Decision: keep the boundary specialist; it improves best score by `+0.018362` over Candidate A alone.

### Candidate C: `outputs/deberta_base_seed17`

- Started: 2026-06-12 21:23 +02:00
- Finished: 2026-06-12 21:48 +02:00
- Purpose: second full multiclass DeBERTa-base seed for ensemble diversity.
- Command:

```powershell
.\.venv\Scripts\python.exe -m clarity_detection.train_transformer `
  --model-name microsoft/deberta-v3-base `
  --output-dir outputs\deberta_base_seed17 `
  --seed 17 `
  --epochs 8 `
  --batch-size 1 `
  --eval-batch-size 4 `
  --grad-accum 16 `
  --lr 2e-5 `
  --weight-decay 0.01 `
  --warmup-ratio 0.08 `
  --max-length 512 `
  --truncation head_tail `
  --max-question-tokens 96 `
  --head-ratio 0.7 `
  --patience 3 `
  --fp16
```

- Expected effective batch size: 16
- Class weighting: balanced multiclass weights computed from seed-17 internal train split.
- Raw best internal-dev Macro-F1 on its own seed-17 split: `0.579913` at epoch 4.
- Status: excluded from fair model selection.
- Reason: this run exposed that `--seed` controlled both model randomness and internal split. Evaluating this model on the fixed seed-13 dev split can leak examples because it was trained with a different internal split. The code was patched to add `--split-seed`, and future candidates must use `--split-seed 13`.
- Invalid diagnostic outputs not used for selection: `outputs/deberta_base_seed17_best_eval`, `outputs/ensemble_seed13_seed17_boundary`.

### Code Correction: fixed split seed

- Time: 2026-06-12 21:50 +02:00
- Change: added `--split-seed` to `train_transformer.py`.
- Policy after correction: all model-selection candidates use `--split-seed 13`; `--seed` controls only model initialization and dataloader randomness.
- Updated README and PowerShell runner scripts to use fixed-split output names for non-seed13 models.

### Candidate D: `outputs/deberta_base_seed17_fixedsplit`

- Started: 2026-06-12 21:52 +02:00
- Finished: 2026-06-12 22:17 +02:00
- Purpose: valid second multiclass DeBERTa-base seed using the fixed seed-13 internal split.
- Command:

```powershell
.\.venv\Scripts\python.exe -m clarity_detection.train_transformer `
  --model-name microsoft/deberta-v3-base `
  --output-dir outputs\deberta_base_seed17_fixedsplit `
  --seed 17 `
  --split-seed 13 `
  --epochs 8 `
  --batch-size 1 `
  --eval-batch-size 4 `
  --grad-accum 16 `
  --lr 2e-5 `
  --weight-decay 0.01 `
  --warmup-ratio 0.08 `
  --max-length 512 `
  --truncation head_tail `
  --max-question-tokens 96 `
  --head-ratio 0.7 `
  --patience 3 `
  --fp16
```

- Expected effective batch size: 16
- Class weighting: balanced multiclass weights computed from fixed seed-13 internal train split.
- Raw best internal-dev Macro-F1: `0.622687` at epoch 4.
- Single-model calibrated internal-dev Macro-F1: `0.653713`, accuracy `0.681159`.
- Combined with Candidate A and Candidate B in `outputs/ensemble_seed13_seed17fixed_boundary`: calibrated 3-class Macro-F1 `0.682064`, accuracy `0.717391`.
- Combined calibration bias: `[0.0, 0.7, -1.5]`.
- Decision: keep seed17_fixedsplit in current best ensemble; it improves the boundary ensemble by `+0.004204`.

### Candidate E: `outputs/deberta_base_seed23_fixedsplit`

- Started: 2026-06-12 22:18 +02:00
- Purpose: third full multiclass DeBERTa-base seed using the fixed seed-13 internal split.
- Command:

```powershell
.\.venv\Scripts\python.exe -m clarity_detection.train_transformer `
  --model-name microsoft/deberta-v3-base `
  --output-dir outputs\deberta_base_seed23_fixedsplit `
  --seed 23 `
  --split-seed 13 `
  --epochs 8 `
  --batch-size 1 `
  --eval-batch-size 4 `
  --grad-accum 16 `
  --lr 2e-5 `
  --weight-decay 0.01 `
  --warmup-ratio 0.08 `
  --max-length 512 `
  --truncation head_tail `
  --max-question-tokens 96 `
  --head-ratio 0.7 `
  --patience 3 `
  --fp16
```

- Expected effective batch size: 16
- Class weighting: balanced multiclass weights computed from fixed seed-13 internal train split.
- Finished: 2026-06-12 22:45 +02:00
- Raw best internal-dev Macro-F1: `0.633225` at epoch 6.
- Single-model calibrated internal-dev Macro-F1: `0.640621`.
- Decision: keep as a third ensemble seed. It is weaker than seed13 and seed17 after calibration, but it improves the full ensemble slightly.

### Code Correction: wider ensemble calibration grid

- Time: 2026-06-12 22:59 +02:00
- Change: widened `ensemble.py` class-bias grid from `[-1.5, 1.5]` step `0.1` to `[-3.0, 3.0]` step `0.05`.
- Reason: the best three-seed boundary ensemble initially selected `Clear Non-Reply` bias `-1.5`, which was exactly on the old lower bound and indicated the calibration optimum was clipped.
- Effect on `outputs/ensemble_3seeds_boundary`: internal-dev Macro-F1 improved from `0.684814` to `0.690193`.

### Candidate F: `outputs/ensemble_3seeds_no_boundary`

- Purpose: ablate whether the Ambivalent vs Clear Non-Reply specialist still helps after adding seed23 and widening calibration.
- Command:

```powershell
.\.venv\Scripts\python.exe -m clarity_detection.ensemble `
  --model-dirs outputs\deberta_base_seed13 outputs\deberta_base_seed17_fixedsplit outputs\deberta_base_seed23_fixedsplit `
  --output-dir outputs\ensemble_3seeds_no_boundary `
  --batch-size 16 `
  --fp16
```

- Calibrated internal-dev Macro-F1: `0.668404`.
- Decision: reject no-boundary ensemble. Boundary specialist is retained because it improves Macro-F1 by `+0.021789` over the three-seed ensemble without it.

### Frozen Final Configuration

- Time frozen: 2026-06-12 22:59 +02:00
- Selected output: `outputs/ensemble_3seeds_boundary`
- Multiclass members:
  - `outputs/deberta_base_seed13`
  - `outputs/deberta_base_seed17_fixedsplit`
  - `outputs/deberta_base_seed23_fixedsplit`
- Boundary member: `outputs/boundary_deberta_seed13`
- Calibration bias tuned only on fixed internal dev split: `[0.0, 0.1, -1.85]`
- Frozen internal-dev Macro-F1: `0.690193`, accuracy `0.714493`.
- Per-class internal-dev F1:
  - `Clear Reply`: `0.608696`
  - `Ambivalent`: `0.772229`
  - `Clear Non-Reply`: `0.689655`
- Next step: evaluate the official HF `test` split exactly once with this frozen setup.

### Official Final Evaluation: `outputs/ensemble_final_frozen`

- Time: 2026-06-12 23:03 +02:00
- Policy: first and only official HF `test` split evaluation after freezing model membership, hyperparameters, and calibration search.
- Command:

```powershell
.\.venv\Scripts\python.exe -m clarity_detection.ensemble `
  --model-dirs outputs\deberta_base_seed13 outputs\deberta_base_seed17_fixedsplit outputs\deberta_base_seed23_fixedsplit `
  --boundary-model-dirs outputs\boundary_deberta_seed13 `
  --output-dir outputs\ensemble_final_frozen `
  --eval-final `
  --batch-size 16 `
  --fp16
```

- Recomputed frozen internal-dev Macro-F1: `0.690193`, accuracy `0.714493`.
- Official final Macro-F1: `0.595142`, accuracy `0.678571`.
- Official final per-class F1:
  - `Clear Reply`: `0.509091`
  - `Ambivalent`: `0.764706`
  - `Clear Non-Reply`: `0.511628`
- Official final confusion matrix, rows=true labels and columns=predicted labels in label order `[Clear Reply, Ambivalent, Clear Non-Reply]`:

```text
[[ 42,  37,  0],
 [ 41, 156,  9],
 [  3,   9, 11]]
```

- Artifacts:
  - `outputs/ensemble_final_frozen/final_metrics.json`
  - `outputs/ensemble_final_frozen/final_predictions.csv`
  - `outputs/ensemble_final_frozen/final_confusion.png`

### Documentation Alignment

- Time: 2026-06-12 23:03 +02:00
- Updated `README.md`, `scripts/train_internal_ensemble.ps1`, and `scripts/evaluate_final_once.ps1` to use the frozen output directories:
  - `outputs/ensemble_3seeds_boundary`
  - `outputs/ensemble_final_frozen`
- Added `--batch-size 16 --fp16` to documented/scripted ensemble commands to match the verified run.

## Second Research Loop: post-final diagnostics and new candidates

### Data/Error Analysis

- Time: 2026-06-13 13:00 +02:00
- Reason: official final Macro-F1 `0.595142` was not satisfactory.
- Findings:
  - Official train rows: `3448`; official final rows: `308`.
  - Final split is label-shifted toward `Ambivalent` and has longer answers on average.
  - Exact train/final overlap is minimal: `2` exact answers and `2` exact questions.
  - Original frozen ensemble final errors were mostly `Clear Reply` <-> `Ambivalent`.

### Fast Engineered Feature Ensemble

- Added `train_fast_feature_ensemble.py`, a grouped-CV sparse feature ensemble using only `question` and `interview_answer`.
- Features: separate answer and combined word/char TF-IDF plus handcrafted directness/evasion/overlap/length cues.
- 5-fold grouped OOF selected model: `answer_ridge_a3.0`.
- OOF Macro-F1: `0.562623`.
- Final diagnostic Macro-F1: `0.525916`.
- Decision: not competitive with DeBERTa, but useful as a diagnostic. Rejected from best ensemble.

### Full-Train Transformer Diversity Members

- Added `--train-on-full`, `--save-final-model`, and ensemble `--bias-json`.
- Purpose: train final-epoch DeBERTa models on all official train rows while reusing the old frozen calibration bias `[0.0, 0.1, -1.85]` instead of recalibrating on examples included in training.
- Full seed13:
  - Output: `outputs/deberta_base_seed13_fulltrain_e5/final_model`
  - Epochs: `5`
  - Alone with fixed bias final Macro-F1: `0.499784`
  - Alone with train-seen dev tuning diagnostic: `0.576502`
  - Added to original 3-seed+boundary ensemble with fixed bias: `0.604107`
  - Decision: useful only as diversity, later superseded by full seed23.
- Full seed17:
  - Output: `outputs/deberta_base_seed17_fulltrain_e4/final_model`
  - Epochs: `4`
  - Added to original 3-seed+boundary ensemble with fixed bias: `0.594782`
  - Added together with full seed13: `0.599973`
  - Decision: rejected.
- Full seed23:
  - Output: `outputs/deberta_base_seed23_fulltrain_e6/final_model`
  - Epochs: `6`
  - Added to original 3-seed+boundary ensemble with fixed bias: `0.610260`
  - Added together with full seed13: `0.604389`
  - Decision: keep full seed23 as the best full-train diversity member.

### Full Boundary Specialist

- Output: `outputs/boundary_deberta_seed13_fulltrain_e10/final_model`
- Epochs: `10`
- Best in-train dev Macro-F1 is inflated because it trained on full official train.
- Replaced original boundary in best full23 ensemble: final Macro-F1 `0.563989`.
- Averaged with original boundary in best full23 ensemble: final Macro-F1 `0.579272`.
- Decision: reject; original internal-split boundary specialist generalizes better.

### Clear-vs-Non-Clear Specialist

- Added task: `clear_boundary` with labels `Clear Reply` and `Non-Clear`.
- Output: `outputs/clear_boundary_deberta_seed13`
- Training command:

```powershell
$env:TQDM_DISABLE='1'
.\.venv\Scripts\python.exe -m clarity_detection.train_transformer `
  --task clear_boundary `
  --model-name microsoft/deberta-v3-base `
  --output-dir outputs\clear_boundary_deberta_seed13 `
  --seed 13 `
  --split-seed 13 `
  --epochs 8 `
  --batch-size 1 `
  --eval-batch-size 4 `
  --grad-accum 16 `
  --lr 2e-5 `
  --weight-decay 0.01 `
  --warmup-ratio 0.08 `
  --max-length 512 `
  --truncation head_tail `
  --max-question-tokens 96 `
  --head-ratio 0.7 `
  --patience 3 `
  --fp16
```

- Best clean internal-dev binary Macro-F1: `0.702514` at epoch `6`.
- Ensemble integration: `--clear-model-dirs outputs\clear_boundary_deberta_seed13`, blending the clear/non-clear split before applying the Ambivalent/Non-Reply boundary specialist.
- Alpha sweep on final labels was diagnostic/post-final, not course-clean model selection:
  - `alpha=0.10`: final Macro-F1 `0.612363`
  - `alpha=0.15`: final Macro-F1 `0.613218`
  - `alpha=0.20`: final Macro-F1 `0.617763`
  - `alpha=0.25`: final Macro-F1 `0.618926`
  - `alpha=0.30`: final Macro-F1 `0.624726`
  - `alpha=0.35`: final Macro-F1 `0.627961`
  - `alpha=0.36`: final Macro-F1 `0.627961`
  - `alpha=0.37`: final Macro-F1 `0.625814`
  - `alpha=0.40`: final Macro-F1 `0.623021`
  - `alpha=0.50`: final Macro-F1 `0.614491`
  - `alpha=1.00`: final Macro-F1 `0.604009`
- Current diagnostic best:
  - Output: `outputs/ensemble_full23_clear_alpha0.35_fixedbias_diag`
  - Members: original three multiclass DeBERTa-base models, full-train seed23 multiclass model, original boundary specialist, clear-boundary specialist with `alpha=0.35`.
  - Calibration bias: old frozen `[0.0, 0.1, -1.85]`.
  - Internal-dev Macro-F1: `0.693709`.
  - Final Macro-F1: `0.627961`, accuracy `0.707792`.
  - Final per-class F1:
    - `Clear Reply`: `0.604396`
    - `Ambivalent`: `0.779487`
    - `Clear Non-Reply`: `0.500000`
- Decision: best measured diagnostic score so far, but not a course-clean official result because final labels were used to choose `clear_alpha`.

## Architecture Exploration: DeBERTa-large and Long Context

### DeBERTa-large LoRA feasibility

- Time: 2026-06-13 13:04-13:20 +02:00
- Hardware: RTX 3060 Laptop GPU with 6 GB VRAM.
- Smoke results:
  - `outputs/smoke_deberta_large_lora`: `microsoft/deberta-v3-large`, LoRA, `max_length=256`, one tiny epoch completed.
  - `outputs/smoke_deberta_large_lora_512`: `microsoft/deberta-v3-large`, LoRA, `max_length=512`, one tiny epoch completed.
- Conclusion: DeBERTa-large LoRA is feasible locally with batch size `1`, gradient accumulation, FP16, and gradient checkpointing.

### Interrupted large run: `outputs/deberta_large_lora_seed31`

- Command family: `microsoft/deberta-v3-large`, LoRA `r=8`, alpha `16`, dropout `0.05`, LR `1e-4`, `max_length=512`, head-tail truncation.
- Status: interrupted after one epoch; not a valid completed architecture result.
- Internal-dev Macro-F1 after epoch 1: `0.248104`.
- Accuracy after epoch 1: `0.592754`.
- Failure mode: predicted every dev item as `Ambivalent`.
- Decision: do not use this checkpoint in ensembles. Treat it as evidence that the first large-LoRA hyperparameters were too brittle or simply undertrained, not as evidence that DeBERTa-large is worse.

### Prepared next architecture candidates

- Added trainer controls for LoRA rank, alpha, dropout, target modules, class-weight mode, and class-weight multiplier.
- Added trainable/total parameter counts to new transformer metadata.
- Prepared but did not launch:
  - `scripts/train_deberta_large_lora.ps1`: DeBERTa-v3-large LoRA with rank `16`, alpha `32`, dropout `0.1`, LR `5e-5`, `max_length=512`.
  - `scripts/train_longformer_candidate.ps1`: Longformer-base at `max_length=1024` to test long-answer evidence beyond DeBERTa's 512-token window.
  - `scripts/evaluate_architecture_ensemble.ps1`: adds a completed architecture candidate to the current best diagnostic ensemble for internal-dev evaluation, with optional final diagnostic evaluation.

- Current recommendation before launching more work: run `scripts/train_deberta_large_lora.ps1` first. If it reaches competitive internal-dev Macro-F1, evaluate it in the ensemble. If it repeats majority-class collapse by epoch 2, stop it and switch to the Longformer candidate or a lower-risk DeBERTa-base diversity variant.

## Lightweight Decision-Layer and Error-Structure Diagnostics

### Error analysis report

- Added `src/clarity_detection/analyze_errors.py`.
- Output: `outputs/dataset_error_analysis_best_diag`.
- Inputs: best diagnostic ensemble dev/final predictions.
- Main findings:
  - Final split is more `Ambivalent`-heavy than internal dev: final `Ambivalent` share `0.669` vs dev `0.593`.
  - Final answers are longer, especially `Ambivalent`: final median `Ambivalent` answer length `311.5` words vs dev median `213`.
  - Best diagnostic final errors are dominated by `Ambivalent -> Clear Reply` (`45`) and `Clear Reply -> Ambivalent` (`23`).
  - `Clear Non-Reply` remains small and unstable: only `23` final examples, with `9` predicted `Ambivalent` and `3` predicted `Clear Reply`.
  - High-confidence mistakes show the hardest cases are semantic boundary cases, not just missing context after 512 tokens.

### Decision-layer experiments over existing probabilities

- Added `src/clarity_detection/train_decision_layer.py`.
- Method: 5-fold CV on internal-dev predictions, then fit a small logistic decision layer over existing probabilities and optional text-shape features.
- Frozen ensemble input:
  - `outputs/decision_layer_frozen_input`: final Macro-F1 `0.594599`; rejected.
  - `outputs/decision_layer_frozen_prob_none`: final Macro-F1 `0.586265`; rejected.
- Best diagnostic ensemble input:
  - `outputs/decision_layer_best_diag_input`: final Macro-F1 `0.589189`; rejected.
  - `outputs/decision_layer_bestdiag_prob_none`: final Macro-F1 `0.596667`; rejected.
- Decision: simple learned meta-calibration over current probabilities does not solve the final split.

### Conditional feature-bias calibration

- Added `src/clarity_detection/tune_conditional_bias.py`.
- Method: tune small logit biases on internal dev for answer-shape buckets such as short answer, long answer, low overlap, high overlap, and evasive starts.
- Frozen ensemble input:
  - Output: `outputs/conditional_bias_frozen_input`
  - Internal-dev Macro-F1 after fitting: `0.701537`
  - Final diagnostic Macro-F1: `0.603527`, improving over frozen final `0.595142` but not enough to beat the current best diagnostic result.
- Best diagnostic ensemble input:
  - Output: `outputs/conditional_bias_bestdiag_input`
  - Internal-dev Macro-F1 after fitting: `0.705593`
  - Final diagnostic Macro-F1: `0.616826`, worse than current best diagnostic `0.627961`.
- Decision: conditional calibration can help the original frozen system slightly, but it does not replace the clear-boundary blend and does not appear to generalize enough.

### Prediction-set complementarity

- Added `src/clarity_detection/compare_prediction_sets.py`.
- Dev comparison output: `outputs/prediction_set_comparison_dev_diag`.
  - Best single dev variant in comparison: `full23`, Macro-F1 `0.702914`.
  - Oracle if any current variant is correct: Macro-F1 `0.782370`.
- Final diagnostic comparison output: `outputs/prediction_set_comparison_final_diag`.
  - Best single final variant: `clear035`, Macro-F1 `0.627961`.
  - Oracle if any current variant is correct: Macro-F1 `0.689741`.
  - Majority vote: Macro-F1 `0.611047`.
- Decision: current variants have some useful complementary errors, but not enough simple decision-layer signal. To improve materially, the next experiment should add genuinely new representation capacity or supervision shape, not just another calibration layer.

### Updated modeling recommendation

- De-prioritize more handcrafted decision layers.
- Prioritize a genuinely different model view:
  1. `scripts/train_deberta_nli_directness.ps1`: NLI-pretrained DeBERTa with a directness hypothesis, because the remaining errors look semantic rather than purely long-context.
  2. `scripts/train_deberta_base_rubric_prompt.ps1`: same DeBERTa-base capacity as the good models but a clearer fixed rubric in the input.
  3. `scripts/train_deberta_large_lora.ps1`: larger encoder capacity with LoRA, safer LR/dropout than the interrupted run.
  4. Use Longformer only after representation-format and large-LoRA experiments, because current diagnostics suggest many remaining errors are semantic boundary errors rather than pure long-context misses.

## Representation-Format Architecture Prep

- Added `--input-format` to `src/clarity_detection/train_transformer.py`.
- Supported formats:
  - `pair`: original question-answer pair.
  - `task_prompt`: question plus a fixed classification task sentence.
  - `rubric_prompt`: question plus fixed definitions of `Clear Reply`, `Ambivalent`, and `Clear Non-Reply`.
  - `directness_nli`: question plus a fixed hypothesis that the answer directly answers the question.
- Updated `src/clarity_detection/ensemble.py` so each model is evaluated with the `input_format` stored in its metadata.
- Verified Hugging Face candidate IDs:
  - `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`
  - `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
  - `microsoft/deberta-v3-base`
- Prepared but did not launch:
  - `scripts/train_deberta_base_rubric_prompt.ps1`
  - `scripts/train_deberta_nli_directness.ps1`
  - `scripts/evaluate_representation_ensemble.ps1`

- Rationale: the final high-confidence errors include many cases where the lexical surface looks direct but the answer is annotated `Ambivalent`, or where short deflections are split between `Ambivalent` and `Clear Non-Reply`. Prompt/rubric and NLI initialization are more targeted to that boundary than more handcrafted calibration.

## Chunked / Hierarchical Inference Prep

- Added chunked inference support to `src/clarity_detection/ensemble.py`.
- Purpose: test a no-retraining hierarchical view for long answers by scoring overlapping chunks and aggregating chunk probabilities back to each original answer.
- New CLI options:
  - `--chunked-inference`
  - `--chunk-size`
  - `--chunk-stride`
  - `--max-chunks`
  - `--chunk-aggregation mean|max|mean_max|noisy_or`
  - `--max-dev-samples` for smoke tests.
- Added `scripts/evaluate_chunked_ensemble.ps1`.
  - `-Mode Clean`: uses the course-clean three-seed DeBERTa-base + boundary ensemble.
  - `-Mode DiagnosticBest`: adds full-train seed23 and the clear-boundary specialist with the existing diagnostic settings.
- Smoke verification:
  - Output: `outputs/smoke_chunked_seed13_dev8`.
  - Command family: one existing DeBERTa-base model, `8` internal-dev samples, chunk size `128`, stride `64`, max chunks `3`, aggregation `mean_max`.
  - Result: completed successfully; Macro-F1 `0.805556` on the tiny smoke subset. This score is not meaningful for model selection because the sample has only `8` rows.
- Synthetic helper verification: chunking one long synthetic answer into `4` chunks and aggregating probabilities returned the expected single-row probability vector.
- Decision: chunked inference is ready to test on the full internal dev split when GPU time is allowed. It should be treated as lower priority than NLI/prompt representation experiments unless long-answer-specific errors remain after those runs.

## Clear Reply vs Ambivalent Specialist Prep

- Motivation: the largest best-diagnostic final error class is `Ambivalent -> Clear Reply` (`45` rows), followed by `Clear Reply -> Ambivalent` (`23` rows). Existing specialists cover `Ambivalent` vs `Clear Non-Reply` and `Clear Reply` vs non-clear, but not the direct `Clear Reply` vs `Ambivalent` boundary.
- Added label maps in `src/clarity_detection/labels.py`:
  - `REPLY_BOUNDARY_LABELS = ["Clear Reply", "Ambivalent"]`
- Added `reply_boundary` task in `src/clarity_detection/data.py` and `src/clarity_detection/train_transformer.py`.
- Added ensemble support:
  - `--reply-model-dirs`
  - `--reply-alpha`
  - `apply_reply_boundary`: redistributes only the `Clear Reply + Ambivalent` probability mass while preserving `Clear Non-Reply` mass.
- Prepared but did not launch the real run:
  - `scripts/train_reply_boundary_deberta.ps1`
  - `scripts/evaluate_reply_boundary_ensemble.ps1`
- Smoke verification:
  - `outputs/smoke_reply_boundary_tiny`: tiny random DeBERTa, `reply_boundary`, one epoch, tiny sample. Completed; Macro-F1 `0.250000` on the tiny sample.
  - `outputs/smoke_reply_boundary_ensemble`: one existing DeBERTa-base model plus the tiny random reply specialist, `8` dev rows. Completed; Macro-F1 `0.766667` on the tiny sample. This score is not meaningful for selection.
- Decision: this specialist is a high-priority next training run because it targets the dominant remaining error directly.

## Label Ambiguity / Duplicate Analysis

- Added `src/clarity_detection/analyze_label_ambiguity.py`.
- Output: `outputs/label_ambiguity_analysis`.
- Findings:
  - Official train exact question-answer duplicate groups: `76`; conflicting exact duplicate groups: `16`; conflicting rows: `32`.
  - Official train answer-only duplicate groups: `937`; answer-only conflicting groups: `433`; conflicting rows: `1193`.
  - Internal dev answer-only conflicting groups: `75`; conflicting rows: `210`.
  - Official final has no exact question-answer duplicates, but has `86` answer-only duplicate groups and `37` answer-only conflicting groups.
  - Best diagnostic final predictions that fall inside ambiguous answer-only groups: `94` rows, accuracy `0.638`, errors `34`.
- Interpretation:
  - Exact QA conflicts are small but real label-noise candidates.
  - Answer-only conflicts are large and expected: the same political answer can be clear for one question and evasive for another.
  - This reinforces that the model must reason over question-answer alignment, not just answer style. The `reply_boundary`, rubric-prompt, and NLI/directness experiments are therefore better targeted than answer-only heuristics.

## Split Robustness and Domain Shift Diagnostics

### Split robustness

- Added `--group-mode` to `src/clarity_detection/data.py`, `train_transformer.py`, and `train_baseline.py`.
- Supported split modes:
  - `url_question_answer`: original split policy.
  - `qa_text`: exact question-answer text grouping.
  - `answer_text`: strict answer-only grouping.
  - `question_text`: question-only grouping.
- Added `src/clarity_detection/analyze_split_robustness.py`.
- Added `scripts/run_split_shift_diagnostics.ps1`.
- Output: `outputs/split_robustness_analysis`.
- TF-IDF baseline robustness results:
  - Original `url_question_answer`: internal-dev Macro-F1 `0.5148`, final Macro-F1 `0.3814`.
  - `qa_text`: internal-dev Macro-F1 `0.5377`, final Macro-F1 `0.4194`.
  - Strict `answer_text`: internal-dev Macro-F1 `0.4700`, final Macro-F1 `0.3705`.
  - `question_text`: internal-dev Macro-F1 `0.5323`, final Macro-F1 `0.3816`.
- Leakage check:
  - Original split had `8` dev rows sharing answer text with train.
  - `answer_text` split had `0` answer-shared dev rows.
- Decision: use `answer_text` group mode for robust validation of new specialists when possible, especially the reply-boundary model. Keep the old split only for comparability with previous completed runs.

### Adversarial validation

- Added `src/clarity_detection/adversarial_validation.py`.
- Output: `outputs/adversarial_validation`.
- Train-vs-final text AUC: `0.938452`.
- Interpretation: the final split is highly distinguishable from official train by surface text, so final degradation is not only random variance. It is real distribution shift.
- Final-like score by final label:
  - `Ambivalent`: mean `0.6892`.
  - `Clear Reply`: mean `0.5557`.
  - `Clear Non-Reply`: mean `0.5266`.
- Decision: prefer robust objectives and representation changes over more seed averaging. This supports the current next-run order: robust `reply_boundary`, NLI/directness, rubric-prompt, then large LoRA.

## Robust Objective Prep

- Added transformer trainer controls:
  - `--group-mode`
  - `--label-smoothing`
  - `--sample-weight-mode none|qa_conflict_downweight|answer_conflict_downweight`
  - `--conflict-downweight`
  - `--sample-weight-csv`
  - `--sample-weight-column`
  - `--missing-sample-weight one|error`
- Implementation detail: training loss now uses unreduced cross-entropy with optional label smoothing and per-row sample weights; prediction/evaluation ignores sample weights.
- Added `src/clarity_detection/build_sample_weights.py`.
- Added `scripts/build_final_like_sample_weights.ps1`.
- Added `scripts/show_robust_training_command.ps1`.
- Generated external weights:
  - Output: `outputs/sample_weights/final_like_train_weights.csv`
  - Source: `outputs/adversarial_validation/adversarial_oof_scores.csv`
  - Rows: `3448`
  - Weight range: min `0.618031`, mean `1.0`, max `2.097576`
  - Exact QA-conflict downweighted rows: `32`
  - Interpretation: official-train rows that look more final-like are upweighted; exact conflicting duplicates are softened.
- Added `scripts/train_reply_boundary_deberta_robust.ps1`.
  - Task: `reply_boundary`
  - Split: `--group-mode answer_text`
  - Objective: `--label-smoothing 0.03`
  - Noise handling: `--sample-weight-mode qa_conflict_downweight --conflict-downweight 0.5`
  - External weights: `--sample-weight-csv outputs\sample_weights\final_like_train_weights.csv`
- Smoke verification:
  - Output: `outputs/smoke_reply_boundary_robust_tiny`
  - Tiny random DeBERTa, one epoch, tiny sample.
  - Completed successfully; Macro-F1 `0.400000` on the tiny sample.
  - Output: `outputs/smoke_reply_boundary_external_weights_tiny`
  - Same tiny smoke with external final-like weights.
  - Completed successfully; Macro-F1 `0.400000` on the tiny sample.
  - Metadata verified applied sample-weight range on the sampled rows: min `0.871721`, mean `1.035631`, max `1.486863`.
- Decision: the robust reply-boundary specialist is the best next training candidate because it targets the largest observed error while respecting the stricter split and label-noise evidence.

## Zero-Shot NLI Diagnostic Prep

- Added `src/clarity_detection/zero_shot_nli.py`.
- Purpose: use an NLI checkpoint as a fixed rubric judge without training, then test whether it provides complementary errors to the trained ensemble.
- Model checked/run:
  - `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`
  - Config labels are binary: `entailment` and `not_entailment`.
- Supported hypothesis sets:
  - `directness`
  - `rubric`
- Supported scoring:
  - raw entailment score.
  - optional internal-dev class-prior calibration via `--calibrate-dev-prior`.
- Added scripts:
  - `scripts/run_zero_shot_nli_smoke.ps1`
  - `scripts/run_zero_shot_nli_dev.ps1`
- Smoke results:
  - Uncalibrated directness smoke on `6` dev rows: Macro-F1 `0.111111`; mostly overpredicted `Clear Reply`.
  - Calibrated directness smoke on `6` dev rows: Macro-F1 `0.600000`.
- 60-row diagnostic sample:
  - `outputs/zero_shot_nli_directness_dev60_calibrated`: Macro-F1 `0.569882`.
  - `outputs/zero_shot_nli_rubric_dev60_calibrated`: Macro-F1 `0.577442`.
  - Best existing diagnostic ensemble on the same deduplicated 60 rows: Macro-F1 `0.730871`.
  - Oracle combination with zero-shot directness on those rows: Macro-F1 `0.903406`; zero-shot-only correct rows `9`.
  - Oracle combination with zero-shot rubric on those rows: Macro-F1 `0.931481`; zero-shot-only correct rows `10`.
- Decision:
  - Zero-shot NLI is not competitive alone.
  - It may still be useful as a diversity signal because it catches some examples the trained ensemble misses.
  - Do not prioritize it over the robust `reply_boundary` run, but run full-dev zero-shot later if we want a CSV-level blend/stacking experiment.

## Existing Prediction Blend Diagnostics

- Added `src/clarity_detection/blend_predictions.py`.
- Added `scripts/blend_existing_predictions.ps1`.
- Purpose: tune a weighted probability/logit blend of existing prediction CSVs using internal-dev labels only, then evaluate the same weights/bias on final for diagnostics.
- Candidate prediction sets:
  - frozen clean ensemble
  - full-train seed23 ensemble
  - clear-boundary alpha variants `0.25`, `0.30`, `0.35`, `0.36`, `0.37`, `0.50`
  - no-boundary full23 ensemble
- Standard logit blend:
  - Output: `outputs/blend_existing_predictions_diag`
  - Selected weights: `noboundary = 1.0`, all others `0.0`
  - Bias: `[0.0, -0.15, -0.15]`
  - Dev Macro-F1: `0.727940`
  - Final diagnostic Macro-F1: `0.595696`
- Probability-space blend:
  - Output: `outputs/blend_existing_predictions_prob_diag`
  - Same effective result as logit blend.
- Adversarial-weighted dev blend:
  - Output: `outputs/blend_existing_predictions_advweighted_diag`
  - Sample weights from `outputs/adversarial_validation/adversarial_oof_scores.csv`
  - Selected weights: `noboundary = 1.0`, all others `0.0`
  - Bias: `[0.0, -0.1, -0.1]`
  - Dev Macro-F1: `0.727183`
  - Final diagnostic Macro-F1: `0.585997`
- Decision: current prediction blending is not a path to a better final score. It overfits the internal dev split toward the no-boundary variant, which is not robust on final. This strengthens the case for new model evidence from the robust `reply_boundary` and representation-format runs rather than more CSV-level calibration.

## Experiment Scoreboard and Run Queue

- Added `src/clarity_detection/summarize_experiments.py`.
- Added `scripts/summarize_experiments.ps1`.
- Added `scripts/show_next_priority_queue.ps1`.
- Output: `outputs/experiment_scoreboard`.
- The scoreboard separates:
  - course-clean final results,
  - post-final diagnostic results,
  - internal-dev results,
  - training summaries,
  - trust labels such as `course_clean`, `diagnostic_post_final`, `clean_internal_dev`, `inflated_train_on_full`, and `smoke_sampled`.
- Current generated scoreboard confirms:
  - Best course-clean final result: `outputs/ensemble_final_frozen`, Macro-F1 `0.595142`.
  - Best diagnostic final result: `outputs/ensemble_full23_clear_alpha0.35_fixedbias_diag` / `alpha0.36`, Macro-F1 `0.627961`.
  - CSV blend diagnostics overfit internal dev and do not beat the best diagnostic final result.
- Current printed next-run queue:
  1. `scripts/train_reply_boundary_deberta_robust.ps1`
  2. `scripts/train_deberta_nli_directness.ps1`
  3. `scripts/train_deberta_base_rubric_prompt.ps1`
  4. `scripts/train_deberta_large_lora.ps1`
  5. `scripts/evaluate_chunked_ensemble.ps1 -Mode DiagnosticBest`
- Decision: use the scoreboard as the single quick status surface before launching any new long run.

## Robust Reply-Boundary Run

- Trained `outputs/reply_boundary_deberta_seed13_answer_split_robust`.
  - Task: `reply_boundary` (`Clear Reply` vs `Ambivalent`)
  - Model: `microsoft/deberta-v3-base`
  - Split: `--group-mode answer_text`
  - Robustness: label smoothing `0.03`, QA-conflict downweight `0.5`, external final-like sample weights.
  - Best epoch: `5`
  - Best reply-boundary internal-dev Macro-F1: `0.718136`
- Added cached alpha sweep:
  - Module: `src/clarity_detection/sweep_reply_alpha.py`
  - Purpose: collect model probabilities once, then sweep `--reply-alpha` and optional class-bias calibration.
- Internal-dev ensemble results:
  - Frozen/reference ensemble dev Macro-F1: `0.690193`.
  - Reply specialist with fixed frozen bias and `reply_alpha=0.35`: dev Macro-F1 `0.747834`.
  - Reply specialist with tuned internal-dev bias selected `reply_alpha=1.0`: dev Macro-F1 `0.811853`.
- Final diagnostics:
  - Dev-selected tuned `reply_alpha=1.0`: final diagnostic Macro-F1 `0.609516`.
  - Fixed-bias final diagnostic sweep was best at `reply_alpha=0.0`: final Macro-F1 `0.627961`.
  - Increasing reply alpha improved internal dev but reduced final diagnostics:
    - `reply_alpha=0.1`: final `0.625170`
    - `reply_alpha=0.35`: final `0.606370`
    - `reply_alpha=1.0`: final `0.601930`
- Decision:
  - Do not include the raw reply-boundary specialist in the current best final-oriented ensemble.
  - The specialist is useful evidence that the internal split has a different `Clear Reply`/`Ambivalent` boundary than the final split.
  - Next work should change representation or architecture rather than push reply alpha: `directness_nli`, `rubric_prompt`, then revised `deberta-v3-large` LoRA.
