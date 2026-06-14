# QEvasion Experiment Scoreboard

This report is generated from current files under `outputs/`.

## Best Course-Clean Final Runs

| name | kind | cleanliness | final_macro_f1 | dev_macro_f1 | task | model_name | trust_for_selection | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ensemble_final_frozen | ensemble | course_clean_final_once | 0.595142 | 0.690193 |  |  | course_clean |  |

## Best Diagnostic Final Runs

| name | kind | cleanliness | final_macro_f1 | dev_macro_f1 | task | model_name | trust_for_selection | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ensemble_full23_clear_alpha0.35_fixedbias_diag | ensemble | diagnostic | 0.627961 | 0.693709 |  |  | diagnostic_post_final |  |
| ensemble_full23_clear_alpha0.36_fixedbias_diag | ensemble | diagnostic | 0.627961 | 0.690829 |  |  | diagnostic_post_final |  |
| ensemble_reply_fixed_final_diag_alpha0 | ensemble | diagnostic | 0.627961 | 0.693709 |  |  | diagnostic_post_final |  |
| ensemble_full23_clear_alpha0.37_fixedbias_diag | ensemble | diagnostic | 0.625814 | 0.693709 |  |  | diagnostic_post_final |  |
| ensemble_reply_fixed_final_diag_alpha0p1 | ensemble | diagnostic | 0.625170 | 0.708771 |  |  | diagnostic_post_final |  |
| ensemble_full23_clear_alpha0.33_fixedbias_diag | ensemble | diagnostic | 0.624726 | 0.693709 |  |  | diagnostic_post_final |  |
| ensemble_full23_clear_alpha0.3_fixedbias_diag | ensemble | diagnostic | 0.624726 | 0.695142 |  |  | diagnostic_post_final |  |
| ensemble_full23_clear_alpha0.4_fixedbias_diag | ensemble | diagnostic | 0.623021 | 0.693905 |  |  | diagnostic_post_final |  |
| ensemble_reply_fixed_final_diag_alpha0p05 | ensemble | diagnostic | 0.621905 | 0.697142 |  |  | diagnostic_post_final |  |
| ensemble_reply_fixed_final_diag_alpha0p2 | ensemble | diagnostic | 0.621212 | 0.718881 |  |  | diagnostic_post_final |  |
| ensemble_reply_fixed_final_diag_alpha0p3 | ensemble | diagnostic | 0.620025 | 0.740861 |  |  | diagnostic_post_final |  |
| ensemble_full23_clear_alpha0.25_fixedbias_diag | ensemble | diagnostic | 0.618926 | 0.696803 |  |  | diagnostic_post_final |  |
| ensemble_full23_clear_alpha0.2_fixedbias_diag | ensemble | diagnostic | 0.617763 | 0.697220 |  |  | diagnostic_post_final |  |
| conditional_bias_bestdiag_input | decision_layer | diagnostic | 0.616826 |  |  |  | diagnostic_post_final |  |
| ensemble_full23_clear_alpha0.5_fixedbias_diag | ensemble | diagnostic | 0.614491 | 0.689780 |  |  | diagnostic_post_final |  |
| ensemble_full23_clear_alpha0.15_fixedbias_diag | ensemble | diagnostic | 0.613218 | 0.703222 |  |  | diagnostic_post_final |  |
| ensemble_full23_clear_alpha0.1_fixedbias_diag | ensemble | diagnostic | 0.612363 | 0.702034 |  |  | diagnostic_post_final |  |
| ensemble_3seeds_plus_full23_fixedbias_diag | ensemble | diagnostic | 0.610260 | 0.702914 |  |  | diagnostic_post_final |  |
| ensemble_reply_tuned_final_diag_alpha1 | ensemble | diagnostic | 0.609516 | 0.811853 |  |  | diagnostic_post_final |  |
| ensemble_reply_fixed_final_diag_alpha0p4 | ensemble | diagnostic | 0.607713 | 0.755611 |  |  | diagnostic_post_final |  |

## Best Internal Dev Runs

| name | kind | cleanliness | dev_macro_f1 | final_macro_f1 | task | model_name | trust_for_selection | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| boundary_deberta_seed13_fulltrain_e10 | model | internal_dev_only | 0.923689 |  | boundary | microsoft/deberta-v3-base | inflated_train_on_full |  |
| deberta_base_seed23_fulltrain_e6 | model | internal_dev_only | 0.839603 |  | multiclass | microsoft/deberta-v3-base | inflated_train_on_full |  |
| deberta_base_seed13_fulltrain_e5_tuned_diag | model | diagnostic | 0.834451 | 0.576502 |  |  | diagnostic_post_final |  |
| boundary_deberta_seed13 | model | internal_dev_only | 0.823957 |  | boundary | microsoft/deberta-v3-base | clean_internal_dev |  |
| deberta_base_seed13_fulltrain_e5 | model | internal_dev_only | 0.813398 |  | multiclass | microsoft/deberta-v3-base | inflated_train_on_full |  |
| ensemble_reply_tuned_alpha1 | ensemble | internal_dev_only | 0.811853 |  |  |  | clean_internal_dev |  |
| ensemble_reply_tuned_final_diag_alpha1 | ensemble | diagnostic | 0.811853 | 0.609516 |  |  | diagnostic_post_final |  |
| ensemble_reply_tuned_alpha0p75 | ensemble | internal_dev_only | 0.809020 |  |  |  | clean_internal_dev |  |
| deberta_base_seed13_fulltrain_e5_fixedbias_eval | model | internal_dev_only | 0.806181 | 0.499784 |  |  | unknown |  |
| smoke_chunked_seed13_dev8 | smoke | smoke | 0.805556 |  |  |  | smoke_sampled |  |
| ensemble_reply_tuned_alpha0p6 | ensemble | internal_dev_only | 0.800935 |  |  |  | clean_internal_dev |  |
| ensemble_reply_fixed_final_diag_alpha0p75 | ensemble | diagnostic | 0.797971 | 0.592725 |  |  | diagnostic_post_final |  |
| ensemble_reply_fixed_final_diag_alpha1 | ensemble | diagnostic | 0.795645 | 0.601930 |  |  | diagnostic_post_final |  |
| ensemble_reply_fixed_final_diag_alpha0p6 | ensemble | diagnostic | 0.795622 | 0.598445 |  |  | diagnostic_post_final |  |
| ensemble_reply_tuned_alpha0p5 | ensemble | internal_dev_only | 0.795127 |  |  |  | clean_internal_dev |  |
| ensemble_reply_fixed_final_diag_alpha0p5 | ensemble | diagnostic | 0.785781 | 0.599973 |  |  | diagnostic_post_final |  |
| deberta_base_seed17_best_eval | model | internal_dev_only | 0.783867 |  |  |  | unknown |  |
| smoke_reply_boundary_ensemble | smoke | smoke | 0.766667 |  |  |  | smoke_sampled |  |
| ensemble_reply_tuned_alpha0p4 | ensemble | internal_dev_only | 0.765535 |  |  |  | clean_internal_dev |  |
| ensemble_reply_tuned_alpha0p35 | ensemble | internal_dev_only | 0.758536 |  |  |  | clean_internal_dev |  |

## Best Trustworthy Internal Dev Runs

| name | kind | cleanliness | dev_macro_f1 | final_macro_f1 | task | model_name | trust_for_selection | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| boundary_deberta_seed13 | model | internal_dev_only | 0.823957 |  | boundary | microsoft/deberta-v3-base | clean_internal_dev |  |
| ensemble_reply_tuned_alpha1 | ensemble | internal_dev_only | 0.811853 |  |  |  | clean_internal_dev |  |
| ensemble_reply_tuned_alpha0p75 | ensemble | internal_dev_only | 0.809020 |  |  |  | clean_internal_dev |  |
| deberta_base_seed13_fulltrain_e5_fixedbias_eval | model | internal_dev_only | 0.806181 | 0.499784 |  |  | unknown |  |
| ensemble_reply_tuned_alpha0p6 | ensemble | internal_dev_only | 0.800935 |  |  |  | clean_internal_dev |  |
| ensemble_reply_tuned_alpha0p5 | ensemble | internal_dev_only | 0.795127 |  |  |  | clean_internal_dev |  |
| deberta_base_seed17_best_eval | model | internal_dev_only | 0.783867 |  |  |  | unknown |  |
| ensemble_reply_tuned_alpha0p4 | ensemble | internal_dev_only | 0.765535 |  |  |  | clean_internal_dev |  |
| ensemble_reply_tuned_alpha0p35 | ensemble | internal_dev_only | 0.758536 |  |  |  | clean_internal_dev |  |
| ensemble_reply_tuned_alpha0p3 | ensemble | internal_dev_only | 0.750044 |  |  |  | clean_internal_dev |  |
| ensemble_with_reply_boundary_alpha0.35 | ensemble | internal_dev_only | 0.747834 |  |  |  | clean_internal_dev |  |
| ensemble_reply_tuned_alpha0p2 | ensemble | internal_dev_only | 0.727633 |  |  |  | clean_internal_dev |  |
| ensemble_reply_tuned_alpha0p1 | ensemble | internal_dev_only | 0.715505 |  |  |  | clean_internal_dev |  |
| ensemble_reply_tuned_alpha0p05 | ensemble | internal_dev_only | 0.709047 |  |  |  | clean_internal_dev |  |
| ensemble_reply_tuned_alpha0 | ensemble | internal_dev_only | 0.704719 |  |  |  | clean_internal_dev |  |
| ensemble_seed13_seed17_boundary | ensemble | internal_dev_only | 0.700049 |  |  |  | clean_internal_dev |  |
| reply_boundary_deberta_seed13_answer_split_robust | model | internal_dev_only | 0.693537 |  | reply_boundary | microsoft/deberta-v3-base | clean_internal_dev |  |
| clear_boundary_deberta_seed13 | model | internal_dev_only | 0.691908 |  | clear_boundary | microsoft/deberta-v3-base | clean_internal_dev |  |
| ensemble_3seeds_boundary | ensemble | internal_dev_only | 0.690193 |  |  |  | clean_internal_dev |  |
| ensemble_seed13_seed17fixed_boundary | ensemble | internal_dev_only | 0.682064 |  |  |  | clean_internal_dev |  |

## Best Training Summaries

| name | kind | cleanliness | best_dev_macro_f1 | dev_macro_f1 | final_macro_f1 | task | model_name | trust_for_selection | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| boundary_deberta_seed13_fulltrain_e10 | model | internal_dev_only | 0.923689 | 0.923689 |  | boundary | microsoft/deberta-v3-base | inflated_train_on_full |  |
| deberta_base_seed23_fulltrain_e6 | model | internal_dev_only | 0.839603 | 0.839603 |  | multiclass | microsoft/deberta-v3-base | inflated_train_on_full |  |
| boundary_deberta_seed13 | model | internal_dev_only | 0.823957 | 0.823957 |  | boundary | microsoft/deberta-v3-base | clean_internal_dev |  |
| deberta_base_seed13_fulltrain_e5 | model | internal_dev_only | 0.813398 | 0.813398 |  | multiclass | microsoft/deberta-v3-base | inflated_train_on_full |  |
| deberta_base_seed17_fulltrain_e4 | model | internal_dev_only | 0.729774 | 0.729774 |  | multiclass | microsoft/deberta-v3-base | inflated_train_on_full |  |
| reply_boundary_deberta_seed13_answer_split_robust | model | internal_dev_only | 0.718136 | 0.693537 |  | reply_boundary | microsoft/deberta-v3-base | clean_internal_dev |  |
| clear_boundary_deberta_seed13 | model | internal_dev_only | 0.702514 | 0.691908 |  | clear_boundary | microsoft/deberta-v3-base | clean_internal_dev |  |
| deberta_base_seed13 | model | internal_dev_only | 0.644893 | 0.633895 |  | multiclass | microsoft/deberta-v3-base | clean_internal_dev |  |
| deberta_base_seed23_fixedsplit | model | internal_dev_only | 0.633225 | 0.608939 |  | multiclass | microsoft/deberta-v3-base | clean_internal_dev |  |
| deberta_base_seed17_fixedsplit | model | internal_dev_only | 0.622687 | 0.588128 |  | multiclass | microsoft/deberta-v3-base | clean_internal_dev |  |
| deberta_base_seed17 | model | internal_dev_only | 0.579913 | 0.553741 |  | multiclass | microsoft/deberta-v3-base | clean_internal_dev |  |
| smoke_boundary_tiny | smoke | smoke | 0.454545 | 0.454545 |  | boundary | hf-internal-testing/tiny-random-DebertaV2ForSequenceClassification | smoke_sampled |  |
| smoke_reply_boundary_external_weights_tiny | smoke | smoke | 0.400000 | 0.400000 |  | reply_boundary | hf-internal-testing/tiny-random-deberta-v2 | smoke_sampled |  |
| smoke_reply_boundary_robust_tiny | smoke | smoke | 0.400000 | 0.400000 |  | reply_boundary | hf-internal-testing/tiny-random-deberta-v2 | smoke_sampled |  |
| smoke_reply_boundary_tiny | smoke | smoke | 0.250000 | 0.250000 |  | reply_boundary | hf-internal-testing/tiny-random-deberta-v2 | smoke_sampled |  |
| smoke_clear_boundary_tiny | smoke | smoke | 0.250000 | 0.250000 |  | clear_boundary | hf-internal-testing/tiny-random-DebertaV2ForSequenceClassification | smoke_sampled |  |
| smoke_deberta_base_fp32 | smoke | smoke | 0.248366 | 0.248366 |  | multiclass | microsoft/deberta-v3-base | smoke_sampled |  |
| smoke_full_train_tiny | smoke | smoke | 0.245614 | 0.245614 |  | multiclass | hf-internal-testing/tiny-random-DebertaV2ForSequenceClassification | smoke_sampled |  |
| smoke_deberta_base | smoke | smoke | 0.158730 | 0.158730 |  | multiclass | microsoft/deberta-v3-base | smoke_sampled |  |
| smoke_deberta_large_lora | smoke | smoke | 0.133333 | 0.133333 |  | multiclass | microsoft/deberta-v3-large | smoke_sampled |  |

## Decision Notes

- Treat `course_clean_final_once` as the only official-style result.
- Treat `diagnostic` final metrics as post-final analysis, not course-clean model selection.
- Current evidence says CSV-level blending and simple decision layers overfit internal dev.
- The robust `reply_boundary` specialist strongly improves internal dev but does not transfer to final diagnostics.
- Next priority: representation/architecture changes (`directness_nli`, `rubric_prompt`, then revised large LoRA).