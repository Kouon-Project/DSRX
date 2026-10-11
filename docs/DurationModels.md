# Duration models

`dur_prediction_args.type` selects `deterministic`, `ddpm`, or `reflow`.
Existing configurations without this key select `deterministic` and keep the
original predictor parameters.

DDPM and ReFlow predict centered log-duration ratios within each `ph2word`
group. `word_dur` supplies the group's frame budget. The codec converts ratios
to nonnegative integer phoneme durations whose sum equals that budget.
Single-phoneme groups receive the full budget directly. When a group has fewer
frames than its configured per-phone minimum requires, zero-frame phones are
allowed.

The encoder conditions on phonemes, MIDI pitch, language and speaker settings,
log word duration, group position and group size. Training targets come from
the annotated phoneme durations. Each multi-phoneme group has equal loss weight.

## Training

Start from `configs/templates/config_duration_reflow.yaml` or
`configs/templates/config_duration_ddpm.yaml`. Set the dataset, dictionary,
validation song prefixes and binary output directory, then run:

```bash
python scripts/binarize.py --config configs/templates/config_duration_reflow.yaml
python scripts/train.py --config configs/templates/config_duration_reflow.yaml --exp_name duration_reflow
```

The templates train duration only. Multiple speakers and languages use the
usual `datasets`, `use_spk_id`, `num_spk`, `use_lang_id`, and `num_lang` settings.

## Inference

```bash
python scripts/infer.py variance song.ds --exp duration_reflow --predict dur --out infer_out
```

The input supplies `ph_seq`, `ph_num`, `note_seq`, `note_dur`, and `note_slur`.
The output adds `ph_dur` in seconds. Generative durations already meet the group
budgets, so the inference path uses them directly.

Set `dur_seed` on an input segment to choose its duration sample. Otherwise the
selected duration configuration's `sampling_seed` is used. ReFlow supports
`euler`, `rk2` and `rk4`; DDPM supports `ddim` and `ancestral`. Ancestral sampling
uses `sampling_steps == timesteps`.

The model API also accepts `dur_reference` and `dur_retake`. A selected phone
retakes its whole group; unselected groups keep the reference durations.

## Validation metrics

Duration validation covers the full validation set, with noise keyed by sample
index and `validation_seed`.

| Metric | Meaning |
| --- | --- |
| `duration/boundary_mae_ms_multi` | Mean internal boundary error in milliseconds, averaged over multi-phone groups |
| `duration/boundary_acc_2f_multi` | Fraction of internal boundaries within two frames, averaged over groups |
| `duration/allocation_score_multi` | Agreement of within-group duration proportions, from zero to one |
| `duration/ph_dur_acc_multi` | Phone-duration accuracy with a two-frame or 20% tolerance, whichever is larger |
| `constraint/word_sum_violation_rate` | Fraction of groups whose predicted total differs from the supplied budget |
| `constraint/word_sum_max_error_frames` | Largest group-total error in frames |
| `constraint/invalid_duration_rate` | Fraction of negative, fractional, non-finite or nonzero padding durations |

`ph_dur_acc` remains available for comparisons with previous runs. Deterministic
models also report `rhythm_corr` before rhythm regulation.
