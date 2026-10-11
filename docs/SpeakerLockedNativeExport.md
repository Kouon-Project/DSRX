# Single-speaker native export

`--lock-speaker` creates a native package with one fixed speaker:

```bash
python scripts/export_openutau.py --exp acoustic_experiment --duration-exp duration_experiment --variance-exp variance_experiment --out release/singer --device portable --lock-speaker singer --voicebank-name singer --no-include-python-checkpoint
```

Use a new or empty output directory. The selected speaker must exist in each
supplied experiment. Merge LoRA checkpoints before exporting a locked package.

The exporter retains the selected speaker vector, clears other rows, removes
the speaker lookup table and freezes the retained vector into each conditioner.
The native tensor interface stays compatible with the editor; the model uses
its frozen speaker vector. Maps and subbanks contain the selected identity with
local speaker ID zero.

Release metadata contains inference settings, neutral asset names and the
selected speaker. The exporter removes TorchScript source-path debug records
and checks the package for training paths, other speaker names and removed
speaker vectors. `release-info.json` records the result.

Portable mode stores FP32 modules for CPU and CUDA. The exported modules use
the same LibTorch interface as a multi-speaker package. Locked packages are
exported as complete voicebanks; component replacement uses the ordinary
multi-speaker export path.
