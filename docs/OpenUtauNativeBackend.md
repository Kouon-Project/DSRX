# OpenUtau native export

`scripts/export_openutau.py` packages DSRX models for the OpenUtau DSRX backend.
The built-in backend loads TorchScript `.pt` modules through LibTorch. A package
can also include checkpoints for the Python TCP backend.

## Export a voicebank

Run in the DSRX inference environment, from the repository root:

```bash
python scripts/export_openutau.py --exp acoustic_experiment --ckpt 100000 --duration-exp duration_experiment --duration-ckpt 30000 --variance-exp variance_experiment --variance-ckpt 60000 --out release/my_voice --device portable --no-include-python-checkpoint
```

Experiment names refer to folders under `ckpt/`. Omit a checkpoint step to use
the latest checkpoint. Duration and variance experiments are optional.

The package contains acoustic and vocoder modules, dictionaries, phoneme and
language maps, speaker embeddings, `character.yaml`, `dsrx.yaml`, and component
manifests. Duration assets are under `dsdur/`, variance assets under
`dsvariance/`, and vocoder assets under `dsvocoder/`.

Supported native components:

| Component | Model |
| --- | --- |
| Acoustic | ReFlow with shallow diffusion |
| Duration | Deterministic predictor, including legacy configs without `type` |
| Variance | ReFlow energy, breathiness, voicing and tension; `predict_dur` and `predict_pitch` set to `false` |
| Vocoder | NSF-HiFiGAN, DDSP or RefineGAN |

DDPM and ReFlow duration models use the Python inference path described in
[Duration models](DurationModels.md).

## Device selection

`--device portable` writes one FP32 weight set for CPU and CUDA. The exporter
rebinds traced device constants to input devices and removes redundant graph
expressions. OpenUtau selects the device when loading the package.

`--device cpu` and `--device cuda` create device-specific packages. To change
an existing package to portable mode, export it again from its checkpoint.
Windows output directories can contain Unicode characters.

## Replace a component

```bash
python scripts/export_openutau.py --exp acoustic_experiment --duration-exp duration_experiment --duration-only --out release/my_voice --device portable
python scripts/export_openutau.py --exp acoustic_experiment --variance-exp variance_experiment --variance-only --out release/my_voice --device portable
python scripts/export_openutau.py --exp acoustic_experiment --vocoder-only --vocoder-type refinegan --vocoder-ckpt models/refinegan.ckpt --out release/my_voice --device portable
```

`--vocoder-type` accepts `auto`, `nsf-hifigan`, `ddsp`, and `refinegan`.
`--vocoder-ckpt` overrides the checkpoint specified in the acoustic config.

## OpenUtau configuration

Use an OpenUtau build with the DSRX backend. `character.yaml` subbank suffixes
select entries in the DSRX speaker map. `dsrx.yaml` connects the native model,
duration, variance and vocoder folders.

Select **Built-in (LibTorch)** or **Python server** in the DSRX preferences or
track override. The built-in device setting accepts Automatic, CUDA and CPU.
Automatic selects available CUDA first. CPU thread count `0` uses half the
logical processors, capped at eight.

For a Python-server package, use `--include-python-checkpoint`. The optional
`--python-path` and `--dsrx-root` arguments configure local backend startup.
Python mode needs the full model configuration saved in the package.

For a fixed-speaker release with trimmed metadata, see
[Single-speaker native export](SpeakerLockedNativeExport.md).
