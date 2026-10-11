# Getting Started

## Installation

### Environments and dependencies

DiffSinger requires Python 3.8 or later. We strongly recommend you create a virtual environment via Conda or venv before installing dependencies.

1. Install The latest PyTorch following the [official instructions](https://pytorch.org/get-started/locally/) according to your OS and hardware.

2. Install other dependencies via the following command:

   ```bash
   pip install -r requirements.txt
   ```

### Materials and assets

Some essential materials and assets are needed before continuing with this repository. See [materials for training and using models](BestPractices.md#materials-for-training-and-using-models) for detailed instructions.

## Configuration

Every model needs a configuration file to run preprocessing, training, inference and deployment. Templates of configurations files are in [configs/templates](../configs/templates). Templates of fine-tuning configurations are in [configs/finetune_templates](../configs/finetune_templates). Please **copy** the templates to your own data directory before you edit them.

Before you continue, it is highly recommended to read through [Best Practices](BestPractices.md), which is a more detailed tutorial on how to configure your experiments.

For more details about configurable parameters, see [Configuration Schemas](ConfigurationSchemas.md).

> Tips: to see which parameters are required or recommended to be edited, you can search by _customizability_ in the configuration schemas.

## Preprocessing

Raw data pieces and transcriptions should be binarized into dataset files before training. Before doing this step, please ensure all required configurations like `raw_data_dir` and `binary_data_dir` are set properly, and all your desired functionalities and features are enabled and configured.

Assume that you have a configuration file called `my_config.yaml`. Run:

```bash
python scripts/binarize.py --config my_config.yaml
```

Preprocessing can be accelerated through multiprocessing. See [binarization_args.num_workers](ConfigurationSchemas.md#binarization_args.num_workers) for more explanations.

## Training

Assume that you have a configuration file called `my_config.yaml` and the name of your model is `my_experiment`. Run:

```bash
python scripts/train.py --config my_config.yaml --exp_name my_experiment --reset
```

Checkpoints will be saved at the `checkpoints/my_experiment/` directory. When interrupting the program and running the above command again, the training resumes automatically from the latest checkpoint.

For more suggestions related to training performance, see [performance tuning](BestPractices.md#performance-tuning).

### TensorBoard

Run the following command to start the TensorBoard:

```bash
tensorboard --logdir checkpoints/
```

> NOTICE
> 
> If you are training a model with multiple GPUs (DDP), please add `--reload_multifile=true` option when launching TensorBoard, otherwise it may not update properly.

## Inference

Inference of DiffSinger is based on DS files. Assume that you have a DS file named `my_song.ds` and your model is named `my_experiment`.

If your model is a variance model, run:

```bash
python scripts/infer.py variance my_song.ds --exp my_experiment
```

or run

```bash
python scripts/infer.py variance --help
```

for more configurable options.

If your model is an acoustic model, run:

```bash
python scripts/infer.py acoustic my_song.ds --exp my_experiment
```

or run

```bash
python scripts/infer.py acoustic --help
```

for more configurable options.

CUDA inference automatically enables the profiled native PyTorch fast path,
including selective FP16 linear layers, a frozen TorchScript velocity backbone
and representative phrase-length warmup. See
[Native PyTorch Inference Optimization](InferenceOptimization.md) for the exact
behavior, configuration switches, batch-backend reporting and the optional
inference-only checkpoint format.

## Deployment

DSRX exports TorchScript packages for the OpenUtau DSRX LibTorch backend.

```bash
python scripts/export_openutau.py --exp my_experiment --out release/my_voice --device portable --no-include-python-checkpoint
```

The package contains model modules, dictionaries, speaker embeddings and editor configuration. See [OpenUtau native export](OpenUtauNativeBackend.md) for duration, variance and vocoder options, and [single-speaker export](SpeakerLockedNativeExport.md) for fixed-speaker releases.

## Other utilities

There are other useful CLI tools in the [scripts/](../scripts) directory not mentioned above:

- drop_spk.py - delete speaker embeddings from checkpoints (for data security reasons when distributing models)
- vocoder.py - bypass the acoustic model and only run the vocoder on given mel-spectrograms
