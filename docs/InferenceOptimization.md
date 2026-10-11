# Native PyTorch Inference Optimization

Native acoustic and variance inference share the same CUDA optimization path.
It is enabled by default for both `scripts/infer.py` and
`inference/batch_backend.py`. Native voicebank export is described in [OpenUtau native export](OpenUtauNativeBackend.md).

The native vocoder path is optimized as well. `NsfHifiGAN` keeps F0 phase
generation in FP32 while storing and executing its convolutional generator in
FP16. The persistent batch backend right-pads phrase requests to profiled frame
buckets and crops the waveform back to the exact requested length; its five
most common short buckets are warmed during model loading. Regular command-line
inference uses the same mixed-precision generator without bucket padding or
vocoder warmup.

`RefineGAN` removes weight normalization after loading and uses CUDA FP16
autocast. Phrases longer than 3,500 mel frames are rendered in overlapping
chunks. A full-phrase FP32 phase prefix keeps the comb source phase continuous,
and 128-frame crossfades prevent chunk seams. This bounds dedicated VRAM on
very long phrases and avoids spilling into Windows shared GPU memory.

## Default CUDA behavior

- Large `Linear` layers (at least 16,384 weight elements) store FP16 weights
  and run FP16 matrix multiplication. Inputs and outputs at each replacement
  boundary remain FP32.
- Rectified-flow velocity backbones are traced and frozen with TorchScript.
- Backbones are warmed at 128, 512, 2,048, 6,000 and 10,000 frames to reduce
  first-shape compilation cost on later OpenUtau phrase requests.
- Acoustic inference enables TF32 matrix multiplication. Variance inference
  keeps FP32 matrix multiplication strict because profiling found it more
  stable for variance curves.
- CPU inference keeps the original eager FP32 implementation.

The representative warmup intentionally moves roughly four seconds of work to
model startup on the profiled RTX 3060 Laptop GPU so phrase-level requests stay
on the hot path.

The startup log contains a summary such as:

```text
| inference optimization: kind=acoustic, selective_fp16=47 linears, torchscript=1/1, tf32=True
```

The batch backend also includes the complete optimization report in its
`model_ready` or `variance_model_ready` event.

These optional keys can be placed in an experiment's `config.yaml`:

```yaml
inference_optimization: true
inference_selective_fp16: true
inference_selective_fp16_min_elements: 16384
inference_torchscript: true
inference_warmup_frames: "128|512|2048|6000|10000"
inference_vocoder_nsf_selective_fp16: true
inference_vocoder_nsf_bucketed_backend: true
inference_vocoder_nsf_buckets: "128|256|384|512|768|1024|1536|2048|3072|4096|6144|8192|10240"
inference_vocoder_nsf_warmup_frames: "128|256|384|512|768"
inference_vocoder_refinegan_autocast: true
inference_vocoder_refinegan_chunking: true
inference_vocoder_refinegan_chunk_frames: 3500
inference_vocoder_refinegan_chunk_overlap: 128
inference_vocoder_refinegan_full_fp16: false
```

Set `inference_optimization: false` to restore the eager model for diagnosis.
The RefineGAN full-FP16 option is intentionally off by default until it passes
model-specific listening tests.

## Inference-only checkpoints

Training checkpoints contain optimizer, scheduler, callback and trainer state
that native inference never reads. Export an exact model-only checkpoint with:

```bash
python scripts/export_inference_ckpt.py --exp my_experiment --ckpt 152000
```

The default output is written beside the source as
`model_ckpt_steps_152000.infer.ckpt`. Native inference automatically prefers
that file over `model_ckpt_steps_152000.ckpt`; training and native export retain
their existing checkpoint selection behavior.

An export to a separate directory copies the adjacent YAML, JSON and TXT model
assets by default:

```bash
python scripts/export_inference_ckpt.py \
  --input ckpt/my_experiment/model_ckpt_steps_152000.ckpt \
  --output path/to/inference_package \
  --manifest path/to/inference_package/manifest.json
```

The exporter writes through a temporary file, reloads it, and verifies every
state entry before replacing the target. Use `--force` only when an existing
export should be replaced.
