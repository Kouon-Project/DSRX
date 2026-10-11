# DiffSinger (Kouon Project forked from OpenVPI maintained 2024-11 ver.)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/openvpi/DiffSinger/blob/main/LICENSE)

This is a refactored and enhanced version of _DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism_ based on the original [paper](https://arxiv.org/abs/2105.02446) and [implementation](https://github.com/MoonInTheRiver/DiffSinger), which provides:

- Cleaner code structure: useless and redundant files are removed and the others are re-organized.
- Better sound quality: the sampling rate of synthesized audio are adapted to 44.1 kHz instead of the original 24 kHz.
- Higher fidelity: improved acoustic models and diffusion sampling acceleration algorithms are integrated.
- More controllability: introduced variance models and parameters for prediction and control of pitch, energy, breathiness, etc.
- Production compatibility: functionalities are designed to match the requirements of production deployment and the SVS communities.

|                                       Overview                                        |                                    Variance Model                                     |                                    Acoustic Model                                     |
|:-------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------:|
| <img src="docs/resources/arch-overview.jpg" alt="arch-overview" style="zoom: 60%;" /> | <img src="docs/resources/arch-variance.jpg" alt="arch-variance" style="zoom: 50%;" /> | <img src="docs/resources/arch-acoustic.jpg" alt="arch-acoustic" style="zoom: 60%;" /> |

## User Guidance

- [Getting started](docs/GettingStarted.md)
- [Duration models: deterministic, DDPM and ReFlow](docs/DurationModels.md)
- [OpenUtau native export](docs/OpenUtauNativeBackend.md)
- [Single-speaker native export](docs/SpeakerLockedNativeExport.md)
- [Native inference optimization](docs/InferenceOptimization.md)

## Architecture & Algorithms

TBD

## Development Resources

TBD

## References

### Original Paper & Implementation

- Paper: [DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism](https://arxiv.org/abs/2105.02446)
- Implementation: [MoonInTheRiver/DiffSinger](https://github.com/MoonInTheRiver/DiffSinger)

### Generative Models & Algorithms

- Denoising Diffusion Probabilistic Models (DDPM): [paper](https://arxiv.org/abs/2006.11239), [implementation](https://github.com/hojonathanho/diffusion)
  - [DDIM](https://arxiv.org/abs/2010.02502) for diffusion sampling acceleration
  - [PNDM](https://arxiv.org/abs/2202.09778) for diffusion sampling acceleration
  - [DPM-Solver++](https://github.com/LuChengTHU/dpm-solver) for diffusion sampling acceleration
  - [UniPC](https://github.com/wl-zhao/UniPC) for diffusion sampling acceleration
- Rectified Flow (RF): [paper](https://arxiv.org/abs/2209.03003), [implementation](https://github.com/gnobitab/RectifiedFlow)

### Dependencies & Submodules

- [RoPE](https://github.com/lucidrains/rotary-embedding-torch) for transformer encoder
- [HiFi-GAN](https://github.com/jik876/hifi-gan) and [NSF](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf) for waveform reconstruction
- [pc-ddsp](https://github.com/yxlllc/pc-ddsp) for waveform reconstruction
- [RMVPE](https://github.com/Dream-High/RMVPE) and yxlllc's [fork](https://github.com/yxlllc/RMVPE) for pitch extraction
- [Vocal Remover](https://github.com/tsurumeso/vocal-remover) and yxlllc's [fork](https://github.com/yxlllc/vocal-remover) for harmonic-noise separation

In this fork:
- [LoRA](https://arxiv.org/abs/2106.09685) for LoRA-finetuning
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) for flash attention on Fastspeech2
- [RefineGAN](https://timedomain-tech.github.io/refinegan) for RefineGAN support

## Disclaimer

Any organization or individual is prohibited from using any functionalities included in this repository to generate someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.

## License

This forked DiffSinger repository is licensed under the [Apache 2.0 License](LICENSE).

