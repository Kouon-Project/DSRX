"""Export a DSRX acoustic model for OpenUtau's in-process LibTorch backend.

The generated package contains TorchScript modules and speaker embeddings.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

import torch
from torch import Tensor, nn


class AcousticConditioner(nn.Module):
    """Give the acoustic conditioner a stable, tensor-only interface."""

    def __init__(self, model: nn.Module, enabled_variances: list[str], locked_speaker: Tensor | None = None):
        super().__init__()
        self.model = model
        self.register_buffer("locked_speaker", locked_speaker)
        self.use_energy = "energy" in enabled_variances
        self.use_breathiness = "breathiness" in enabled_variances
        self.use_voicing = "voicing" in enabled_variances
        self.use_tension = "tension" in enabled_variances

    def forward(
        self,
        tokens: Tensor,
        durations: Tensor,
        f0: Tensor,
        energy: Tensor,
        breathiness: Tensor,
        voicing: Tensor,
        tension: Tensor,
        gender: Tensor,
        velocity: Tensor,
        speaker_embedding: Tensor,
        languages: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.locked_speaker is not None:
            speaker_embedding = self.locked_speaker
        variances: Dict[str, Tensor] = {}
        if self.use_energy:
            variances["energy"] = energy
        if self.use_breathiness:
            variances["breathiness"] = breathiness
        if self.use_voicing:
            variances["voicing"] = voicing
        if self.use_tension:
            variances["tension"] = tension
        condition, auxiliary_mel = self.model(
            tokens,
            durations,
            f0,
            variances,
            gender,
            velocity,
            speaker_embedding,
            languages,
        )
        return condition, auxiliary_mel


class FixedReflowSampler(nn.Module):
    """Expose fixed-step rectified-flow sampling to LibTorch."""

    def __init__(self, model: nn.Module, steps: int):
        super().__init__()
        self.model = model
        self.steps = steps

    def forward(self, condition: Tensor, auxiliary_mel: Tensor, depth: Tensor) -> Tensor:
        return self.model(condition, auxiliary_mel, depth, self.steps)


class VarianceConditioner(nn.Module):
    """Combine DSRX's linguistic and variance conditioners behind one ABI."""

    def __init__(
        self,
        linguistic_model: nn.Module,
        variance_preprocess: nn.Module,
        predicted_variances: list[str],
        locked_speaker: Tensor | None = None,
    ):
        super().__init__()
        self.linguistic_model = linguistic_model
        self.variance_preprocess = variance_preprocess
        self.predicted_variances = predicted_variances
        self.register_buffer("locked_speaker", locked_speaker)

    def forward(
        self,
        tokens: Tensor,
        durations: Tensor,
        pitch: Tensor,
        speaker_embedding: Tensor,
        languages: Tensor,
    ) -> Tensor:
        if self.locked_speaker is not None:
            speaker_embedding = self.locked_speaker
        encoder_out, _ = self.linguistic_model(tokens, durations, languages)
        variances: Dict[str, Tensor] = {}
        for name in self.predicted_variances:
            variances[name] = torch.zeros_like(pitch)
        retake = torch.ones(
            (
                pitch.shape[0],
                pitch.shape[1],
                len(self.predicted_variances),
            ),
            dtype=torch.bool,
            device=pitch.device,
        )
        return self.variance_preprocess(
            encoder_out,
            durations,
            pitch,
            variances,
            retake,
            speaker_embedding,
        )


class DeterministicDurationModel(nn.Module):
    """Combine DSRX's word encoder and deterministic predictor behind one ABI."""

    def __init__(self, linguistic_model: nn.Module, duration_predictor: nn.Module,
                 locked_speaker: Tensor | None = None):
        super().__init__()
        self.linguistic_model = linguistic_model
        self.duration_predictor = duration_predictor
        self.register_buffer("locked_speaker", locked_speaker)

    def forward(
        self,
        tokens: Tensor,
        word_div: Tensor,
        word_dur: Tensor,
        ph_midi: Tensor,
        speaker_embedding: Tensor,
        languages: Tensor,
    ) -> Tensor:
        if self.locked_speaker is not None:
            speaker_embedding = self.locked_speaker
        encoder_out, x_masks = self.linguistic_model(
            tokens,
            word_div,
            word_dur,
            languages,
        )
        return self.duration_predictor(
            encoder_out,
            x_masks,
            ph_midi,
            speaker_embedding,
        )


class FixedVarianceReflowSampler(nn.Module):
    """Expose fixed-step multi-variance rectified-flow sampling to LibTorch."""

    def __init__(self, model: nn.Module, steps: int):
        super().__init__()
        self.model = model
        self.steps = steps

    def forward(self, condition: Tensor) -> Tensor:
        return self.model(condition, steps=self.steps)


class ScriptLengthRegulator(nn.Module):
    """A scriptable regulator that keeps arange on the input device."""

    def forward(self, durations: Tensor) -> Tensor:
        token_indices = torch.arange(
            1,
            durations.shape[1] + 1,
            device=durations.device,
        )[None, :, None]
        cumulative = torch.cumsum(durations, dim=1)
        previous = torch.cat(
            (torch.zeros_like(cumulative[:, :1]), cumulative[:, :-1]),
            dim=1,
        )
        positions = torch.arange(
            durations.sum(dim=1).max(),
            device=durations.device,
        )[None, None]
        mask = (positions >= previous[:, :, None]) & (positions < cumulative[:, :, None])
        return (token_indices * mask).sum(dim=1)


class DDSPVocoderWrapper(nn.Module):
    """Flatten DDSP's nested output into OpenUtau's tensor-only ABI."""

    def __init__(self, model: nn.Module, mel_multiplier: float):
        super().__init__()
        self.model = model
        self.mel_multiplier = mel_multiplier

    def forward(self, mel: Tensor, f0: Tensor) -> Tensor:
        signal, _, _ = self.model(
            mel * self.mel_multiplier,
            f0.unsqueeze(-1),
        )
        return signal.reshape(signal.shape[0], -1).float()


class RefineGANVocoderWrapper(nn.Module):
    """Keep RefineGAN phase input explicit for native long-phrase chunking."""

    def __init__(self, generator: nn.Module, mel_multiplier: float):
        super().__init__()
        self.generator = generator
        self.mel_multiplier = mel_multiplier

    def forward(self, mel: Tensor, f0: Tensor, phase_offset: Tensor) -> Tensor:
        mel_channels = (mel.transpose(1, 2) * self.mel_multiplier).to(
            self.generator.mel_conv.weight.dtype
        )
        f0_channels = f0.unsqueeze(1).float()
        return self.generator(
            mel_channels,
            f0_channels,
            phase_offset=phase_offset,
        ).squeeze(1).float()


def freeze_module(module: torch.jit.ScriptModule, name: str, *, required: bool = False) -> torch.jit.ScriptModule:
    """Freeze for inference; locked releases fail closed if freezing fails."""
    try:
        return torch.jit.freeze(module.eval())
    except Exception as exc:
        if required:
            raise RuntimeError(f"A locked release must successfully freeze {name}.") from exc
        print(f"| warn: unable to freeze {name}: {type(exc).__name__}: {exc}")
        return module.eval()


def save_torchscript(module: torch.jit.ScriptModule, path: Path) -> None:
    # Native TorchScript filename APIs on Windows can reject Chinese paths.
    # A Python stream supports them without changing the serialized weights.
    with path.open("wb") as stream:
        torch.jit.save(module, stream)


def normalize_vocoder_type(value: str | None) -> str:
    normalized = (value or "NsfHifiGAN").strip().lower().replace("_", "-")
    aliases = {
        "nsfhifigan": "nsf-hifigan",
        "nsf-hifigan": "nsf-hifigan",
        "ddsp": "ddsp",
        "refinegan": "refinegan",
        "refine-gan": "refinegan",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported DSRX vocoder: {value}")
    return aliases[normalized]


def resolve_repo_path(repo: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def install_portable_rotary() -> None:
    """Remove einops shape scalars and RoPE cache mutation from the traced path."""
    import modules.commons.rotary_embedding_torch as rotary

    def rotate_half_native(value: Tensor) -> Tensor:
        even = value[..., 0::2]
        odd = value[..., 1::2]
        return torch.stack((-odd, even), dim=-1).flatten(-2)

    def apply_rotary_native(
        freqs: Tensor,
        value: Tensor,
        start_index: int = 0,
        scale: float = 1.0,
        seq_dim: int = -2,
        freqs_seq_dim: int | None = None,
    ) -> Tensor:
        del seq_dim, freqs_seq_dim
        rotation_size = freqs.shape[-1]
        left = value[..., :start_index]
        middle = value[..., start_index : start_index + rotation_size]
        right = value[..., start_index + rotation_size :]
        transformed = (
            middle * freqs.cos() * scale
            + rotate_half_native(middle) * freqs.sin() * scale
        )
        return torch.cat((left, transformed, right), dim=-1).to(value.dtype)

    def forward_native(
        self: nn.Module,
        positions: Tensor,
        seq_len: int | None = None,
        offset: int = 0,
    ) -> Tensor:
        del seq_len, offset
        frequencies = torch.einsum(
            "..., f -> ... f",
            positions.to(self.freqs.dtype),
            self.freqs,
        )
        return frequencies.repeat_interleave(2, dim=-1)

    rotary.rotate_half = rotate_half_native
    rotary.apply_rotary_emb = apply_rotary_native
    rotary.RotaryEmbedding.forward = forward_native


def make_conditioner_inputs(
    *,
    token_count: int,
    frame_count: int,
    hidden_size: int,
    device: torch.device,
) -> tuple[Tensor, ...]:
    durations = torch.full(
        (1, token_count),
        frame_count // token_count,
        dtype=torch.long,
        device=device,
    )
    durations[:, -1] += frame_count - int(durations[0].sum().item())
    return (
        torch.ones((1, token_count), dtype=torch.long, device=device),
        durations,
        torch.full((1, frame_count), 220.0, dtype=torch.float32, device=device),
        torch.zeros((1, frame_count), dtype=torch.float32, device=device),
        torch.full((1, frame_count), -60.0, dtype=torch.float32, device=device),
        torch.full((1, frame_count), -96.0, dtype=torch.float32, device=device),
        torch.zeros((1, frame_count), dtype=torch.float32, device=device),
        torch.zeros((1, frame_count), dtype=torch.float32, device=device),
        torch.ones((1, frame_count), dtype=torch.float32, device=device),
        torch.zeros((1, frame_count, hidden_size), dtype=torch.float32, device=device),
        torch.zeros((1, token_count), dtype=torch.long, device=device),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a DSRX model as an OpenUtau native package."
    )
    parser.add_argument("--exp", required=True, help="DSRX experiment name")
    parser.add_argument("--ckpt", type=int, help="checkpoint step; latest when omitted")
    parser.add_argument(
        "--duration-exp",
        help="optional deterministic DSRX duration experiment packaged under dsdur",
    )
    parser.add_argument(
        "--duration-ckpt",
        type=int,
        help="duration checkpoint step; latest when omitted",
    )
    parser.add_argument(
        "--duration-only",
        action="store_true",
        help="add or replace deterministic dsdur in an existing native package",
    )
    parser.add_argument(
        "--variance-exp",
        help="optional DSRX variance experiment packaged under dsvariance",
    )
    parser.add_argument(
        "--variance-ckpt",
        type=int,
        help="variance checkpoint step; latest when omitted",
    )
    parser.add_argument(
        "--variance-only",
        action="store_true",
        help="add or replace dsvariance in an existing native package",
    )
    parser.add_argument("--out", type=Path, required=True, help="output package directory")
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "portable"),
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda/cpu: device-specific export; portable: one FP32 package shared by CPU and CUDA",
    )
    parser.add_argument(
        "--vocoder-type",
        choices=("auto", "nsf-hifigan", "ddsp", "refinegan"),
        default="auto",
        help="override config.yaml's vocoder for the built-in package",
    )
    parser.add_argument(
        "--vocoder-ckpt",
        type=Path,
        help="override config.yaml's vocoder_ckpt",
    )
    parser.add_argument(
        "--vocoder-only",
        action="store_true",
        help="replace only dsvocoder in an existing native package",
    )
    parser.add_argument(
        "--voicebank-name",
        help="name written to character.txt/dsrx.yaml; defaults to the experiment name",
    )
    parser.add_argument(
        "--default-speaker",
        help="default subbank speaker; defaults to the first spk_map entry",
    )
    parser.add_argument(
        "--lock-speaker",
        help="fixed single-speaker native release with scrubbed metadata; requires a fresh "
             "output directory and --no-include-python-checkpoint",
    )
    parser.add_argument(
        "--include-python-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="copy the selected .infer.ckpt into the package when available",
    )
    parser.add_argument(
        "--python-path",
        type=Path,
        help="optional Python executable written to dsrx.yaml for managed startup",
    )
    parser.add_argument(
        "--dsrx-root",
        type=Path,
        help="optional DSRX repository path written to dsrx.yaml for managed startup",
    )
    return parser.parse_args()


def load_json_map(path: Path) -> dict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def build_acoustic_model(
    *,
    phoneme_dictionary,
    device: torch.device,
    ckpt_steps: int | None,
    strict_weights: bool = False,
):
    """Build and load the acoustic model for native deployment."""
    from deployment.modules.toplevel import DiffSingerAcousticNative
    from utils import load_ckpt
    from utils.hparams import hparams
    from utils.lora import inject_lora, load_lora_state_dict, merge_lora_into_model

    model = DiffSingerAcousticNative(
        vocab_size=len(phoneme_dictionary),
        out_dims=hparams["audio_num_mel_bins"],
        cross_lingual_token_idx=sorted({
            phoneme_dictionary.encode_one(phone)
            for phone in phoneme_dictionary.cross_lingual_phonemes
        }),
    ).eval().to(device)
    lora_config = hparams.get("lora", {})
    if strict_weights and isinstance(lora_config, dict) and lora_config.get("enabled", False):
        raise NotImplementedError("Locking a LoRA model requires a separately merged checkpoint first.")
    if isinstance(lora_config, dict) and lora_config.get("enabled", False):
        inject_lora(
            model,
            rank=int(lora_config.get("rank", 8)),
            alpha=int(lora_config.get("alpha", 16)),
            target_modules=lora_config.get("target_modules", ["linear"]),
        )
        model = model.to(device)
        base_ckpt = lora_config.get("base_ckpt")
        load_ckpt(
            model,
            base_ckpt or hparams["work_dir"],
            ckpt_steps=None if base_ckpt else ckpt_steps,
            prefix_in_ckpt="model",
            strict=strict_weights,
            device=device,
        )
        from utils.training_utils import get_latest_checkpoint_path

        latest = get_latest_checkpoint_path(Path(hparams["work_dir"]))
        if latest:
            try:
                load_ckpt(
                    model,
                    Path(latest),
                    ckpt_steps=None,
                    prefix_in_ckpt="model",
                    strict=False,
                    device=device,
                )
            except Exception:
                state = torch.load(latest, map_location=device).get("state_dict", {})
                load_lora_state_dict(model, state, prefix="model", strict=False)
        model = model.to(device)
        if lora_config.get("merge_before_export", True):
            merge_lora_into_model(model)
    else:
        load_ckpt(
            model,
            hparams["work_dir"],
            ckpt_steps=ckpt_steps,
            prefix_in_ckpt="model",
            strict=strict_weights,
            device=device,
        )
    if hparams.get("use_artifact_embed", False):
        model.fs2.register_buffer(
            "frozen_artifact_level",
            torch.zeros(1, dtype=torch.long, device=device),
        )
    return model


def build_variance_model(
    *,
    phoneme_dictionary,
    device: torch.device,
    ckpt_steps: int | None,
):
    """Build and load the variance model for native deployment."""
    from deployment.modules.toplevel import DiffSingerVarianceNative
    from utils import load_ckpt
    from utils.hparams import hparams
    from utils.lora import inject_lora, load_lora_state_dict, merge_lora_into_model

    model = DiffSingerVarianceNative(
        vocab_size=len(phoneme_dictionary),
        cross_lingual_token_idx=sorted({
            phoneme_dictionary.encode_one(phone)
            for phone in phoneme_dictionary.cross_lingual_phonemes
        }),
    ).eval().to(device)
    lora_config = hparams.get("lora", {})
    if isinstance(lora_config, dict) and lora_config.get("enabled", False):
        inject_lora(
            model,
            rank=int(lora_config.get("rank", 8)),
            alpha=int(lora_config.get("alpha", 16)),
            target_modules=lora_config.get("target_modules", ["linear"]),
        )
        model = model.to(device)
        base_ckpt = lora_config.get("base_ckpt")
        load_ckpt(
            model,
            base_ckpt or hparams["work_dir"],
            ckpt_steps=None if base_ckpt else ckpt_steps,
            prefix_in_ckpt="model",
            strict=False,
            device=device,
        )
        from utils.training_utils import get_latest_checkpoint_path

        latest = get_latest_checkpoint_path(Path(hparams["work_dir"]))
        if latest:
            try:
                load_ckpt(
                    model,
                    Path(latest),
                    ckpt_steps=None,
                    prefix_in_ckpt="model",
                    strict=False,
                    device=device,
                )
            except Exception:
                state = torch.load(latest, map_location=device).get("state_dict", {})
                load_lora_state_dict(model, state, prefix="model", strict=False)
        model = model.to(device)
        if lora_config.get("merge_before_export", True):
            merge_lora_into_model(model)
    else:
        load_ckpt(
            model,
            hparams["work_dir"],
            ckpt_steps=ckpt_steps,
            prefix_in_ckpt="model",
            strict=True,
            device=device,
            prefer_inference=True,
        )
    return model


@torch.no_grad()
def speaker_embedding(model: nn.Module, speaker_id: int, device: torch.device) -> Tensor:
    speaker = torch.tensor([[speaker_id]], dtype=torch.long, device=device)
    return model.fs2.spk_embed(speaker).squeeze(0).squeeze(0)


@torch.no_grad()
def variance_speaker_embedding(
    model: nn.Module,
    speaker_id: int,
    device: torch.device,
) -> Tensor:
    speaker = torch.tensor([speaker_id], dtype=torch.long, device=device)
    return model.spk_embed(speaker).squeeze(0)


def save_and_validate_vocoder(
    *,
    module: torch.jit.ScriptModule,
    path: Path,
    device: torch.device,
    mel_bins: int,
    hop_size: int,
    refinegan: bool = False,
) -> None:
    module = freeze_module(module, path.stem)
    save_torchscript(module, path)
    with path.open("rb") as stream:
        loaded = torch.jit.load(stream, map_location=device).eval()
    mel = torch.randn((1, 31, mel_bins), device=device)
    f0 = torch.full((1, 31), 220.0, device=device)
    if refinegan:
        wav = loaded(mel, f0, torch.zeros((1, 1, 1), device=device))
    else:
        wav = loaded(mel, f0)
    if wav.shape[0] != 1 or wav.shape[-1] != 31 * hop_size:
        raise RuntimeError(
            f"Exported vocoder returned {tuple(wav.shape)}, expected [1, {31 * hop_size}]."
        )
    del loaded, wav, mel, f0


def export_nsf_hifigan(
    *,
    repo: Path,
    checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    mel_bins: int,
    sample_rate: int,
    hop_size: int,
    mel_base: str,
) -> dict[str, Any]:
    from deployment.modules.nsf_hifigan import NSFHiFiGANNative
    from utils import load_ckpt
    from utils.hparams import hparams

    config_path = checkpoint.with_name("config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config["sampling_rate"]) != sample_rate:
        raise ValueError("NSF-HiFiGAN sampling_rate does not match the acoustic model.")
    if int(config["hop_size"]) != hop_size or int(config["num_mels"]) != mel_bins:
        raise ValueError("NSF-HiFiGAN hop_size/num_mels do not match the acoustic model.")

    wrapper = NSFHiFiGANNative(config).eval().to(device)
    load_ckpt(
        wrapper.generator,
        str(checkpoint),
        prefix_in_ckpt=None,
        key_in_ckpt="generator",
        strict=True,
        device=device,
    )
    wrapper.generator.remove_weight_norm()
    selective_fp16 = bool(
        hparams.get("inference_optimization", True)
        and device.type == "cuda"
        and hparams.get("inference_vocoder_nsf_selective_fp16", True)
    )
    if selective_fp16:
        wrapper.generator.half()
    example = (
        torch.randn((1, 24, mel_bins), device=device),
        torch.full((1, 24), 220.0, device=device),
    )
    script = torch.jit.trace(wrapper, example, check_trace=False, strict=False)
    model_path = output_dir / "nsf_hifigan.pt"
    save_and_validate_vocoder(
        module=script,
        path=model_path,
        device=device,
        mel_bins=mel_bins,
        hop_size=hop_size,
    )
    buckets = hparams.get(
        "inference_vocoder_nsf_buckets",
        [128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240],
    )
    warmup = hparams.get(
        "inference_vocoder_nsf_warmup_frames",
        # OpenUtau renders interactively. Eagerly running representative shapes
        # delays the first requested phrase and does not warm its actual bucket
        # when the shape differs. Let the first real phrase initialize only the
        # bucket it needs unless a deployment explicitly opts into warmup.
        [],
    )
    return {
        "type": "nsf-hifigan",
        "model": model_path.name,
        "mel_base": str(mel_base),
        "selective_fp16": selective_fp16,
        "phase_dtype": "float32",
        "weight_norm_removed": True,
        "bucketed_backend": bool(
            hparams.get("inference_optimization", True)
            and device.type == "cuda"
            and hparams.get("inference_vocoder_nsf_bucketed_backend", True)
        ),
        "buckets": [int(value) for value in buckets],
        "warmup_frames": [int(value) for value in warmup],
    }


def export_ddsp(
    *,
    checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    mel_bins: int,
    sample_rate: int,
    hop_size: int,
    mel_base: str,
) -> dict[str, Any]:
    import yaml

    config_path = checkpoint.with_name("config.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data = config.get("data", {})
    if int(data.get("sampling_rate", sample_rate)) != sample_rate:
        raise ValueError("DDSP sampling_rate does not match the acoustic model.")
    if int(data.get("block_size", hop_size)) != hop_size:
        raise ValueError("DDSP block_size does not match the acoustic model hop_size.")
    if int(data.get("n_mels", mel_bins)) != mel_bins:
        raise ValueError("DDSP n_mels does not match the acoustic model.")

    source = torch.jit.load(str(checkpoint), map_location=device).eval()
    multiplier = 0.434294 if str(mel_base) == "e" else 1.0
    wrapper = DDSPVocoderWrapper(source, multiplier).eval().to(device)
    script = torch.jit.trace(
        wrapper,
        (
            torch.randn((1, 24, mel_bins), device=device),
            torch.full((1, 24), 220.0, device=device),
        ),
        check_trace=False,
        strict=False,
    )
    model_path = output_dir / "ddsp.pt"
    save_and_validate_vocoder(
        module=script,
        path=model_path,
        device=device,
        mel_bins=mel_bins,
        hop_size=hop_size,
    )
    return {
        "type": "ddsp",
        "model": model_path.name,
        "mel_base": str(mel_base),
    }


def export_refinegan(
    *,
    checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    mel_bins: int,
    sample_rate: int,
    hop_size: int,
    mel_base: str,
) -> dict[str, Any]:
    from modules.vocoders.refinegan import RefineGAN

    from utils.hparams import hparams

    original_checkpoint = hparams.get("vocoder_ckpt")
    hparams["vocoder_ckpt"] = str(checkpoint)
    try:
        vocoder = RefineGAN()
        vocoder.to_device(device)
    finally:
        hparams["vocoder_ckpt"] = original_checkpoint
    generator = vocoder.generator.eval()
    if int(generator.sampling_rate) != sample_rate:
        raise ValueError("RefineGAN sampling_rate does not match the acoustic model.")
    if int(generator.hop_length) != hop_size:
        raise ValueError("RefineGAN hop_length does not match the acoustic model.")
    if int(generator.mel_conv.in_channels) != mel_bins:
        raise ValueError("RefineGAN mel bins do not match the acoustic model.")

    multiplier = 1.0 if str(mel_base) == "e" else 2.302585
    wrapper = RefineGANVocoderWrapper(generator, multiplier).eval().to(device)
    example = (
        torch.randn((1, 24, mel_bins), device=device),
        torch.full((1, 24), 220.0, device=device),
        torch.zeros((1, 1, 1), device=device),
    )
    trace_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if vocoder._autocast
        else nullcontext()
    )
    with trace_context:
        script = torch.jit.trace(wrapper, example, check_trace=False, strict=False)
    model_path = output_dir / "refinegan.pt"
    save_and_validate_vocoder(
        module=script,
        path=model_path,
        device=device,
        mel_bins=mel_bins,
        hop_size=hop_size,
        refinegan=True,
    )
    return {
        "type": "refinegan",
        "model": model_path.name,
        "mel_base": str(mel_base),
        "autocast_fp16": bool(vocoder._autocast),
        "full_fp16": bool(vocoder._full_fp16),
        "weight_norm_removed": True,
        "chunking": bool(vocoder._chunking),
        "chunk_frames": int(vocoder._chunk_frames),
        "chunk_overlap": int(vocoder._chunk_overlap),
        "phase_continuity": bool(vocoder._chunking),
    }


def export_vocoder(
    *,
    vocoder_type: str,
    repo: Path,
    checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    mel_bins: int,
    sample_rate: int,
    hop_size: int,
    mel_base: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "checkpoint": checkpoint,
        "output_dir": output_dir,
        "device": device,
        "mel_bins": mel_bins,
        "sample_rate": sample_rate,
        "hop_size": hop_size,
        "mel_base": mel_base,
    }
    print(f"| exporting {vocoder_type} vocoder from {checkpoint}")
    if vocoder_type == "nsf-hifigan":
        fields = export_nsf_hifigan(repo=repo, **common)
    elif vocoder_type == "ddsp":
        fields = export_ddsp(**common)
    elif vocoder_type == "refinegan":
        fields = export_refinegan(**common)
    else:
        raise ValueError(f"Unsupported vocoder type: {vocoder_type}")
    manifest = {
        "format_version": 1,
        "runtime": "libtorch-torchscript",
        "device": device.type,
        "sample_rate": sample_rate,
        "hop_size": hop_size,
        "mel_bins": mel_bins,
        **fields,
    }
    manifest_path = output_dir / "dsrx-vocoder.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def copy_python_checkpoint(work_dir: Path, out: Path, ckpt_steps: int | None) -> str | None:
    if ckpt_steps is not None:
        candidate = work_dir / f"model_ckpt_steps_{ckpt_steps}.infer.ckpt"
    else:
        candidates = sorted(
            work_dir.glob("model_ckpt_steps_*.infer.ckpt"),
            key=lambda path: int(path.name.split("steps_", 1)[1].split(".", 1)[0]),
        )
        candidate = candidates[-1] if candidates else Path()
    if not candidate.is_file():
        print("| warn: no .infer.ckpt was found; Python backend artifact was not copied")
        return None
    destination = out / candidate.name
    if candidate.resolve() != destination.resolve():
        shutil.copy2(candidate, destination)
    return destination.name


def copy_python_vocoder(checkpoint: Path, out: Path) -> str:
    """Copy the training-runtime vocoder and any adjacent config files."""
    destination_dir = out / "dsvocoder" / "python"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / checkpoint.name
    if checkpoint.resolve() != destination.resolve():
        shutil.copy2(checkpoint, destination)
    for config_name in ("config.yaml", "config.yml", "config.json", "config_v1.json"):
        source = checkpoint.parent / config_name
        target = destination_dir / config_name
        if source.is_file() and source.resolve() != target.resolve():
            shutil.copy2(source, target)
    return destination.relative_to(out).as_posix()


def update_dsrx_yaml(
    *,
    out: Path,
    name: str | None = None,
    default_speaker: str | None = None,
    python_path: Path | None = None,
    dsrx_root: Path | None = None,
    python_vocoder_checkpoint: str | None = None,
    duration_work_dir: str | None = None,
    variance_work_dir: str | None = None,
) -> None:
    import yaml

    dsrx_yaml = out / "dsrx.yaml"
    metadata = {}
    if dsrx_yaml.is_file():
        metadata = yaml.safe_load(dsrx_yaml.read_text(encoding="utf-8")) or {}
    metadata.setdefault("format_version", 1)
    if name:
        metadata.setdefault("name", name)
    metadata.setdefault("native_model_dir", ".")
    metadata.setdefault("vocoder_dir", "dsvocoder")
    if default_speaker:
        metadata.setdefault("default_speaker", default_speaker)
    if python_path:
        metadata["python_path"] = str(python_path.expanduser().resolve())
    if dsrx_root:
        metadata["dsrx_root"] = str(dsrx_root.expanduser().resolve())
    if python_vocoder_checkpoint:
        metadata["python_vocoder_ckpt"] = python_vocoder_checkpoint
    if duration_work_dir:
        metadata["duration_work_dir"] = duration_work_dir
    if variance_work_dir:
        metadata["variance_work_dir"] = variance_work_dir
    dsrx_yaml.write_text(
        yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def write_voicebank_metadata(
    *,
    out: Path,
    name: str,
    speakers: list[str],
    default_speaker: str,
    python_path: Path | None,
    dsrx_root: Path | None,
    python_vocoder_checkpoint: str | None,
) -> None:
    for dirname in ("dsdur", "dspitch", "dsvariance", "dsvocoder"):
        (out / dirname).mkdir(parents=True, exist_ok=True)
    character_txt = out / "character.txt"
    if not character_txt.exists():
        character_txt.write_text(f"name={name}\n", encoding="utf-8")
    character_yaml = out / "character.yaml"
    if not character_yaml.exists():
        subbanks = []
        for speaker in speakers:
            subbanks.append({
                "color": "" if speaker == default_speaker else speaker,
                "prefix": "",
                "suffix": speaker,
            })
        import yaml

        character_yaml.write_text(
            yaml.safe_dump(
                {
                    "singer_type": "diffsinger",
                    "text_file_encoding": "utf-8",
                    "subbanks": subbanks,
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    update_dsrx_yaml(
        out=out,
        name=name,
        default_speaker=default_speaker,
        python_path=python_path,
        dsrx_root=dsrx_root,
        python_vocoder_checkpoint=python_vocoder_checkpoint,
    )


def variance_clamps(predicted_variances: list[str], hparams: dict) -> dict[str, list[float]]:
    values = {
        "energy": [float(hparams["energy_db_min"]), 0.0],
        "breathiness": [float(hparams["breathiness_db_min"]), 0.0],
        "voicing": [float(hparams["voicing_db_min"]), 0.0],
        "tension": [
            float(hparams["tension_logit_min"]),
            float(hparams["tension_logit_max"]),
        ],
    }
    return {name: values[name] for name in predicted_variances}


def make_variance_conditioner_inputs(
    *,
    token_count: int,
    frame_count: int,
    hidden_size: int,
    device: torch.device,
) -> tuple[Tensor, ...]:
    durations = torch.full(
        (1, token_count),
        frame_count // token_count,
        dtype=torch.long,
        device=device,
    )
    durations[:, -1] += frame_count - int(durations[0].sum().item())
    return (
        torch.ones((1, token_count), dtype=torch.long, device=device),
        durations,
        torch.full((1, frame_count), 60.0, dtype=torch.float32, device=device),
        torch.zeros((1, frame_count, hidden_size), dtype=torch.float32, device=device),
        torch.zeros((1, token_count), dtype=torch.long, device=device),
    )


def make_duration_inputs(
    *,
    word_divisions: list[int],
    word_durations: list[int],
    hidden_size: int,
    device: torch.device,
) -> tuple[Tensor, ...]:
    if len(word_divisions) != len(word_durations) or not word_divisions:
        raise ValueError("Duration trace words must be non-empty and aligned.")
    token_count = sum(word_divisions)
    return (
        torch.ones((1, token_count), dtype=torch.long, device=device),
        torch.tensor([word_divisions], dtype=torch.long, device=device),
        torch.tensor([word_durations], dtype=torch.long, device=device),
        torch.full((1, token_count), 60, dtype=torch.long, device=device),
        torch.zeros(
            (1, token_count, hidden_size),
            dtype=torch.float32,
            device=device,
        ),
        torch.zeros((1, token_count), dtype=torch.long, device=device),
    )


def duration_symbol_type(symbol: str) -> str:
    """Return the G2P role needed by OpenUtau's DiffSinger phonemizers."""
    language, _, phone = symbol.partition("/")
    if not phone:
        phone = language
        language = ""
    vowels = {
        "": {"AP", "SP"},
        "en": {
            "aa", "ae", "ah", "ao", "aw", "ax", "ay", "eh", "er",
            "ey", "ih", "iy", "ow", "oy", "uh", "uw", "exAP",
        },
        "ja": {"a", "e", "i", "o", "u", "N", "exAP"},
        "zh": {
            "A", "E", "I", "N", "O", "U", "a", "aa", "e", "i",
            "i0", "iA", "iE", "iO", "ia", "ie", "io", "ir", "o",
            "oo", "u", "uA", "uE", "ua", "ue", "uo", "v", "v0",
            "vE", "ve", "exAP",
        },
    }
    if phone in vowels.get(language, set()) or symbol in {"AP", "SP"}:
        return "vowel"
    if symbol in {"en/w", "en/y", "ja/w", "ja/y", "zh/w", "zh/y"}:
        return "semivowel"
    if symbol in {"en/l", "en/r", "ja/r", "zh/l", "zh/r"}:
        return "liquid"
    return "consonant"


def write_duration_dsdicts(
    *,
    work_dir: Path,
    duration_out: Path,
    phoneme_map: dict[str, int],
    languages: list[str],
) -> None:
    """Convert DSRX tab dictionaries into OpenUtau dsdict YAML files."""
    import yaml

    for language in languages:
        source = work_dir / f"dictionary-{language}.txt"
        if not source.is_file():
            continue

        def resolve_phone(phone: str) -> str | None:
            if phone in phoneme_map:
                return phone
            candidate = f"{language}/{phone}"
            return candidate if candidate in phoneme_map else None

        entries: list[dict[str, Any]] = []
        used_symbols: set[str] = {phone for phone in ("AP", "SP") if phone in phoneme_map}
        for raw_line in source.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t", 1)
            if len(fields) != 2:
                continue
            grapheme, raw_phones = fields
            phones = [resolve_phone(phone) for phone in raw_phones.split()]
            if not phones or any(phone is None for phone in phones):
                missing = [
                    phone
                    for phone, resolved in zip(raw_phones.split(), phones)
                    if resolved is None
                ]
                print(
                    f"| warn: skip unsupported {language} dictionary entry "
                    f"{grapheme!r}: {missing}"
                )
                continue
            resolved = [str(phone) for phone in phones]
            used_symbols.update(resolved)
            entries.append({"grapheme": grapheme, "phonemes": resolved})

        # Phonetic hints must accept every token in this language, including
        # phones absent from the grapheme dictionary.
        used_symbols.update(
            symbol
            for symbol in phoneme_map
            if symbol in {"AP", "SP"} or symbol.startswith(f"{language}/")
        )
        data = {
            "symbols": [
                {"symbol": symbol, "type": duration_symbol_type(symbol)}
                for symbol in sorted(used_symbols)
            ],
            "entries": entries,
        }
        (duration_out / f"dsdict-{language}.yaml").write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )


def copy_duration_checkpoint(
    work_dir: Path,
    out: Path,
    ckpt_steps: int | None,
) -> str | None:
    if ckpt_steps is not None:
        candidate = work_dir / f"model_ckpt_steps_{ckpt_steps}.ckpt"
    else:
        candidates = sorted(
            work_dir.glob("model_ckpt_steps_*.ckpt"),
            key=lambda path: int(path.name.split("steps_", 1)[1].split(".", 1)[0]),
        )
        candidate = candidates[-1] if candidates else Path()
    if not candidate.is_file():
        print("| warn: no duration training checkpoint was found for the Python runtime")
        return None
    destination = out / candidate.name
    if candidate.resolve() != destination.resolve():
        shutil.copy2(candidate, destination)
    return destination.name


def export_duration_package(
    *,
    repo: Path,
    out: Path,
    exp_name: str,
    ckpt_steps: int | None,
    device: torch.device,
    include_python_checkpoint: bool,
    speaker_lock=None,
) -> str:
    from deployment.speaker_locked_export import release_manifest, verify_locked_forward, write_release_metadata
    from modules.duration import get_duration_type
    from utils.hparams import hparams, set_hparams
    from utils.phoneme_utils import load_phoneme_dictionary

    print(f"| exporting deterministic DSRX duration experiment {exp_name}")
    set_hparams(
        config=str(repo / "ckpt" / exp_name / "config.yaml"),
        exp_name=exp_name,
        print_hparams=False,
    )
    install_portable_rotary()
    if not hparams.get("predict_dur", False):
        raise ValueError("The selected duration checkpoint has predict_dur=false.")
    duration_type = get_duration_type(hparams["dur_prediction_args"])
    if duration_type != "deterministic":
        raise NotImplementedError(
            "Native duration export requires dur_prediction_args.type=deterministic; "
            "use scripts/infer.py variance for DDPM or ReFlow duration checkpoints."
        )

    work_dir = (repo / hparams["work_dir"]).resolve()
    duration_out = out / "dsdur"
    duration_out.mkdir(parents=True, exist_ok=True)
    phoneme_dictionary = load_phoneme_dictionary()
    model = build_variance_model(
        phoneme_dictionary=phoneme_dictionary,
        device=device,
        ckpt_steps=ckpt_steps,
    )
    if getattr(model.fs2, "duration_type", "deterministic") != "deterministic":
        raise RuntimeError("Loaded checkpoint is not a deterministic duration model.")

    hidden_size = int(hparams["hidden_size"])
    spk_map = load_json_map(work_dir / "spk_map.json")
    fixed_speaker = speaker_lock.apply(model, spk_map) if speaker_lock else None
    if speaker_lock:
        spk_map = {speaker_lock.speaker: spk_map[speaker_lock.speaker]}
    duration_model = DeterministicDurationModel(
        model.view_as_linguistic_encoder(),
        model.view_as_dur_predictor(),
        fixed_speaker,
    ).eval()
    example = make_duration_inputs(
        word_divisions=[2, 2, 1],
        word_durations=[8, 3, 4],
        hidden_size=hidden_size,
        device=device,
    )
    check = make_duration_inputs(
        word_divisions=[1, 3, 2, 2],
        word_durations=[4, 12, 7, 10],
        hidden_size=hidden_size,
        device=device,
    )
    print("| tracing deterministic duration model")
    duration_script = torch.jit.trace(
        duration_model,
        example,
        check_inputs=[check],
        strict=False,
    )
    duration_script = freeze_module(duration_script, "deterministic duration model", required=bool(speaker_lock))
    if speaker_lock:
        verify_locked_forward(duration_model, duration_script, check, 4)
    prediction_check = duration_script(*check)
    if tuple(prediction_check.shape) != (1, sum([1, 3, 2, 2])):
        raise RuntimeError(
            f"Exported duration model returned unexpected shape {tuple(prediction_check.shape)}."
        )
    if not torch.isfinite(prediction_check).all() or (prediction_check < 0).any():
        raise RuntimeError("Exported duration model returned invalid values.")
    model_path = duration_out / "duration_deterministic.pt"
    save_torchscript(duration_script, model_path)

    phonemes_path = duration_out / ("phonemes.json" if speaker_lock else f"{exp_name}.phonemes.json")
    phoneme_dictionary.dump(phonemes_path)
    phoneme_map = load_json_map(phonemes_path)
    lang_map = load_json_map(work_dir / "lang_map.json")
    languages_path = duration_out / ("languages.json" if speaker_lock else f"{exp_name}.languages.json")
    languages_path.write_text(
        json.dumps(lang_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    configured_dictionaries = hparams.get("dictionaries") or {}
    dictionary_languages = list(configured_dictionaries) if configured_dictionaries else list(lang_map)
    for language in dictionary_languages:
        source = work_dir / f"dictionary-{language}.txt"
        if source.is_file():
            shutil.copy2(source, duration_out / source.name)
    write_duration_dsdicts(
        work_dir=work_dir,
        duration_out=duration_out,
        phoneme_map=phoneme_map,
        languages=dictionary_languages,
    )
    for filename in ("config.yaml", "lang_map.json", "spk_map.json"):
        if speaker_lock and filename != "lang_map.json":
            continue
        source = work_dir / filename
        if source.is_file():
            shutil.copy2(source, duration_out / filename)

    if speaker_lock:
        write_release_metadata(duration_out, hparams, speaker_lock.speaker)
    speaker_files: dict[str, str] = {}
    for speaker, speaker_id in spk_map.items():
        filename = f"speaker.{speaker}.emb"
        embedding = fixed_speaker.reshape(-1) if speaker_lock else variance_speaker_embedding(model, int(speaker_id), device)
        embedding.detach().cpu().numpy().astype("float32").tofile(duration_out / filename)
        speaker_files[speaker] = filename

    import yaml

    dsconfig = {
        "phonemes": phonemes_path.name,
        "languages": languages_path.name,
        "use_lang_id": bool(hparams.get("use_lang_id", False)),
        "hidden_size": hidden_size,
        "predict_dur": True,
        "speakers": list(spk_map),
        "sample_rate": int(hparams["audio_sample_rate"]),
        "hop_size": int(hparams["hop_size"]),
    }
    (duration_out / "dsconfig.yaml").write_text(
        yaml.safe_dump(dsconfig, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    python_checkpoint = (
        copy_duration_checkpoint(work_dir, duration_out, ckpt_steps)
        if include_python_checkpoint
        else None
    )
    manifest = {
        "format_version": 1,
        "runtime": "libtorch-torchscript",
        "model_type": "deterministic",
        "device": device.type,
        "pytorch_version": torch.__version__,
        "model": model_path.name,
        "phonemes": phonemes_path.name,
        "languages": languages_path.name,
        "speakers": speaker_files,
        "cross_lingual_phonemes": sorted(phoneme_dictionary.cross_lingual_phonemes),
        "sample_rate": int(hparams["audio_sample_rate"]),
        "hop_size": int(hparams["hop_size"]),
        "hidden_size": hidden_size,
        "use_language_ids": bool(hparams.get("use_lang_id", False)),
        "python_checkpoint": python_checkpoint,
    }
    manifest_path = duration_out / "dsrx-duration.json"
    release_manifest(manifest, speaker_lock.speaker if speaker_lock else None)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    update_dsrx_yaml(out=out, duration_work_dir="dsdur")
    return manifest_path.relative_to(out).as_posix()


def export_variance_package(
    *,
    repo: Path,
    out: Path,
    exp_name: str,
    ckpt_steps: int | None,
    device: torch.device,
    include_python_checkpoint: bool,
    speaker_lock=None,
) -> str:
    from deployment.speaker_locked_export import release_manifest, verify_locked_forward, write_release_metadata
    from inference.optimization import optimize_model_for_inference
    from utils.hparams import hparams, set_hparams
    from utils.phoneme_utils import load_phoneme_dictionary

    print(f"| exporting DSRX variance experiment {exp_name}")
    # Passing an explicit config is important here: set_hparams' command-line
    # parser treats this script's --exp as an abbreviation of --exp_name.
    # Without the explicit path a second call would silently reload acoustic
    # hparams instead of the requested variance experiment.
    set_hparams(
        config=str(repo / "ckpt" / exp_name / "config.yaml"),
        exp_name=exp_name,
        print_hparams=False,
    )
    install_portable_rotary()
    if hparams.get("diffusion_type") != "reflow":
        raise NotImplementedError("OpenUtau native variance export currently requires reflow.")
    if hparams.get("predict_dur", False) or hparams.get("predict_pitch", False):
        raise NotImplementedError(
            "dsdur and dspitch are separate package modules; the dsvariance export "
            "requires predict_dur=false and predict_pitch=false."
        )

    work_dir = (repo / hparams["work_dir"]).resolve()
    variance_out = out / "dsvariance"
    variance_out.mkdir(parents=True, exist_ok=True)
    phoneme_dictionary = load_phoneme_dictionary()
    model = build_variance_model(
        phoneme_dictionary=phoneme_dictionary,
        device=device,
        ckpt_steps=ckpt_steps,
    )
    optimization_report = optimize_model_for_inference(
        model,
        model_kind="variance",
        device=device,
    )
    predicted_variances = list(model.variance_prediction_list)
    if not predicted_variances:
        raise ValueError("The selected variance checkpoint predicts no variance parameters.")

    hidden_size = int(hparams["hidden_size"])
    spk_map = load_json_map(work_dir / "spk_map.json")
    fixed_speaker = speaker_lock.apply(model, spk_map) if speaker_lock else None
    if speaker_lock:
        spk_map = {speaker_lock.speaker: spk_map[speaker_lock.speaker]}
    linguistic_model = model.view_as_linguistic_encoder()
    variance_preprocess = model.view_as_variance_preprocess()
    conditioner = VarianceConditioner(
        linguistic_model,
        variance_preprocess,
        predicted_variances,
        fixed_speaker,
    ).eval()
    example = make_variance_conditioner_inputs(
        token_count=4,
        frame_count=48,
        hidden_size=hidden_size,
        device=device,
    )
    check = make_variance_conditioner_inputs(
        token_count=7,
        frame_count=79,
        hidden_size=hidden_size,
        device=device,
    )
    print("| tracing variance conditioner")
    conditioner_script = torch.jit.trace(
        conditioner,
        example,
        check_inputs=[check],
        strict=False,
    )
    conditioner_script = freeze_module(conditioner_script, "variance conditioner", required=bool(speaker_lock))
    if speaker_lock:
        verify_locked_forward(conditioner, conditioner_script, check, 3)
    condition_check = conditioner_script(*check)
    assert tuple(condition_check.shape) == (1, 79, hidden_size)
    conditioner_path = variance_out / "variance_conditioner.pt"
    save_torchscript(conditioner_script, conditioner_path)

    print("| tracing fixed-step variance rectified-flow sampler")
    predictor_model = model.view_as_variance_predictor()
    variance_count = len(predicted_variances)
    repeat_bins = predictor_model.variance_predictor.out_dims
    condition_example = torch.randn((1, 48, hidden_size), device=device)
    noise_example = torch.randn(
        (1, variance_count, repeat_bins, 48),
        device=device,
    )
    time_example = (
        torch.rand((1,), device=device).float()
        * predictor_model.variance_predictor.time_scale_factor
    )
    predictor_model.variance_predictor.set_backbone(
        torch.jit.trace(
            predictor_model.variance_predictor.backbone,
            (noise_example, time_example, condition_example.transpose(1, 2)),
            check_inputs=[(
                torch.randn((1, variance_count, repeat_bins, 79), device=device),
                torch.rand((1,), device=device).float()
                * predictor_model.variance_predictor.time_scale_factor,
                torch.randn((1, hidden_size, 79), device=device),
            )],
            strict=False,
        )
    )
    sampling_steps = int(hparams.get("sampling_steps", 20))
    predictor_script = torch.jit.trace(
        FixedVarianceReflowSampler(predictor_model, sampling_steps).eval(),
        (condition_example,),
        check_trace=False,
        strict=False,
    )
    predictor_script = freeze_module(predictor_script, "variance rectified-flow sampler", required=bool(speaker_lock))
    predictor_check = predictor_script(
        torch.randn((1, 79, hidden_size), device=device)
    )
    assert tuple(predictor_check.shape) == (1, variance_count, 79)
    predictor_path = variance_out / "variance_reflow.pt"
    save_torchscript(predictor_script, predictor_path)

    dictionaries = hparams.get("dictionaries")
    if dictionaries:
        for language in dictionaries:
            filename = f"dictionary-{language}.txt"
            shutil.copy2(work_dir / filename, variance_out / filename)
    else:
        shutil.copy2(work_dir / "dictionary.txt", variance_out / "dictionary.txt")
    phonemes_path = variance_out / ("phonemes.json" if speaker_lock else f"{exp_name}.phonemes.json")
    phoneme_dictionary.dump(phonemes_path)
    lang_map = load_json_map(work_dir / "lang_map.json")
    languages_path = variance_out / ("languages.json" if speaker_lock else f"{exp_name}.languages.json")
    languages_path.write_text(
        json.dumps(lang_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for filename in ("config.yaml", "lang_map.json", "spk_map.json"):
        if speaker_lock and filename != "lang_map.json":
            continue
        source = work_dir / filename
        if source.is_file():
            shutil.copy2(source, variance_out / filename)

    if speaker_lock:
        write_release_metadata(variance_out, hparams, speaker_lock.speaker)
    speaker_files: dict[str, str] = {}
    for speaker, speaker_id in spk_map.items():
        filename = f"speaker.{speaker}.emb"
        embedding = fixed_speaker.reshape(-1) if speaker_lock else variance_speaker_embedding(model, int(speaker_id), device)
        embedding.detach().cpu().numpy().astype("float32").tofile(variance_out / filename)
        speaker_files[speaker] = filename

    python_checkpoint = (
        copy_python_checkpoint(work_dir, variance_out, ckpt_steps)
        if include_python_checkpoint
        else None
    )
    manifest = {
        "format_version": 1,
        "runtime": "libtorch-torchscript",
        "device": device.type,
        "pytorch_version": torch.__version__,
        "conditioner": conditioner_path.name,
        "predictor": predictor_path.name,
        "phonemes": phonemes_path.name,
        "languages": languages_path.name,
        "speakers": speaker_files,
        "predicted_variances": predicted_variances,
        "clamps": variance_clamps(predicted_variances, hparams),
        "cross_lingual_phonemes": sorted(phoneme_dictionary.cross_lingual_phonemes),
        "sample_rate": int(hparams["audio_sample_rate"]),
        "hop_size": int(hparams["hop_size"]),
        "hidden_size": hidden_size,
        "steps": sampling_steps,
        "use_language_ids": bool(
            hparams.get("use_lang_id", False)
            and phoneme_dictionary.cross_lingual_phonemes
        ),
        "python_checkpoint": python_checkpoint,
        "optimization": optimization_report.to_dict(),
    }
    manifest_path = variance_out / "dsrx-variance.json"
    release_manifest(manifest, speaker_lock.speaker if speaker_lock else None)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    update_dsrx_yaml(out=out, variance_work_dir="dsvariance")
    return manifest_path.relative_to(out).as_posix()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    from deployment.speaker_locked_export import (
        SpeakerLock, release_manifest, validate_locked_export,
        verify_locked_forward, write_release_metadata,
    )

    validate_locked_export(args)
    speaker_lock = SpeakerLock(args.lock_speaker) if args.lock_speaker else None
    if speaker_lock:
        import yaml

        for experiment in filter(None, (args.exp, args.duration_exp, args.variance_exp)):
            source_dir = repo / "ckpt" / experiment
            if args.lock_speaker not in load_json_map(source_dir / "spk_map.json"):
                raise ValueError(f"The selected speaker is absent from {experiment}.")
            config = yaml.safe_load((source_dir / "config.yaml").read_text(encoding="utf-8"))
            if config.get("lora", {}).get("enabled", False):
                raise NotImplementedError("Merge LoRA weights before exporting a locked release.")
    if args.duration_only or args.variance_only or args.vocoder_only:
        existing = out / "dsrx-native.json"
        if existing.is_file() and json.loads(existing.read_text(encoding="utf-8")).get("speaker_lock"):
            raise ValueError("Locked releases cannot be updated with an unlocked module; create a fresh locked export.")
    out.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export was requested, but PyTorch cannot access CUDA.")
    portable = args.device == "portable"
    # Trace the original FP32 model. CUDA-specific half-precision rewrites bake
    # casts into the graph and must not become the shared CPU/GPU weight copy.
    device = torch.device("cpu" if portable else args.device)

    from utils.hparams import hparams, set_hparams

    if args.duration_only:
        if not args.duration_exp:
            raise ValueError("--duration-only requires --duration-exp.")
        if args.variance_only or args.vocoder_only:
            raise ValueError(
                "--duration-only cannot be combined with --variance-only or --vocoder-only."
            )
        root_manifest_path = out / "dsrx-native.json"
        if not root_manifest_path.is_file():
            raise FileNotFoundError(
                f"--duration-only requires an existing {root_manifest_path}."
            )
        root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
        root_manifest["duration_manifest"] = export_duration_package(
            repo=repo,
            out=out,
            exp_name=args.duration_exp,
            ckpt_steps=args.duration_ckpt,
            device=device,
            include_python_checkpoint=args.include_python_checkpoint,
        )
        root_manifest_path.write_text(
            json.dumps(root_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if portable:
            from deployment.portable_torchscript import convert_package_to_portable
            convert_package_to_portable(out, root_manifest["duration_manifest"])
        print(f"| OpenUtau deterministic duration module replaced in {out}")
        return

    if args.variance_only:
        if not args.variance_exp:
            raise ValueError("--variance-only requires --variance-exp.")
        root_manifest_path = out / "dsrx-native.json"
        if not root_manifest_path.is_file():
            raise FileNotFoundError(
                f"--variance-only requires an existing {root_manifest_path}."
            )
        root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
        root_manifest["variance_manifest"] = export_variance_package(
            repo=repo,
            out=out,
            exp_name=args.variance_exp,
            ckpt_steps=args.variance_ckpt,
            device=device,
            include_python_checkpoint=args.include_python_checkpoint,
        )
        root_manifest_path.write_text(
            json.dumps(root_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if portable:
            from deployment.portable_torchscript import convert_package_to_portable
            convert_package_to_portable(out, root_manifest["variance_manifest"])
        print(f"| OpenUtau native variance module replaced in {out}")
        return

    set_hparams(config=str(repo / "ckpt" / args.exp / "config.yaml"),
                exp_name=args.exp, print_hparams=False)
    if args.vocoder_only:
        root_manifest_path = out / "dsrx-native.json"
        if not root_manifest_path.is_file():
            raise FileNotFoundError(
                f"--vocoder-only requires an existing {root_manifest_path}."
            )
        root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
        vocoder_type = normalize_vocoder_type(
            hparams.get("vocoder") if args.vocoder_type == "auto" else args.vocoder_type
        )
        vocoder_checkpoint = resolve_repo_path(
            repo,
            args.vocoder_ckpt or hparams["vocoder_ckpt"],
        )
        vocoder_manifest_path = export_vocoder(
            vocoder_type=vocoder_type,
            repo=repo,
            checkpoint=vocoder_checkpoint,
            output_dir=out / "dsvocoder",
            device=device,
            mel_bins=int(root_manifest["mel_bins"]),
            sample_rate=int(root_manifest["sample_rate"]),
            hop_size=int(root_manifest["hop_size"]),
            mel_base=str(hparams.get("mel_base", "10")),
        )
        root_manifest.pop("vocoder", None)
        root_manifest["vocoder_manifest"] = vocoder_manifest_path.relative_to(out).as_posix()
        root_manifest_path.write_text(
            json.dumps(root_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if portable:
            from deployment.portable_torchscript import convert_package_to_portable
            convert_package_to_portable(out, root_manifest["vocoder_manifest"])
        if args.include_python_checkpoint:
            python_vocoder_checkpoint = copy_python_vocoder(vocoder_checkpoint, out)
            update_dsrx_yaml(
                out=out,
                python_path=args.python_path,
                dsrx_root=args.dsrx_root,
                python_vocoder_checkpoint=python_vocoder_checkpoint,
            )
        print(f"| OpenUtau native vocoder replaced in {out}")
        return
    if hparams.get("diffusion_type") != "reflow":
        raise NotImplementedError("OpenUtau native export currently requires a reflow acoustic model.")
    if not hparams.get("use_shallow_diffusion", False):
        raise NotImplementedError("OpenUtau native export currently requires shallow diffusion.")

    install_portable_rotary()
    from inference.optimization import optimize_model_for_inference
    from utils.phoneme_utils import load_phoneme_dictionary

    work_dir = (repo / hparams["work_dir"]).resolve()
    spk_map = load_json_map(work_dir / "spk_map.json")
    lang_map = load_json_map(work_dir / "lang_map.json")
    phoneme_dictionary = load_phoneme_dictionary()
    use_language_ids = bool(
        hparams.get("use_lang_id", False)
        and phoneme_dictionary.cross_lingual_phonemes
    )
    model = build_acoustic_model(
        phoneme_dictionary=phoneme_dictionary,
        device=device,
        ckpt_steps=args.ckpt,
        strict_weights=bool(speaker_lock),
    )
    fixed_speaker = speaker_lock.apply(model, spk_map) if speaker_lock else None
    if speaker_lock:
        spk_map = {speaker_lock.speaker: spk_map[speaker_lock.speaker]}
    optimization_report = optimize_model_for_inference(
        model,
        model_kind="acoustic",
        device=device,
    )
    hidden_size = int(hparams["hidden_size"])
    mel_bins = int(hparams["audio_num_mel_bins"])
    enabled_variances = list(model.fs2.variance_embed_list)

    conditioner_model = model.view_as_fs2_aux()
    conditioner_model.fs2.lr = torch.jit.script(ScriptLengthRegulator())
    conditioner = AcousticConditioner(conditioner_model, enabled_variances, fixed_speaker).eval()
    example = make_conditioner_inputs(
        token_count=4,
        frame_count=48,
        hidden_size=hidden_size,
        device=device,
    )
    check = make_conditioner_inputs(
        token_count=7,
        frame_count=79,
        hidden_size=hidden_size,
        device=device,
    )
    print("| tracing acoustic conditioner")
    conditioner_script = torch.jit.trace(
        conditioner,
        example,
        check_inputs=[check],
        strict=False,
    )
    conditioner_script = freeze_module(conditioner_script, "acoustic conditioner", required=bool(speaker_lock))
    if speaker_lock:
        verify_locked_forward(conditioner, conditioner_script, check, 9)
    condition_check, auxiliary_check = conditioner_script(*check)
    assert tuple(condition_check.shape) == (1, 79, hidden_size)
    assert tuple(auxiliary_check.shape) == (1, 79, mel_bins)
    conditioner_path = out / "acoustic_conditioner.pt"
    save_torchscript(conditioner_script, conditioner_path)

    print("| tracing fixed-step rectified-flow sampler")
    condition_example = torch.randn((1, 48, hidden_size), device=device)
    noise_example = torch.randn((1, 1, mel_bins, 48), device=device)
    time_example = torch.rand((1,), device=device).float() * model.diffusion.time_scale_factor
    major_mel_decoder = model.view_as_reflow()
    major_mel_decoder.diffusion.set_backbone(
        torch.jit.trace(
            major_mel_decoder.diffusion.backbone,
            (noise_example, time_example, condition_example.transpose(1, 2)),
            check_inputs=[(
                torch.randn((1, 1, mel_bins, 79), device=device),
                torch.rand((1,), device=device).float() * model.diffusion.time_scale_factor,
                torch.randn((1, hidden_size, 79), device=device),
            )],
            strict=False,
        )
    )
    sampling_steps = int(hparams.get("sampling_steps", 20))
    reflow_script = torch.jit.trace(
        FixedReflowSampler(major_mel_decoder, sampling_steps).eval(),
        (
            condition_example,
            torch.randn((1, 48, mel_bins), device=device),
            torch.tensor(0.4, device=device),
        ),
        check_trace=False,
        strict=False,
    )
    reflow_script = freeze_module(reflow_script, "rectified-flow sampler", required=bool(speaker_lock))
    reflow_check = reflow_script(
        torch.randn((1, 79, hidden_size), device=device),
        torch.randn((1, 79, mel_bins), device=device),
        torch.tensor(0.4, device=device),
    )
    assert tuple(reflow_check.shape) == (1, 79, mel_bins)
    reflow_path = out / "acoustic_reflow.pt"
    save_torchscript(reflow_script, reflow_path)

    vocoder_type = normalize_vocoder_type(
        hparams.get("vocoder") if args.vocoder_type == "auto" else args.vocoder_type
    )
    vocoder_checkpoint = resolve_repo_path(
        repo,
        args.vocoder_ckpt or hparams["vocoder_ckpt"],
    )
    vocoder_manifest_path = export_vocoder(
        vocoder_type=vocoder_type,
        repo=repo,
        checkpoint=vocoder_checkpoint,
        output_dir=out / "dsvocoder",
        device=device,
        mel_bins=mel_bins,
        sample_rate=int(hparams["audio_sample_rate"]),
        hop_size=int(hparams["hop_size"]),
        mel_base=str(hparams.get("mel_base", "10")),
    )

    dictionaries = hparams.get("dictionaries")
    if dictionaries:
        for language in dictionaries:
            filename = f"dictionary-{language}.txt"
            shutil.copy2(work_dir / filename, out / filename)
    else:
        shutil.copy2(work_dir / "dictionary.txt", out / "dictionary.txt")
    phonemes_path = out / ("phonemes.json" if speaker_lock else f"{hparams['exp_name']}.phonemes.json")
    phoneme_dictionary.dump(phonemes_path)
    languages_path = out / ("languages.json" if speaker_lock else f"{hparams['exp_name']}.languages.json")
    languages_path.write_text(
        json.dumps(lang_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    shutil.copy2(work_dir / "lang_map.json", out / "lang_map.json")
    if speaker_lock:
        write_release_metadata(out, hparams, speaker_lock.speaker)
    else:
        shutil.copy2(work_dir / "spk_map.json", out / "spk_map.json")
    speaker_files: dict[str, str] = {}
    for speaker, speaker_id in spk_map.items():
        filename = f"speaker.{speaker}.emb"
        embedding = fixed_speaker.reshape(-1) if speaker_lock else speaker_embedding(model, int(speaker_id), device)
        embedding.detach().cpu().numpy().astype("float32").tofile(out / filename)
        speaker_files[speaker] = filename

    if not speaker_lock:
        shutil.copy2(work_dir / "config.yaml", out / "config.yaml")
    python_checkpoint = (
        copy_python_checkpoint(work_dir, out, args.ckpt)
        if args.include_python_checkpoint
        else None
    )
    python_vocoder_checkpoint = (
        copy_python_vocoder(vocoder_checkpoint, out)
        if args.include_python_checkpoint
        else None
    )
    speaker_names = [
        speaker
        for speaker, _ in sorted(
            spk_map.items(),
            key=lambda item: (int(item[1]), item[0]),
        )
    ]
    if not speaker_names:
        raise ValueError("spk_map.json does not declare any speakers.")
    default_speaker = args.default_speaker or speaker_names[0]
    if default_speaker not in spk_map:
        raise ValueError(f"Unknown --default-speaker: {default_speaker}")
    write_voicebank_metadata(
        out=out,
        name=args.voicebank_name or str(hparams["exp_name"]),
        speakers=speaker_names,
        default_speaker=default_speaker,
        python_path=args.python_path,
        dsrx_root=args.dsrx_root,
        python_vocoder_checkpoint=python_vocoder_checkpoint,
    )
    manifest = {
        "format_version": 1,
        "runtime": "libtorch-torchscript",
        "device": device.type,
        "pytorch_version": torch.__version__,
        "conditioner": conditioner_path.name,
        "acoustic": reflow_path.name,
        "vocoder_manifest": vocoder_manifest_path.relative_to(out).as_posix(),
        "phonemes": phonemes_path.name,
        "languages": languages_path.name,
        "speakers": speaker_files,
        "enabled_variances": enabled_variances,
        "cross_lingual_phonemes": sorted(phoneme_dictionary.cross_lingual_phonemes),
        "sample_rate": int(hparams["audio_sample_rate"]),
        "hop_size": int(hparams["hop_size"]),
        "hidden_size": hidden_size,
        "mel_bins": mel_bins,
        "depth": float(1.0 - model.diffusion.t_start),
        "steps": sampling_steps,
        "use_key_shift": bool(hparams.get("use_key_shift_embed", False)),
        "key_shift_range": hparams.get("augmentation_args", {})
            .get("random_pitch_shifting", {})
            .get("range", [-12.0, 12.0]),
        "use_speed": bool(hparams.get("use_speed_embed", False)),
        "speed_range": hparams.get("augmentation_args", {})
            .get("random_time_stretching", {})
            .get("range", [0.5, 2.0]),
        "use_language_ids": use_language_ids,
        "python_checkpoint": python_checkpoint,
        "python_vocoder_checkpoint": python_vocoder_checkpoint,
        "optimization": optimization_report.to_dict(),
    }
    if args.duration_exp:
        manifest["duration_manifest"] = export_duration_package(
            repo=repo,
            out=out,
            exp_name=args.duration_exp,
            ckpt_steps=args.duration_ckpt,
            device=device,
            include_python_checkpoint=args.include_python_checkpoint,
            speaker_lock=speaker_lock,
        )
    if args.variance_exp:
        manifest["variance_manifest"] = export_variance_package(
            repo=repo,
            out=out,
            exp_name=args.variance_exp,
            ckpt_steps=args.variance_ckpt,
            device=device,
            include_python_checkpoint=args.include_python_checkpoint,
            speaker_lock=speaker_lock,
        )
    manifest_path = out / "dsrx-native.json"
    release_manifest(manifest, speaker_lock.speaker if speaker_lock else None)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if portable:
        from deployment.portable_torchscript import convert_package_to_portable

        report = convert_package_to_portable(out)
        print(f"| portable CPU/CUDA package: rebound {sum(report.values())} device constants")
    if speaker_lock:
        report = speaker_lock.finalize(out)
        print(f"| locked native release privacy audit passed: {report['torchscript_modules_checked']} modules")
    print(f"| OpenUtau native package exported to {out}")


if __name__ == "__main__":
    main()
