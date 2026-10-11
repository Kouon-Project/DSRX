"""Export a fixed speaker and trim training metadata from native packages."""

from __future__ import annotations

import json
import pickletools
import re
import shutil
import struct
import zipfile
from pathlib import Path

import torch
from torch import Tensor, nn


class SpeakerLock:
    """Private audit state shared by the acoustic, duration and variance exports."""

    def __init__(self, speaker: str):
        self.speaker = speaker
        self.forbidden_names: set[str] = set()
        self.removed_vectors: list[bytes] = []

    def apply(self, model: nn.Module, speaker_map: dict) -> Tensor:
        vector, removed = isolate_speaker(model, speaker_map, self.speaker)
        self.forbidden_names.update(set(speaker_map) - {self.speaker})
        self.removed_vectors.extend(removed)
        return vector

    def finalize(self, package: Path) -> dict:
        return finalize_release(package, self.speaker, self.forbidden_names, self.removed_vectors)


def validate_locked_export(args) -> None:
    speaker = args.lock_speaker
    if not speaker:
        return
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", speaker):
        raise ValueError("--lock-speaker requires 1-64 letters, digits, underscores or hyphens, starting with a letter or digit.")
    if args.duration_only or args.variance_only or args.vocoder_only:
        raise ValueError("A locked release requires a complete export.")
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        raise ValueError("A locked release requires a new or empty output directory.")
    if args.include_python_checkpoint:
        raise ValueError("--lock-speaker requires --no-include-python-checkpoint.")
    if args.python_path or args.dsrx_root:
        raise ValueError("A locked native release must not include Python or repository paths.")
    if args.default_speaker and args.default_speaker != speaker:
        raise ValueError("The default speaker must match --lock-speaker.")
    args.default_speaker = speaker
    args.voicebank_name = args.voicebank_name or speaker


@torch.no_grad()
def isolate_speaker(model: nn.Module, speaker_map: dict, speaker: str) -> tuple[Tensor, list[bytes]]:
    """Keep one exact row, overwrite all others (including unmapped slots), then
    remove the entire lookup table. Only the returned fixed vector is exported.

    A private generator keeps this scrub from changing inference RNG state.
    Returned removed vectors are used only for an in-memory release audit.
    """
    if speaker not in speaker_map:
        raise ValueError(f"The selected model has no speaker {speaker!r}.")
    tables = [(name, child) for name, child in model.named_modules()
              if name.rsplit(".", 1)[-1] == "spk_embed"]
    if len(tables) != 1 or not isinstance(tables[0][1], nn.Embedding):
        raise ValueError("Speaker locking requires exactly one known embedding table.")
    name, table = tables[0]
    speaker_id = int(speaker_map[speaker])
    if not 0 <= speaker_id < table.num_embeddings:
        raise ValueError("Speaker ID is outside the embedding table.")
    # Fail on aliases: a named second identity must not share the retained row.
    if any(int(value) == speaker_id for key, value in speaker_map.items() if key != speaker):
        raise ValueError("The selected speaker ID has another named alias.")
    retained = table.weight[speaker_id].detach().clone()
    drop_ids = [index for index in range(table.num_embeddings) if index != speaker_id]
    removed = [table.weight[index].detach().float().cpu().numpy().tobytes() for index in drop_ids]
    generator = torch.Generator(device="cpu").manual_seed(0)
    noise = torch.randn((len(drop_ids), table.embedding_dim), generator=generator,
                        dtype=table.weight.dtype, device="cpu").to(table.weight.device)
    table.weight[drop_ids] = noise
    torch.testing.assert_close(table.weight[speaker_id], retained, rtol=0, atol=0)
    parent_name, _, attribute = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    delattr(parent, attribute)
    return retained.reshape(1, 1, -1), removed


@torch.no_grad()
def verify_locked_forward(eager: nn.Module, script, inputs: tuple[Tensor, ...], speaker_index: int) -> None:
    """Compare with ordinary retained-speaker inference and test ignored input."""
    fixed = eager.locked_speaker
    if fixed is None:
        return
    baseline_inputs = list(inputs)
    baseline_inputs[speaker_index] = fixed.expand_as(inputs[speaker_index])
    eager.locked_speaker = None
    try:
        expected = eager(*baseline_inputs)
    finally:
        eager.locked_speaker = fixed
    actual = script(*inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    altered = list(inputs)
    altered[speaker_index] = torch.full_like(inputs[speaker_index], float("nan"))
    torch.testing.assert_close(script(*altered), actual, rtol=0, atol=0)
    if list(list(script.graph.inputs())[speaker_index + 1].uses()):
        raise RuntimeError("Locked model still consumes its external speaker input.")
    # Check speaker tables by name; frozen RoPE parameters may remain after loading.
    if any("spk_embed" in name for name, _ in
           list(script.named_parameters()) + list(script.named_buffers())):
        raise RuntimeError("Locked model still contains a speaker lookup table.")


def inference_config(hparams: dict) -> dict:
    """Select the inference settings used by the editor's config readers."""
    result = {key: int(hparams[key]) for key in (
        "audio_sample_rate", "hop_size", "hidden_size", "audio_num_mel_bins",
    ) if key in hparams}
    for key in (
        "use_lang_id", "use_spk_id", "use_key_shift_embed", "use_speed_embed",
        "use_energy_embed", "use_breathiness_embed", "use_voicing_embed", "use_tension_embed",
        "predict_dur", "predict_pitch", "predict_energy", "predict_breathiness",
        "predict_voicing", "predict_tension",
    ):
        if key in hparams:
            result[key] = bool(hparams[key])
    result["num_spk"] = 1
    augmentation = {}
    for key in ("random_pitch_shifting", "random_time_stretching"):
        limits = hparams.get("augmentation_args", {}).get(key, {}).get("range")
        if limits is not None:
            augmentation[key] = {"range": [float(value) for value in limits]}
    if augmentation:
        result["augmentation_args"] = augmentation
    return result


def write_release_metadata(out: Path, hparams: dict, speaker: str) -> None:
    import yaml

    (out / "config.yaml").write_text(
        yaml.safe_dump(inference_config(hparams), allow_unicode=True, sort_keys=False), encoding="utf-8")
    (out / "spk_map.json").write_text(json.dumps({speaker: 0}, indent=2), encoding="utf-8")


def release_manifest(manifest: dict, speaker: str | None) -> dict:
    if speaker:
        for key in ("python_checkpoint", "python_vocoder_checkpoint", "optimization"):
            manifest.pop(key, None)
        manifest["speaker_lock"] = {
            "mode": "fixed", "speaker": speaker, "external_embedding_input": "ignored",
        }
    return manifest


def strip_torchscript_debug(path: Path) -> int:
    """Remove source-path debug records and preserve code and tensor bytes."""
    temporary = path.with_suffix(".redacted.tmp")
    removed = 0
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(temporary, "w") as target:
        for entry in source.infolist():
            if entry.filename.endswith(".debug_pkl"):
                removed += 1
                continue
            # Rebuild ZipInfo to avoid retaining archive comments/extra metadata.
            clean = zipfile.ZipInfo(entry.filename, date_time=(2000, 1, 1, 0, 0, 0))
            clean.compress_type = entry.compress_type
            with source.open(entry) as reader, target.open(clean, "w", force_zip64=True) as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
    temporary.replace(path)
    return removed


def _pickle_strings(data: bytes):
    for operation, argument, _ in pickletools.genops(data):
        if operation.name in {"UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8", "STRING"}:
            yield str(argument)


def audit_release(package: Path, speaker: str, forbidden_names: set[str], removed_vectors: list[bytes]) -> dict:
    """Inspect metadata, archive strings and raw tensor storages."""
    import yaml

    # A drive letter must not be the last character of a URL scheme (https:/).
    path_pattern = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|\\\\[A-Za-z0-9]|/(?:Users|home|mnt)/")
    names = sorted(name for name in forbidden_names if not name.isdecimal() and name != speaker)
    name_pattern = (re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(re.escape(name) for name in names)
                               + r")(?![A-Za-z0-9_])", re.IGNORECASE) if names else None)

    def inspect_text(value: str, location: str, *, dictionary: bool = False):
        if path_pattern.search(value):
            raise ValueError(f"Private absolute path remains in {location}.")
        # Dictionary words are language resources, not speaker identity metadata.
        if not dictionary and name_pattern and name_pattern.search(value):
            raise ValueError(f"Other-speaker identity metadata remains in {location}.")

    modules = 0
    file_count = 0
    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(package).as_posix()
        file_count += 1
        inspect_text(relative, "file name")
        if path.suffix == ".pt":
            modules += 1
            with zipfile.ZipFile(path) as archive:
                for entry in archive.infolist():
                    if entry.filename.endswith(".debug_pkl"):
                        raise ValueError(f"Unredacted debug metadata in {relative}.")
                    data = archive.read(entry)
                    if entry.filename.endswith(".py"):
                        inspect_text(data.decode("utf-8"), relative)
                    elif entry.filename.endswith(".pkl"):
                        for value in _pickle_strings(data):
                            inspect_text(value, relative)
                    elif re.search(r"/(?:constants|data)/\d+$", entry.filename):
                        if any(vector and vector in data for vector in removed_vectors):
                            raise ValueError(f"A removed speaker vector remains in {relative}.")
        elif path.suffix == ".emb":
            if path.name != f"speaker.{speaker}.emb":
                raise ValueError("A non-target speaker embedding remains in the release.")
        elif path.suffix.lower() == ".png":
            # Artwork is an allowed release asset, but embedded EXIF/XMP/text
            # can reintroduce personal names, paths and editing history.
            allowed_chunks = {b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS", b"gAMA",
                              b"cHRM", b"sRGB", b"iCCP", b"cICP", b"pHYs", b"sBIT", b"bKGD"}
            with path.open("rb") as stream:
                if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                    raise ValueError(f"Invalid PNG artwork in {relative}.")
                while True:
                    header = stream.read(8)
                    if len(header) != 8:
                        raise ValueError(f"Incomplete PNG artwork in {relative}.")
                    length, kind = struct.unpack(">I4s", header)
                    if kind not in allowed_chunks:
                        raise ValueError(f"Unreviewed PNG metadata in {relative}: {kind!r}.")
                    if kind == b"cICP" and length != 4:
                        raise ValueError(f"Invalid PNG color information in {relative}.")
                    if stream.tell() + length + 4 > path.stat().st_size:
                        raise ValueError(f"Incomplete PNG chunk in {relative}.")
                    stream.seek(length + 4, 1)
                    if kind == b"IEND":
                        if stream.read(1):
                            raise ValueError(f"Unexpected trailing PNG data in {relative}.")
                        break
        elif path.suffix in {".json", ".yaml", ".yml", ".txt", ".md"}:
            contents = path.read_text(encoding="utf-8-sig")
            dictionary = path.name.startswith(("dictionary", "dsdict-"))
            inspect_text(contents, relative, dictionary=dictionary)
            if path.name == "spk_map.json" and json.loads(contents) != {speaker: 0}:
                raise ValueError(f"Non-isolated speaker map in {relative}.")
            if path.name in {"dsrx-native.json", "dsrx-duration.json", "dsrx-variance.json"}:
                metadata = json.loads(contents)
                if set(metadata.get("speakers", {})) != {speaker}:
                    raise ValueError(f"Non-isolated speakers in {relative}.")
                if metadata.get("speaker_lock", {}).get("speaker") != speaker:
                    raise ValueError(f"Speaker lock is missing in {relative}.")
            if path.name == "config.yaml":
                metadata = yaml.safe_load(contents)
                if metadata != inference_config(metadata):
                    raise ValueError(f"Non-inference config fields remain in {relative}.")
        else:
            raise ValueError(f"Unexpected release artifact {relative}.")
    return {"files_checked": file_count, "torchscript_modules_checked": modules,
            "private_metadata_found": False, "removed_speaker_vectors_found": False}


def finalize_release(package: Path, speaker: str, forbidden_names: set[str], removed_vectors: list[bytes]) -> dict:
    # Retain dictionary entries and remove source comments.
    for path in package.rglob("dictionary*.txt"):
        records = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
                   if line.strip() and not line.lstrip().startswith("#")]
        path.write_text("\n".join(records) + "\n", encoding="utf-8")
    debug_records = sum(strip_torchscript_debug(path) for path in package.rglob("*.pt"))
    report = audit_release(package, speaker, forbidden_names, removed_vectors)
    report.update({"format_version": 1, "speaker": speaker, "native_only": True,
                   "speaker_lock": "fixed", "source_debug_records_removed": debug_records})
    (package / "release-info.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
