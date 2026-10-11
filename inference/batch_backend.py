import argparse
import gc
import json
import os
import socketserver
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("PYTHONPATH", str(ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.ds_acoustic import DiffSingerAcousticInfer
from inference.ds_variance import DiffSingerVarianceInfer
from modules.fastspeech.param_adaptor import VARIANCE_CHECKLIST
from utils.hparams import hparams, set_hparams
from utils.infer_utils import cross_fade, parse_commandline_spk_mix, save_wav


ReportFn = Callable[[str, Dict], None]


def _pad_2d(tensor: torch.Tensor, target_len: int, value: float = 0.0) -> torch.Tensor:
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.size(1) == target_len:
        return tensor
    return F.pad(tensor, (0, target_len - tensor.size(1)), value=value)


def _pad_3d(tensor: torch.Tensor, target_len: int, value: float = 0.0) -> torch.Tensor:
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.size(1) == target_len:
        return tensor
    return F.pad(tensor, (0, 0, 0, target_len - tensor.size(1)), value=value)


def _expand_or_pad(tensor: torch.Tensor, target_len: int, value: float = 0.0) -> torch.Tensor:
    if tensor.size(1) == target_len:
        return tensor
    if tensor.size(1) == 1:
        repeats = [1 for _ in tensor.shape]
        repeats[1] = target_len
        return tensor.repeat(*repeats)
    if tensor.dim() == 2:
        return _pad_2d(tensor, target_len, value=value)
    return _pad_3d(tensor, target_len, value=value)


def _cross_fade_nd(a: np.ndarray, b: np.ndarray, idx: int) -> np.ndarray:
    """
    Cross fade two arrays along the first axis. Supports 2D (frames x channels) or 1D.
    """
    if a.ndim == 1 and b.ndim == 1:
        return cross_fade(a, b, idx)
    fade_len = a.shape[0] - idx
    result = np.zeros((idx + b.shape[0],) + a.shape[1:], dtype=np.float32)
    result[:idx] = a[:idx]
    k = np.linspace(0.0, 1.0, num=fade_len, endpoint=True, dtype=np.float32)[:, None]
    result[idx:a.shape[0]] = (1.0 - k) * a[idx:] + k * b[:fade_len]
    result[a.shape[0]:] = b[fade_len:]
    return result


class BatchInferenceBackend:
    """
    TCP-based inference backend with cold/hot start, batching and stitching.
    """

    def __init__(
        self,
        *,
        batch_size: int = 2,
        ckpt_steps: Optional[int] = None,
        device: Optional[str] = None,
        lazy_load: bool = False,
        auto_unload_minutes: Optional[float] = None,
        auto_unload_vram_gb: Optional[float] = None,
        monitor_interval: float = 5.0,
    ):
        self.batch_size = batch_size
        self.ckpt_steps = ckpt_steps
        self.device = device
        self.lazy_load = lazy_load
        self.auto_unload_minutes = auto_unload_minutes
        self.auto_unload_vram_gb = auto_unload_vram_gb
        self.monitor_interval = monitor_interval

        self._infer: Optional[DiffSingerAcousticInfer] = None
        self._variance_infer: Optional[DiffSingerVarianceInfer] = None
        self._variance_predictions: Optional[set] = None
        self._lock = threading.Lock()
        self._last_active = time.time()

        if not self.lazy_load:
            self._load_model(lambda *_args, **_kwargs: None)
        self._start_monitor()

    @property
    def infer(self) -> DiffSingerAcousticInfer:
        assert self._infer is not None, "Model is not loaded."
        return self._infer

    def _load_model(self, reporter: ReportFn):
        if self._infer is not None:
            return
        reporter("status", {"state": "loading_model"})
        self._infer = DiffSingerAcousticInfer(
            device=self.device,
            load_model=True,
            load_vocoder=True,
            ckpt_steps=self.ckpt_steps,
        )
        vocoder_optimization = self._infer.warmup_vocoder()
        optimization = getattr(self._infer, "inference_optimization", None)
        reporter(
            "status",
            {
                "state": "model_ready",
                "optimization": optimization.to_dict() if optimization else None,
                "vocoder_optimization": vocoder_optimization,
            },
        )
        self._last_active = time.time()

    def _load_variance_model(self, reporter: ReportFn, predictions: set):
        if self._variance_infer is not None and self._variance_predictions == predictions:
            return
        if self._variance_infer is not None and self._variance_predictions != predictions:
            del self._variance_infer
            self._variance_infer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        reporter("status", {"state": "loading_variance_model"})
        self._variance_infer = DiffSingerVarianceInfer(
            device=self.device,
            ckpt_steps=self.ckpt_steps,
            predictions=predictions,
        )
        self._variance_predictions = predictions
        optimization = getattr(self._variance_infer, "inference_optimization", None)
        reporter(
            "status",
            {
                "state": "variance_model_ready",
                "optimization": optimization.to_dict() if optimization else None,
            },
        )
        self._last_active = time.time()

    def _unload_model(self, reason: str):
        if self._infer is None and self._variance_infer is None:
            return
        print(f"| backend unloading model ({reason})")
        with self._lock:
            try:
                if self._infer is not None:
                    del self._infer
                    self._infer = None
                if self._variance_infer is not None:
                    del self._variance_infer
                    self._variance_infer = None
                    self._variance_predictions = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                print(f"| backend unload failed: {exc}")

    def _get_vram_used_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        try:
            device = torch.device(self.device if self.device else "cuda")
            free, total = torch.cuda.mem_get_info(device)
            used = total - free
            return used / (1024 ** 3)
        except Exception:
            return 0.0

    def _start_monitor(self):
        if self.auto_unload_minutes is None and self.auto_unload_vram_gb is None:
            return

        def _loop():
            while True:
                time.sleep(self.monitor_interval)
                if self._infer is None and self._variance_infer is None:
                    continue
                now = time.time()
                idle = (now - self._last_active) / 60.0
                vram = self._get_vram_used_gb()
                should_unload = False
                reason = ""
                if self.auto_unload_minutes is not None and idle >= self.auto_unload_minutes:
                    should_unload = True
                    reason = f"idle {idle:.2f} min >= {self.auto_unload_minutes} min"
                if (
                    not should_unload
                    and self.auto_unload_vram_gb is not None
                    and vram >= self.auto_unload_vram_gb
                ):
                    should_unload = True
                    reason = f"vram {vram:.2f} GB >= {self.auto_unload_vram_gb} GB"
                if should_unload:
                    self._unload_model(reason)

        t = threading.Thread(target=_loop, daemon=True)
        t.start()

    def _prepare_segments(self, params: List[dict]) -> List[Dict]:
        prepared = []
        for idx, param in enumerate(params):
            batch = self.infer.preprocess_input(param, idx)
            if hparams.get("use_spk_id", False):
                # Ensure spk_mix_embed exists for batching
                if "spk_mix_embed" not in batch and "spk_mix_id" in batch and "spk_mix_value" in batch:
                    with torch.inference_mode():
                        batch["spk_mix_embed"] = torch.sum(
                            self.infer.model.fs2.spk_embed(batch["spk_mix_id"])
                            * batch["spk_mix_value"].unsqueeze(3),
                            dim=2,
                            keepdim=False,
                        )
            prepared.append(
                {
                    "meta": {
                        "offset": float(param.get("offset", 0.0)),
                        "seed": int(param.get("seed", -1)),
                        "index": idx,
                    },
                    "batch": batch,
                    "mel_len": batch["mel2ph"].size(1),
                    "txt_len": batch["tokens"].size(1),
                }
            )
        return prepared

    def _collate(self, items: List[Dict]) -> Tuple[Dict[str, torch.Tensor], List[int]]:
        max_mel = max(i["mel_len"] for i in items)
        max_txt = max(i["txt_len"] for i in items)

        all_keys = set()
        for item in items:
            all_keys |= set(item["batch"].keys())
        reference = {
            key: next(item["batch"][key] for item in items if key in item["batch"])
            for key in all_keys
        }

        collated: Dict[str, List[torch.Tensor]] = {k: [] for k in all_keys}
        lengths: List[int] = []
        for item in items:
            batch = item["batch"]
            lengths.append(item["mel_len"])
            for key in all_keys:
                tensor = batch.get(key)
                if tensor is None:
                    ref = reference[key]
                    tensor = torch.zeros_like(ref, device=self.infer.device)
                if key in {"tokens", "languages"}:
                    padded = _pad_2d(tensor, max_txt, value=0)
                elif key in {"mel2ph", "f0"}:
                    padded = _pad_2d(tensor, max_mel, value=0)
                elif key in {"key_shift", "speed"}:
                    padded = _expand_or_pad(tensor, max_mel, value=0)
                elif key in {"spk_mix_value", "spk_mix_embed"}:
                    padded = _expand_or_pad(tensor, max_mel, value=0)
                elif key in {"spk_mix_id"}:
                    padded = tensor
                else:
                    padded = _expand_or_pad(tensor, max_mel, value=0)
                collated[key].append(padded)

        stacked = {k: torch.cat(v, dim=0) for k, v in collated.items()}
        return stacked, lengths

    def _stitch_audio(self, pieces: List[Dict]) -> np.ndarray:
        # pieces: [{'offset': float, 'audio': np.ndarray}]
        pieces = sorted(pieces, key=lambda x: x["offset"])
        result = np.zeros(0, dtype=np.float32)
        current_length = 0
        sr = hparams["audio_sample_rate"]
        for piece in pieces:
            start = int(round(piece["offset"] * sr))
            audio = piece["audio"]
            silent = start - current_length
            if silent >= 0:
                result = np.append(result, np.zeros(silent, dtype=np.float32))
                result = np.append(result, audio)
            else:
                result = cross_fade(result, audio, current_length + silent)
            current_length = max(current_length, start + audio.shape[0])
        return result

    def _stitch_mel(self, pieces: List[Dict]) -> np.ndarray:
        if not pieces:
            return np.zeros((0, 0), dtype=np.float32)
        pieces = sorted(pieces, key=lambda x: x["offset"])
        result = np.zeros((0, pieces[0]["mel"].shape[1]), dtype=np.float32)
        current_length = 0
        timestep = hparams["hop_size"] / hparams["audio_sample_rate"]
        for piece in pieces:
            start = int(round(piece["offset"] / timestep))
            mel = piece["mel"]
            silent = start - current_length
            if silent >= 0:
                result = np.concatenate(
                    [result, np.zeros((silent, result.shape[1]), dtype=np.float32)], axis=0
                )
                result = np.concatenate([result, mel], axis=0)
            else:
                result = _cross_fade_nd(result, mel, current_length + silent)
            current_length = max(current_length, start + mel.shape[0])
        return result

    def infer_segments(
        self,
        params: List[dict],
        *,
        batch_size: Optional[int] = None,
        seed: int = -1,
        return_type: str = "wav",
        lang_override: Optional[str] = None,
        spk_mix_str: Optional[str] = None,
        reporter: ReportFn,
    ) -> Dict:
        """
        Run batched inference and stitch outputs according to offsets.
        """
        if not params:
            raise ValueError("No segments provided for inference.")
        if return_type not in {"wav", "mel"}:
            raise ValueError(f"Unsupported return_type: {return_type}")
        params = [dict(p) for p in params]  # shallow copy for mutation
        if lang_override is not None:
            for p in params:
                p["lang"] = lang_override
        spk_mix = None
        if spk_mix_str:
            spk_mix = parse_commandline_spk_mix(spk_mix_str)
            for p in params:
                p["ph_spk_mix_backup"] = p.get("ph_spk_mix")
                p["spk_mix_backup"] = p.get("spk_mix")
                p["ph_spk_mix"] = p["spk_mix"] = spk_mix
        self._load_model(reporter)
        effective_batch = batch_size or self.batch_size
        reporter("status", {"state": "preprocessing", "segments": len(params)})
        prepared = self._prepare_segments(params)

        results_audio: List[Dict] = []
        results_mel: List[Dict] = []
        total_start = time.perf_counter()
        batch_index = 0
        for start in range(0, len(prepared), effective_batch):
            chunk = prepared[start : start + effective_batch]
            # respect per-segment seeds; if multiple seeds differ, fall back to single inference
            if len(chunk) == 0:
                continue
            chunk_seeds = []
            for item in chunk:
                local_seed = item["meta"]["seed"]
                if local_seed < 0:
                    local_seed = seed
                chunk_seeds.append(local_seed if local_seed is not None else -1)
            unique_seed_set = {s for s in chunk_seeds if s >= 0}
            has_unseeded = any(s < 0 for s in chunk_seeds)
            if len(unique_seed_set) > 1:
                sub_chunks = [[item] for item in chunk]
                sub_seeds = [item["meta"]["seed"] if item["meta"]["seed"] >= 0 else seed for item in chunk]
            else:
                sub_chunks = [chunk]
                sub_seeds = [next(iter(unique_seed_set)) if unique_seed_set and not has_unseeded else None]

            for sub, sub_seed in zip(sub_chunks, sub_seeds):
                batch_start = time.perf_counter()
                if sub_seed is not None and sub_seed >= 0:
                    torch.manual_seed(sub_seed & 0xFFFF_FFFF)
                    torch.cuda.manual_seed_all(sub_seed & 0xFFFF_FFFF)
                reporter(
                    "status",
                    {
                        "state": "running_batch",
                        "batch_index": batch_index,
                        "batch_size": len(sub),
                        "offset_range": [
                            float(sub[0]["meta"]["offset"]),
                            float(sub[-1]["meta"]["offset"]),
                        ],
                    },
                )
                collated, mel_lengths = self._collate(sub)
                collated = {k: (v.to(self.infer.device) if torch.is_tensor(v) else v) for k, v in collated.items()}
                with torch.inference_mode():
                    mel_pred = self.infer.forward_model(collated)

                # run vocoder or collect mel
                for i, item in enumerate(sub):
                    mel_len = mel_lengths[i]
                    seg_meta = item["meta"]
                    mel_slice = mel_pred[i : i + 1, :mel_len]
                    if return_type == "wav":
                        f0_slice = collated["f0"][i : i + 1, :mel_len]
                        with torch.inference_mode():
                            audio = (
                                self.infer.run_vocoder(
                                    mel_slice, f0=f0_slice, use_vocoder_buckets=True
                                )[0]
                                .cpu()
                                .numpy()
                                .astype(np.float32)
                            )
                        results_audio.append({"offset": seg_meta["offset"], "audio": audio})
                    else:
                        results_mel.append(
                            {"offset": seg_meta["offset"], "mel": mel_slice.cpu().squeeze(0).numpy()}
                        )
                reporter(
                    "status",
                    {
                        "state": "batch_done",
                        "batch_index": batch_index,
                        "batch_size": len(sub),
                        "elapsed": round(time.perf_counter() - batch_start, 3),
                        "offset_range": [
                            float(sub[0]["meta"]["offset"]),
                            float(sub[-1]["meta"]["offset"]),
                        ],
                    },
                )
                batch_index += 1

        if return_type == "wav":
            stitched = self._stitch_audio(results_audio)
            return {
                "waveform": stitched,
                "duration": round(stitched.shape[0] / hparams["audio_sample_rate"], 3),
                "runtime": round(time.perf_counter() - total_start, 3),
            }
        stitched_mel = self._stitch_mel(results_mel)
        return {
            "mel": stitched_mel,
            "frames": stitched_mel.shape[0],
            "runtime": round(time.perf_counter() - total_start, 3),
        }

    def infer_variance(
        self,
        params: List[dict],
        *,
        seed: int = -1,
        lang_override: Optional[str] = None,
        spk_mix_str: Optional[str] = None,
        out_dir: Path,
        title: str,
        predictions: set,
        reporter: ReportFn,
    ) -> Path:
        if not params:
            raise ValueError("No segments provided for variance inference.")
        params = [dict(p) for p in params]
        if lang_override is not None:
            for p in params:
                p["lang"] = lang_override
        if spk_mix_str:
            spk_mix = parse_commandline_spk_mix(spk_mix_str)
            for p in params:
                p["ph_spk_mix_backup"] = p.get("ph_spk_mix")
                p["spk_mix_backup"] = p.get("spk_mix")
                p["ph_spk_mix"] = p["spk_mix"] = spk_mix
        self._load_variance_model(reporter, predictions)
        reporter("status", {"state": "variance_preprocessing", "segments": len(params)})
        start = time.perf_counter()
        self._variance_infer.run_inference(
            params, out_dir=out_dir, title=title, num_runs=1, seed=seed
        )
        elapsed = round(time.perf_counter() - start, 3)
        self._last_active = time.time()
        reporter("status", {"state": "variance_done", "elapsed": elapsed})
        save_path = out_dir / f"{title}.ds"
        return save_path


class _InferenceTCPHandler(socketserver.StreamRequestHandler):
    backend: BatchInferenceBackend = None  # injected before serving

    def _send(self, event: str, payload: Dict):
        data = {"event": event, **payload}
        raw = (json.dumps(data) + "\n").encode("utf-8")
        self.wfile.write(raw)
        self.wfile.flush()

    def handle(self):
        try:
            raw = self.rfile.readline()
            if not raw:
                return
            request = json.loads(raw.decode("utf-8").strip())
        except Exception as exc:
            self._send("error", {"message": f"Invalid request: {exc}"})
            return

        action = request.get("action")
        if action == "ping":
            self._send("pong", {"message": "alive"})
            return
        if action != "infer":
            self._send("error", {"message": f"Unsupported action: {action}"})
            return

        job_id = request.get("job_id") or str(uuid.uuid4())
        params = request.get("segments") or []
        options = request.get("options") or {}
        batch_size = options.get("batch_size")
        seed = int(options.get("seed", -1))
        return_type = options.get("return_type", "wav")
        output_path = options.get("output_path")
        lang_override = options.get("lang")
        spk_mix_str = options.get("spk")
        task = options.get("task", "acoustic")
        predict_opt = options.get("predict")

        def reporter(event: str, payload: Dict):
            self._send(event, {"job_id": job_id, **payload})

        reporter("status", {"state": "received", "segments": len(params)})
        with self.backend._lock:
            try:
                if task == "variance":
                    if predict_opt in (None, "all", ""):
                        predictions = {"dur", "pitch", *VARIANCE_CHECKLIST}
                    else:
                        if isinstance(predict_opt, str):
                            predictions = {p.strip() for p in predict_opt.split(",") if p.strip()}
                        elif isinstance(predict_opt, list):
                            predictions = set(predict_opt)
                        else:
                            predictions = set(VARIANCE_CHECKLIST)
                        # normalize common aliases
                        norm = set()
                        for p in predictions:
                            if p.lower() in {"dur", "duration"}:
                                norm.add("dur")
                            elif p.lower() == "pitch":
                                norm.add("pitch")
                            else:
                                norm.add(p)
                        predictions = norm
                    if output_path:
                        output_file = Path(output_path)
                        out_dir = output_file.parent
                        title = output_file.stem
                    else:
                        output_dir = Path(options.get("output_dir", "infer_output"))
                        out_dir = output_dir
                        title = job_id
                    out_dir.mkdir(parents=True, exist_ok=True)
                    save_path = self.backend.infer_variance(
                        params,
                        seed=seed,
                        lang_override=lang_override,
                        spk_mix_str=spk_mix_str,
                        out_dir=out_dir,
                        title=title,
                        predictions=predictions,
                        reporter=reporter,
                    )
                    reporter(
                        "result",
                        {
                            "job_id": job_id,
                            "path": str(save_path),
                            "mode": "ds",
                        },
                    )
                else:
                    result = self.backend.infer_segments(
                        params,
                        batch_size=batch_size,
                        seed=seed,
                        return_type=return_type,
                        lang_override=lang_override,
                        spk_mix_str=spk_mix_str,
                        reporter=reporter,
                    )
                    if output_path:
                        output_file = Path(output_path)
                    else:
                        suffix = ".pt" if return_type == "mel" else ".wav"
                        output_dir = Path(options.get("output_dir", "infer_output"))
                        output_file = output_dir / f"{job_id}{suffix}"
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    if return_type == "wav":
                        reporter("status", {"state": "saving", "path": str(output_file)})
                        save_wav(result["waveform"], output_file, hparams["audio_sample_rate"])
                        reporter(
                            "result",
                            {
                                "job_id": job_id,
                                "path": str(output_file),
                                "seconds": result["duration"],
                                "runtime": result.get("runtime"),
                                "mode": "wav",
                            },
                        )
                    else:
                        reporter("status", {"state": "saving", "path": str(output_file)})
                        torch.save(result, output_file)
                        reporter(
                            "result",
                            {
                                "job_id": job_id,
                                "path": str(output_file),
                                "frames": result["frames"],
                                "runtime": result.get("runtime"),
                                "mode": "mel",
                            },
                        )
            except Exception as exc:
                reporter("error", {"message": str(exc), "trace": traceback.format_exc()})


def _start_server(host: str, port: int, backend: BatchInferenceBackend):
    _InferenceTCPHandler.backend = backend
    with socketserver.ThreadingTCPServer((host, port), _InferenceTCPHandler) as server:
        print(f"| backend listening on {host}:{port}")
        server.serve_forever()


def parse_args():
    parser = argparse.ArgumentParser(description="Batched DiffSinger inference backend")
    parser.add_argument("--config", type=str, default="", help="Path to config yaml")
    parser.add_argument("--exp", type=str, default="", help="Experiment name under ckpt/")
    parser.add_argument("--hparams", type=str, default="", help="Override hparams, k=v,k2=v2")
    parser.add_argument("--vocoder-ckpt", type=str, default="", help="Absolute packaged vocoder checkpoint override")
    parser.add_argument("--ckpt", type=int, default=None, help="Checkpoint step override")
    parser.add_argument("--batch-size", type=int, default=2, help="Max batch size for inference")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=10086, help="Bind port")
    parser.add_argument("--lazy-load", action="store_true", help="Delay model load until first request")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g., cuda:0)")
    parser.add_argument("--auto-unload-minutes", type=float, default=None, help="Auto unload after idle minutes")
    parser.add_argument("--auto-unload-vram-gb", type=float, default=None, help="Auto unload if VRAM exceeds GB")
    parser.add_argument("--monitor-interval", type=float, default=5.0, help="Monitor interval in seconds")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.config and not args.exp:
        raise SystemExit("Either --config or --exp must be provided.")

    config_path = (
        Path(args.config)
        if args.config
        else ROOT / "ckpt" / args.exp / "config.yaml"
    )
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    set_hparams(
        config=str(config_path),
        exp_name=args.exp,
        hparams_str=args.hparams,
        global_hparams=True,
        print_hparams=True,
    )
    if not args.exp:
        # A standalone saved config is expected to live beside its checkpoint.
        hparams["work_dir"] = str(config_path.resolve().parent)
    if args.vocoder_ckpt:
        hparams["vocoder_ckpt"] = str(Path(args.vocoder_ckpt).expanduser().resolve())

    backend = BatchInferenceBackend(
        batch_size=args.batch_size,
        ckpt_steps=args.ckpt,
        device=args.device,
        lazy_load=args.lazy_load,
        auto_unload_minutes=args.auto_unload_minutes,
        auto_unload_vram_gb=args.auto_unload_vram_gb,
        monitor_interval=args.monitor_interval,
    )
    _start_server(args.host, args.port, backend)


if __name__ == "__main__":
    main()
