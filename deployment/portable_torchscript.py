"""Device-neutral TorchScript packages for OpenUtau's LibTorch runtime.

Tracing evaluates ``tensor.device`` in Python and can capture a literal CPU or
CUDA device. Loading the weights with map_location does not change those IR
constants. Replace them with the device of a tensor argument, after freezing.
Weights remain a single FP32 copy stored on CPU; LibTorch moves that copy when
loading. Do not run this pass on CUDA-optimized/FP16 exports.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def _nodes(block):
    for node in list(block.nodes()):
        yield node
        for child in node.blocks():
            yield from _nodes(child)


def make_device_agnostic(module: torch.jit.ScriptModule) -> int:
    """Rebind explicit device arguments without changing tensor values/dtypes."""
    replaced = 0
    for child in module.modules():
        for method_name in child._c._method_names():
            graph = child._c._get_method(method_name).graph
            constants = [
                node for node in _nodes(graph)
                if node.kind() == "prim::Constant"
                and node.output().type().kind() == "DeviceObjType"
            ]
            if not constants:
                continue
            tensor_input = next(
                (value for value in graph.inputs() if value.type().kind() == "TensorType"),
                None,
            )
            if tensor_input is None:
                raise ValueError(f"{method_name}: a device constant has no tensor argument.")
            device_node = graph.create("prim::device", [tensor_input])
            device_node.output().setType(torch._C.DeviceObjType.get())
            device_node.insertBefore(next(graph.nodes()))
            for node in constants:
                device = node.output().toIValue()
                if device.type not in {"cpu", "cuda"}:
                    raise ValueError(f"Unsupported traced device {device} in {method_name}.")
                node.output().replaceAllUsesWith(device_node.output())
                node.destroy()
                replaced += 1
            torch._C._jit_pass_dce(graph)
            graph.lint()
    return replaced


def simplify_inference_graph(module: torch.jit.ScriptModule) -> int:
    """Remove redundant expressions with alias-aware common-subexpression elimination."""
    removed = 0
    for child in module.modules():
        for method_name in child._c._method_names():
            graph = child._c._get_method(method_name).graph
            before = len(list(_nodes(graph)))
            torch._C._jit_pass_cse(graph)
            torch._C._jit_pass_dce(graph)
            graph.lint()
            removed += before - len(list(_nodes(graph)))
    return removed


def convert_package_to_portable(package: Path, manifest_file: str = "dsrx-native.json") -> dict[str, int]:
    """Convert a newly exported FP32 CPU package, including its linked modules."""
    package = package.resolve()
    manifests: dict[Path, dict] = {}
    models: set[Path] = set()

    def resolve(parent: Path, filename: str) -> Path:
        path = (parent / filename).resolve()
        if not path.is_relative_to(package) or not path.is_file():
            raise ValueError(f"Missing or out-of-package portable model asset: {filename}")
        return path

    def collect(path: Path):
        if path in manifests:
            return
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        if manifest.get("runtime") != "libtorch-torchscript":
            raise ValueError(f"Not a LibTorch manifest: {path}")
        if manifest.get("device") not in {"cpu", "portable"}:
            raise ValueError("Portable export requires original FP32 CPU weights, not CUDA/FP16 weights.")
        if (any(manifest.get(key, False) for key in ("selective_fp16", "full_fp16", "autocast_fp16"))
                or manifest.get("optimization", {}).get("selective_fp16", False)):
            raise ValueError(f"A portable model cannot contain fixed FP16 execution: {path}")
        manifests[path] = manifest
        for key in ("model", "conditioner", "acoustic", "predictor", "vocoder"):
            if manifest.get(key):
                models.add(resolve(path.parent, manifest[key]))
        for key in ("duration_manifest", "variance_manifest", "vocoder_manifest"):
            if manifest.get(key):
                collect(resolve(path.parent, manifest[key]))

    collect(resolve(package, manifest_file))
    report = {}
    simplification_report = {}
    for path in sorted(models):
        # Python file objects also avoid TorchScript's native Windows Unicode
        # filename limitation. The exported package can live in a Chinese path.
        with path.open("rb") as stream:
            module = torch.jit.load(stream, map_location="cpu").eval()
        tensors = list(module.parameters()) + list(module.buffers())
        for child in module.modules():
            for name in child._c._method_names():
                tensors.extend(node.output().toIValue()
                               for node in _nodes(child._c._get_method(name).graph)
                               if node.kind() == "prim::Constant"
                               and node.output().type().kind() == "TensorType")
        if any(value.dtype in (torch.float16, torch.bfloat16) for value in tensors if value is not None):
            raise ValueError(f"Portable export requires original FP32 tensors, not half-precision weights: {path}")
        count = make_device_agnostic(module)
        simplification_report[path.relative_to(package).as_posix()] = simplify_inference_graph(module)
        temporary = path.with_suffix(".portable.tmp")
        with temporary.open("wb") as stream:
            torch.jit.save(module, stream)
        temporary.replace(path)
        report[path.relative_to(package).as_posix()] = count
    for path, manifest in manifests.items():
        manifest["device"] = "portable"
        manifest["supported_devices"] = ["cpu", "cuda"]
        manifest["precision"] = "float32"
        manifest["portable_export_version"] = 1
        manifest["common_subexpression_elimination"] = True
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (package / Path(manifest_file).parent / "portable-export.json").write_text(
        json.dumps({"device_constants_rebound": report, "redundant_graph_nodes_removed": simplification_report}, indent=2),
        encoding="utf-8"
    )
    return report
