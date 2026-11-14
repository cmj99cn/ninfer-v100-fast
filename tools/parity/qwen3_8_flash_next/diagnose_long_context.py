from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tools.reference.qwen3_8_flash_next.model import RefModel
from tools.reference.qwen3_8_flash_next.ops import bf16_from_u16, qsa_indexer

BINARY = ROOT / "build" / "tests" / "ninfer_flash_next_text_forward_test"
NCCL = "/home/cmj/nccl-lib/whl/nvidia/nccl/lib"

RESIDUAL_NAMES = {
    "residual_in",
    "residual_out",
    "normalized_in",
    "normalized_out",
    "stage",
    "normalized_moe",
}
COLUMN_NAMES = {"attention_block", "mixer", "moe_block", "moe"}
INJECTION_NAMES = {"attention_injection", "moe_injection"}

BOUNDS = {
    "residual_in": 2.0**-8,
    "residual_out": 2.0**-8,
    "normalized_in": 2.0**-8,
    "normalized_out": 2.0**-8,
    "stage": 2.0**-8,
    "normalized_moe": 2.0**-8,
    "attention_block": 2.0**-8,
    "moe_block": 2.0**-8,
    "mixer": 2.0**-8,
    "moe": 2.0**-8,
    "attention_injection": 2.0**-8,
    "moe_injection": 2.0**-8,
    "index_query": 2.0**-8,
    "index_keys": 2.0**-8,
}


def prompt_for(tokens):
    return [1024 + (index * 7) % 200000 for index in range(tokens)]


def parse_qsa_dump(path):
    prompt = None
    entries = []
    section = None
    with open(path, "r", encoding="utf-8") as handle:
        first = handle.readline().strip()
        if first != "flash-next-dump v1":
            raise ValueError("unexpected dump header " + first)
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == "prompt":
                prompt = [int(value) for value in parts[2:]]
            elif parts[0] == "ple":
                section = "ple"
            elif parts[0] == "qsa":
                section = "qsa"
            elif section == "qsa":
                position = int(parts[0])
                count = int(parts[1])
                entries.append((position, np.asarray([int(v) for v in parts[2 : 2 + count]], dtype=np.int64)))
    return prompt, entries


def run_cpp(artifact, tokens, dump_dir, layer_dir):
    environment = dict(os.environ)
    for key in ("NINFER_FLASH_NEXT_LONG_PROMPT", "NINFER_FLASH_NEXT_REFERENCE_TOKEN",
                "NINFER_FLASH_NEXT_DUMP", "NINFER_FLASH_NEXT_LAYER_DUMP"):
        environment.pop(key, None)
    environment["NINFER_FLASH_NEXT_ARTIFACT"] = artifact
    environment["NINFER_FLASH_NEXT_LONG_PROMPT"] = str(tokens)
    environment["NINFER_FLASH_NEXT_DUMP"] = str(dump_dir)
    environment["NINFER_FLASH_NEXT_LAYER_DUMP"] = str(layer_dir)
    environment["LD_LIBRARY_PATH"] = NCCL + ":" + environment.get("LD_LIBRARY_PATH", "")
    start = time.perf_counter()
    result = subprocess.run([str(BINARY)], capture_output=True, text=True, env=environment)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(
            "the C++ long-prompt binary exited " + str(result.returncode) + "\n"
            + result.stdout + "\n" + result.stderr
        )
    return elapsed, result.stdout


def ref_to_cpp(name, array, tokens):
    value = np.asarray(array, dtype=np.float32)
    if name in RESIDUAL_NAMES:
        return np.ascontiguousarray(value.transpose(2, 1, 0))
    if name in COLUMN_NAMES:
        return np.ascontiguousarray(value.transpose(1, 0))
    if name in INJECTION_NAMES:
        return np.ascontiguousarray(value.transpose(1, 0))
    if name == "index_query":
        return np.ascontiguousarray(value.transpose(2, 1, 0))
    if name == "index_keys":
        return np.ascontiguousarray(value.transpose(1, 0))
    raise ValueError("unknown reference tensor " + name)


def write_reference_dump(directory, tensors, tokens):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.txt"
    with open(manifest, "w", encoding="utf-8") as handle:
        for (layer, name), array in sorted(tensors.items()):
            value = ref_to_cpp(name, array, tokens)
            path = directory / f"L{layer}_{name}"
            value.astype(np.float32).ravel(order="F").tofile(path)
            handle.write(
                f"{layer} {name} 1 4 {value.shape[0]} {value.shape[1]} {value.shape[2] if value.ndim > 2 else 1} 1 {value.nbytes}\n"
            )


def load_dump(directory):
    entries = {}
    with open(Path(directory) / "manifest.txt", "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            layer = int(parts[0])
            name = parts[1]
            dtype = int(parts[2])
            dims = [int(value) for value in parts[4:8]]
            byte_count = int(parts[8])
            raw = np.fromfile(Path(directory) / f"L{layer}_{name}", dtype=np.uint8, count=byte_count)
            if dtype == 0:
                value = bf16_from_u16(raw.view("<u2"))
            elif dtype == 1:
                value = raw.view("<f4")
            elif dtype == 2:
                value = raw.view("<i4")
            else:
                raise ValueError("unknown dump dtype " + str(dtype))
            physical = value.reshape(tuple(reversed(dims)))
            logical = physical.transpose(tuple(range(len(dims) - 1, -1, -1)))
            entries[(layer, name)] = np.ascontiguousarray(logical)
    return entries


def compare_tensor(cpp, reference):
    left = cpp.astype(np.float64).reshape(-1)
    right = reference.astype(np.float64).reshape(-1)
    difference = left - right
    denominator = float(np.linalg.norm(right))
    relative = float(np.linalg.norm(difference) / denominator) if denominator > 0 else float("nan")
    maximum = float(np.max(np.abs(difference))) if difference.size else 0.0
    rms = float(np.sqrt(np.mean(right * right))) if right.size else 0.0
    return relative, maximum, rms


def cpp_blocks(index_keys):
    pages = index_keys.shape[3]
    matrix = index_keys.reshape(index_keys.shape[0], 16, 1, pages)
    block_count = pages * 16
    out = np.empty((index_keys.shape[0], block_count), dtype=np.float32)
    for block in range(block_count):
        out[:, block] = matrix[:, block % 16, 0, block // 16]
    return out


def score_matrix(query, keys):
    dots = np.einsum("thd,bd->thb", query.astype(np.float64), keys.astype(np.float64))
    positive = np.maximum(0.0, dots)
    return np.sum(positive, axis=1)


def selections_from_scores(scores, tokens, budget=512):
    counts = np.minimum((np.arange(tokens, dtype=np.int64) + 1) // 4, scores.shape[1])
    selections = []
    for token in range(tokens):
        count = int(counts[token])
        pair = np.lexsort((np.arange(count), -scores[token, :count]))
        selections.append(np.sort(pair[: min(budget, count)]).astype(np.int64))
    return selections, counts


def boundary_gaps(scores, tokens):
    counts = np.minimum((np.arange(tokens, dtype=np.int64) + 1) // 4, scores.shape[1])
    gaps = np.full(tokens, np.nan)
    for token in range(tokens):
        count = int(counts[token])
        if count <= 512:
            continue
        values = scores[token, :count]
        order = np.lexsort((np.arange(count), -values))
        gaps[token] = values[order[511]] - values[order[512]]
    return gaps


FORWARD_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "residual_in",
            "normalized_in",
            "attention_block",
            "attention_injection",
            "mixer",
            "stage",
            "normalized_moe",
            "moe_block",
            "moe_injection",
            "moe",
            "residual_out",
            "normalized_out",
            "index_query",
            "index_keys",
        )
    )
}


def logical_matrix(value):
    value = np.asarray(value)
    while value.ndim > 2 and value.shape[-1] == 1:
        value = value[..., 0]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2100)
    parser.add_argument("--artifact", default=os.environ.get("NINFER_FLASH_NEXT_ARTIFACT"))
    parser.add_argument("--reference-cache", default=None)
    parser.add_argument("--dump-dir", default=None)
    parser.add_argument("--keep", action="store_true")
    arguments = parser.parse_args()

    artifact = arguments.artifact
    if not artifact or not Path(artifact).is_file():
        print("SKIP: NINFER_FLASH_NEXT_ARTIFACT is not set or does not exist")
        return 77
    if not BINARY.is_file():
        print("FAIL: the C++ parity binary is missing: " + str(BINARY))
        return 1
    if subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode != 0:
        print("SKIP: no CUDA device")
        return 77

    tokens = arguments.tokens
    prompt = prompt_for(tokens)
    work = Path(tempfile.mkdtemp(prefix="flash_next_diag_"))
    layer_dir = work / "cpp"
    layer_dir.mkdir()
    index_dump = work / "qsa.txt"

    cpp_time, cpp_stdout = run_cpp(artifact, tokens, index_dump, layer_dir)
    stored_prompt, entries = parse_qsa_dump(index_dump)
    if stored_prompt != prompt:
        print("FAIL: the C++ dump prompt does not match")
        return 1
    cpp = load_dump(layer_dir)

    reference_dir = Path(arguments.reference_cache) if arguments.reference_cache else work / "ref"
    reference_manifest = reference_dir / "manifest.txt"
    reference_time = 0.0
    if reference_manifest.is_file():
        reference = load_dump(reference_dir)
    else:
        start = time.perf_counter()
        with RefModel(artifact) as model:
            result = model.forward(prompt, capture=True, capture_tensors=True)
        reference_time = time.perf_counter() - start
        write_reference_dump(reference_dir, result["tensors"], tokens)
        reference = load_dump(reference_dir)

    print(f"tokens {tokens}")
    print(f"cpp time {cpp_time:.3f} s")
    print(f"reference time {reference_time:.3f} s")
    print("cpp " + " ".join(cpp_stdout.split()))

    print("\nper-tensor error")
    print(f"{'layer':>5} {'tensor':<20} {'rel_l2':>11} {'max_abs':>11} {'rms':>11} {'bound':>10} {'flag':>5}")
    first_exceed = None
    for key in sorted(cpp):
        layer, name = key
        if key not in reference:
            continue
        if name == "index_keys":
            blocks = logical_matrix(reference[key])
            left = cpp_blocks(cpp[key])[:, : blocks.shape[1]]
            right = blocks
        elif name == "index_query":
            left = logical_matrix(cpp[key])
            right = logical_matrix(reference[key])
        else:
            left = cpp[key]
            right = reference[key]
        if left.shape != right.shape:
            print(f"{layer:>5} {name:<20} shape mismatch {left.shape} {right.shape}")
            continue
        relative, maximum, rms = compare_tensor(left, right)
        bound = BOUNDS[name]
        exceed = relative > bound
        if exceed and (first_exceed is None or
                       (layer, FORWARD_ORDER[name]) < (first_exceed[0], FORWARD_ORDER[first_exceed[1]])):
            first_exceed = (layer, name, relative, bound)
        print(f"{layer:>5} {name:<20} {relative:11.4e} {maximum:11.4e} {rms:11.4e} {bound:10.4e} {'X' if exceed else '.':>5}")

    print("\nper-stream residual RMS (reference)")
    print(f"{'layer':>5} {'min_rms':>12} {'stream':>7} {'token':>7}")
    for layer in range(4):
        value = reference[(layer, "residual_in")].astype(np.float64)
        magnitudes = np.sqrt(np.mean(value * value, axis=0))
        index = int(np.argmin(magnitudes))
        print(f"{layer:>5} {magnitudes.min():12.4e} {index // tokens:7d} {index % tokens:7d}")

    positions = np.arange(tokens, dtype=np.int64)
    q_cpp = logical_matrix(cpp[(3, "index_query")]).transpose(2, 1, 0)
    k_cpp = cpp_blocks(cpp[(3, "index_keys")]).transpose(1, 0)
    q_ref = logical_matrix(reference[(3, "index_query")]).transpose(2, 1, 0)
    k_ref = logical_matrix(reference[(3, "index_keys")]).transpose(1, 0)
    blocks_ref = k_ref.shape[0]
    scores_cpp = score_matrix(q_cpp, k_cpp[:blocks_ref])
    scores_ref = score_matrix(q_ref, k_ref)
    selections_cpp_formula, counts = selections_from_scores(scores_cpp, tokens)
    selections_ref, _ = selections_from_scores(scores_ref, tokens)
    gaps_cpp = boundary_gaps(scores_cpp, tokens)
    gaps_ref = boundary_gaps(scores_ref, tokens)

    cpp_entries = {position: blocks for position, blocks in entries}
    cpp_formula_match = 0
    model_divergent = []
    for token in range(tokens):
        actual = cpp_entries.get(token)
        formula = selections_cpp_formula[token]
        if actual is not None and np.array_equal(actual, formula):
            cpp_formula_match += 1
        expected = selections_ref[token]
        if counts[token] > 512 and actual is not None and not np.array_equal(actual, expected):
            model_divergent.append(token)

    pruned_tokens = int(np.sum(counts > 512))
    print("\nlayer 3 indexer")
    print(f"pruned tokens {pruned_tokens}")
    print(f"cpp formula reproduction {cpp_formula_match}/{tokens}")
    print(f"model-level divergent {len(model_divergent)}/{pruned_tokens}")
    print(f"{'token':>6} {'gap_ref':>12} {'gap_cpp':>12} {'max_dscore':>12} {'l2_dscore':>12}")
    delta = 0.0
    perturbations = []
    for token in model_divergent:
        count = int(counts[token])
        difference = np.abs(scores_cpp[token, :count] - scores_ref[token, :count])
        maximum = float(difference.max())
        l2 = float(np.linalg.norm(scores_cpp[token, :count] - scores_ref[token, :count]))
        perturbations.append(maximum)
        delta = max(delta, float(gaps_ref[token]))
        print(f"{token:>6} {gaps_ref[token]:12.4e} {gaps_cpp[token]:12.4e} {maximum:12.4e} {l2:12.4e}")
    if model_divergent:
        print(f"delta (max reference boundary gap on divergent tokens) {delta:.6e}")
        print(f"max score perturbation on divergent tokens {max(perturbations):.6e}")
        outside = [
            token
            for token in np.flatnonzero(counts > 512)
            if np.isfinite(gaps_ref[token]) and gaps_ref[token] > delta
        ]
        divergent_set = set(model_divergent)
        print(
            "tokens with reference boundary gap > delta: "
            + str(len(outside))
            + ", divergent among them: "
            + str(sum(1 for token in outside if int(token) in divergent_set))
        )
        non_divergent = [
            float(gaps_ref[token])
            for token in np.flatnonzero(counts > 512)
            if int(token) not in divergent_set
        ]
        if non_divergent:
            print(f"smallest boundary gap on a non-divergent token {min(non_divergent):.6e}")

    if first_exceed is None:
        print("\nVERDICT b: all compared tensors are inside their implementation-profile bound")
    else:
        layer, name, relative, bound = first_exceed
        print(
            f"\nfirst tensor beyond its profile bound: layer {layer} {name} "
            f"rel_l2 {relative:.6e} bound {bound:.6e}"
        )

    if arguments.keep or arguments.dump_dir:
        print("work directory " + str(work))
    elif not arguments.keep:
        import shutil

        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
