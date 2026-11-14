from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tools.reference.qwen3_8_flash_next.ops import run_self_tests

PROMPT = [1024, 7, 98140, 33, 4097, 512, 100, 2048]
BINARY = ROOT / "build" / "tests" / "ninfer_flash_next_text_forward_test"
NCCL = "/home/cmj/nccl-lib/whl/nvidia/nccl/lib"


def skip(message):
    print("SKIP: " + message)
    return 77


def parse_dump(path):
    prompt = None
    rows = {}
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
            elif section == "ple":
                index = int(parts[0])
                rows[index] = [int(value) for value in parts[1:]]
            elif section == "qsa":
                position = int(parts[0])
                count = int(parts[1])
                entries.append((position, set(int(value) for value in parts[2 : 2 + count])))
    return prompt, rows, entries


def main():
    artifact = os.environ.get("NINFER_FLASH_NEXT_ARTIFACT")
    if not artifact:
        return skip("NINFER_FLASH_NEXT_ARTIFACT is not set")
    if not Path(artifact).is_file():
        return skip("the artifact does not exist: " + artifact)
    if not BINARY.is_file():
        print("FAIL: the C++ parity binary is missing: " + str(BINARY))
        return 1
    probe = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if probe.returncode != 0:
        return skip("no CUDA device")

    failures = [name for name, ok in run_self_tests(verbose=False) if not ok]
    if failures:
        print("FAIL: reference operator hand checks: " + ", ".join(failures))
        return 1

    from tools.reference.qwen3_8_flash_next.model import RefModel

    with RefModel(artifact) as model:
        result = model.forward(PROMPT, capture=True)
        logits = result["logits"]
        order = np.argsort(-logits, kind="stable")[:2]
        token = int(order[0])
        margin = float(logits[order[0]] - logits[order[1]])
        reference_rows = result["ple_rows"]
        selections = result["qsa"]
        if len(selections) != 1:
            print("FAIL: expected one QSA layer in the reduced stack")
            return 1
        reference_blocks = next(iter(selections.values()))

    handle = tempfile.NamedTemporaryFile(prefix="flash_next_parity_", suffix=".txt", delete=False)
    dump_path = handle.name
    handle.close()
    environment = dict(os.environ)
    environment["NINFER_FLASH_NEXT_REFERENCE_TOKEN"] = str(token)
    environment["NINFER_FLASH_NEXT_DUMP"] = dump_path
    environment["LD_LIBRARY_PATH"] = NCCL + ":" + environment.get("LD_LIBRARY_PATH", "")
    try:
        completed = subprocess.run([str(BINARY)], capture_output=True, text=True, env=environment)
        if completed.returncode == 77:
            return skip("the C++ parity binary skipped (missing artifact or CUDA)")
        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr)
            print("FAIL: the C++ parity binary exited " + str(completed.returncode))
            return 1
        prompt, rows, entries = parse_dump(dump_path)
    finally:
        try:
            os.unlink(dump_path)
        except OSError:
            pass

    if prompt != PROMPT:
        print("FAIL: the dump prompt does not match")
        return 1

    row_mismatches = []
    for index in range(len(PROMPT)):
        expected = [int(value) for value in reference_rows[index]]
        if rows.get(index) != expected:
            row_mismatches.append(index)
    block_mismatches = []
    by_position = {position: blocks for position, blocks in entries}
    for index in range(len(PROMPT)):
        expected = set(int(value) for value in reference_blocks[index])
        actual = by_position.get(index)
        if actual != expected:
            block_mismatches.append((index, sorted(expected), None if actual is None else sorted(actual)))

    print("reference token " + str(token))
    print("logit margin " + format(margin, ".9g"))
    print("ple row parity " + ("OK" if not row_mismatches else "FAIL " + str(row_mismatches)))
    print("qsa block parity " + ("OK" if not block_mismatches else "FAIL " + str(block_mismatches)))
    if margin < 1.0e-3:
        print("WARNING: the top two logits are nearly tied (margin < 1e-3)")

    if row_mismatches or block_mismatches:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
