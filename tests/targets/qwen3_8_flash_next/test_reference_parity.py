from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tools.reference.qwen3_8_flash_next.ops import qsa_indexer, run_self_tests

PROMPT = [1024, 7, 98140, 33, 4097, 512, 100, 2048]
LONG_TOKENS = 2052
BUDGET_TOKENS = 2100
LONG_PROMPT = [1024 + (index * 7) % 200000 for index in range(LONG_TOKENS)]
BUDGET_PROMPT = [1024 + (index * 7) % 200000 for index in range(BUDGET_TOKENS)]
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


def load_index_dump(path, tokens):
    with open(path, "rb") as handle:
        stored_tokens, pages, query_words, key_words = struct.unpack("<iiQQ", handle.read(24))
        if stored_tokens != tokens:
            raise ValueError("the index dump token count does not match")
        query = np.frombuffer(handle.read(query_words * 4), dtype="<f4")
        keys = np.frombuffer(handle.read(key_words * 4), dtype="<f4")
    query = query.reshape(tokens, 4, 128)
    keys = keys.reshape(pages, 1, 16, 128)
    blocks = np.stack([keys[block // 16, 0, block % 16, :] for block in range(tokens // 4)], axis=0)
    return query, blocks


def compare_blocks(reference_blocks, entries, tokens):
    by_position = {position: blocks for position, blocks in entries}
    mismatches = []
    for index in range(tokens):
        expected = set(int(value) for value in reference_blocks[index])
        actual = by_position.get(index)
        if actual != expected:
            mismatches.append((index, sorted(expected), None if actual is None else sorted(actual)))
    return mismatches


def run_binary(environment, extra):
    merged = dict(environment)
    merged.update(extra)
    merged["LD_LIBRARY_PATH"] = NCCL + ":" + merged.get("LD_LIBRARY_PATH", "")
    return subprocess.run([str(BINARY)], capture_output=True, text=True, env=merged)


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
        short_start = time.perf_counter()
        short = model.forward(PROMPT, capture=True)
        short_time = time.perf_counter() - short_start
        short_order = np.argsort(-short["logits"], kind="stable")[:2]
        short_token = int(short_order[0])
        short_margin = float(short["logits"][short_order[0]] - short["logits"][short_order[1]])
        if short_margin < 1.0e-2:
            if short_margin < 1.0e-3:
                print("FAIL: the short top two logits are nearly tied (margin "
                      + format(short_margin, ".9g") + " < 1e-3)")
            else:
                print("FAIL: the short top two logits margin " + format(short_margin, ".9g")
                      + " is below the 1e-2 acceptance bound")
            return 1
        short_rows = short["ple_rows"]
        short_qsa = short["qsa"]
        if len(short_qsa) != 1:
            print("FAIL: expected one QSA layer in the reduced stack")
            return 1
        short_blocks = next(iter(short_qsa.values()))

        long_start = time.perf_counter()
        long = model.forward(LONG_PROMPT, capture=True)
        long_time = time.perf_counter() - long_start
        long_order = np.argsort(-long["logits"], kind="stable")[:2]
        long_token = int(long_order[0])
        long_margin = float(long["logits"][long_order[0]] - long["logits"][long_order[1]])
        long_qsa = long["qsa"]
        if len(long_qsa) != 1:
            print("FAIL: expected one QSA layer in the long reduced stack")
            return 1
        long_blocks = next(iter(long_qsa.values()))
        budget_rows = model.ple_rows(BUDGET_PROMPT)

    short_handle = tempfile.NamedTemporaryFile(prefix="flash_next_short_", suffix=".txt", delete=False)
    short_dump = short_handle.name
    short_handle.close()
    long_handle = tempfile.NamedTemporaryFile(prefix="flash_next_long_", suffix=".txt", delete=False)
    long_dump = long_handle.name
    long_handle.close()
    budget_handle = tempfile.NamedTemporaryFile(prefix="flash_next_budget_", suffix=".txt", delete=False)
    budget_dump = budget_handle.name
    budget_handle.close()
    environment = dict(os.environ)
    for key in ("NINFER_FLASH_NEXT_LONG_PROMPT", "NINFER_FLASH_NEXT_REFERENCE_TOKEN",
                "NINFER_FLASH_NEXT_DUMP"):
        environment.pop(key, None)
    try:
        short_run = run_binary(
            environment,
            {
                "NINFER_FLASH_NEXT_REFERENCE_TOKEN": str(short_token),
                "NINFER_FLASH_NEXT_DUMP": short_dump,
            },
        )
        if short_run.returncode == 77:
            return skip("the C++ parity binary skipped (missing artifact or CUDA)")
        if short_run.returncode != 0:
            print(short_run.stdout)
            print(short_run.stderr)
            print("FAIL: the C++ short parity binary exited " + str(short_run.returncode))
            return 1
        short_prompt, short_dump_rows, short_entries = parse_dump(short_dump)

        long_run_start = time.perf_counter()
        long_run = run_binary(
            environment,
            {
                "NINFER_FLASH_NEXT_LONG_PROMPT": str(LONG_TOKENS),
                "NINFER_FLASH_NEXT_REFERENCE_TOKEN": str(long_token),
                "NINFER_FLASH_NEXT_DUMP": long_dump,
            },
        )
        long_run_time = time.perf_counter() - long_run_start
        if long_run.returncode == 77:
            return skip("the C++ long parity run skipped (missing artifact or CUDA)")
        if long_run.returncode != 0:
            print(long_run.stdout)
            print(long_run.stderr)
            print("FAIL: the C++ long parity binary exited " + str(long_run.returncode))
            return 1
        long_prompt, long_dump_rows, long_entries = parse_dump(long_dump)
        long_index_query, long_index_keys = load_index_dump(long_dump + ".index", LONG_TOKENS)

        budget_run_start = time.perf_counter()
        budget_run = run_binary(
            environment,
            {
                "NINFER_FLASH_NEXT_LONG_PROMPT": str(BUDGET_TOKENS),
                "NINFER_FLASH_NEXT_DUMP": budget_dump,
            },
        )
        budget_run_time = time.perf_counter() - budget_run_start
        if budget_run.returncode == 77:
            return skip("the C++ budget parity run skipped (missing artifact or CUDA)")
        if budget_run.returncode != 0:
            print(budget_run.stdout)
            print(budget_run.stderr)
            print("FAIL: the C++ budget parity binary exited " + str(budget_run.returncode))
            return 1
        budget_prompt, budget_dump_rows, budget_entries = parse_dump(budget_dump)
        budget_index_query, budget_index_keys = load_index_dump(budget_dump + ".index", BUDGET_TOKENS)
    finally:
        for path in (short_dump, short_dump + ".index", long_dump, long_dump + ".index", budget_dump,
                     budget_dump + ".index"):
            try:
                os.unlink(path)
            except OSError:
                pass

    if short_prompt != PROMPT:
        print("FAIL: the short dump prompt does not match")
        return 1
    if long_prompt != LONG_PROMPT:
        print("FAIL: the long dump prompt does not match")
        return 1
    if budget_prompt != BUDGET_PROMPT:
        print("FAIL: the budget dump prompt does not match")
        return 1

    row_mismatches = []
    for index in range(len(PROMPT)):
        expected = [int(value) for value in short_rows[index]]
        if short_dump_rows.get(index) != expected:
            row_mismatches.append(index)
    long_row_mismatches = []
    for index in range(len(LONG_PROMPT)):
        expected = [int(value) for value in long["ple_rows"][index]]
        if long_dump_rows.get(index) != expected:
            long_row_mismatches.append(index)
    budget_row_mismatches = []
    for index in range(len(BUDGET_PROMPT)):
        expected = [int(value) for value in budget_rows[index]]
        if budget_dump_rows.get(index) != expected:
            budget_row_mismatches.append(index)
    short_block_mismatches = compare_blocks(short_blocks, short_entries, len(PROMPT))
    model_long_mismatches = compare_blocks(long_blocks, long_entries, LONG_TOKENS)
    indexed_blocks = qsa_indexer(
        long_index_query, long_index_keys, np.arange(LONG_TOKENS), 512
    )
    long_block_mismatches = compare_blocks(indexed_blocks, long_entries, LONG_TOKENS)

    long_sizes = [len(blocks) for blocks in indexed_blocks]
    blocks_per_complete = LONG_TOKENS // 4
    pruned = max(long_sizes) == 512 and blocks_per_complete > 512
    budget_blocks = qsa_indexer(
        budget_index_query, budget_index_keys, np.arange(BUDGET_TOKENS), 512
    )
    budget_mismatches = compare_blocks(budget_blocks, budget_entries, BUDGET_TOKENS)
    budget_sizes = [len(blocks) for blocks in budget_blocks]
    budget_complete = BUDGET_TOKENS // 4
    budget_pruned = max(budget_sizes) == 512 and budget_complete > 512
    print("reference token " + str(short_token))
    print("logit margin " + format(short_margin, ".9g"))
    print("reference time " + format(short_time, ".3f") + " s")
    print("ple row parity " + ("OK" if not row_mismatches else "FAIL " + str(row_mismatches)))
    print("qsa block parity " + ("OK" if not short_block_mismatches else "FAIL " + str(short_block_mismatches)))
    print("long reference token " + str(long_token))
    print("long logit margin " + format(long_margin, ".9g"))
    print("long reference time " + format(long_time, ".3f") + " s")
    print("long c++ time " + format(long_run_time, ".3f") + " s")
    print("long block size max " + str(max(long_sizes)) + " of C=" + str(blocks_per_complete))
    print("long pruned " + ("yes" if pruned else "no"))
    print("long qsa block parity " + ("OK" if not long_block_mismatches else "FAIL " + str(long_block_mismatches[:3])))
    print("long model-level qsa block parity " + ("OK" if not model_long_mismatches else "FAIL " + str(model_long_mismatches[:3])))
    print("long ple row parity " + ("OK" if not long_row_mismatches else "FAIL " + str(long_row_mismatches[:3]) + " of " + str(len(long_row_mismatches))))
    print("budget c++ time " + format(budget_run_time, ".3f") + " s")
    print("budget block size max " + str(max(budget_sizes)) + " of C=" + str(budget_complete))
    print("budget pruned " + ("yes" if budget_pruned else "no"))
    print("budget qsa block parity " + ("OK" if not budget_mismatches else "FAIL " + str(budget_mismatches[:3])))
    print("budget ple row parity " + ("OK" if not budget_row_mismatches else "FAIL " + str(budget_row_mismatches[:3]) + " of " + str(len(budget_row_mismatches))))
    if long_margin < 1.0e-3:
        print("WARNING: the long top two logits are nearly tied (margin < 1e-3)")

    if (row_mismatches or long_row_mismatches or budget_row_mismatches or short_block_mismatches
            or long_block_mismatches or model_long_mismatches or not pruned or budget_mismatches
            or not budget_pruned):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
