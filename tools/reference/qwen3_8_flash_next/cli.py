from __future__ import annotations

import argparse
import os

from .config import CFG
from .model import RefModel
from .ops import run_self_tests
from .text import greedy, parse_ids, prefill


def build_parser():
    parser = argparse.ArgumentParser(description="Qwen3.8-Flash-Next FP32/FP64 reference")
    parser.add_argument("--weights", default=os.environ.get("NINFER_FLASH_NEXT_ARTIFACT"))
    parser.add_argument("--ids", help="comma or space separated prompt token ids")
    parser.add_argument("--decode", type=int, default=0, help="greedy continuation tokens")
    parser.add_argument("--self-test", action="store_true", help="run the operator hand checks")
    parser.add_argument("--debug", action="store_true", help="print PLE rows and QSA selections")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        failures = [name for name, ok in run_self_tests(verbose=True) if not ok]
        if failures:
            print("FAIL " + ", ".join(failures))
            return 1
        print("OK reference operator hand checks")
        return 0
    if not args.weights:
        parser.error("--weights or NINFER_FLASH_NEXT_ARTIFACT is required")
    if not args.ids:
        parser.error("--ids is required")
    ids = parse_ids(args.ids)
    with RefModel(args.weights) as model:
        token, margin, result = prefill(model, ids)
        print(f"token {token} margin {margin:.9g}")
        if args.debug:
            rows = result["ple_rows"]
            for index in range(rows.shape[0]):
                print("ple", index, " ".join(str(int(value)) for value in rows[index]))
            for layer, selections in result["qsa"].items():
                for index, selected in enumerate(selections):
                    print("qsa", layer, index, " ".join(str(int(value)) for value in selected))
        if args.decode > 0:
            produced, sequence, _ = greedy(model, ids, args.decode)
            print("tokens " + " ".join(str(token) for token in produced))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
