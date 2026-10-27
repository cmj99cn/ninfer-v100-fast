#!/usr/bin/env python3
"""Cold prefill probe: one request per repetition, distinct seeds (no prefix reuse).

Usage: python3 tools/v100/bench_prefill_cold.py [tokens] [reps] [port]
Assumes the aligned server (tools/v100/ninfer-aligned-tp2.sh) is running.
"""
import sys, json, statistics
sys.path.insert(0, "/home/cmj")
from speed_bench import make_prompt, post_plain

tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 7419
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
port = sys.argv[3] if len(sys.argv) > 3 else "9010"
URL = f"http://127.0.0.1:{port}/v1/chat/completions"
KEY, MODEL = "ninfer-eval-key", "ninfer-nvfp4"
base = {"model": MODEL, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}

walls, tps_list = [], []
for i in range(reps):
    prompt = make_prompt(tokens, seed=f"cold-probe-{i}-{tokens}. ")
    body = {**base, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1}
    r = post_plain(URL, KEY, body, timeout=900)
    pt = r["prompt_tokens"]
    tps = pt / r["wall"]
    walls.append(r["wall"])
    tps_list.append(tps)
    print(f"rep{i}: wall={r['wall']:.3f}s prompt={pt} tps={tps:.0f}", flush=True)
print(json.dumps({"tokens": tokens, "wall_med": statistics.median(walls),
                  "tps_med": statistics.median(tps_list)}))
