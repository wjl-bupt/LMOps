#!/usr/bin/env python3
"""
build_teacher_response_parquet.py — add the GAD discriminator's `teacher_response`
text column to a verl-format math parquet.

GAD trains the critic as a Bradley-Terry discriminator (teacher solution > student
solution). It needs, per prompt, the TEACHER's own solution text as the "real" example.
This script generates that with the teacher model (Qwen3-4B) via vLLM and appends it as
a `teacher_response` string column, preserving the original Arrow schema of every other
column.

Usage:
    python3 build_teacher_response_parquet.py \
        --in ../datasets/dapo-math-17k-dedup.parquet \
        --out ../datasets/dapo-math-17k-gad-mini.parquet \
        --teacher /dockerdata/junewluo/models/Qwen3-4B \
        --n 64 --max-tokens 2048
"""
import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--n", type=int, default=64, help="number of prompts to keep/generate (0 = all)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor_parallel_size for the teacher (use >1 for large teachers like 32B)")
    ap.add_argument("--sample", action="store_true",
                    help="randomly sample --n rows (seeded, reproducible) instead of taking the first --n. "
                         "Use this to avoid ordering bias in the source parquet (DAPO is ordered by answer magnitude).")
    ap.add_argument("--seed", type=int, default=42, help="random seed used when --sample is set")
    args = ap.parse_args()

    table = pq.read_table(args.inp)
    if args.n and args.n > 0 and args.n < table.num_rows:
        if args.sample:
            idx = np.sort(
                np.random.default_rng(args.seed).choice(table.num_rows, size=args.n, replace=False)
            )
            table = table.take(pa.array(idx))
            print(f"[teacher] random-sampled {args.n} rows (seed={args.seed}) from {args.inp}")
        else:
            table = table.slice(0, args.n)
            print(f"[teacher] took first {args.n} rows from {args.inp}")
    n = table.num_rows

    # Build teacher prompts by applying the chat template (thinking OFF, to match training).
    prompt_lists = table.column("prompt").to_pylist()  # each: [{"role","content"}, ...]
    tok = AutoTokenizer.from_pretrained(args.teacher)
    prompts = [
        tok.apply_chat_template(
            [{"role": m["role"], "content": m["content"]} for m in p],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for p in prompt_lists
    ]

    llm = LLM(model=args.teacher, dtype="bfloat16",
              tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_tokens + 2048)
    sp = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=args.max_tokens)
    outs = llm.generate(prompts, sp)
    responses = [o.outputs[0].text for o in outs]

    table = table.append_column("teacher_response", pa.array(responses, type=pa.string()))
    pq.write_table(table, args.out)

    lens = [len(r) for r in responses]
    print(f"[teacher] wrote {args.out}: {n} rows, teacher_response added.")
    print(f"[teacher] response char-length min/mean/max: "
          f"{min(lens)}/{sum(lens)//len(lens)}/{max(lens)}")


if __name__ == "__main__":
    main()
