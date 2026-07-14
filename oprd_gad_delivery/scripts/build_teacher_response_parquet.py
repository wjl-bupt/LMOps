#!/usr/bin/env python3
"""Add a `teacher_response` TEXT column to a training parquet for GAD.

The GAD discriminator needs the teacher's own solution to each prompt as its "real"
example. This helper shows the expected schema. Plug in your teacher-generation step
(vLLM offline generation with the WHITE-BOX teacher on the same prompts) where marked.

Usage:
    python3 build_teacher_response_parquet.py --in dapo_math_17k.parquet --out dapo_math_17k_gad.parquet

Notes:
  * `teacher_response` must be the raw response TEXT (no chat template, no prompt) —
    rl_dataset.py tokenizes it with the STUDENT tokenizer (add_special_tokens=False)
    and right-pads with eos to data.max_response_length.
  * Size data.max_response_length to cover typical teacher solution length; longer
    responses are truncated per data.truncation (use "right" so the tail is dropped,
    or raise max_response_length to keep the final answer).
"""
import argparse

import pandas as pd


def generate_teacher_responses(prompts: list[str]) -> list[str]:
    """REPLACE THIS with real teacher generation (offline vLLM on the white-box teacher).

    Example (pseudocode):
        from vllm import LLM, SamplingParams
        llm = LLM(model=TEACHER_PATH)
        outs = llm.chat([[{"role": "user", "content": p}] for p in prompts],
                        SamplingParams(temperature=0.7, max_tokens=2048))
        return [o.outputs[0].text for o in outs]
    """
    del prompts  # placeholder — remove once real generation is wired in
    raise NotImplementedError("Wire in your teacher generation here (offline vLLM).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--prompt-col", default="prompt", help="column holding the user prompt text")
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    prompts = df[args.prompt_col].tolist()
    df["teacher_response"] = generate_teacher_responses(prompts)
    df.to_parquet(args.out)
    print(f"Wrote {args.out} with teacher_response column ({len(df)} rows).")


if __name__ == "__main__":
    main()
