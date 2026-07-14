#!/usr/bin/env python3
"""Add a `teacher_response` TEXT column to a verl-format training parquet, for GAD.

The GAD discriminator needs the WHITE-BOX teacher's own solution to each prompt as its
"real" example. This script:
  1. reads a verl-format parquet (e.g. DAPO-Math-17k: columns `prompt` [chat list],
     `data_source`, `reward_model.ground_truth`, `ability`, `extra_info`),
  2. extracts the user prompt text from the `prompt` chat list,
  3. generates a teacher solution per prompt (WIRE IN your offline vLLM teacher below),
  4. writes the same rows back out WITH an added `teacher_response` TEXT column
     (all original columns preserved, so reward_model.ground_truth still works for
     baselines / validation).

Usage:
    python3 build_teacher_response_parquet.py \
        --in  ../datasets/dapo-math-17k.parquet \
        --out ../datasets/dapo-math-17k-gad.parquet \
        --teacher /path/to/Qwen3-4B

Notes:
  * `teacher_response` is raw response TEXT (no chat template, no prompt). rl_dataset.py
    tokenizes it with the STUDENT tokenizer (add_special_tokens=False) and right-pads with
    eos to data.max_response_length.
  * Only the TRAINING parquet needs this column; validation parquets (AIME24 etc.) do not.
  * Size data.max_response_length to cover the teacher's solution length. Also set
    data.truncation=right for GAD runs (the OPRD scripts default to 'error', which would
    raise on an over-long teacher_response) — or keep it and ensure lengths fit.
"""
import argparse

import pandas as pd


def extract_prompt_text(prompt_field) -> str:
    """Pull the user text out of a verl `prompt` field (chat-message list) or a plain string."""
    if isinstance(prompt_field, str):
        return prompt_field
    # verl format: list/array of {"role": ..., "content": ...}
    try:
        turns = list(prompt_field)
        # last user turn (DAPO has a single user turn)
        for turn in reversed(turns):
            if isinstance(turn, dict) and turn.get("role") == "user":
                return turn["content"]
        # fallback: concatenate all contents
        return "\n".join(t["content"] for t in turns if isinstance(t, dict) and "content" in t)
    except Exception as e:
        raise ValueError(f"Unrecognized prompt field format: {type(prompt_field)}") from e


def generate_teacher_responses(prompts: list[str], teacher_path: str) -> list[str]:
    """REPLACE with real offline generation on the WHITE-BOX teacher (vLLM).

    Example:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(teacher_path)
        llm = LLM(model=teacher_path, tensor_parallel_size=1)
        chats = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True)
                 for p in prompts]
        outs = llm.generate(chats, SamplingParams(temperature=0.7, top_p=0.95, max_tokens=2048))
        return [o.outputs[0].text for o in outs]
    """
    del prompts, teacher_path  # placeholder — remove once real generation is wired in
    raise NotImplementedError("Wire in your white-box teacher generation here (offline vLLM).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="verl-format training parquet")
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--teacher", required=True, help="white-box teacher model path/name")
    ap.add_argument("--prompt-col", default="prompt", help="verl prompt column (chat list) or a text col")
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    print(f"Loaded {len(df)} rows; columns: {list(df.columns)}")
    prompts = [extract_prompt_text(p) for p in df[args.prompt_col].tolist()]
    df["teacher_response"] = generate_teacher_responses(prompts, args.teacher)
    df.to_parquet(args.out)
    print(f"Wrote {args.out} with an added teacher_response column ({len(df)} rows).")


if __name__ == "__main__":
    main()
