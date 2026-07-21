#!/usr/bin/env python3
"""
dedup_train_parquet.py — collapse the DAPO-Math-17k parquet to one row per unique prompt.

The published BytedTsinghua-SIA/DAPO-Math-17k parquet repeats each of the ~17k unique
problems ~100x (1,791,700 rows total). verl's data.filter_overlong_prompts=True scans
*every* row, and with shuffle=False + train_batch_size=B it treats all rows as distinct
samples -> total_training_steps = num_rows / B (~224k steps for the bloated file).

This dedup keeps the FIRST occurrence of each unique prompt, preserving the exact Arrow
schema (nested prompt/reward_model/extra_info structs are untouched). Non-destructive:
writes a new file, never modifies the input.

Usage:
    python3 dedup_train_parquet.py \
        --in  ../datasets/dapo-math-17k.parquet \
        --out ../datasets/dapo-math-17k-dedup.parquet
"""
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input parquet")
    ap.add_argument("--out", dest="out", required=True, help="output parquet")
    ap.add_argument("--key", default="prompt",
                    help="column to dedup on (default: prompt)")
    args = ap.parse_args()

    table = pq.read_table(args.inp)
    n_before = table.num_rows

    # Build a hashable dedup key by stringifying the (possibly nested) key column.
    key_series = pd.read_parquet(args.inp, columns=[args.key])[args.key].astype(str)
    keep_mask = ~key_series.duplicated(keep="first")  # True where this is the first time we see the prompt

    # Slice the ORIGINAL Arrow table with the mask -> schema/types preserved exactly.
    table_dedup = table.filter(pa.array(keep_mask.to_numpy()))
    n_after = table_dedup.num_rows

    pq.write_table(table_dedup, args.out)

    print(f"[dedup] input : {args.inp}")
    print(f"[dedup] output: {args.out}")
    print(f"[dedup] rows  : {n_before} -> {n_after}  (removed {n_before - n_after}, "
          f"{100 * (n_before - n_after) / n_before:.1f}% duplicates)")


if __name__ == "__main__":
    main()
