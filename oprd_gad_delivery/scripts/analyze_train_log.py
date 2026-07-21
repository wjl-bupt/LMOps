#!/usr/bin/env python3
"""
analyze_train_log.py — extract training-progress changes from an OPRD/GAD run log.

Prints three things, parsed from the tee'd stdout log (logs/run_*.log):
  1. Eval curve  — AIME24 acc (best@k / maj@k) at each validation point (baseline + every test_freq).
  2. Training trend — sampled steps: pg_loss / rep_loss / kl_loss / grad_norm / d_loss / d_acc / resp_len / clip%.
  3. Trick activity — GAN/GAIL discriminator-gating (gad/d_skip_count, gad/d_update_skipped) if present.

Usage:
    python3 analyze_train_log.py logs/run_20260717_153153.log
"""
import re
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "logs/run_20260717_153153.log"
txt = open(LOG, encoding="utf-8", errors="ignore").read()


def num(line, key):
    m = re.search(re.escape(key) + r":(-?[0-9.eE+\-]+)", line)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def fmt(v, f):
    return "NA" if v is None else format(v, f)


# ---- 1. eval curve (validation accuracy) ----
best = [float(x) for x in re.findall(r"val-core/[a-z_]+/acc/best@\d+/mean:([0-9.]+)", txt)]
maj = [float(x) for x in re.findall(r"val-core/[a-z_]+/acc/maj@\d+/mean:([0-9.]+)", txt)]
print(f"# log: {LOG}\n")
print("## 1) 评估曲线 (AIME24 acc)")
if best:
    print("| # | 验证点 | best@k | maj@k | Δbest vs baseline |")
    print("|---|---|---|---|---|")
    b0 = best[0]
    for i, bb in enumerate(best):
        lab = "baseline" if i == 0 else f"val#{i}"
        mm = f"{maj[i]:.4f}" if i < len(maj) else ""
        print(f"| {i} | {lab} | {bb:.4f} | {mm} | {bb-b0:+.4f} |")
else:
    print("(还没有验证点)")

# ---- 2. training trend (per-step metrics) ----
KEYS = ["training/global_step", "actor/pg_loss", "actor/rep_loss", "actor/kl_loss",
        "actor/grad_norm", "critic/d_loss", "critic/d_acc", "response_length/mean",
        "response_length/clip_ratio", "gad/d_skip_count"]
rows = {}
for line in txt.splitlines():
    if "training/global_step:" in line and "step:" in line:
        d = {k: num(line, k) for k in KEYS}
        if d["training/global_step"] is not None:
            rows[int(d["training/global_step"])] = d
steps = [rows[k] for k in sorted(rows)]
print("\n## 2) 训练趋势 (采样步)")
if steps:
    print("| step | pg_loss | rep_loss | kl_loss | grad_norm | d_loss | d_acc | resp_len | clip% |")
    print("|---|---|---|---|---|---|---|---|---|")
    allsteps = sorted(rows)
    # sample ~12 evenly-spaced steps + always the last
    pick = sorted(set(allsteps[:: max(1, len(allsteps) // 12)] + [allsteps[-1]]))
    for st in pick:
        r = rows[st]
        print(f"| {st} | {fmt(r['actor/pg_loss'],'+.4f')} | {fmt(r['actor/rep_loss'],'.4f')} | "
              f"{fmt(r['actor/kl_loss'],'.4f')} | {fmt(r['actor/grad_norm'],'.2f')} | "
              f"{fmt(r['critic/d_loss'],'.4g')} | {fmt(r['critic/d_acc'],'.3f')} | "
              f"{fmt(r['response_length/mean'],'.0f')} | {fmt((r['response_length/clip_ratio'] or 0)*100,'.0f')} |")
    print(f"\n(已解析 {len(steps)} 个训练步)")
else:
    print("(还没有训练步)")

# ---- 3. discriminator-gating trick activity ----
skips = [r["gad/d_skip_count"] for r in steps if r.get("gad/d_skip_count") is not None]
print("\n## 3) 判别器门控 (Trick②) 活跃度")
if skips:
    fired = sum(1 for s in skips if s and s > 0)
    print(f"记录步数={len(skips)} | 触发跳过的步数={fired} | 最大连续跳过={max(skips):.0f}")
else:
    print("(未记录 gad/d_skip_count —— 门控未开或该版本未跑到)")
