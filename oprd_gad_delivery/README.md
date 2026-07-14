# OPRD + GAD — Combined White-box Cross-arch Distillation

Merges **GAD** (Generative Adversarial Distillation — a discriminator reward optimized by GRPO)
into the **OPRD** (On-Policy Representation Distillation — teacher hidden-state MSE) codebase.

**Method:** representation distillation is the *primary* signal; the adversarial discriminator
reward is *auxiliary*.

```
L_actor =  rep_distillation_coef * MSE(h_student, sg(h_teacher))     # OPRD — primary, deterministic
        +  gad_coef             * PG(D(y_student))  via GRPO         # GAD  — auxiliary, adversarial
        +  kl_loss_coef         * KL(student || ref)                 # anchor
```

- **Setting:** white-box teacher (e.g. Qwen3-4B) → small student (e.g. Qwen3-1.7B), math benchmarks.
- **Baselines (native, unchanged):** OPRD-only, OPD.
- **Positioning:** this is a *white-box* method (needs teacher hidden states); it is **not** compared
  against black-box GAD. GAD contributes only its adversarial-reward mechanism.

## Contents

| Path | What |
|---|---|
| `oprd_gad.patch` | All source changes, as a `git apply`-able patch against the OPRD fork (base commit `93816fd`). |
| `scripts/gad_oprd_distillation.sh` | **Combined OPRD+GAD** launcher. |
| `scripts/baseline_oprd_only.sh` | OPRD-only baseline. |
| `scripts/baseline_opd.sh` | OPD (and OPD+OPRD) baseline. |
| `scripts/build_teacher_response_parquet.py` | Helper to add the `teacher_response` column. |
| `tests/test_gad_components.py` | CPU unit tests (BT loss + last-token masking); runs without GPU. |

## Setup (on the GPU machine)

```bash
# 1. Clone the OPRD fork at the base commit the patch targets
git clone https://github.com/ShenzhiYang2000/OPRD.git
cd OPRD && git checkout 93816fd            # verl 0.7.0.dev base

# 2. Apply the merge (run from the OPRD repo root — patch paths are verl/verl/... and on_policy_distillation.sh)
git apply /path/to/oprd_gad_delivery/oprd_gad.patch

# 3. Install (per OPRD README) — verl 0.7.0.dev + vllm + ray + flash-attn, etc.

# 4. Drop the launchers next to on_policy_distillation.sh
cp /path/to/oprd_gad_delivery/scripts/*.sh .
```

## Data

The combined run's training parquet needs a **`teacher_response` TEXT column** (the white-box
teacher's own solution to each prompt) — the discriminator's "real" example. See
`scripts/build_teacher_response_parquet.py` (wire in offline vLLM teacher generation).
Validation parquets do **not** need it (all reads are guarded). `teacher_response` is tokenized
with the **student** tokenizer and right-padded with eos to `data.max_response_length`.

## Run

```bash
# Combined OPRD + GAD (lightweight in-training low-rank bridge)
ACTOR_MODEL_PATH=/models/Qwen3-1.7B-Base REWARD_MODEL_PATH=/models/Qwen3-4B \
GAD_COEF=0.5 REP_DISTILLATION_COEF=1.0 REP_LOW_RANK=8 \
bash gad_oprd_distillation.sh

# Baselines
bash baseline_oprd_only.sh
bash baseline_opd.sh
```

## Architecture (four coexisting workers, no collision)

| worker | role | gate |
|---|---|---|
| actor / rollout | student (generator) | — |
| `reward_model` | frozen white-box **teacher** → hidden states for MSE | `reward_model.enable=True` |
| `critic` | **discriminator** (Bradley-Terry, → `D(y)` reward) | `critic.enable=True` + `use_gad_discriminator=True` |
| `ref` | KL anchor | `actor.use_kl_loss=True` |

## Source changes (in `oprd_gad.patch`)

- `core_algos.py` — `compute_discriminator_loss` (Bradley-Terry `-logσ(D(y_t)-D(y_s))`).
- `workers/config/actor.py`, `workers/config/critic.py`, `config/critic/critic.yaml` — `use_gad_discriminator`, `gad_coef`, `gad_gate_pg`.
- `workers/critic/dp_critic.py` — discriminator mode: `_forward_micro_batch(compute_teacher=)`, last-real-token score (`_slice_response_values`), BT `update_critic`. GAE value path preserved.
- `utils/dataset/rl_dataset.py` — optional `teacher_response` tokenization (guarded).
- `workers/rollout/vllm_rollout/vllm_rollout_spmd.py` — build `teacher_input_ids/attention_mask/position_ids` (guarded).
- `trainer/ppo/ray_trainer.py` — pop `teacher_response` into the gen batch; under GAD, `D(y_student)` → `token_level_scores` → GRPO advantage (external reward_fn skipped).
- `workers/actor/dp_actor.py` — scale the PG term by `gad_coef` (rep MSE term unchanged).
- `on_policy_distillation.sh` — `GAD_ARGS` block (off by default → baselines unchanged).

All GAD behavior is gated behind `use_gad_discriminator` + `adv_estimator=grpo`; OPRD-only and OPD are byte-for-byte unchanged when GAD is off.

## Verification status

- **Done (CPU/static):** `python3 tests/test_gad_components.py` passes (BT loss value/gradient, last-token masking); all edited `.py` compile; `bash -n` on all scripts; patch applies cleanly on base `93816fd`.
- **Requires GPU (not yet run — no GPU on the authoring machine):** FSDP co-sharding of actor + teacher(RM) + discriminator(critic); vLLM rollout with teacher sequences; end-to-end reward → advantage → PG + rep MSE; OOM behavior. **First GPU gate:** tiny smoke config (student/teacher/discriminator ≈0.5B, `rollout.n=2`, 3 steps) before the real 4B→1.5B run.

## Tuning / sharp edges (see plan for detail)

- **λ (`gad_coef`) start small (0.1–1)**; keep the rep term dominant. Watch `actor/rep_loss` vs `actor/pg_loss` and `critic/d_acc`.
- **Discriminator cold-start:** use `trainer.critic_warmup` and/or ramp `gad_coef` from 0; the MSE term carries early learning.
- **Memory:** three big forwards/step (teacher + discriminator×2 + actor). Keep the discriminator small; `param_offload=True` on RM/critic; `critic.use_dynamic_bsz=False`.
- **`teacher_response` length:** size `data.max_response_length` to cover teacher solutions (right-truncation drops the tail / final answer).
- **Reward lags one step** (discriminator scores before its own update) — expected, matches GAD.
- **Lightweight vs frozen bridge:** default uses in-training PCA `P_T` + joint `P_S` (`rep_freeze_ps=False`). For the stronger frozen bridge, build `ps_bank.pt` offline and set `REP_LOW_RANK_INIT_CHECKPOINT` + `rep_freeze_ps=True`.
