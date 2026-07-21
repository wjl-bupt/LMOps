# OPRD + GAD —— 白盒跨架构组合蒸馏

> English version: [README.md](README.md)

把 **GAD**(Generative Adversarial Distillation —— 判别器 reward,用 GRPO 优化)缝进
**OPRD**(On-Policy Representation Distillation —— 教师隐状态 MSE)的代码库。

**方法:** 表征蒸馏是*主*信号,对抗判别器 reward 是*辅*信号。

```
L_actor =  rep_distillation_coef * MSE(h_student, sg(h_teacher))     # OPRD —— 主,确定性
        +  gad_coef             * PG(D(y_student))  via GRPO         # GAD  —— 辅,对抗
        +  kl_loss_coef         * KL(student || ref)                 # 锚
```

- **场景:** 白盒教师(如 Qwen3-4B)→ 小学生(如 Qwen3-1.7B),math benchmark。
- **Baseline(原生、未改动):** OPRD-only、OPD。
- **定位:** 这是*白盒*方法(需要教师隐状态),**不与黑盒 GAD 对比**;GAD 只贡献它的对抗-reward 机制。

## 更新(2026-07):流程加固 + GAN/GAIL trick + 部署

在 8× H20 上把整条流程跑通后(Qwen3-32B 教师 → Qwen3-4B 学生)所做的改动。除特别说明外都是**可选 / 默认不变**;OPD / OPRD baseline 保持隔离。

**环境修复(已并入 `bootstrap`)。** vLLM 栈会拉进 **scipy 1.18**(要求 numpy≥2),在锁定的 numpy 1.26 下经 `np.long` 导致 `import transformers` 崩溃。env 阶段现在执行 `pip install scipy==1.15.3 matplotlib`(`is_plot=True` 时需要 matplotlib)。

**GAN/GAIL 稳定化 trick(可选,默认关)** —— 用来对抗判别器饱和 / 训练发散(未加 trick 的 32B→4B 一次运行出现 `d_acc→1.0`、KL 爆炸、学生 AIME24 从 0.16 掉到 0):
| env | 默认 | 作用 |
|---|---|---|
| `GAD_REWARD_SHAPING` | `raw` | `gail` = 有界 `logσ(D)` reward(原始 `D` 无界,判别器饱和时会爆)|
| `GAD_D_GATE` | `False` | 判别器自适应门控:上一步 `d_acc > GAD_D_ACC_HI` 时跳过判别器更新 |
| `GAD_D_ACC_HI` | `0.6` | 门控阈值 |
| `GAD_D_MAX_SKIP` | `5` | **容灾**:判别器最多连跳 N 步(避免被饿死)|

新增指标 `gad/d_update_skipped`、`gad/d_skip_count`。全部门控在 `use_gad_discriminator` 之下。代码:reward 整形 + 门控在 `ray_trainer.py`;字段声明在 `workers/config/actor.py`。

**`teacher_response` 过长不再崩。** `rl_dataset.py` 现在用 `truncation="right"` 编码 `teacher_response`(原来是全局 `truncation='error'` → 教师解答超过 `max_response_length` 时崩)。过长的教师参考被截断,不再致命。*(取代下文旧的"过长会崩"坑。)*

**`gad_oprd_distillation.sh` 自含 step 0。** 若 `teacher_response` 数据缺失会自动生成(`GAD_TRAIN_DATASET`),经 Hydra 注入 student/teacher/discriminator 模型路径,`MODEL_DTYPE` 默认 `bfloat16`。env:`TEACHER_MODEL_PATH`(默认 Qwen3-32B)、`STUDENT_MODEL_PATH`(默认 Qwen3-4B)、`GAD_BASE_DATASET`、`TEACHER_GEN_N`、`TEACHER_GEN_TP`、`FORCE_REBUILD_TEACHER`。

**`build_teacher_response_parquet.py`** 新增 `--tp N`(大教师如 32B 的张量并行)和 `--sample [--seed]`(随机抽样——DAPO **按答案量级排序**,取前 N 有偏)。

**更多可 env 配置项**(`on_policy_distillation.sh`):`KL_LOSS_COEF`/`KL_LOSS_TYPE`(默认 0.005 / low_var_kl);`VAL_N`(验证 n,默认 **4**,原 16);`MAX_VAL_RESP_LENGTH`(默认 **8192**,原 15360);`TEST_FREQ`(默认 **50**,原 2)。OPRD-only:`REP_DISTILLATION_LAST_K`/`REP_DISTILLATION_LAYERS` 可覆盖。

**数据路径改为绝对(`DATA_ROOT`)。** `TRAIN_DATASET`、`TEST_DATA_DIR`、`GAD_BASE_DATASET` 由 `DATA_ROOT`(默认 `/dockerdata/junewluo/datasets`)派生,不再用相对 CWD 的 `../datasets`——所以 repo 放哪都能跑。用 `export DATA_ROOT=/该机/datasets` 覆盖。

**reward 函数路径修复**(`on_policy_distillation.sh`):`verl/utils/...` → `verl/verl/utils/...`(嵌套目录)。不用再命令行覆盖。

**部署 = 自包含 checkout。** `bootstrap` 的 `TARGET_DIR` 默认改为 `<delivery>/.oprd`(原 `$HOME/OPRD_gad`);overlay 现在 `cp scripts/*`(原 `*.sh`),`.py` 助手也会进 repo 根。注意:设 `PROJECT_PATH` 把几百 G 的训练产物放到 `.oprd` **外**;分发 bundle 时**别带**预先 clone 的 `.oprd`(bootstrap 每台机重建);若 delivery 被 git 跟踪,`.gitignore` 加 `.oprd/`。

**新增助手脚本**(`scripts/`):`dedup_train_parquet.py`(DAPO 约 100× 重复——1,791,700 行 / 17,398 唯一 prompt;训练前去重)和 `analyze_train_log.py`(`python3 analyze_train_log.py logs/run_*.log` → 评估曲线 + 训练趋势 + 门控活跃度)。

**显存画像(32B 教师 → 4B 学生,8× H20)。** 全层 rep-MSE + `last_k=2000` 会 OOM;用 `REP_DISTILLATION_LAYERS=last`、`REP_DISTILLATION_LAST_K=256`、`TEACHER_PARAM_OFFLOAD=False`、`MODEL_DTYPE=bfloat16`、`data.dataloader_num_workers=1`。`MINI_BATCH_SIZE=64` 在 step 19 OOM(~93/95GB 边缘碎片化);`32` 更稳,`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解碎片。注意这是**共享** GPU,起跑前 `nvidia-smi` 看有无其他租户。

---

## 目录内容

| 路径 | 说明 |
|---|---|
| `DEPLOY.md` | **新机器部署 runbook** —— 环境 → 模型 → 数据 → 冒烟 → 实验。 |
| `bootstrap_oprd_gad.sh` | **一键**安装+运行:clone OPRD@基线 → 合并改动 → 建 conda 环境 → 自检 → 跑。 |
| `BASE_COMMIT` | 合并所 pin 的 OPRD 基线 commit(bootstrap 脚本读取它)。 |
| `oprd_gad.patch` | 全部源码改动,针对 OPRD fork(基线 commit `93816fd`)的 `git apply` 补丁。 |
| `scripts/gad_oprd_distillation.sh` | **组合 OPRD+GAD** 启动脚本。 |
| `scripts/baseline_oprd_only.sh` | OPRD-only baseline。 |
| `scripts/baseline_opd.sh` | OPD(及 OPD+OPRD)baseline。 |
| `scripts/smoke_debug.sh` | 单卡 tiny 冒烟 + 异常自动进 pdb 的调试运行(约 3 步)。 |
| `scripts/build_teacher_response_parquet.py` | 生成 `teacher_response` 列的辅助脚本。 |
| `scripts/download_data.sh` | 下载 DAPO-Math-17k(训练)+ AIME24(评测),verl 格式。 |
| `tests/test_gad_components.py` | CPU 单测(BT loss + 末位掩码);无 GPU 可跑。 |
| `modified_files_full/` | 10 个改后文件的**完整版**(只读参考;真正要用的是 patch)。 |

## 环境配置(在 GPU 机上)

OPRD fork 用的是**比 GAD 更新的栈**(Python 3.12 / vLLM 0.11 / torch 2.8 /
flash-attn 2.8.1)——**不要复用 GAD 的 docker 镜像**。

### 一键 bootstrap(推荐)

在 GPU 机上、从本交付目录执行:

```bash
bash bootstrap_oprd_gad.sh          # clone OPRD@BASE_COMMIT + 合并改动 + 建 conda 环境 + 自检
# 之后,准备好带 teacher_response 列的数据后:
RUN_SCRIPT=smoke_debug.sh bash bootstrap_oprd_gad.sh run
```

它会把 OPRD pin 到 `BASE_COMMIT`,用**文件覆盖**方式合并改动(不受 CRLF / 补丁上下文影响),
拷贝启动脚本,建好 `verl` conda 环境并跑自检。幂等、可分阶段:
`bash bootstrap_oprd_gad.sh clone patch | env | check | run | all`。可用环境变量覆盖:
`TARGET_DIR`(默认 `<delivery>/.oprd`)、`CONDA_ENV`(`verl`)、`PATCH_MODE`(`overlay`|`apply`)、
`RUN_SCRIPT`、`RUN_ARGS`、`OPRD_REPO_URL`。

下面的手动步骤做的是同一件事。

```bash
# 1. 拿到源码 = clone OPRD fork + 打我们的补丁
git clone https://github.com/ShenzhiYang2000/OPRD.git
cd OPRD
git checkout 93816fd                                     # 补丁针对的 verl 0.7.0.dev 基线
# 从 OPRD 仓库根目录执行 git apply(补丁路径是 verl/verl/... 和 on_policy_distillation.sh)
git apply /path/to/oprd_gad_delivery/oprd_gad.patch
cp /path/to/oprd_gad_delivery/scripts/*.sh .             # 启动脚本,和 on_policy_distillation.sh 并列

# 2. 环境(OPRD 官方步骤)
conda create -n verl python==3.12 -y
conda activate verl
cd verl/
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh # vllm0.11 / torch2.8 / flash-attn2.8.1 / flashinfer / ray ...
pip install math-verify
pip install -e . --no-deps                               # 让 `python -m verl.trainer.main_ppo` 能 import 到 verl
cd ..

# 3. 模型 & 数据路径
export MODEL_DIR=/path/to/models
export DATA_DIR=/path/to/datasets
```

说明:
- `USE_MEGATRON=0` —— 我们用 FSDP,不用 Megatron(省掉难装的依赖)。
- 真实实验需要**多卡**(脚本默认 8 卡);单卡只够冒烟。

### CUDA / 驱动兼容性(目标机:驱动 535.161.08、"CUDA 12.4"、8× H20 96 GB)

- `nvidia-smi` 里的 `CUDA Version: 12.4` 是**驱动能支持的最高 CUDA runtime**,不是硬上限。PyTorch/vLLM
  自带**各自的** CUDA runtime;靠 **CUDA 12.x 次版本兼容(minor-version compatibility)**,脚本装的
  torch 2.8(cu126/cu128 构建)+ vLLM 0.11 在 12.4(r535)驱动上能正常跑。H20(Hopper,`sm_90`)完全受支持。
- 所以**按脚本原样装即可**——不需要专门的 CUDA-12.4 构建;也**不要**装系统级 CUDA 12.4 toolkit
  (torch 用不到;`USE_MEGATRON=0` 已避开需要从源码编译、才会用到系统 toolkit 的部分)。
- 装完立刻验证:
  ```bash
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  python -c "import flash_attn, vllm; print('flash_attn', flash_attn.__version__, '| vllm', vllm.__version__)"
  ```
- 仅当遇到 `CUDA driver version is insufficient for CUDA runtime version`(在 r535 上基本不会)时:重装**同版本**
  但更低 CUDA-minor 的 torch 构建,例如
  `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126`,并让 flash-attn / flashinfer /
  vLLM 的构建与之匹配。只在默认真的失败时才这么做。
- 8× H20(96 GB)非常充裕:真实实验直接用满 8 卡(`N_GPUS_PER_NODE=8`,脚本默认);冒烟用 1 卡。

## 数据

**训练 = DAPO-Math-17k;评测 = AIME24** —— 两者在 HuggingFace 上都已是 **verl 格式**(无需预处理)。
注意 AIME24 是*验证*基准,不是训练数据。

**Schema(verl 格式)。** 每一行:
```python
{
  "data_source": "math_dapo",
  "prompt": [{"role": "user", "content": "...  output the final answer within \\boxed{}."}],  # chat 列表;prompt_key="prompt"
  "ability": "math",
  "reward_model": {"style": "rule", "ground_truth": "42"},   # 验证 & OPD/OPRD baseline 用它判对错
  "extra_info": {"index": 0, "split": "train"},
  "teacher_response": "We start by ... therefore \\boxed{42}."   # 仅训练集需要,为 GAD 而加(判别器的"真样本")
}
```
验证集(AIME24)**不需要** `teacher_response`(读取处都做了 guard)。它用**学生** tokenizer 编码,
右侧用 eos 补齐到 `data.max_response_length`。

**下载**(从 OPRD 仓库根目录执行,`../datasets` 才能正确解析;幂等):
```bash
bash scripts/download_data.sh
#  -> ../datasets/dapo-math-17k.parquet   和   ../datasets/test_data/AIME24/test.parquet
export TEST_FILE='["../datasets/test_data/AIME24/test.parquet"]'   # 让启动脚本在 AIME24 上评测
```
(等价地,`verl/recipe/dapo/prepare_dapo_data.sh` 也是 wget 同样这两个 HF parquet。)

**加 `teacher_response` 列(仅 GAD 需要):**
```bash
python3 scripts/build_teacher_response_parquet.py \
  --in ../datasets/dapo-math-17k.parquet --out ../datasets/dapo-math-17k-gad.parquet \
  --teacher $REWARD_MODEL_PATH
TRAIN_DATASET=../datasets/dapo-math-17k-gad.parquet bash gad_oprd_distillation.sh
```

**两个坑:**
- 启动脚本默认 `data.truncation=error`;过长的 `teacher_response` 会**直接崩**。二选一:把
  `data.max_response_length` 设到能覆盖教师解题长度,或传 `data.truncation=right`(注意:右截断会丢掉
  末尾的 `\boxed{}` 答案 —— 更推荐把长度设大)。
- GAD 训练的 reward 是 `D(y)`,**不是** `ground_truth`(我们跳过了外部 reward_fn),但 parquet 里要**保留**
  `reward_model.ground_truth` —— 验证和 OPD/OPRD baseline 要用它。`build_teacher_response_parquet.py`
  是**加列**,不覆盖其它列。

## 运行

```bash
# 组合 OPRD + GAD(轻量的训练中低秩桥)
ACTOR_MODEL_PATH=/models/Qwen3-1.7B-Base REWARD_MODEL_PATH=/models/Qwen3-4B \
GAD_COEF=0.5 REP_DISTILLATION_COEF=1.0 REP_LOW_RANK=8 \
bash gad_oprd_distillation.sh

# Baseline
bash baseline_oprd_only.sh
bash baseline_opd.sh
```

## 调试(Debugging)

三层,由易到难。

**Level 0 —— 纯算法逻辑(任何机器,不需要 GPU/ray):**
```bash
python3 -m pdb tests/test_gad_components.py
```
单步看 Bradley-Terry 判别器 loss 和末位打分掩码。该文件里的 `real-import` 用例还会对*真实的* verl
函数做数值对拍——在裸 CPU 机上 SKIP,装好环境后 PASS。

**Level 1 —— import 冒烟(装完环境后):**
```bash
python -c "from verl.trainer.ppo.core_algos import compute_discriminator_loss; \
           from verl.workers.critic.dp_critic import DataParallelPPOCritic; print('import ok')"
```

**Level 2 —— 单卡端到端调试(核心):**
```bash
bash smoke_debug.sh     # 1 卡,约 0.5B 模型,resp_len=512,rollout.n=2,3 步
```
`smoke_debug.sh` 预置了 `RAY_DEBUG_POST_MORTEM=1` + `HYDRA_FULL_ERROR=1`,所以 **worker 里任何异常都会
自动掉进 pdb**(post-mortem)——调 verl 崩溃最实用的方式。需要一个已经带 `teacher_response` 列的训练 parquet。

**断点 & Ray 多进程的坑。** verl 跑在 Ray 上:
- `ray_trainer.fit()` 里的代码(含**我们注入 reward 的那段** `if use_gad_discriminator:`)跑在 **driver
  主进程**,普通 `breakpoint()` / IDE 断点**能命中**。
- `dp_critic` / `dp_actor` 里的代码跑在 **Ray worker 子进程**,主进程断点**打不中**。要在这里断点:
  (1) 靠 `RAY_DEBUG_POST_MORTEM=1`(已设),或 (2) 在代码里写 `breakpoint()`,运行时
  `export RAY_DEBUG=legacy`,另开终端用 `ray debug` 附加。
- 调 loss *数值*优先用 Level 0(逻辑一模一样,CPU 单步),再用 Level 2 验证接线。

建议下的断点(打完补丁后):

| 位置 | 看什么 |
|---|---|
| `ray_trainer.py` → `if use_gad_discriminator:`(compute_values 块) | `D(y)` 是否正确变成 `token_level_scores` |
| `dp_critic.py: update_critic`(`if use_gad:` 分支) | `d_loss` / `d_acc`,teacher vs student 打分 |
| `dp_critic.py: _slice_response_values` | 是否只有最后一个真实 token 非零 |
| `dp_actor.py`(`policy_loss = pg_loss` 附近) | `gad_coef·PG + rep_distillation_coef·rep` 是否正确相加 |

### 补丁暴露的冒烟/调试开关

`on_policy_distillation.sh` 现在支持这些环境变量(不设时默认值不变),单卡运行无需改脚本,
并把额外的 Hydra 覆盖通过 `"$@"` 转发:

| env | 默认 | 冒烟值 |
|---|---|---|
| `N_GPUS_PER_NODE` | 8 | 1 |
| `MAX_RESP_LENGTH` | 16384 | 512 |
| `N_RESPONSES`(GRPO 组大小) | 2 | 2 |
| `MINI_BATCH_SIZE` | 8 | 2 |
| `TEST_FREQ` / `SAVE_FREQ` | 2 / 200 | 调大(跳过) |

追加一次性覆盖的例子(追加的 key 不能和脚本里已设的重复,比如步数上限):
```bash
bash smoke_debug.sh trainer.total_training_steps=3   # smoke_debug.sh 里已默认带上这一条
```

## 架构(四个 worker 共存,不冲突)

| worker | 角色 | 开关 |
|---|---|---|
| actor / rollout | 学生(生成器) | —— |
| `reward_model` | 冻结的白盒**教师** → 出隐状态供 MSE | `reward_model.enable=True` |
| `critic` | **判别器**(Bradley-Terry,→ `D(y)` reward) | `critic.enable=True` + `use_gad_discriminator=True` |
| `ref` | KL 锚 | `actor.use_kl_loss=True` |

## 源码改动(都在 `oprd_gad.patch` 里)

- `core_algos.py` —— `compute_discriminator_loss`(Bradley-Terry `-logσ(D(y_t)-D(y_s))`)。
- `workers/config/actor.py`、`workers/config/critic.py`、`config/critic/critic.yaml` —— `use_gad_discriminator`、`gad_coef`、`gad_gate_pg`。
- `workers/critic/dp_critic.py` —— 判别器模式:`_forward_micro_batch(compute_teacher=)`、末位打分(`_slice_response_values`)、BT `update_critic`。GAE value 路径保留。
- `utils/dataset/rl_dataset.py` —— 可选的 `teacher_response` 编码(guarded)。
- `workers/rollout/vllm_rollout/vllm_rollout_spmd.py` —— 构造 `teacher_input_ids/attention_mask/position_ids`(guarded)。
- `trainer/ppo/ray_trainer.py` —— 把 `teacher_response` 送进 gen batch;GAD 下 `D(y_student)` → `token_level_scores` → GRPO 优势(跳过外部 reward_fn)。
- `workers/actor/dp_actor.py` —— PG 项乘以 `gad_coef`(rep MSE 项不变)。
- `on_policy_distillation.sh` —— `GAD_ARGS` 块(默认关 → baseline 不变);把 `N_GPUS_PER_NODE / MAX_RESP_LENGTH / N_RESPONSES / TEST_FREQ / SAVE_FREQ` 暴露为环境变量(默认值不变),并向 python 命令转发 `"$@"` 以支持临时 Hydra 覆盖。

所有 GAD 行为都门控在 `use_gad_discriminator` + `adv_estimator=grpo` 之下;关掉 GAD 时,OPRD-only 和 OPD 与原版逐字节一致。

## 验证状态

- **已完成(CPU/静态):** `python3 tests/test_gad_components.py` 通过(BT loss 数值/梯度、末位掩码);所有改过的 `.py` 编译通过;所有脚本 `bash -n` 通过;补丁在基线 `93816fd` 上干净 apply。
- **需 GPU(尚未运行 —— 作者机无 GPU):** actor + teacher(RM) + discriminator(critic) 的 FSDP 同分片;带 teacher 序列的 vLLM rollout;端到端 reward → advantage → PG + rep MSE;OOM 行为。**首个 GPU gate:** `bash scripts/smoke_debug.sh`(≈0.5B 模型,`rollout.n=2`,3 步),之后再上真正的 4B→1.5B。

## 调参 / 尖角(细节见 plan)

- **λ(`gad_coef`)从小起(0.1–1)**;保持 rep 项主导。盯 `actor/rep_loss` vs `actor/pg_loss` 和 `critic/d_acc`。
- **判别器冷启动:** 用 `trainer.critic_warmup` 和/或让 `gad_coef` 从 0 爬升;此间靠 MSE 项学习。
- **显存:** 每步三个大前向(teacher + discriminator×2 + actor)。判别器选小骨架;RM/critic 开 `param_offload=True`;`critic.use_dynamic_bsz=False`。
- **`teacher_response` 长度:** 把 `data.max_response_length` 设到能覆盖教师解题长度(右截断会丢尾部 / 最终答案)。
- **reward 滞后一步**(判别器在自身更新前打分)—— 符合预期,与 GAD 一致。
- **轻量桥 vs 冻结桥:** 默认用训练中 PCA `P_T` + 联合 `P_S`(`rep_freeze_ps=False`)。想用更强的冻结桥,先离线建好 `ps_bank.pt`,再设 `REP_LOW_RANK_INIT_CHECKPOINT` + `rep_freeze_ps=True`。
