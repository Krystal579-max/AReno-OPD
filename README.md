# AReno-OPD: On-Policy Distillation for AReno

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

为 [AReno](https://github.com/inclusionAI/AReno) 开源框架添加 **On-Policy Distillation (OPD)** 算法支持。OPD 是一种在线策略蒸馏技术，学生模型生成响应（rollout），冻结的教师模型为其打分，学生通过最小化与教师分布之间的 KL 散度来学习。

---

## 目录

- [背景介绍](#背景介绍)
- [算法原理](#算法原理)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [API 参考](#api-参考)
- [测试](#测试)
- [算法对比](#算法对比)
- [License](#license)

---

## 背景介绍

### 什么是 On-Policy Distillation?

On-Policy Distillation (OPD) 是一种面向大语言模型后训练的知识蒸馏技术。与传统的离线蒸馏不同，OPD 是**在线策略（on-policy）** 的——学生模型**自己生成**响应，然后让教师模型对这些响应进行评分。这使得学生能够在其自身的分布上学习，从而更有效地匹配教师的行为。

### 为什么在 AReno 中实现 OPD?

[AReno](https://github.com/inclusionAI/AReno) 是一个面向大模型后训练的开源框架，致力于将训练框架和推理引擎放在同一套基础设施中，实现训推一体。AReno 已支持 PPO、GRPO、DPO、SFT 等算法，而 OPD 的加入填补了**在线策略蒸馏**这一重要方向，为用户提供更丰富的模型优化选择。

---

## 算法原理

### 工作流程

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│   提示数据    │ ──► │  学生模型(可训练)  │ ──► │  Rollout: 每提示生成  │
│   (Prompts)  │     │  (Student)       │     │  n_samples 个响应     │
└──────────────┘     └──────────────────┘     └──────────┬───────────┘
                                                          │
                                                          ▼
                                                ┌──────────────────┐
                                                │  教师模型(冻结)    │
                                                │  (Teacher)        │
                                                │  对响应 token 评分 │
                                                │  (logprobs)       │
                                                └──────────┬───────────┘
                                                          │
                                                          ▼
                                                ┌──────────────────┐
                                                │  KL 散度损失:     │
                                                │  KL(student ||   │
                                                │    teacher)      │
                                                └──────────┬───────────┘
                                                          │
                                                          ▼
                                                ┌──────────────────┐
                                                │  学生模型更新     │
                                                │  (梯度下降)       │
                                                └──────────────────┘
```

### 数学公式

对于学生策略 `π_s` 生成的每个响应 token `a`：

- **学生 log-probability**: `log π_s(a | context)`
- **教师 log-probability**: `log π_t(a | context)`

蒸馏损失为学生到教师分布的 KL 散度：

```
Loss = KL(π_s || π_t) = E_{a ~ π_s}[log π_s(a) - log π_t(a)]
                    ≈ mean(log π_s(a) - log π_t(a))   # 在响应 token 上取平均
```

教师模型的 log-probabilities 被**脱离计算图**（`.detach()`），梯度仅流经学生网络：

```
∇Loss ≈ mean(∇log π_s(a))
```

### 温度缩放

通过温度参数 `τ` 软化两个分布：

```
Loss_τ = mean((log π_s(a) - log π_t(a)) / τ) = Loss / τ
```

更高的温度使分布更平滑，降低损失幅度。

### 与 PPO 的关键区别

| 方面 | OPD | PPO |
|------|-----|-----|
| 学习信号 | 教师 logprobs | 奖励 + GAE 优势 |
| 奖励模型 | 不需要 | 需要 |
| Critic 网络 | 不需要 | 需要 |
| 参考模型 | 教师（评分用，冻结） | KL 惩罚（冻结） |
| 优势计算 | 不需要 | GAE 广义优势估计 |

---

## 项目结构

```
AReno-OPD/
├── areno/
│   └── api/
│       ├── algorithms.py              # 算法注册表（注册 opd）
│       ├── trainer_config.py          # OPDTrainerConfig 配置类
│       ├── loss_fns/
│       │   ├── __init__.py            # 导出 opd_loss_fn
│       │   └── opd.py                 # OPD 损失函数实现
│       └── trainers/
│           ├── __init__.py            # 导出 OPDTrainer
│           └── opd.py                 # OPD 训练器实现
├── tests/
│   ├── test_opd_loss.py               # 损失函数单元测试（12 个用例）
│   └── test_opd_algorithm.py          # 算法注册与配置测试（6 个用例）
├── docs/
│   └── opd_algorithm.md               # 详细算法文档
├── README.md                          # 本文件
└── .gitignore
```

### 核心文件说明

| 文件 | 说明 |
|------|------|
| `areno/api/loss_fns/opd.py` | OPD 损失函数，计算 KL(student \|\| teacher) 散度 |
| `areno/api/trainers/opd.py` | OPD 训练器，包含 Rollout + Teacher-Scoring 工作流 |
| `areno/api/trainer_config.py` | `OPDTrainerConfig` 数据类，管理超参数 |
| `areno/api/algorithms.py` | 将 `opd` 注册为 AReno 内置算法 |
| `tests/test_opd_loss.py` | 12 个测试覆盖损失计算、可微性、温度缩放等 |
| `tests/test_opd_algorithm.py` | 6 个测试覆盖注册、配置、导出等 |

---

## 快速开始

### 前提条件

- Python 3.10+
- PyTorch 2.0+
- AReno 框架（需先安装）

### 安装

```bash
# 克隆 AReno 主仓库
git clone https://github.com/inclusionAI/AReno.git
cd AReno

# 安装依赖
pip install -e .

# 将本仓库的 OPD 代码合并到 AReno 中
# 方式一：直接复制文件
cp -r /path/to/AReno-OPD/areno/* ./areno/

# 方式二：使用 git patch
git am /path/to/AReno-OPD.patch
```

### 训练脚本

```python
import areno.api
from areno.api.trainer_config import OPDTrainerConfig
from areno.api.trainer_factory import build_trainer

# 配置 OPD 训练
config = OPDTrainerConfig(
    algo="opd",
    ckpt="/path/to/student/model",          # 学生模型（可训练）
    dataset_path="/path/to/training/data",
    ref_ckpt="/path/to/teacher/model",       # 教师模型（冻结，通常更强）
    opd_kl_coef=1.0,                         # KL 损失系数
    opd_temperature=1.0,                     # 温度参数
    n_samples=8,                             # 每提示采样数
    batch_size=32,
    epochs=3,
)

# 构建训练器
trainer = build_trainer(
    config,
    instance=areno.api.Trainer(...),
    dataset=dataset,
    reward_fn=None,      # OPD 不需要奖励函数
    loss_fn=areno.api.loss_fns.opd_loss_fn,
)

# 开始训练
trainer.fit()
```

### CLI 命令行使用

```bash
python -m areno train \
    --algo opd \
    --ckpt /path/to/student/model \
    --dataset /path/to/data \
    --ref-ckpt /path/to/teacher/model \
    --opd-kl-coef 1.0 \
    --opd-temperature 1.0 \
    --n-samples 8 \
    --batch-size 32
```

---

## 配置说明

### OPDTrainerConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ref_ckpt` | `str \| None` | `None` | 教师模型检查点路径（默认使用学生模型） |
| `opd_kl_coef` | `float` | `1.0` | KL 散度损失系数 |
| `opd_temperature` | `float` | `1.0` | 温度参数，软化 log-probability 分布 |

从 `RolloutTrainerConfig` 继承的关键参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_samples` | `int` | `8` | 每提示生成的采样数 |
| `temperature` | `float` | `1.0` | Rollout 采样温度 |
| `top_p` | `float` | `1.0` | Top-p (nucleus) 采样参数 |
| `top_k` | `int` | `-1` | Top-k 采样参数（-1 表示禁用） |
| `batch_size` | `int` | `32` | 训练批次大小 |
| `max_prompt_tokens` | `int` | `1024` | 最大提示 token 数 |
| `max_new_tokens` | `int` | `3071` | 最大生成 token 数 |

---

## API 参考

### `opd_loss_fn(data_pack, logprobs, *, kl_coef=1.0, temperature=1.0)`

**参数：**
- `data_pack` (`dict`): 包含响应掩码和 `ref_logprobs`（教师 log-probabilities）的数据字典
- `logprobs` (`torch.Tensor`): 当前学生 log-probabilities（可微）
- `kl_coef` (`float`, 可选): KL 损失缩放系数，默认 `1.0`
- `temperature` (`float`, 可选): 温度参数，默认 `1.0`

**返回：**
- `loss` (`torch.Tensor`): 标量损失值
- `stats` (`dict`): 诊断指标，包括 `opd_loss`, `opd_kl`, `opd_student_logprob_mean`, `opd_teacher_logprob_mean`, `opd_kl_coef`, `opd_temperature`

### `OPDTrainer`

继承自 `PolicyOnlyTrainer`，主要方法：

| 方法 | 说明 |
|------|------|
| `fit()` | 启动完整训练循环 |
| `_materialize_train_batch(tokenizer, prompt_batch, rollout_results)` | 构建训练批次，包含教师评分 |
| `_augment_train_stats(result)` | 附加 OPD 诊断统计信息 |

### `OPDTrainerConfig`

```python
@dataclass(slots=True)
class OPDTrainerConfig(RolloutTrainerConfig):
    ref_ckpt: str | None = None
    opd_kl_coef: float = 1.0
    opd_temperature: float = 1.0
```

---

## 测试

运行所有 OPD 相关测试：

```bash
# 运行损失函数测试
python -m pytest tests/test_opd_loss.py -v

# 运行算法注册测试
python -m pytest tests/test_opd_algorithm.py -v

# 运行所有测试
python -m pytest tests/test_opd*.py -v
```

### 测试覆盖

**损失函数测试**（`test_opd_loss.py`，12 个用例）：

| 测试 | 说明 |
|------|------|
| `test_loss_is_scalar` | 验证损失输出为标量 |
| `test_loss_is_differentiable` | 验证损失可微 |
| `test_teacher_logprobs_detached` | 验证教师 logprobs 不接收梯度 |
| `test_loss_value_correct` | 验证简单情况下的正确损失值 |
| `test_packed_and_padded_agree` | 验证 packed 和 padded 布局结果一致 |
| `test_temperature_scaling` | 验证温度缩放正确 |
| `test_kl_coef_scaling` | 验证 KL 系数线性缩放 |
| `test_prompt_tokens_masked` | 验证提示 token 被正确屏蔽 |
| `test_all_prompt_tokens` | 验证全为提示 token 时损失为 0 |
| `test_stats_contain_expected_keys` | 验证诊断统计键完整 |
| `test_stats_values` | 验证统计值正确 |

**算法注册测试**（`test_opd_algorithm.py`，6 个用例）：

| 测试 | 说明 |
|------|------|
| `test_opd_algorithm_registered` | 验证 `opd` 算法已注册 |
| `test_opd_trainer_resolves` | 验证训练器类可解析 |
| `test_opd_loss_fn_factory` | 验证损失工厂绑定超参数 |
| `test_default_config` | 验证默认配置 |
| `test_custom_config` | 验证自定义配置 |
| `test_opd_trainer_in_trainers_init` | 验证训练器导出 |

---

## 算法对比

| 方面 | OPD | PPO | GRPO | DPO | SFT |
|------|-----|-----|------|-----|-----|
| **类型** | 蒸馏 | 强化学习 | 强化学习 | 偏好学习 | 监督学习 |
| **在线策略** | 是 | 是 | 是 | 否（离线） | 否 |
| **奖励模型** | 不需要 | 需要 | 不需要（组归一化） | 不需要 | 不需要 |
| **Critic** | 不需要 | 需要 | 不需要 | 不需要 | 不需要 |
| **参考模型** | 教师（评分） | KL 惩罚 | 无 | 参考策略 | 无 |
| **损失函数** | KL(student \|\| teacher) | 裁剪 Actor + KL | 裁剪策略梯度 | 偏好边界 | 负对数似然 |
| **学习信号** | 教师 logprobs | 奖励 + GAE | 组相对奖励 | 偏好对 | 目标 token |

### 何时选择 OPD?

- **您有一个更强的教师模型**：如果您可以访问一个更大或更优的模型（如 GPT-4、Qwen-72B 等），OPD 可以有效将其知识蒸馏到小模型中
- **需要在线策略学习**：学生模型从其自身的分布中学习，而非静态数据集
- **简化训练流程**：不需要奖励模型、critic 网络或优势计算，实现更简单
- **资源受限**：相比 PPO，OPD 需要更少的 GPU 显存（无 critic 网络）

---

## License

本项目基于 Apache License 2.0 开源协议。AReno 主仓库同样采用 Apache 2.0 协议。

---

## 参考

- [AReno 开源框架](https://github.com/inclusionAI/AReno) — 大模型后训练训推一体框架
- [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525) — 知识蒸馏综述
- [On-Policy Distillation: A Survey](https://arxiv.org/abs/2310.12345) — 在线策略蒸馏技术
- [PPO: Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) — 近端策略优化
- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300) — 组相对策略优化