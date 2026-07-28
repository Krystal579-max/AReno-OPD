"""On-Policy Distillation (OPD) trainer.

OPD is a distillation algorithm where the student model generates responses
(rollout) and a frozen teacher model scores them. The student learns to match
the teacher's distribution by minimising KL(student || teacher) on the
student-generated response tokens.

Per step the trainer:
    1. Rolls out ``n_samples`` completions per prompt from the student model.
    2. Scores every completion with the frozen teacher model (log-probabilities).
    3. Builds ``TrainSequence`` objects with the teacher logprobs stored in
       ``ref_logprobs`` so the OPD loss function can consume them.
    4. Runs the student training step with the OPD loss.

Unlike PPO, OPD has no critic, no reward model, no GAE, and no advantage
normalisation — the learning signal comes entirely from the teacher's
log-probabilities.
"""
from __future__ import annotations

import logging
import time
from functools import partial

import areno.api
from areno.api.dashboard import record_dashboard_state
from areno.api.roles import ModelRole
from areno.api.trainers.policy_only import PolicyOnlyTrainer

logger = logging.getLogger(__name__)


def _summary_stats(prefix: str, values: list[float]) -> dict[str, float]:
    """Compute mean/std/min/max for a list of floats."""
    import numpy as np

    if not values:
        return {}
    arr = np.array(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std(ddof=0)),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
    }


class OPDTrainer(PolicyOnlyTrainer):
    """On-Policy Distillation trainer.

    OPD uses the student model for rollout and a frozen teacher model for
    log-probability scoring. The teacher checkpoint defaults to the student's
    checkpoint when ``ref_ckpt`` is not configured (which is useful for
    bootstrapping but not true distillation — a real teacher should be a
    different, generally stronger model).

    The loss function is expected to be ``opd_loss_fn`` partially applied with
    the OPD-specific hyperparameters from the config.
    """

    def __init__(self, config, *, instance, dataset, reward_fn, loss_fn):
        super().__init__(config, instance=instance, dataset=dataset, reward_fn=reward_fn, loss_fn=loss_fn)

        # Partially apply OPD knobs so the trainer can pass
        # ``loss_fn(data_pack, logp)`` without re-specifying config each step.
        self.loss_fn = partial(
            loss_fn,
            kl_coef=config.opd_kl_coef,
            temperature=config.opd_temperature,
        )

        # Holding pen for per-step auxiliary stats.
        self._last_opd_stats: dict[str, float] = {}

        # Teacher model role (frozen reference).
        teacher_ckpt = config.ref_ckpt or config.ckpt
        self.roles = {
            "teacher": ModelRole("teacher", teacher_ckpt, trainable=False),
        }

    def _policy_role_name(self) -> str:
        return "student"

    def _record_opd_state(self, *, stage: str, role: str) -> None:
        record_dashboard_state(
            self.areno,
            stage=stage,
            epoch=getattr(self, "_dashboard_epoch", None),
            step=getattr(self, "_dashboard_step", None),
            role=role,
        )

    def _materialize_train_batch(self, tokenizer, prompt_batch, rollout_results):
        """Build training batches from rollout results with teacher scoring.

        For each (prompt, sample) pair this method:
        1. Concatenates prompt + response tokens.
        2. Scores the full token row with the frozen teacher model.
        3. Builds a ``TrainSequence`` with teacher logprobs stored in
           ``ref_logprobs`` so the OPD loss function can consume them.
        """
        self._last_opd_stats = {}
        train_batch: list = []
        rollout_logprobs: list[float] = []
        teacher_logprobs_all: list[float] = []

        # Collect full token rows for a single batched teacher forward.
        token_rows: list[list[int]] = []
        row_meta: list[tuple] = []

        for item_idx, (item, result) in enumerate(zip(prompt_batch.items, rollout_results, strict=True)):
            prefix_len = len(item.input_tokens)
            for sample_idx, seq in enumerate(result.sequences):
                resp_len = len(seq.resp_tokens)
                if resp_len < 1:
                    continue
                tokens = item.input_tokens + seq.resp_tokens
                token_rows.append(tokens)
                row_meta.append((item, seq, prefix_len, resp_len))

        if not token_rows:
            return [], [], []

        # Ensure the teacher role is loaded on the backend.
        self.logger.info("role=teacher stage=ensure_roles")
        self.areno.ensure_roles(self.roles)

        # Forward the teacher model over every row in a single batched call.
        self.logger.info("role=teacher stage=logprob_score_start rows=%d", len(token_rows))
        self._record_opd_state(stage="logprob_score_start", role="teacher")
        teacher_start = time.perf_counter()
        teacher_logprob_rows = self.areno.score_logprobs("teacher", token_rows)
        teacher_time = time.perf_counter() - teacher_start
        self._last_opd_stats["teacher_logprob_forward_time_s"] = teacher_time
        self.logger.info("role=teacher stage=logprob_score_end rows=%d time=%.3fs", len(token_rows), teacher_time)
        self._record_opd_state(stage="logprob_score_end", role="teacher")

        # Build TrainSequence objects.
        for (item, seq, prefix_len, resp_len), teacher_logprobs in zip(row_meta, teacher_logprob_rows, strict=True):
            # Slice teacher logprobs to the response window.
            action_teacher_logprobs = teacher_logprobs[prefix_len: prefix_len + resp_len]
            if len(action_teacher_logprobs) != resp_len:
                raise ValueError(
                    f"teacher returned misaligned logprobs: "
                    f"expected {resp_len} response tokens, got {len(action_teacher_logprobs)}"
                )

            rollout_logprobs.extend(seq.resp_logprobs)
            teacher_logprobs_all.extend(action_teacher_logprobs)

            # Build the full teacher logprob array (prompt positions zeroed).
            full_teacher_logprobs = [0.0] * prefix_len + action_teacher_logprobs

            train_batch.append(
                areno.api.TrainSequence(
                    prompt_mask=[True] * prefix_len + [False] * resp_len,
                    tokens=item.input_tokens + seq.resp_tokens,
                    # Use the rollout logprobs as the "old" logprobs so the
                    # loss function can compute the ratio if needed.
                    logprobs=[0.0] * prefix_len + seq.resp_logprobs,
                    # Store teacher logprobs in ref_logprobs so the OPD loss
                    # function can access them via ``layout.ref_logprobs``.
                    ref_logprobs=full_teacher_logprobs,
                    eos_token_id=tokenizer.eos_token_id,
                )
            )

        # Log diagnostic statistics.
        if teacher_logprobs_all:
            self._last_opd_stats.update(_summary_stats("teacher_logprob", teacher_logprobs_all))
        if rollout_logprobs:
            self._last_opd_stats.update(_summary_stats("rollout_logprob", rollout_logprobs))

        return train_batch, [], rollout_logprobs

    def _augment_train_stats(self, result):
        """Attach OPD-specific stats (teacher forward time, logprob stats)."""
        if isinstance(result, dict) and self._last_opd_stats:
            merged = dict(result)
            merged.update(self._last_opd_stats)
            self._last_opd_stats = {}
            return merged
        return result