"""On-Policy Distillation (OPD) loss function.

OPD is a distillation algorithm where the student model generates responses
(rollout) and the teacher model scores them. The student learns to match the
teacher's distribution by minimizing the KL divergence between the student and
teacher log-probabilities on the student-generated response tokens.

    loss = KL(student || teacher) = mean(student_logprob - teacher_logprob)

where the teacher logprobs are detached from the graph (the teacher is frozen).
"""
from __future__ import annotations

from areno.api.loss_fns.layout import masked_mean, response_layout


def opd_loss_fn(
    data_pack,
    logprobs,
    *,
    kl_coef: float = 1.0,
    temperature: float = 1.0,
):
    """On-Policy Distillation loss: KL(student || teacher) on response tokens.

    The student model generates responses (rollout). The teacher (reference)
    model scores the same tokens. The loss minimises the KL divergence from the
    student to the teacher distribution on the student-generated tokens.

    Parameters
    ----------
    data_pack : dict
        Packed or padded data dictionary containing at least the response mask
        and ``ref_logprobs`` (teacher log-probabilities).
    logprobs : torch.Tensor
        Current student log-probabilities (differentiable).
    kl_coef : float, optional
        Coefficient scaling the KL loss term (default: ``1.0``).
    temperature : float, optional
        Temperature applied to both student and teacher logprobs before
        computing the KL divergence. Higher temperatures soften the
        distribution (default: ``1.0``).

    Returns
    -------
    loss : torch.Tensor
        Scalar loss value.
    stats : dict[str, torch.Tensor]
        Diagnostic metrics including ``opd_loss``, ``opd_kl``,
        ``opd_student_logprob_mean``, and ``opd_teacher_logprob_mean``.
    """
    import torch

    kl_coef = float(kl_coef)
    temperature = float(temperature)

    # Resolve the response-layout view (packed or padded).
    layout = response_layout(data_pack, logprobs, need_ref_logprobs=True)

    # Apply temperature scaling.
    scaled_logprobs = logprobs / temperature
    teacher_logprobs = layout.ref_logprobs / temperature

    # KL(student || teacher) = log π_s(a) - log π_t(a).
    # The teacher is frozen so its logprobs are detached.
    kl = scaled_logprobs - teacher_logprobs.detach()

    # Masked mean over response tokens only, scaled by the KL coefficient.
    loss = masked_mean(kl, layout) * kl_coef

    # Diagnostic statistics.
    with torch.no_grad():
        kld = masked_mean(kl, layout)
        student_lp = masked_mean(scaled_logprobs, layout)
        teacher_lp = masked_mean(teacher_logprobs, layout)

    return loss, {
        "opd_loss": loss.detach(),
        "opd_kl": kld.detach(),
        "opd_student_logprob_mean": student_lp.detach(),
        "opd_teacher_logprob_mean": teacher_lp.detach(),
        "opd_kl_coef": torch.tensor(kl_coef, device=logprobs.device),
        "opd_temperature": torch.tensor(temperature, device=logprobs.device),
    }