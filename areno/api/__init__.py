"""AReno API: OPD algorithm integration module.

This package extends the AReno training framework with the On-Policy
Distillation (OPD) algorithm, including:
- OPD loss function (KL divergence)
- OPD trainer with teacher-scoring workflow
- OPD configuration dataclass
- Algorithm registration
"""

from areno.api.loss_fns import opd_loss_fn
from areno.api.trainers import OPDTrainer

__all__ = ["opd_loss_fn", "OPDTrainer"]