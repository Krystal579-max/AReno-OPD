"""Unit tests for the OPD loss function."""
from __future__ import annotations

import pytest
import torch


def _make_packed_data_pack(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    seq_ids: torch.Tensor | None = None,
) -> dict:
    """Build a packed data_pack dict for the OPD loss function."""
    data_pack = {
        "packed_response_mask": response_mask,
        "packed_logprobs": logprobs,
        "packed_ref_logprobs": ref_logprobs,
    }
    if seq_ids is not None:
        data_pack["packed_seq_ids"] = seq_ids
        data_pack["packed_num_sequences"] = int(seq_ids[-1].item()) + 1
    return data_pack


def _make_padded_data_pack(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    prompt_mask: torch.Tensor,
) -> dict:
    """Build a padded data_pack dict for the OPD loss function."""
    return {
        "prompt_mask": prompt_mask,
        "logprobs": logprobs,
        "ref_logprobs": ref_logprobs,
    }


class TestOPDLossFunction:
    """Test suite for ``opd_loss_fn``."""

    @pytest.fixture(autouse=True)
    def _import_opd(self):
        from areno.api.loss_fns.opd import opd_loss_fn

        self.opd_loss_fn = opd_loss_fn

    # ------------------------------------------------------------------
    # Basic loss computation
    # ------------------------------------------------------------------

    def test_loss_is_scalar(self):
        """The loss must return a scalar (0-d) tensor."""
        logprobs = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=torch.float32)
        ref_logprobs = torch.tensor([-0.5, -1.5, -2.5, -3.5, -4.5], dtype=torch.float32)
        response_mask = torch.tensor([0, 1, 1, 1, 1], dtype=torch.float32)
        data_pack = _make_packed_data_pack(logprobs, ref_logprobs, response_mask)
        loss, stats = self.opd_loss_fn(data_pack, logprobs)
        assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"

    def test_loss_is_differentiable(self):
        """The loss must have a gradient w.r.t. ``logprobs``."""
        logprobs = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32, requires_grad=True)
        ref_logprobs = torch.tensor([-0.5, -1.5, -2.5], dtype=torch.float32)
        response_mask = torch.tensor([1, 1, 1], dtype=torch.float32)
        data_pack = _make_packed_data_pack(logprobs, ref_logprobs, response_mask)
        loss, _ = self.opd_loss_fn(data_pack, logprobs)
        loss.backward()
        assert logprobs.grad is not None, "Gradient should be computed"
        assert logprobs.grad.shape == logprobs.shape

    def test_teacher_logprobs_detached(self):
        """Teacher logprobs should not receive gradients."""
        logprobs = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32, requires_grad=True)
        ref_logprobs = torch.tensor([-0.5, -1.5, -2.5], dtype=torch.float32, requires_grad=True)
        response_mask = torch.tensor([1, 1, 1], dtype=torch.float32)
        data_pack = _make_packed_data_pack(logprobs, ref_logprobs, response_mask)
        loss, _ = self.opd_loss_fn(data_pack, logprobs)
        loss.backward()
        assert ref_logprobs.grad is None, "Teacher logprobs should be detached"

    def test_loss_value_correct(self):
        """Verify the loss value for a simple case.
        KL(student || teacher) = mean(student_logp - teacher_logp)
        For student=[-1, -2], teacher=[-0.5, -1.5]:
            KL = ((-1 - (-0.5)) + (-2 - (-1.5))) / 2 = (-0.5 + -0.5) / 2 = -0.5
        """
        logprobs = torch.tensor([-1.0, -2.0], dtype=torch.float32)
        ref_logprobs = torch.tensor([-0.5, -1.5], dtype=torch.float32)
        response_mask = torch.tensor([1, 1], dtype=torch.float32)
        data_pack = _make_packed_data_pack(logprobs, ref_logprobs, response_mask)
        loss, stats = self.opd_loss_fn(data_pack, logprobs)
        expected = -0.5
        assert torch.isclose(loss, torch.tensor(expected), atol=1e-6), f"Expected {expected}, got {loss}"

    # ------------------------------------------------------------------
    # Packed vs padded layout
    # ------------------------------------------------------------------

    def test_packed_and_padded_agree(self):
        """Packed and padded layouts should produce the same loss for the
        same data with a single sequence."""
        logprobs = torch.tensor([-1.0, -2.0, -3.0, -4.0], dtype=torch.float32)
        ref_logprobs = torch.tensor([-0.5, -1.5, -2.5, -3.5], dtype=torch.float32)
        # Packed: 1 prompt token, 3 response tokens.
        packed_mask = torch.tensor([0, 1, 1, 1], dtype=torch.float32)
        packed = _make_packed_data_pack(logprobs, ref_logprobs, packed_mask)
        loss_packed, _ = self.opd_loss_fn(packed, logprobs)
        # Padded: shape (1, 4), prompt_mask = [True, False, False, False].
        padded_logprobs = logprobs.unsqueeze(0)
        padded_ref = ref_logprobs.unsqueeze(0)
        prompt_mask = torch.tensor([[True, False, False, False]])
        padded = _make_padded_data_pack(padded_logprobs, padded_ref, prompt_mask)
        loss_padded, _ = self.opd_loss_fn(padded, padded_logprobs)
        assert torch.isclose(loss_packed, loss_padded, atol=1e-6), (
            f"Packed loss {loss_packed} != padded loss {loss_padded}"
        )

    # ------------------------------------------------------------------
    # Temperature scaling
    # ------------------------------------------------------------------

    def test_temperature_scaling(self):
        """Higher temperature should reduce the loss magnitude."""
        logprobs = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32)
        ref_logprobs = torch.tensor([-0.5, -1.5, -2.5], dtype=torch.float32)
        response_mask = torch.tensor([1, 1, 1], dtype=torch.float32)
        data_pack = _make_packed_data_pack(logprobs, ref_logprobs, response_mask)
        loss_t1, _ = self.opd_loss_fn(data_pack, logprobs, temperature=1.0)
        loss_t2, _ = self.opd_loss_fn(data_pack, logprobs, temperature=2.0)
        # Loss at temp=2.0 should be half of loss at temp=1.0 (since both
        # student and teacher logprobs are divided by temperature).
        assert torch.isclose(loss_t2, loss_t1 / 2.0, atol=1e-6), (
            f"Expected {loss_t1 / 2.0}, got {loss_t2}"
        )

    # ------------------------------------------------------------------
    # KL coefficient
    # ------------------------------------------------------------------

    def test_kl_coef_scaling(self):
        """The KL coefficient should linearly scale the loss."""
        logprobs = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32)
        ref_logprobs = torch.tensor([-0.5, -1.5, -2.5], dtype=torch.float32)
        response_mask = torch.tensor([1, 1, 1], dtype=torch.float32)
        data_pack = _make_packed_data_pack(logprobs, ref_logprobs, response_mask)
        loss_1, _ = self.opd_loss_fn(data_pack, logprobs, kl_coef=1.0)
        loss_2, _ = self.opd_loss_fn(data_pack, logprobs, kl_coef=2.0)
        assert torch.isclose(loss_2, loss_1 * 2.0, atol=1e-6), (
            f"Expected {loss_1 * 2.0}, got {loss_2}"
        )

    # ------------------------------------------------------------------
    # Response mask filtering
    # ------------------------------------------------------------------

    def test_prompt_tokens_masked(self):
        """Prompt tokens (response_mask=0) should not contribute to loss."""
        logprobs = torch.tensor([-10.0, -1.0, -2.0], dtype=torch.float32)
        ref_logprobs = torch.tensor([-10.0, -0.5, -1.5], dtype=torch.float32)
        # Only the last 2 tokens are response tokens.
        response_mask = torch.tensor([0, 1, 1], dtype=torch.float32)
        data_pack = _make_packed_data_pack(logprobs, ref_logprobs, response_mask)
        loss, stats = self.opd_loss_fn(data_pack, logprobs)
        # Expected: mean over tokens 1,2: ((-1 - (-0.5)) + (-2 - (-1.5))) / 2 = -0.5
        expected = -0.5
        assert torch.isclose(loss, torch.tensor(expected), atol=1e-6), f"Expected {expected}, got {loss}"

    def test_all_prompt_tokens(self):
        """When all tokens are prompt, the loss should be zero (no valid tokens)."""
        logprobs = torch.tensor([-1.0, -2.0], dtype=torch.float32)
        ref_logprobs = torch.tensor([-0.5, -1.5], dtype=torch.float32)
        response_mask = torch.tensor([0, 0], dtype=torch.float32)
        data_pack = _make_packed_data_pack(logprobs, ref_logprobs, response_mask)
        loss, stats = self.opd_loss_fn(data_pack, logprobs)
        # With no valid response tokens, the loss should be 0 (masked_mean
        # divides by valid_count.clamp(min=1) but the masked sum is 0).
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6), f"Expected 0, got {loss}"

    # ------------------------------------------------------------------
    # Diagnostic stats
    # ------------------------------------------------------------------

    def test_stats_contain_expected_keys(self):
        """The stats dict should contain all expected diagnostic keys."""
        logprobs = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32)
        ref_logprobs = torch.tensor([-0.5, -1.5, -2.5], dtype=torch.float32)
        response_mask = torch.tensor([1, 1, 1], dtype=torch.float32)
        data_pack = _make_packed_data_pack(logprobs, ref_logprobs, response_mask)
        loss, stats = self.opd_loss_fn(data_pack, logprobs)
        expected_keys = {
            "opd_loss", "opd_kl", "opd_student_logprob_mean",
            "opd_teacher_logprob_mean", "opd_kl_coef", "opd_temperature",
        }
        assert expected_keys.issubset(stats.keys()), f"Missing keys: {expected_keys - set(stats.keys())}"

    def test_stats_values(self):
        """Verify the stats values for a known case."""
        logprobs = torch.tensor([-1.0, -2.0], dtype=torch.float32)
        ref_logprobs = torch.tensor([-0.5, -1.5], dtype=torch.float32)
        response_mask = torch.tensor([1, 1], dtype=torch.float32)
        data_pack = _make_packed_data_pack(logprobs, ref_logprobs, response_mask)
        loss, stats = self.opd_loss_fn(data_pack, logprobs)
        assert torch.isclose(stats["opd_kl"], torch.tensor(-0.5), atol=1e-6)
        assert torch.isclose(stats["opd_student_logprob_mean"], torch.tensor(-1.5), atol=1e-6)
        assert torch.isclose(stats["opd_teacher_logprob_mean"], torch.tensor(-1.0), atol=1e-6)
        assert torch.isclose(stats["opd_kl_coef"], torch.tensor(1.0), atol=1e-6)
        assert torch.isclose(stats["opd_temperature"], torch.tensor(1.0), atol=1e-6)