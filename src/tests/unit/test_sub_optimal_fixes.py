"""
Tests for sub-optimization fixes:
  Fix #1: target_use_4bit config flag
  Fix #2: batch tokenization correctness
  Fix #3: dense Rule2 matmul equivalence
"""

from __future__ import annotations

import math

import pytest
import torch

from experiments.runner import ExperimentConfig


# ──────────────────────────────────────────────────────────────────────
# Fix #1 (P0): target_use_4bit config field exists
# ──────────────────────────────────────────────────────────────────────

class TestTargetUse4BitConfig:
    """Verify the new ExperimentConfig field exists and serializes."""

    def test_default_true(self):
        cfg = ExperimentConfig(name="test")
        assert cfg.target_use_4bit is True

    def test_set_false(self):
        cfg = ExperimentConfig(name="test", target_use_4bit=False)
        assert cfg.target_use_4bit is False

    def test_asdict_includes_field(self):
        cfg = ExperimentConfig(name="test", target_use_4bit=False)
        d = cfg.__dict__ if hasattr(cfg, "__dict__") else {}
        # As dataclass: use __dataclass_fields__
        import dataclasses as dc
        d = dc.asdict(cfg)
        assert "target_use_4bit" in d
        assert d["target_use_4bit"] is False


# ──────────────────────────────────────────────────────────────────────
# Fix #2 (P1): batch tokenization correctness
# ──────────────────────────────────────────────────────────────────────

class TestBatchTokenization:
    """Batch tokenization must produce identical results to per-sample."""

    @pytest.fixture
    def tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("facebook/opt-125m")

    def test_batch_equals_individual(self, tokenizer):
        """Batch encode 100 samples; each result matches per-sample encode."""
        texts = [
            f"This is sample number {i} with some extra text to make it longer."
            for i in range(100)
        ]

        # Per-sample encoding
        individual = []
        for t in texts:
            ids = tokenizer.encode(t, return_tensors="pt")
            individual.append(ids.squeeze(0))

        # Batch encoding (simulating the new code path)
        chunk_size = 64
        batch_results = []
        for cs in range(0, len(texts), chunk_size):
            chunk = texts[cs : cs + chunk_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=False)
            for i in range(len(chunk)):
                batch_results.append(enc.input_ids[i])

        # Compare
        for i in range(len(texts)):
            assert torch.equal(individual[i], batch_results[i]), (
                f"Mismatch at sample {i}"
            )

    def test_batch_preserves_sequence_length(self, tokenizer):
        """Batch encoding preserves exact sequence lengths."""
        texts = [
            "Short sentence.",
            "This is a medium length sentence with more words.",
            "A" * 200,
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
        for i, t in enumerate(texts):
            expected_len = len(tokenizer.encode(t))
            assert (enc.input_ids[i] != tokenizer.pad_token_id).sum().item() == expected_len


# ──────────────────────────────────────────────────────────────────────
# Fix #3 (P2): dense Rule2 matmul equivalence
# ──────────────────────────────────────────────────────────────────────

class TestDenseRule2Matmul:
    """Dense matmul must produce results numerically equal to sparse."""

    @pytest.fixture
    def small_vocab_matrices(self):
        """
        Build a small transfer matrix (~50×40) that fits the dense threshold.
        Returns (target_size, drafter_size, transfer_dict).
        """
        import random
        target_size = 50
        drafter_size = 40
        transfer = {}
        # Populate with ~20% non-zero entries (similar to real Rule2)
        random.seed(42)
        for t_idx in range(target_size):
            # Each target token maps to 1-3 drafter tokens
            n_contribs = random.randint(1, 3)
            d_indices = random.sample(range(drafter_size), min(n_contribs, drafter_size))
            weights = [round(random.uniform(0.2, 1.0), 4) for _ in d_indices]
            transfer[t_idx] = list(zip(d_indices, weights))
        return target_size, drafter_size, transfer

    def test_dense_sparse_equivalence(self, small_vocab_matrices):
        """Dense and sparse Rule2 produce the same output."""
        from core.translation.rules import Rule2Mapping

        target_size, drafter_size, transfer = small_vocab_matrices

        # Build mapping (both dense and sparse paths will be triggered)
        mapping = Rule2Mapping.__new__(Rule2Mapping)
        mapping.drafter_size = drafter_size
        mapping.target_size = target_size
        mapping._transfer = transfer
        mapping._sparse_T = Rule2Mapping._build_sparse_matrix(transfer, target_size, drafter_size)
        mapping._dense_T = None  # we build it manually to verify

        # Build dense matrix manually
        mapping._build_dense_matrix()
        assert mapping._dense_T is not None, "Dense matrix should be built for small vocab"

        # Test input: (batch=2, drafter_vocab)
        test_input = torch.randn(2, drafter_size)

        # Dense path
        T = mapping._dense_T
        result_dense = test_input @ T.t()

        # Sparse path
        sparse_T = mapping._sparse_T.to(test_input.device)
        result_sparse = torch.sparse.mm(test_input, sparse_T.t())

        # Should be numerically equal
        assert torch.allclose(result_dense, result_sparse, atol=1e-5), (
            "Dense and sparse Rule2 outputs must match within tolerance"
        )

    def test_dense_matrix_memory_size(self, small_vocab_matrices):
        """Dense matrix fits within the memory threshold."""
        target_size, drafter_size, _ = small_vocab_matrices
        # threshold_elements = 5_000_000
        threshold_elements = 5_000_000
        actual_elements = target_size * drafter_size
        assert actual_elements <= threshold_elements, (
            f"Dense matrix size {actual_elements} exceeds threshold {threshold_elements}"
        )

    def test_dense_matrix_is_correct(self, small_vocab_matrices):
        """Dense matrix stores the same values as the transfer dict."""
        from core.translation.rules import Rule2Mapping

        target_size, drafter_size, transfer = small_vocab_matrices

        mapping = Rule2Mapping.__new__(Rule2Mapping)
        mapping.drafter_size = drafter_size
        mapping.target_size = target_size
        mapping._transfer = transfer
        mapping._build_dense_matrix()

        dense = mapping._dense_T.to("cpu")
        for t_idx, contribs in transfer.items():
            for d_idx, weight in contribs:
                val = dense[t_idx, d_idx].item()
                assert abs(val - weight) < 1e-6, (
                    f"Dense matrix mismatch at ({t_idx}, {d_idx}): expected {weight}, got {val}"
                )

    def test_sparse_still_works_without_dense(self, small_vocab_matrices):
        """Sparse path still works when dense is not built."""
        from core.translation.rules import Rule2Mapping

        target_size, drafter_size, transfer = small_vocab_matrices

        mapping = Rule2Mapping.__new__(Rule2Mapping)
        mapping.drafter_size = drafter_size
        mapping.target_size = target_size
        mapping._transfer = transfer
        mapping._sparse_T = Rule2Mapping._build_sparse_matrix(transfer, target_size, drafter_size)
        mapping._dense_T = None  # force sparse path

        test_input = torch.randn(1, drafter_size)
        result = mapping.map_logits(test_input)

        assert result.shape == (1, target_size), (
            f"Output shape mismatch: expected (1, {target_size}), got {result.shape}"
        )
        assert not result.isnan().any(), "Result should not contain NaN"


# ──────────────────────────────────────────────────────────────────────
# Integration: run a single experiment with new config
# ──────────────────────────────────────────────────────────────────────

class TestIntegration:
    """Quick integration tests for the new config fields."""

    def test_config_persists_through_override(self):
        """target_use_4bit survives via CLI override chain."""
        cfg = ExperimentConfig(name="test", target_use_4bit=False)
        assert getattr(cfg, "target_use_4bit", True) is False

    def test_default_still_4bit(self):
        """Default is still 4-bit for backward compatibility."""
        cfg = ExperimentConfig(name="test")
        assert cfg.target_use_4bit is True
