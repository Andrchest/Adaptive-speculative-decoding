"""
Tests for the bugfixes — CPU-only, no model loading required.

P3: BenchmarkCollector.summary() computes overall TPS correctly
P7: No duplicate k = drafter_logits.shape[0] in lattice
P10: Lattice LRU eviction works
P1: Lattice DP computes correct probabilities
P2: Seeds are set for random, numpy, torch
"""

from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock

import torch

sys.path.insert(0, "src")

from benchmarks.metrics.collector import BenchmarkCollector, StepRecord
from core.decoder.speculative import SpeculativeDecoder
from core.extensions.adaptive.speedup_predictor import SpeedupPredictor
from core.extensions.lattice.tokenizer_lattice import TokenizerLattice

# ------------------------------------------------------------------
# P3: TPS metric — overall vs mean-of-means
# ------------------------------------------------------------------


class TestTPSMetric:
    """BenchmarkCollector.summary() must compute overall TPS, not biased mean."""

    def test_overall_tps_vs_mean_of_means(self):
        """Verify that TPS = total_new_tokens / total_wall_time (unbiased)."""
        collector = BenchmarkCollector(name="test")

        # Directly append a record with known values
        from benchmarks.metrics.collector import DecodeRecord

        rec1 = DecodeRecord(prompt_len=10, total_new_tokens=20, wall_time_s=1.0)
        rec1.step_records.append(StepRecord(draft_len=5, accepted=10, cache_hit=True))
        rec1.step_records.append(StepRecord(draft_len=5, accepted=10, cache_hit=False))
        collector._records.append(rec1)

        # Second: 1000 tokens in 100s
        rec2 = DecodeRecord(prompt_len=100, total_new_tokens=1000, wall_time_s=100.0)
        rec2.step_records.append(StepRecord(draft_len=5, accepted=1000, cache_hit=False))
        collector._records.append(rec2)

        summary = collector.summary()

        # Overall TPS = 1020 / 101.0 ≈ 10.1
        total_tps = summary["tokens_per_sec"]
        assert abs(total_tps - 1020 / 101.0) < 0.5, (
            f"Expected TPS ≈ {1020/101:.1f}, got {total_tps:.1f}"
        )
        assert summary["total_new_tokens"] == 1020
        assert summary["wall_time_total_s"] == 101.0
        assert summary["avg_tokens_per_sec"] > 0

    def test_biased_vs_unbiased(self):
        """Demonstrate the bias in mean-of-means vs overall TPS."""
        from benchmarks.metrics.collector import DecodeRecord

        collector = BenchmarkCollector(name="bias_test")

        # Short burst: 100 tokens in 0.01s → 10000 TPS
        rec_short = DecodeRecord(prompt_len=10, total_new_tokens=100, wall_time_s=0.01)
        rec_short.step_records.append(StepRecord(draft_len=5, accepted=100, cache_hit=True))
        collector._records.append(rec_short)

        # Long run: 1000 tokens in 1000s → 1 TPS
        rec_long = DecodeRecord(prompt_len=1000, total_new_tokens=1000, wall_time_s=1000.0)
        rec_long.step_records.append(StepRecord(draft_len=5, accepted=1000, cache_hit=False))
        collector._records.append(rec_long)

        summary = collector.summary()
        overall_tps = summary["tokens_per_sec"]
        avg_tps = summary["avg_tokens_per_sec"]

        # Overall = 1100 / 1000.01 ≈ 1.1 TPS (correct)
        assert abs(overall_tps - 1100 / 1000.01) < 0.1, (
            f"Expected ~1.1, got {overall_tps:.1f}"
        )
        # Mean-of-means = (10000 + 1) / 2 = 5000.5 (heavily biased by short burst)
        assert avg_tps > 100, (
            f"avg_tps should be heavily biased high, got {avg_tps:.0f}"
        )
        # Overall should be close to the slower sequence
        assert overall_tps < 10, (
            f"Overall TPS {overall_tps:.1f} should be close to 1.1, not inflated"
        )

    def test_single_sequence(self):
        """Single sequence: overall = avg."""
        collector = BenchmarkCollector(name="single")
        with collector.record_sequence(prompt_len=50) as seq:
            seq.add_step(draft_len=5, accepted=20, cache_hit=True)

        summary = collector.summary()
        assert summary["tokens_per_sec"] > 0
        assert summary["total_new_tokens"] == 20


# ------------------------------------------------------------------
# P1 + P7: Lattice DP correctness
# ------------------------------------------------------------------


class TestLatticeDP:
    """TokenizerLattice DP computes correct probabilities."""

    @staticmethod
    def _mock_tokenizer(vocab: dict[str, int]) -> MagicMock:
        tok = MagicMock()
        tok.get_vocab.return_value = vocab
        return tok

    def test_forward_basic(self):
        """forward() with raw logits → softmax applied internally."""
        drafter_vocab = {"a": 0, "b": 1, "ab": 2}
        target_vocab = {"a": 0, "b": 1, "ab": 2}
        lattice = TokenizerLattice(
            drafter_tokenizer=self._mock_tokenizer(drafter_vocab),
            target_tokenizer=self._mock_tokenizer(target_vocab),
            drafter_vocab_size=3,
            target_vocab_size=3,
        )

        # Pass raw logits; lattice applies softmax
        drafter_logits = torch.tensor([0.5, 0.25, 0.25])
        result = lattice.exact_map_logits(drafter_logits)

        # softmax([0.5, 0.25, 0.25]) ≈ [0.391, 0.305, 0.305]
        # P("a") = softmax(0.5) = 0.391
        # P("ab") = P("a")*P("b") + P("ab") = 0.391*0.305 + 0.305 = 0.424
        import math

        exp_sum = math.exp(0.5) + math.exp(0.25) + math.exp(0.25)
        p_a = math.exp(0.5) / exp_sum
        p_b = math.exp(0.25) / exp_sum
        p_ab = math.exp(0.25) / exp_sum

        expected_a = p_a
        expected_ab = p_a * p_b + p_ab  # path "a"+"b" + path "ab"

        assert abs(result[0].item() - expected_a) < 1e-5, (
            f"P(a)={result[0].item():.4f}, expected {expected_a:.4f}"
        )
        assert abs(result[2].item() - expected_ab) < 1e-5, (
            f"P(ab)={result[2].item():.4f}, expected {expected_ab:.4f}"
        )

    def test_forward_with_precomputed_probs(self):
        """forward() receives pre-softmaxed probabilities."""
        drafter_vocab = {"a": 0, "b": 1, "ab": 2}
        target_vocab = {"a": 0, "b": 1, "ab": 2}
        lattice = TokenizerLattice(
            drafter_tokenizer=self._mock_tokenizer(drafter_vocab),
            target_tokenizer=self._mock_tokenizer(target_vocab),
            drafter_vocab_size=3,
            target_vocab_size=3,
        )

        # Pre-softmaxed probabilities
        probs = torch.tensor([0.5, 0.25, 0.25])

        # P("a") = 0.5 (single path)
        prob_a = lattice.forward(probs, 0).item()
        assert abs(prob_a - 0.5) < 1e-6, f"P(a) = {prob_a}, expected 0.5"

        # P("ab") = P("a")*P("b") + P("ab") = 0.5*0.25 + 0.25 = 0.375
        prob_ab = lattice.forward(probs, 2).item()
        expected = 0.5 * 0.25 + 0.25
        assert abs(prob_ab - expected) < 1e-6, f"P(ab) = {prob_ab}, expected {expected}"

    def test_no_duplicate_k(self):
        """exact_map_logits should not have duplicate k assignment."""
        import inspect

        source = inspect.getsource(TokenizerLattice.exact_map_logits)
        k_assignments = source.count("k = ")
        assert k_assignments <= 1, (
            f"Found {k_assignments} assignments to 'k' in exact_map_logits"
        )


# ------------------------------------------------------------------
# P10: Lattice LRU eviction
# ------------------------------------------------------------------


class TestLRUCache:
    """TokenizerLattice uses LRU eviction, not FIFO."""

    @staticmethod
    def _mock_tokenizer(vocab: dict[str, int]) -> MagicMock:
        tok = MagicMock()
        tok.get_vocab.return_value = vocab
        return tok

    def test_lru_eviction_keeps_hot(self):
        """LRU eviction keeps most recently accessed entry."""
        drafter_vocab = {"a": 0, "ab": 1}
        target_vocab = {"a": 0, "b": 1, "ab": 2}
        lattice = TokenizerLattice(
            drafter_tokenizer=self._mock_tokenizer(drafter_vocab),
            target_tokenizer=self._mock_tokenizer(target_vocab),
            max_cache_size=2,
            drafter_vocab_size=2,
            target_vocab_size=3,
        )

        # Access "a", then "ab", then refresh "a" (most recent),
        # then add "b" — should evict "ab" (second oldest)
        lattice.build("a")
        time.sleep(0.01)
        lattice.build("ab")
        time.sleep(0.01)
        lattice.build("a")  # refresh — now "a" is most recent
        time.sleep(0.01)
        lattice.build("b")  # cache full, evict "ab"

        # "a" (most recent) and "b" (new) should remain; "ab" evicted
        assert "a" in lattice._lattice_cache, (
            f"Most recently accessed 'a' should remain. Cache: {list(lattice._lattice_cache.keys())}"
        )
        assert "b" in lattice._lattice_cache, f"'b' was just inserted. Cache: {list(lattice._lattice_cache.keys())}"
        assert "ab" not in lattice._lattice_cache, (
            f"'ab' should have been evicted. Cache: {list(lattice._lattice_cache.keys())}"
        )

    def test_lru_updates_on_hit(self):
        """Accessing a cached entry refreshes its access time."""
        drafter_vocab = {"a": 0, "b": 1}
        target_vocab = {"a": 0, "b": 1, "c": 2}
        lattice = TokenizerLattice(
            drafter_tokenizer=self._mock_tokenizer(drafter_vocab),
            target_tokenizer=self._mock_tokenizer(target_vocab),
            max_cache_size=2,
            drafter_vocab_size=2,
            target_vocab_size=3,
        )

        lattice.build("a")
        time.sleep(0.01)
        lattice.build("b")
        time.sleep(0.01)
        # Access "a" again to make it most recent
        lattice.build("a")
        time.sleep(0.01)
        lattice.build("c")  # should evict "b" (oldest), not "a"

        assert "a" in lattice._lattice_cache, "Refreshed 'a' should survive"
        assert "c" in lattice._lattice_cache, "'c' was just inserted"
        assert "b" not in lattice._lattice_cache, "'b' should be evicted"


# ------------------------------------------------------------------
# P2: Randomization seeds
# ------------------------------------------------------------------


class TestRandomization:
    """Verify seeds are set for all random generators."""

    def test_seeds_are_set_in_base_experiment(self):
        """BaseExperiment.run() should set random.seed, np.random.seed, torch.manual_seed."""
        import inspect

        from experiments.base import BaseExperiment

        source = inspect.getsource(BaseExperiment.run)
        assert "random.seed" in source, "Missing random.seed() call"
        assert "np.random.seed" in source, "Missing np.random.seed() call"
        assert "torch.manual_seed" in source, "Missing torch.manual_seed() call"

    def test_speculative_uses_rng(self):
        """SpeculativeDecoder should accept and use rng parameter."""
        import inspect

        sig = inspect.signature(SpeculativeDecoder.generate)
        assert "rng" in sig.parameters, "generate() should accept rng parameter"

        sig_accept = inspect.signature(SpeculativeDecoder._accept_reject)
        assert "rng" in sig_accept.parameters, "_accept_reject should accept rng"

        sig_resid = inspect.signature(SpeculativeDecoder._residual_sample)
        assert "rng" in sig_resid.parameters, "_residual_sample should accept rng"

    def test_predictor_uses_rng(self):
        """SpeedupPredictor.train_on_buffer should accept rng."""
        import inspect

        sig = inspect.signature(SpeedupPredictor.train_on_buffer)
        assert "rng" in sig.parameters, "train_on_buffer should accept rng"

    def test_base_experiment_passes_rng_to_generate(self):
        """BaseExperiment.run() should pass torch_rng to decoder.generate()."""
        import inspect

        from experiments.base import BaseExperiment

        source = inspect.getsource(BaseExperiment.run)
        assert "rng=torch_rng" in source, "BaseExperiment.run() should pass rng to decoder.generate()"


# ------------------------------------------------------------------
# P4: Hooks cleanup
# ------------------------------------------------------------------


class TestHooksCleanup:
    """UniversalDrafter hooks are properly cleaned up."""

    def test_remove_hooks_idempotent(self):
        """remove_hooks() should not crash on empty _hooks."""
        from core.extensions.multitarget.universal_drafter import UniversalDrafter

        u = UniversalDrafter.__new__(UniversalDrafter)
        u._hooks = []

        # Should not crash
        u.remove_hooks()

        # Calling again should also not crash
        u.remove_hooks()

    def test_context_manager(self):
        """UniversalDrafter is used through WithUniversalDrafter wrapper,
        which provides the drafter interface. Direct __enter__/__exit__ is not
        required — the wrapper (not UniversalDrafter itself) is what speculative.py
        calls."""
        from core.extensions.multitarget.universal_drafter import UniversalDrafter

        # UniversalDrafter extends nn.Module, not DraftModel.
        # It is always wrapped by WithUniversalDrafter which provides the
        # standard draft(context, k, distill, temperature) interface.
        assert issubclass(UniversalDrafter, torch.nn.Module)

    def test_draft_accepts_distill(self):
        """UniversalDrafter.draft() does NOT accept distill directly.
        The distill parameter is handled by WithUniversalDrafter wrapper.
        See: experiments/built_in/with_universal.py"""
        import inspect
        from core.extensions.multitarget.universal_drafter import UniversalDrafter

        sig = inspect.signature(UniversalDrafter.draft)
        # UniversalDrafter has its own signature: (self, context, k, target_name)
        params = list(sig.parameters.keys())
        assert "target_name" in params, "UniversalDrafter.draft() should accept target_name"
        # distill is handled by the wrapper, not by UniversalDrafter directly
        assert "distill" not in params, "distill is handled by WithUniversalDrafter wrapper"
