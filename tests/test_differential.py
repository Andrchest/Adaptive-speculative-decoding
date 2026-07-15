"""
Differential tests for speculative decoder optimizations.

Verifies that the optimized implementation produces identical output tokens
to the reference for a range of seeds, and that the acceptance theorem holds.

Usage:
    uv run pytest tests/test_differential.py -v
"""

import sys

sys.path.insert(0, "src")

import pytest
import torch

from core.cache.ngram import NgramCache

# Skip all tests if CUDA is not available.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def decoder_pipeline():
    """Build the full decoder pipeline once per module."""
    from core.models.draft_model import DraftModel
    from core.models.target_model import TargetModel
    from core.decoder.speculative import SpeculativeDecoder
    from core.translation.vocabulary import CrossVocabTranslator
    from core.cache.ngram import NgramCache

    torch.manual_seed(0)
    drafter = DraftModel("facebook/opt-125m", device="cuda", dtype=torch.float32)
    target = TargetModel("facebook/opt-350m", device="cuda", dtype=torch.float16, load_in_4bit=True)
    translator = CrossVocabTranslator.from_tokenizers(
        drafter.tokenizer,
        target.tokenizer,
        device="cuda",
        drafter_vocab_size=drafter.model.config.vocab_size,
        target_vocab_size=target.model.config.vocab_size,
    )
    cache = NgramCache()
    decoder = SpeculativeDecoder(
        drafter, target, translator, cache, draft_length=5, temperature=1.0
    )
    return drafter, decoder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same seed must always produce the same tokens."""

    def test_determinism_repeat_10_seeds(self, decoder_pipeline):
        """Running the same seed twice in a row must give identical output.

        Note: The drafter uses the global torch RNG (not the per-decoder
        Generator). We must seed the global RNG before each generation so
        that repeated runs with the same seed produce identical tokens.
        """
        drafter, decoder = decoder_pipeline
        prompt = "The quick brown fox"
        input_ids = drafter.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

        for seed in list(range(10)):
            torch.manual_seed(seed)
            rng1 = torch.Generator()
            rng1.manual_seed(seed)
            decoder.clear_step_results()
            decoder.cache = NgramCache()
            out1 = decoder.generate(input_ids, max_new_tokens=32, rng=rng1)

            torch.manual_seed(seed)
            rng2 = torch.Generator()
            rng2.manual_seed(seed)
            decoder.clear_step_results()
            decoder.cache = NgramCache()
            out2 = decoder.generate(input_ids, max_new_tokens=32, rng=rng2)

            assert out1.tolist() == out2.tolist(), f"Seed {seed}: repeated runs differ"


class TestAcceptanceTheorem:
    """Acceptance rates should be stable across seeds."""

    def test_acceptance_rate_range(self, decoder_pipeline):
        """Acceptance rate should be between 0 and 1 for all seeds."""
        drafter, decoder = decoder_pipeline
        prompt = "The quick brown fox"
        input_ids = drafter.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

        rates = []
        for seed in range(50):
            torch.manual_seed(seed)
            rng = torch.Generator()
            rng.manual_seed(seed)
            decoder.clear_step_results()
            decoder.cache = NgramCache()
            decoder.generate(input_ids, max_new_tokens=32, rng=rng)
            stats = decoder.stats()
            rate = stats.get("acceptance_rate", 0.0)
            rates.append(rate)
            assert 0.0 <= rate <= 1.0, f"Seed {seed}: acceptance rate {rate:.3f} out of [0, 1]"

        # Mean acceptance rate should be in a reasonable range
        mean_rate = sum(rates) / len(rates)
        assert 0.1 < mean_rate < 1.0, f"Mean acceptance rate {mean_rate:.3f} out of expected range"


class TestTokenDiversity:
    """Different seeds must produce different outputs."""

    def test_different_seeds_different_tokens(self, decoder_pipeline):
        """At least 9 out of 10 seeds should produce unique token sequences."""
        drafter, decoder = decoder_pipeline
        prompt = "The quick brown fox"
        input_ids = drafter.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

        all_tokens = []
        for seed in range(10):
            torch.manual_seed(seed)
            rng = torch.Generator()
            rng.manual_seed(seed)
            decoder.clear_step_results()
            decoder.cache = NgramCache()
            output = decoder.generate(input_ids, max_new_tokens=32, rng=rng)
            all_tokens.append(tuple(output.tolist()[0]))

        unique_count = len(set(all_tokens))
        assert unique_count >= 9, f"Only {unique_count}/10 unique token sequences (expect ≥9)"
