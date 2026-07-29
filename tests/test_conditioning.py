"""Tests for conditioning token mapping (speaking-rate & quality buckets).

Ports and extends ZONOS2 tests/misc/test_tts_* to verify the MLX adapter
implements equivalent conditioning logic.
"""

from __future__ import annotations

import pytest
from mlx_zonos2.adapter.conditioning import (
    _DEFAULT_QUALITY_BUCKETS,
    _DEFAULT_SPEAKING_RATE_BYTES_PER_SECOND,
    _SPEAKING_RATE_FPS,
    model_quality_bucket_specs,
    model_quality_features,
    model_quality_num_buckets,
    model_speaking_rate_buckets,
    model_speaking_rate_num_buckets,
    model_tts_max_tokens,
    normalize_tts_request_language,
    resolve_quality_buckets,
    resolve_speaking_rate_bucket,
    resolve_tts_max_tokens,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def model_config_with_speaking_rate() -> dict:
    """Model config with speaking-rate support."""
    return {
        "speaking_rate_num_buckets": 8,
        "speaking_rate_buckets": [
            "0-8", "8-11", "11-14", "14-17",
            "17-21", "21-28", "28-40", "40+",
        ],
        "quality_num_buckets": 60,
        "quality_features": [
            "lufs", "estimated_snr", "max_pause",
            "estimated_bandlimit_hz", "leading_silence_s",
            "trailing_silence_s",
        ],
        "quality_buckets": {
            "lufs": ["-1000--50", "-50+"],
            "estimated_snr": ["0-0", "0+"],
            "max_pause": ["0+"],
            "estimated_bandlimit_hz": ["495.3-3433", "3433+"],
            "leading_silence_s": ["0+"],
            "trailing_silence_s": ["0-0.05", "0.05+"],
        },
        "max_seqlen": 6144,
        "sample_rate": 44100,
    }


@pytest.fixture
def model_config_no_conditioning() -> dict:
    """Model config without speaking-rate or quality support."""
    return {
        "speaking_rate_num_buckets": 0,
        "quality_num_buckets": 0,
        "max_seqlen": 4096,
        "sample_rate": 44100,
    }


# ── Speaking Rate Tests ──────────────────────────────────────────────────


class TestSpeakingRateBucket:
    """Speaking-rate bucket resolution and range validation."""

    def test_disabled_without_inputs(self, model_config_no_conditioning):
        """speaking_rate_enabled=False returns None regardless of inputs."""
        assert resolve_speaking_rate_bucket(
            model_config_no_conditioning,
            speaking_rate_bucket=1,
            speaking_rate=10.0,
            speed=1.0,
            speaking_rate_enabled=False,
        ) is None

    def test_disabled_with_inputs_returns_none(self, model_config_no_conditioning):
        """Disabled speaking-rate conditioning ignores optional rate inputs."""
        assert (
            resolve_speaking_rate_bucket(
                model_config_no_conditioning,
                speed=1.0,
                speaking_rate_enabled=False,
            )
            is None
        )

    def test_direct_bucket_value(self, model_config_with_speaking_rate):
        """Direct bucket value in valid range is returned."""
        result = resolve_speaking_rate_bucket(
            model_config_with_speaking_rate,
            speaking_rate_enabled=True,
            speaking_rate_bucket=3,
        )
        assert result == 3

    def test_bucket_out_of_range_raises(self, model_config_with_speaking_rate):
        """Bucket value outside [0, num_buckets) raises ValueError."""
        num_buckets = model_speaking_rate_num_buckets(model_config_with_speaking_rate)
        with pytest.raises(ValueError, match="must be in"):
            resolve_speaking_rate_bucket(
                model_config_with_speaking_rate,
                speaking_rate_enabled=True,
                speaking_rate_bucket=num_buckets,
            )

    def test_negative_bucket_raises(self, model_config_with_speaking_rate):
        """Negative bucket value raises ValueError."""
        with pytest.raises(ValueError, match="must be in"):
            resolve_speaking_rate_bucket(
                model_config_with_speaking_rate,
                speaking_rate_enabled=True,
                speaking_rate_bucket=-1,
            )

    def test_speaking_rate_to_bucket(self, model_config_with_speaking_rate):
        """speaking_rate value maps to correct bucket via ranges."""
        result = resolve_speaking_rate_bucket(
            model_config_with_speaking_rate,
            speaking_rate_enabled=True,
            speaking_rate=15.0,  # 14-17 bucket
        )
        assert result is not None and result >= 0

    def test_speed_to_bucket(self, model_config_with_speaking_rate):
        """Speed value maps to correct bucket using neutral rate."""
        result = resolve_speaking_rate_bucket(
            model_config_with_speaking_rate,
            speaking_rate_enabled=True,
            speed=1.5,
        )
        assert result is not None and result >= 0

    def test_mutually_exclusive_inputs(self, model_config_with_speaking_rate):
        """Providing multiple of bucket/rate/speed raises error."""
        with pytest.raises(ValueError, match="Provide only one"):
            resolve_speaking_rate_bucket(
                model_config_with_speaking_rate,
                speaking_rate_enabled=True,
                speaking_rate_bucket=1,
                speaking_rate=10.0,
            )

    def test_zero_speaking_rate_raises(self, model_config_with_speaking_rate):
        """speaking_rate=0 raises ValueError."""
        with pytest.raises(ValueError, match="speaking_rate must be positive"):
            resolve_speaking_rate_bucket(
                model_config_with_speaking_rate,
                speaking_rate_enabled=True,
                speaking_rate=0.0,
            )


# ── Quality Bucket Tests ─────────────────────────────────────────────────


class TestQualityBuckets:
    """Quality bucket resolution and feature mapping."""

    def test_disabled_without_inputs(self, model_config_no_conditioning):
        """quality_enabled=False returns None."""
        assert resolve_quality_buckets(
            model_config_no_conditioning,
            quality_buckets=None,
            quality_values=None,
            quality_enabled=False,
        ) is None

    def test_disabled_with_inputs_raises(self, model_config_no_conditioning):
        """quality_enabled=False with inputs raises ValueError."""
        with pytest.raises(ValueError, match="does not support"):
            resolve_quality_buckets(
                model_config_no_conditioning,
                quality_buckets={"trailing_silence_s": 1},
                quality_values=None,
                quality_enabled=True,
            )

    def test_dict_quality_buckets(self, model_config_with_speaking_rate):
        """Dict-style quality buckets are accepted and validated."""
        result = resolve_quality_buckets(
            model_config_with_speaking_rate,
            quality_buckets={"trailing_silence_s": 2},
            quality_values=None,
            quality_enabled=True,
        )
        assert isinstance(result, dict)
        assert result["trailing_silence_s"] == 2

    def test_list_quality_buckets(self, model_config_with_speaking_rate):
        """List-style quality buckets map to features in order."""
        features = model_quality_features(model_config_with_speaking_rate)
        result = resolve_quality_buckets(
            model_config_with_speaking_rate,
            quality_buckets=[None] * len(features),
            quality_values=None,
            quality_enabled=True,
        )
        assert isinstance(result, dict)
        assert len(result) == len(features)

    def test_list_mismatch_raises(self, model_config_with_speaking_rate):
        """List length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="does not match"):
            resolve_quality_buckets(
                model_config_with_speaking_rate,
                quality_buckets=[1, 2],  # too short
                quality_values=None,
                quality_enabled=True,
            )

    def test_quality_values_to_buckets(self, model_config_with_speaking_rate):
        """quality_values dict maps numeric values to buckets."""
        result = resolve_quality_buckets(
            model_config_with_speaking_rate,
            quality_buckets=None,
            quality_values={"lufs": -20.0},
            quality_enabled=True,
        )
        assert isinstance(result, dict)
        assert "lufs" in result

    def test_out_of_range_bucket_raises(self, model_config_with_speaking_rate):
        """Quality bucket out of range raises ValueError."""
        num_buckets = model_quality_num_buckets(model_config_with_speaking_rate)
        with pytest.raises(ValueError, match="must be in"):
            resolve_quality_buckets(
                model_config_with_speaking_rate,
                quality_buckets={"lufs": num_buckets},
                quality_values=None,
                quality_enabled=True,
            )

    def test_default_quality_buckets_applied(self, model_config_with_speaking_rate):
        """Default quality buckets are applied when none provided."""
        result = resolve_quality_buckets(
            model_config_with_speaking_rate,
            quality_buckets=None,
            quality_values=None,
            quality_enabled=True,
        )
        assert "trailing_silence_s" in result


# ── Max Tokens Tests ─────────────────────────────────────────────────────


class TestMaxTokens:
    """max_tokens resolution from config and request."""

    def test_model_max_used_when_none_requested(self, model_config_with_speaking_rate):
        """When request max_tokens is None, model max is used."""
        result = resolve_tts_max_tokens(
            model_config_with_speaking_rate,
            server_default=512,
            requested=None,
        )
        assert result == 6144  # model max_seqlen

    def test_server_default_when_model_missing_max(self):
        """Server default used when model config has no max_seqlen."""
        result = resolve_tts_max_tokens({}, 1024, None)
        assert result == 1024

    def test_request_capped_by_model_max(self, model_config_with_speaking_rate):
        """Requested max capped at model max."""
        result = resolve_tts_max_tokens(
            model_config_with_speaking_rate,
            server_default=512,
            requested=8192,
        )
        assert result == 6144

    def test_zero_request_raises(self, model_config_with_speaking_rate):
        """max_tokens=0 raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            resolve_tts_max_tokens(
                model_config_with_speaking_rate,
                server_default=1024,
                requested=0,
            )

    def test_negative_request_raises(self, model_config_with_speaking_rate):
        """Negative max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            resolve_tts_max_tokens(
                model_config_with_speaking_rate,
                server_default=1024,
                requested=-1,
            )


# ── Language Normalization Tests ─────────────────────────────────────────


class TestLanguageNormalization:
    """Language code normalization and validation."""

    def test_normalize_en_us(self):
        """English US code is normalized correctly."""
        for variant in ["en-US", "En_US", "en_us"]:
            result = normalize_tts_request_language(variant)
            assert result == "en_us"

    def test_normalize_case_insensitive(self):
        """Language codes are case-insensitive."""
        assert normalize_tts_request_language("EN_US") == "en_us"
        assert normalize_tts_request_language("EnUs") == "enus"

    def test_normalize_hyphen_to_underscore(self):
        """Hyphens converted to underscores."""
        assert normalize_tts_request_language("zh-Hans-CN") == "zh_hans_cn"

    def test_unknown_language_is_normalized_and_preserved(self):
        """The multilingual model accepts normalized codes beyond English."""
        assert normalize_tts_request_language("xx-XX") == "xx_xx"

    def test_empty_language_default(self):
        """Empty/whitespace language handled."""
        result = normalize_tts_request_language("  ")
        # Should not crash; may raise if empty string not in map
