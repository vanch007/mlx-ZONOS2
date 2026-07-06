from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

_SPEAKING_RATE_FPS = 86.0 * (44070.0 / 44000.0)
_DEFAULT_SPEAKING_RATE_BYTES_PER_SECOND = 15.0
_SPEAKING_RATE_CLOSED_BUCKET_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$")
_SPEAKING_RATE_OPEN_BUCKET_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*\+\s*$")
_QUALITY_NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_QUALITY_EXACT_BUCKET_RE = re.compile(rf"^\s*({_QUALITY_NUMBER_RE})\s*$")
_QUALITY_CLOSED_BUCKET_RE = re.compile(
    rf"^\s*({_QUALITY_NUMBER_RE})\s*-\s*({_QUALITY_NUMBER_RE})\s*$"
)
_QUALITY_OPEN_BUCKET_RE = re.compile(rf"^\s*({_QUALITY_NUMBER_RE})\s*\+\s*$")
_QUALITY_METRIC_FIELDS = (
    "lufs",
    "estimated_snr",
    "max_pause",
    "estimated_bandlimit_hz",
    "leading_silence_s",
    "trailing_silence_s",
)
_DEFAULT_QUALITY_BUCKETS = {"trailing_silence_s": 3}


def model_speaking_rate_num_buckets(model_config: dict[str, Any]) -> int:
    return int(model_config.get("speaking_rate_num_buckets", 0) or 0)


def model_speaking_rate_buckets(model_config: dict[str, Any]) -> list[str]:
    raw = model_config.get("speaking_rate_buckets") or ()
    return [str(item) for item in raw]


def model_quality_num_buckets(model_config: dict[str, Any]) -> int:
    return int(model_config.get("quality_num_buckets", 0) or 0)


def model_quality_features(model_config: dict[str, Any]) -> list[str]:
    raw = model_config.get("quality_features")
    if raw:
        return [str(item) for item in raw]
    buckets = model_config.get("quality_buckets") or {}
    if buckets:
        return [str(feature) for feature in buckets]
    return list(_QUALITY_METRIC_FIELDS)


def model_quality_bucket_specs(model_config: dict[str, Any]) -> dict[str, list[str]]:
    raw = model_config.get("quality_buckets") or {}
    return {str(feature): [str(spec) for spec in specs] for feature, specs in raw.items()}


def model_tts_max_tokens(model_config: dict[str, Any], server_default: int) -> int:
    model_max = int(model_config.get("max_seqlen", server_default) or server_default)
    return max(1, model_max)


def resolve_tts_max_tokens(
    model_config: dict[str, Any], server_default: int, requested: int | None
) -> int:
    model_max = model_tts_max_tokens(model_config, server_default)
    if requested is None:
        return model_max
    requested = int(requested)
    if requested <= 0:
        raise ValueError("max_tokens must be positive.")
    return min(requested, model_max)


def _parse_speaking_rate_bucket(spec: str) -> tuple[float, float | None]:
    closed = _SPEAKING_RATE_CLOSED_BUCKET_RE.match(str(spec))
    if closed is not None:
        return float(closed.group(1)), float(closed.group(2))
    open_ended = _SPEAKING_RATE_OPEN_BUCKET_RE.match(str(spec))
    if open_ended is not None:
        return float(open_ended.group(1)), None
    raise ValueError(f"Invalid speaking-rate bucket {spec!r}; expected ranges like '0-3' or '60+'.")


def _speaking_rate_bucket_ranges(model_config: dict[str, Any]) -> list[tuple[float, float | None]]:
    ranges = [_parse_speaking_rate_bucket(spec) for spec in model_speaking_rate_buckets(model_config)]
    if not ranges:
        return ranges
    first_low, _ = ranges[0]
    if not math.isclose(first_low, 0.0, abs_tol=1e-9):
        raise ValueError("speaking-rate buckets must start at 0.")
    previous_high: float | None = None
    for idx, (low, high) in enumerate(ranges):
        if low < 0.0:
            raise ValueError("speaking-rate buckets must use non-negative ranges.")
        if high is not None and high <= low:
            raise ValueError(f"speaking-rate bucket {idx} has an empty or inverted range.")
        if previous_high is None and idx > 0:
            raise ValueError("speaking-rate buckets cannot define ranges after an open-ended bucket.")
        if previous_high is not None and not math.isclose(low, previous_high, abs_tol=1e-9):
            raise ValueError("speaking-rate buckets must be contiguous and ordered.")
        previous_high = high
    if ranges[-1][1] is not None:
        raise ValueError("speaking-rate buckets must end with an open-ended range like '60+'.")
    return ranges


def _speaking_rate_bucket_for_rate(
    rate_bytes_per_second: float,
    *,
    num_buckets: int,
    ranges: list[tuple[float, float | None]],
) -> int:
    if rate_bytes_per_second <= 0:
        raise ValueError("speaking_rate must be positive.")
    if ranges:
        for idx, (_, high) in enumerate(ranges):
            if high is None or (
                rate_bytes_per_second < high
                and not math.isclose(rate_bytes_per_second, high, rel_tol=1e-12, abs_tol=1e-9)
            ):
                return idx
        return len(ranges) - 1
    rate_bytes_per_frame = rate_bytes_per_second / _SPEAKING_RATE_FPS
    bucket = int(rate_bytes_per_frame * num_buckets)
    return min(max(bucket, 0), num_buckets - 1)


def _neutral_speaking_rate_bytes_per_second(
    ranges: list[tuple[float, float | None]],
) -> float:
    if not ranges:
        return _DEFAULT_SPEAKING_RATE_BYTES_PER_SECOND
    low, high = ranges[len(ranges) // 2]
    if high is None:
        return max(low, _DEFAULT_SPEAKING_RATE_BYTES_PER_SECOND)
    return (low + high) / 2.0


def resolve_speaking_rate_bucket(
    model_config: dict[str, Any],
    *,
    speaking_rate_bucket: int | None = None,
    speaking_rate: float | None = None,
    speed: float | None = None,
    speaking_rate_enabled: bool = False,
) -> int | None:
    if not speaking_rate_enabled:
        return None
    supplied = [
        speaking_rate_bucket is not None,
        speaking_rate is not None,
        speed is not None,
    ]
    if sum(supplied) == 0:
        return None
    if sum(supplied) > 1:
        raise ValueError("Provide only one of speaking_rate_bucket, speaking_rate, or speed.")

    num_buckets = model_speaking_rate_num_buckets(model_config)
    if num_buckets <= 0:
        if speed is not None and speaking_rate_bucket is None and speaking_rate is None:
            return None
        raise ValueError("Current model does not support speaking-rate conditioning.")

    if speaking_rate_bucket is not None:
        bucket = int(speaking_rate_bucket)
        if bucket < 0 or bucket >= num_buckets:
            raise ValueError(
                f"speaking_rate_bucket must be in [0, {num_buckets - 1}], got {bucket}."
            )
        return bucket

    ranges = _speaking_rate_bucket_ranges(model_config)
    if ranges and len(ranges) != num_buckets:
        raise ValueError(
            f"Model has {num_buckets} speaking-rate buckets, but config defines {len(ranges)} ranges."
        )

    if speaking_rate is not None:
        return _speaking_rate_bucket_for_rate(
            float(speaking_rate),
            num_buckets=num_buckets,
            ranges=ranges,
        )

    assert speed is not None
    speed_value = float(speed)
    if speed_value <= 0:
        raise ValueError("speed must be positive.")
    return _speaking_rate_bucket_for_rate(
        _neutral_speaking_rate_bytes_per_second(ranges) * speed_value,
        num_buckets=num_buckets,
        ranges=ranges,
    )


def _parse_quality_bucket(spec: str) -> tuple[float, float | None]:
    exact = _QUALITY_EXACT_BUCKET_RE.match(str(spec))
    if exact is not None:
        value = float(exact.group(1))
        return value, value
    closed = _QUALITY_CLOSED_BUCKET_RE.match(str(spec))
    if closed is not None:
        return float(closed.group(1)), float(closed.group(2))
    open_ended = _QUALITY_OPEN_BUCKET_RE.match(str(spec))
    if open_ended is not None:
        return float(open_ended.group(1)), None
    raise ValueError(f"Invalid quality bucket {spec!r}.")


def _quality_bucket_for_value(value: float, specs: list[str]) -> int:
    for idx, spec in enumerate(specs):
        low, high = _parse_quality_bucket(spec)
        if high is None:
            if value >= low or math.isclose(value, low, rel_tol=1e-12, abs_tol=1e-9):
                return idx
        elif (value > low or math.isclose(value, low, rel_tol=1e-12, abs_tol=1e-9)) and (
            value < high or math.isclose(value, high, rel_tol=1e-12, abs_tol=1e-9)
        ):
            return idx
    return len(specs) - 1


def _normalize_quality_bucket_map(
    quality_buckets: dict[str, int | None] | list[int | None] | None,
    features: list[str],
) -> dict[str, int | None]:
    if quality_buckets is None:
        return dict(_DEFAULT_QUALITY_BUCKETS)
    if isinstance(quality_buckets, Mapping):
        return {str(key): (None if value is None else int(value)) for key, value in quality_buckets.items()}
    if isinstance(quality_buckets, list):
        if len(quality_buckets) != len(features):
            raise ValueError(
                f"quality_buckets list length {len(quality_buckets)} "
                f"does not match feature count {len(features)}."
            )
        return {
            feature: (None if value is None else int(value))
            for feature, value in zip(features, quality_buckets, strict=True)
        }
    raise ValueError("quality_buckets must be a dict or list.")


def resolve_quality_buckets(
    model_config: dict[str, Any],
    *,
    quality_buckets: dict[str, int | None] | list[int | None] | None,
    quality_values: dict[str, float | None] | list[float | None] | None,
    quality_enabled: bool,
) -> dict[str, int | None] | None:
    if not quality_enabled:
        return None
    num_buckets = model_quality_num_buckets(model_config)
    if num_buckets <= 0:
        if quality_buckets is None and quality_values is None:
            return None
        raise ValueError("Current model does not support quality conditioning.")

    features = model_quality_features(model_config)
    specs = model_quality_bucket_specs(model_config)

    if quality_values is not None:
        if isinstance(quality_values, Mapping):
            values = {str(k): v for k, v in quality_values.items()}
        elif isinstance(quality_values, list):
            if len(quality_values) != len(features):
                raise ValueError("quality_values list length does not match feature count.")
            values = dict(zip(features, quality_values, strict=True))
        else:
            raise ValueError("quality_values must be a dict or list.")
        resolved: dict[str, int | None] = {}
        for feature in features:
            value = values.get(feature)
            if value is None:
                resolved[feature] = None
                continue
            feature_specs = specs.get(feature)
            if not feature_specs:
                raise ValueError(f"No quality bucket specs configured for feature {feature!r}.")
            bucket = _quality_bucket_for_value(float(value), feature_specs)
            if bucket < 0 or bucket >= num_buckets:
                raise ValueError(f"Derived quality bucket for {feature!r} is out of range.")
            resolved[feature] = bucket
        return resolved

    normalized = _normalize_quality_bucket_map(quality_buckets, features)
    for feature, bucket in normalized.items():
        if bucket is None:
            continue
        if bucket < 0 or bucket >= num_buckets:
            raise ValueError(
                f"quality_buckets[{feature!r}] must be in [0, {num_buckets - 1}], got {bucket}."
            )
    return normalized


def normalize_tts_request_language(language: str) -> str:
    # ZONOS2 model supports many languages; only English has text normalization
    # For non-English languages, server should still accept the code
    # Text normalization is only applied for English languages
    normalized = str(language or "").strip().lower().replace("-", "_")
    return normalized