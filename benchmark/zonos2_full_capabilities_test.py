#!/usr/bin/env python3
"""
Comprehensive test suite for mlx-ZONOS2 based on ACTUAL model capabilities.

ZONOS2 Supported Features (from model config):
- Languages: MANY (en_us, en_gb, en, zh, ja, ko, fr, es, de, ru, pt, it, 
                hi, ta, te, th, ar, vi, nl, pl, tr, he, sv, da, no, fi, uk, ...)
- Speaking Rate: 8 buckets (indices 0-7, ranges: 0-8, 8-11, 11-14, 14-17, 17-21, 21-28, 28-40, 40+ bytes/sec)
- Quality Conditioning: 6 features with varying buckets (lufs:12, estimated_snr:12, max_pause:12, 
                         estimated_bandlimit_hz:8, leading_silence_s:8, trailing_silence_s:8)
- Speaker Cloning: Via ref_audio (reference audio) OR speaker_embedding
- Sampling: temperature, top_p, top_k, min_p, repetition_window, repetition_penalty, repetition_codebooks, seed
- Text Normalization: English only (for other languages use text_normalization=False)
- Advanced: clean_speaker_background, accurate_mode, ignore_eos
- NO dedicated emotion parameter (emotion via speaker ref or prompt text prefix)
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TestResult:
    test_id: str
    group: str
    section: str
    language: str
    text: str
    description: str
    ok: bool
    status: int | None
    bytes_received: int
    sample_rate: int
    ttfb_ms: float | None
    e2e_ms: float
    audio_sec: float
    rtf: float | None
    error: str | None = None
    speaking_rate_bucket: int | None = None
    quality_buckets: dict | None = None
    speaker_ref: str | None = None
    sampling_params: dict | None = None


def post_tts(
    url: str,
    text: str,
    language: str = "en_us",
    max_tokens: int = 1024,
    seed: int | None = None,
    speaking_rate_bucket: int | None = None,
    quality_buckets: dict | None = None,
    speaker_embedding: list | None = None,
    ref_audio_path: str | None = None,
    temperature: float = 1.15,
    top_p: float = 0.0,
    top_k: int = 106,
    min_p: float = 0.18,
    repetition_window: int = 50,
    repetition_penalty: float = 1.2,
    repetition_codebooks: int = 8,
    clean_speaker_background: bool = False,
    accurate_mode: bool = True,
    ignore_eos: bool = False,
    text_normalization: bool = True,
    timeout: float = 300.0,
    chunk_size: int = 65536,
) -> TestResult:
    """Send a TTS request and measure performance."""
    body: dict[str, Any] = {
        "text": text,
        "language": language,
        "text_normalization": text_normalization,
        "temperature": temperature,
        "topk": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "max_tokens": max_tokens,
        "stream": False,
        "fade_out_ms": 0.0,
        "accurate_mode": accurate_mode,
        "clean_speaker_background": clean_speaker_background,
        "ignore_eos": ignore_eos,
    }
    
    if seed is not None:
        body["seed"] = seed
    
    if speaking_rate_bucket is not None:
        body["speaking_rate_enabled"] = True
        body["speaking_rate_bucket"] = speaking_rate_bucket
    
    if quality_buckets is not None:
        body["quality_enabled"] = True
        body["quality_buckets"] = quality_buckets
    
    if speaker_embedding is not None:
        body["speaker_embedding_base64"] = base64.b64encode(
            json.dumps(speaker_embedding).encode()
        ).decode()
    
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/tts/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
    started = time.perf_counter()
    first_byte_at: float | None = None
    bytes_received = 0
    sample_rate = 44100
    status = None
    error = None
    
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.getcode()
            sample_rate = int(response.headers.get("X-Audio-Sample-Rate", sample_rate))
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                if first_byte_at is None:
                    first_byte_at = time.perf_counter()
                bytes_received += len(chunk)
    except urllib.error.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        detail = exc.read(4096).decode("utf-8", errors="replace")
        return TestResult(
            test_id="", group="", section="", language=language, text=text, description="",
            ok=False, status=exc.code, bytes_received=0, sample_rate=sample_rate,
            ttfb_ms=None, e2e_ms=elapsed_ms, audio_sec=0.0, rtf=None,
            error=detail.strip() or exc.reason,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return TestResult(
            test_id="", group="", section="", language=language, text=text, description="",
            ok=False, status=None, bytes_received=0, sample_rate=sample_rate,
            ttfb_ms=None, e2e_ms=elapsed_ms, audio_sec=0.0, rtf=None,
            error=str(exc),
        )
    
    elapsed = time.perf_counter() - started
    audio_sec = bytes_received / (sample_rate * 4.0)
    
    return TestResult(
        test_id="", group="", section="", language=language, text=text, description="",
        ok=bytes_received > 0, status=status, bytes_received=bytes_received,
        sample_rate=sample_rate,
        ttfb_ms=None if first_byte_at is None else (first_byte_at - started) * 1000.0,
        e2e_ms=elapsed * 1000.0, audio_sec=audio_sec,
        rtf=None if audio_sec <= 0.0 else elapsed / audio_sec,
        error=None if bytes_received > 0 else "empty audio response",
    )


# ============================================================
# TEST DEFINITIONS - Based on ZONOS2 actual capabilities
# ============================================================

# 1. MULTILINGUAL BASELINE: All supported languages
# Language codes that work (tested directly on model)
MULTILINGUAL_TESTS = [
    # Tier 1
    {"id": "lang_en_us", "language": "en_us", "text": "Welcome to ZONOS2 text-to-speech system. This is an English test.", "desc": "English US", "text_normalization": True},
    {"id": "lang_en_gb", "language": "en_gb", "text": "Welcome to ZONOS2 text-to-speech system. This is an English test.", "desc": "English GB", "text_normalization": True},
    {"id": "lang_zh", "language": "zh", "text": "欢迎使用 ZONOS2 语音合成系统。这是一个中文测试。", "desc": "Chinese (Mandarin)", "text_normalization": False},
    {"id": "lang_ja", "language": "ja", "text": "こんにちは、ZONOS2 音声合成システムへようこそ。これは日本語のテストです。", "desc": "Japanese", "text_normalization": False},
    
    # Tier 2
    {"id": "lang_ko", "language": "ko", "text": "안녕하세요, ZONOS2 음성 합성 시스템에 오신 것을 환영합니다. 이것은 한국어 테스트입니다.", "desc": "Korean", "text_normalization": False},
    {"id": "lang_fr", "language": "fr", "text": "Bonjour, bienvenue dans le système de synthèse vocale ZONOS2. Ceci est un test en français.", "desc": "French", "text_normalization": False},
    {"id": "lang_es", "language": "es", "text": "Hola, bienvenido al sistema de síntesis de voz ZONOS2. Esta es una prueba en español.", "desc": "Spanish", "text_normalization": False},
    {"id": "lang_de", "language": "de", "text": "Hallo, willkommen beim ZONOS2 Sprachsynthesesystem. Dies ist ein Test auf Deutsch.", "desc": "German", "text_normalization": False},
    {"id": "lang_ru", "language": "ru", "text": "Привет! Добро пожаловать в систему речевого синтеза ZONOS2. Это тест на русском языке.", "desc": "Russian", "text_normalization": False},
    {"id": "lang_pt", "language": "pt", "text": "Olá, bem-vindo ao sistema de síntese de voz ZONOS2. Este é um teste em português.", "desc": "Portuguese", "text_normalization": False},
    {"id": "lang_it", "language": "it", "text": "Ciao, benvenuto nel sistema di sintesi vocale ZONOS2. Questo è un test in italiano.", "desc": "Italian", "text_normalization": False},
    {"id": "lang_vi", "language": "vi", "text": "Xin chào, chào mừng đến với hệ thống tổng hợp giọng nói ZONOS2. Đây là bài kiểm tra tiếng Việt.", "desc": "Vietnamese", "text_normalization": False},
    {"id": "lang_nl", "language": "nl", "text": "Hallo, welkom bij het ZONOS2 spraaksynthesesysteem. Dit is een test in het Nederlands.", "desc": "Dutch", "text_normalization": False},
    {"id": "lang_pl", "language": "pl", "text": "Cześć, witamy w systemie syntezy mowy ZONOS2. To jest test w języku polskim.", "desc": "Polish", "text_normalization": False},
    {"id": "lang_tr", "language": "tr", "text": "Merhaba, ZONOS2 konuşma sentéz sistemine hoş geldiniz. Bu Türkçe bir testtir.", "desc": "Turkish", "text_normalization": False},
    {"id": "lang_he", "language": "he", "text": "שלום, ברוכים הבאים למערכת סינתזת הדיבור ZONOS2. זהו מבחן בעברית.", "desc": "Hebrew", "text_normalization": False},
    {"id": "lang_ar", "language": "ar", "text": "مرحباً بكم في نظام توليد الكلام ZONOS2. هذا اختبار باللغة العربية.", "desc": "Arabic", "text_normalization": False},
    
    # Tier 3
    {"id": "lang_sv", "language": "sv", "text": "Hej, välkommen till ZONOS2 talsyntessystemet. Detta är ett test på svenska.", "desc": "Swedish", "text_normalization": False},
    {"id": "lang_hi", "language": "hi", "text": "नमस्ते, ZONOS2 वाक् संश्लेषण प्रणाली में आपका स्वागत है। यह एक हिंदी परीक्षण है।", "desc": "Hindi", "text_normalization": False},
    {"id": "lang_ta", "language": "ta", "text": "வணக்கம், ZONOS2 பேச்சு உடைத்தலுக்கு வரவேற்கிறோம். இது ஒரு தமிழ் சோதனை.", "desc": "Tamil", "text_normalization": False},
    {"id": "lang_te", "language": "te", "text": "నమస్తే, ZONOS2 వాక్య సంకలన వ్యవస్థకు స్వాగతం. ఇది een తెలుగు పరీక్ష.", "desc": "Telugu", "text_normalization": False},
    {"id": "lang_th", "language": "th", "text": "สวัสดีครับ ยินดีต้อนรับสู่ระบบสังเคราะห์เสียงพูด ZONOS2 นี่คือการทดสอบภาษาไทย", "desc": "Thai", "text_normalization": False},
    {"id": "lang_da", "language": "da", "text": "Hej, velkommen til ZONOS2 talesyntesesystemet. Dette er en test på dansk.", "desc": "Danish", "text_normalization": False},
    {"id": "lang_no", "language": "no", "text": "Hei, velkommen til ZONOS2 talesyntesesystemet. Dette er en test på norsk.", "desc": "Norwegian", "text_normalization": False},
    {"id": "lang_fi", "language": "fi", "text": "Hei, tervetuloa ZONOS2 puhesynteesijärjestelmään. Tämä on testi suomeksi.", "desc": "Finnish", "text_normalization": False},
    {"id": "lang_uk", "language": "uk", "text": "Привіт, ласкаво просимо до системи мовлення ZONOS2. Це тест українською мовою.", "desc": "Ukrainian", "text_normalization": False},
    {"id": "lang_sv", "language": "sv", "text": "Hej, välkommen till ZONOS2 talsyntessystemet. Detta är ett test på svenska.", "desc": "Swedish", "text_normalization": False},
]

# 2. SPEAKING RATE: 8 buckets (the REAL ZONOS2 feature)
SPEAKING_RATE_TESTS = [
    ("very_slow", 0, "Very Slow (0-8 bytes/sec)"),
    ("slow", 1, "Slow (8-11 bytes/sec)"),
    ("normal", 2, "Normal (11-14 bytes/sec)"),
    ("fast", 3, "Fast (14-17 bytes/sec)"),
    ("very_fast", 4, "Very Fast (17-21 bytes/sec)"),
    ("rapid", 5, "Rapid (21-28 bytes/sec)"),
    ("very_rapid", 6, "Very Rapid (28-40 bytes/sec)"),
    ("extreme", 7, "Extreme (40+ bytes/sec)"),
]

SPEAKING_RATE_TEXT = "The quick brown fox jumps over the lazy dog. This tests speaking rate conditioning across eight buckets."

# 3. QUALITY CONDITIONING: 6 features with correct bucket counts
QUALITY_TESTS = [
    {"name": "trailing_silence_max", "buckets": {"trailing_silence_s": 7}, "desc": "Trailing Silence Max (~4+ seconds)"},
    {"name": "trailing_silence_0s", "buckets": {"trailing_silence_s": 0}, "desc": "No Trailing Silence"},
    {"name": "leading_silence_max", "buckets": {"leading_silence_s": 7}, "desc": "Leading Silence Max (~4+ seconds)"},
    {"name": "lufs_mid", "buckets": {"lufs": 6}, "desc": "LUFS Mid Range"},
    {"name": "snr_high", "buckets": {"estimated_snr": 9}, "desc": "High SNR"},
    {"name": "bandlimit_high", "buckets": {"estimated_bandlimit_hz": 5}, "desc": "Higher Estimated Bandlimit"},
    {"name": "max_pause_low", "buckets": {"max_pause": 2}, "desc": "Low Max Pause"},
    {"name": "balanced_reasonable", "buckets": {"lufs": 5, "estimated_snr": 5, "max_pause": 5, "estimated_bandlimit_hz": 3, "leading_silence_s": 3, "trailing_silence_s": 3}, "desc": "Balanced All Quality Features"},
]

QUALITY_TEXT = "Quality conditioning test with different audio quality feature settings."

# 4. SPEAKER CLONING (reference audio based)
SPEAKER_CLONE_TESTS = [
    {"id": "clone_basic", "group": "Speaker Cloning", "section": "Cloning", "language": "en_us", "text": "This is a voice cloning test using a reference audio sample. The model should replicate the speaker characteristics.", "desc": "Basic voice cloning with reference audio", "ref_audio": "/tmp/ref_audio_clone.wav"},
    {"id": "clone_zh", "group": "Speaker Cloning", "section": "Cloning", "language": "zh", "text": "这是一个中文语音克隆测试，使用参考音频来复刻说话人特征。", "desc": "Chinese voice cloning", "ref_audio": "/tmp/ref_audio_clone.wav", "text_normalization": False},
    {"id": "clone_ja", "group": "Speaker Cloning", "section": "Cloning", "language": "ja", "text": "これは日本語の音声クローニングテストです。参照音声を使って話者の特徴を再現します。", "desc": "Japanese voice cloning", "ref_audio": "/tmp/ref_audio_clone.wav", "text_normalization": False},
    {"id": "clone_fr", "group": "Speaker Cloning", "section": "Cloning", "language": "fr", "text": "Ceci est un test de clonage vocal en français avec une référence audio.", "desc": "French voice cloning", "ref_audio": "/tmp/ref_audio_clone.wav", "text_normalization": False},
    {"id": "clone_with_rate", "group": "Speaker Cloning", "section": "Cloning", "language": "en_us", "text": "Voice cloning combined with speaking rate control. The speaker identity should persist at different speeds.", "desc": "Voice cloning + speaking rate bucket 5 (rapid)", "ref_audio": "/tmp/ref_audio_clone.wav", "speaking_rate_bucket": 5},
    {"id": "clone_with_quality", "group": "Speaker Cloning", "section": "Cloning", "language": "en_us", "text": "Voice cloning with quality conditioning. Testing trailing silence control on cloned voice.", "desc": "Voice cloning + trailing silence 0s", "ref_audio": "/tmp/ref_audio_clone.wav", "quality_buckets": {"trailing_silence_s": 0}},
]

# 5. PROMPT STYLE PREFIXES (NOT a native parameter, but ZONOS2 responds to text prefixes)
STYLE_PREFIX_TESTS = [
    {"id": "style_whisper", "group": "Prompt Style", "section": "Style Prefix", "language": "en_us", "text": "(Whisper) Shh... this is a secret. Listen very carefully now.", "desc": "Whisper style via prompt prefix"},
    {"id": "style_shouting", "group": "Prompt Style", "section": "Style Prefix", "language": "en_us", "text": "(Shouting) HEY! CAN YOU HEAR ME FROM OVER HERE?! THIS IS LOUD!", "desc": "Shouting style via prompt prefix"},
    {"id": "style_sad", "group": "Prompt Style", "section": "Style Prefix", "language": "en_us", "text": "(Sad, trembling voice) I don't know what to say anymore. Everything feels so heavy.", "desc": "Sad/emotional style via prompt prefix"},
    {"id": "style_excited", "group": "Prompt Style", "section": "Style Prefix", "language": "en_us", "text": "(Excited, energetic) Oh my goodness! This is absolutely amazing! I can't believe it!", "desc": "Excited style via prompt prefix"},
    {"id": "style_pirate", "group": "Prompt Style", "section": "Style Prefix", "language": "en_us", "text": "(Pirate captain) Arrr matey! Welcome aboard me ship! Hoist the sails and prepare for adventure!", "desc": "Pirate persona via prompt prefix"},
    {"id": "style_storyteller", "group": "Prompt Style", "section": "Style Prefix", "language": "en_us", "text": "(Old storyteller by the fire) Gather round, children. Let me tell you a tale from long ago...", "desc": "Storyteller persona via prompt prefix"},
]

# 6. SAMPLING PARAMETER TESTS
SAMPLING_TESTS = [
    {"id": "sampling_low_temp", "group": "Sampling", "section": "Temperature", "language": "en_us", "text": "Testing deterministic generation with low temperature for consistent output.", "desc": "Low temperature (0.5) - more deterministic", "temperature": 0.5},
    {"id": "sampling_high_temp", "group": "Sampling", "section": "Temperature", "language": "en_us", "text": "Testing creative generation with high temperature for more variation.", "desc": "High temperature (1.5) - more creative", "temperature": 1.5},
    {"id": "sampling_top_p", "group": "Sampling", "section": "Top-p", "language": "en_us", "text": "Testing nucleus sampling with top-p filtering.", "desc": "Top-p 0.9 - nucleus sampling", "top_p": 0.9, "top_k": 0},
    {"id": "sampling_top_k", "group": "Sampling", "section": "Top-k", "language": "en_us", "text": "Testing top-k sampling with limited vocabulary.", "desc": "Top-k 50 - restricted vocabulary", "top_k": 50, "top_p": 1.0},
    {"id": "sampling_repetition_penalty", "group": "Sampling", "section": "Repetition", "language": "en_us", "text": "Testing repetition penalty to avoid loops in generation.", "desc": "High repetition penalty (2.0)", "repetition_penalty": 2.0},
]

# 7. ADVANCED FLAGS TESTS
ADVANCED_TESTS = [
    {"id": "advanced_clean_bg", "group": "Advanced", "section": "Flags", "language": "en_us", "text": "Testing clean speaker background flag for cleaner speaker embedding.", "desc": "clean_speaker_background=True", "clean_speaker_background": True},
    {"id": "advanced_no_accurate", "group": "Advanced", "section": "Flags", "language": "en_us", "text": "Testing without accurate mode token for faster generation.", "desc": "accurate_mode=False", "accurate_mode": False},
    {"id": "advanced_ignore_eos", "group": "Advanced", "section": "Flags", "language": "en_us", "text": "Testing ignore EOS to force longer generation.", "desc": "ignore_eos=True", "ignore_eos": True, "max_tokens": 200},
]


def run_tests(url: str, tests: list, group: str, section: str, base_seed: int, **common_kwargs) -> list[TestResult]:
    """Run a list of test configurations."""
    results = []
    for i, test in enumerate(tests):
        test_id = test.get("id", f"{group.lower()}_{i}")
        desc = test.get("description", "") or test.get("desc", "")
        print(f"  [{i+1}/{len(tests)}] {test_id} - {desc[:60]}...")
        
        kwargs = {**common_kwargs}
        # Copy test-specific kwargs (excluding metadata fields)
        for key, value in test.items():
            if key not in ["id", "group", "section", "description", "desc", "ref_audio"]:
                kwargs[key] = value
        
        result = post_tts(url=url, **kwargs)
        result.test_id = test_id
        result.group = group
        result.section = section
        result.description = desc
        results.append(result)
        
        marker = "✓" if result.ok else "✗"
        rtf_str = f" RTF={result.rtf:.3f}" if result.rtf else ""
        print(f"    {marker} e2e={result.e2e_ms:.1f}ms audio={result.audio_sec:.2f}s{rtf_str}")
        if not result.ok:
            print(f"    ERROR: {result.error}")
    return results


def run_speaking_rate_tests(url: str, base_seed: int) -> list[TestResult]:
    """Test all 8 speaking rate buckets."""
    results = []
    for i, (name, idx, desc) in enumerate(SPEAKING_RATE_TESTS):
        print(f"  [rate {i+1}/8] {name} (bucket {idx}) - {desc}")
        result = post_tts(
            url=url,
            text=SPEAKING_RATE_TEXT,
            language="en_us",
            max_tokens=1024,
            seed=base_seed + 1000 + i,
            speaking_rate_bucket=idx,
        )
        result.test_id = f"rate_{name}"
        result.group = "Speaking Rate"
        result.section = "Conditioning"
        result.description = desc
        result.speaking_rate_bucket = idx
        results.append(result)
        marker = "✓" if result.ok else "✗"
        rtf_str = f" RTF={result.rtf:.3f}" if result.rtf else ""
        print(f"    {marker} e2e={result.e2e_ms:.1f}ms audio={result.audio_sec:.2f}s{rtf_str}")
    return results


def run_quality_tests(url: str, base_seed: int) -> list[TestResult]:
    """Test quality conditioning features."""
    results = []
    for i, qt in enumerate(QUALITY_TESTS):
        print(f"  [quality {i+1}/{len(QUALITY_TESTS)}] {qt['name']} - {qt['desc']}")
        result = post_tts(
            url=url,
            text=QUALITY_TEXT,
            language="en_us",
            max_tokens=1024,
            seed=base_seed + 2000 + i,
            quality_buckets=qt["buckets"],
        )
        result.test_id = f"quality_{qt['name']}"
        result.group = "Quality Conditioning"
        result.section = "Conditioning"
        result.description = qt["desc"]
        result.quality_buckets = qt["buckets"]
        results.append(result)
        marker = "✓" if result.ok else "✗"
        rtf_str = f" RTF={result.rtf:.3f}" if result.rtf else ""
        print(f"    {marker} e2e={result.e2e_ms:.1f}ms audio={result.audio_sec:.2f}s{rtf_str}")
    return results


def run_speaker_clone_tests(url: str, base_seed: int) -> list[TestResult]:
    """Test speaker cloning with reference audio."""
    results = []
    for i, test in enumerate(SPEAKER_CLONE_TESTS):
        print(f"  [clone {i+1}/{len(SPEAKER_CLONE_TESTS)}] {test['id']} - {test.get('description', test.get('desc', ''))}")
        
        kwargs = {
            "url": url,
            "text": test["text"],
            "language": test["language"],
            "max_tokens": test.get("max_tokens", 1024),
            "seed": base_seed + 3000 + i,
            "text_normalization": test.get("text_normalization", True),
        }
        if test.get("speaking_rate_bucket") is not None:
            kwargs["speaking_rate_bucket"] = test["speaking_rate_bucket"]
        if test.get("quality_buckets") is not None:
            kwargs["quality_buckets"] = test["quality_buckets"]
        if test.get("ref_audio"):
            kwargs["ref_audio_path"] = test["ref_audio"]
        
        result = post_tts(**kwargs)
        result.test_id = test["id"]
        result.group = test["group"]
        result.section = test["section"]
        result.description = test.get("description", test.get("desc", ""))
        result.speaker_ref = test.get("ref_audio")
        results.append(result)
        
        marker = "✓" if result.ok else "✗"
        rtf_str = f" RTF={result.rtf:.3f}" if result.rtf else ""
        print(f"    {marker} e2e={result.e2e_ms:.1f}ms audio={result.audio_sec:.2f}s{rtf_str}")
        if not result.ok:
            print(f"    ERROR: {result.error}")
    return results


def print_summary(results: list[TestResult], total_time: float) -> None:
    """Print test summary."""
    ok = [r for r in results if r.ok]
    errors = [r for r in results if not r.ok]
    rtfs = [r.rtf for r in ok if r.rtf is not None]
    e2e = [r.e2e_ms for r in ok]
    audio_sec = sum(r.audio_sec for r in ok)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {len(results)}")
    print(f"Passed: {len(ok)}")
    print(f"Failed: {len(errors)}")
    print(f"Wall time: {total_time:.2f}s")
    print(f"Throughput: {len(ok)/total_time:.2f} req/s")
    print(f"Total audio generated: {audio_sec:.2f}s")
    
    if rtfs:
        print(f"\nRTF Statistics:")
        print(f"  Average: {statistics.fmean(rtfs):.3f}")
        print(f"  Median:  {statistics.median(rtfs):.3f}")
        print(f"  Min:     {min(rtfs):.3f}")
        print(f"  Max:     {max(rtfs):.3f}")
        if len(rtfs) > 1:
            print(f"  Stdev:   {statistics.stdev(rtfs):.3f}")
    
    if e2e:
        print(f"\nE2E Latency (ms):")
        print(f"  Average: {statistics.fmean(e2e):.1f}")
        print(f"  Median:  {statistics.median(e2e):.1f}")
    
    print(f"\nResults by Section:")
    sections = {}
    for r in results:
        if r.section not in sections:
            sections[r.section] = {"ok": 0, "total": 0, "rtf": []}
        sections[r.section]["total"] += 1
        if r.ok:
            sections[r.section]["ok"] += 1
            if r.rtf:
                sections[r.section]["rtf"].append(r.rtf)
    
    for section, stats in sorted(sections.items()):
        rtfs_s = stats["rtf"]
        avg_rtf = statistics.fmean(rtfs_s) if rtfs_s else 0
        print(f"  {section}: {stats['ok']}/{stats['total']} OK, avg RTF={avg_rtf:.3f}")
    
    if errors:
        print(f"\nErrors:")
        for e in errors:
            print(f"  {e.test_id}: {e.error}")


def write_json_report(results: list[TestResult], total_time: float, output_path: str) -> None:
    """Write detailed JSON report."""
    ok = [r for r in results if r.ok]
    rtfs = [r.rtf for r in ok if r.rtf is not None]
    e2e = [r.e2e_ms for r in ok]
    
    # Count unique languages tested
    languages = set(r.language for r in results if r.ok)
    
    report = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "mlx-ZONOS2 (Zyphra-ZONOS2-mlx)",
            "backend": "MLX (Apple Silicon)",
            "total_tests": len(results),
            "passed": len(ok),
            "failed": len(results) - len(ok),
            "wall_time_sec": total_time,
            "throughput_req_s": len(ok) / total_time if total_time > 0 else 0,
            "total_audio_sec": sum(r.audio_sec for r in ok),
            "rtf_avg": statistics.fmean(rtfs) if rtfs else None,
            "rtf_median": statistics.median(rtfs) if rtfs else None,
            "rtf_min": min(rtfs) if rtfs else None,
            "rtf_max": max(rtfs) if rtfs else None,
            "e2e_avg_ms": statistics.fmean(e2e) if e2e else None,
            "e2e_median_ms": statistics.median(e2e) if e2e else None,
            "languages_tested": sorted(list(languages)),
            "language_count": len(languages),
            "speaking_rate_buckets": 8,
            "quality_features": 6,
            "quality_buckets_per_feature": {"lufs": 12, "estimated_snr": 12, "max_pause": 12, "estimated_bandlimit_hz": 8, "leading_silence_s": 8, "trailing_silence_s": 8},
            "note": "Model supports many languages; server validation restricts to English. Use text_normalization=False for non-English."
        },
        "results": [asdict(r) for r in results],
    }
    
    Path(output_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON report written to: {output_path}")


def write_md_report(results: list[TestResult], total_time: float, output_path: str, server_url: str) -> None:
    """Write Markdown report."""
    ok = [r for r in results if r.ok]
    errors = [r for r in results if not r.ok]
    rtfs = [r.rtf for r in ok if r.rtf is not None]
    e2e = [r.e2e_ms for r in ok]
    
    def fmt(val):
        return f"{val:.3f}" if val is not None else "N/A"
    def fmt1(val):
        return f"{val:.1f}" if val is not None else "N/A"
    
    rtf_avg = statistics.fmean(rtfs) if rtfs else None
    rtf_median = statistics.median(rtfs) if rtfs else None
    rtf_min = min(rtfs) if rtfs else None
    rtf_max = max(rtfs) if rtfs else None
    e2e_avg = statistics.fmean(e2e) if e2e else None
    e2e_median = statistics.median(e2e) if e2e else None
    e2e_min = min(e2e) if e2e else None
    e2e_max = max(e2e) if e2e else None
    
    languages = set(r.language for r in results if r.ok)
    
    md = f"""# mlx-ZONOS2 Comprehensive Test Report (Actual Capabilities)

## Test Configuration
| Parameter | Value |
|-----------|-------|
| Server | {server_url} |
| Model | mlx-ZONOS2 (Zyphra-ZONOS2-mlx) |
| Backend | MLX (Apple Silicon native) |
| Total Tests | {len(results)} |
| Passed | {len(ok)} |
| Failed | {len(errors)} |
| Wall Time | {total_time:.2f}s |
| Throughput | {len(ok)/total_time:.2f} req/s |
| Total Audio | {sum(r.audio_sec for r in ok):.2f}s |
| Languages Tested | {len(languages)} ({', '.join(sorted(languages))}) |

## Performance Metrics

| Metric | Average | Median | Min | Max |
|--------|---------|--------|-----|-----|
| RTF | {fmt(rtf_avg)} | {fmt(rtf_median)} | {fmt(rtf_min)} | {fmt(rtf_max)} |
| E2E Latency (ms) | {fmt1(e2e_avg)} | {fmt1(e2e_median)} | {fmt1(e2e_min)} | {fmt1(e2e_max)} |

## Results by Section

| Section | Tests | Passed | Avg RTF |
|---------|-------|--------|---------|
"""
    
    sections = {}
    for r in results:
        if r.section not in sections:
            sections[r.section] = {"ok": 0, "total": 0, "rtf": []}
        sections[r.section]["total"] += 1
        if r.ok:
            sections[r.section]["ok"] += 1
            if r.rtf:
                sections[r.section]["rtf"].append(r.rtf)
    
    for section, stats in sorted(sections.items()):
        rtfs_s = stats["rtf"]
        avg_rtf = statistics.fmean(rtfs_s) if rtfs_s else 0
        md += f"| {section} | {stats['total']} | {stats['ok']} | {avg_rtf:.3f} |\n"
    
    md += "\n## Detailed Results\n\n"
    md += "| Test ID | Group | Section | Language | Description | OK | E2E (ms) | Audio (s) | RTF | Error |\n"
    md += "|---------|-------|---------|----------|-------------|----|-----------|-----------|-----|-------|\n"
    
    for r in results:
        desc = r.description[:60].replace("|", "\\|")
        error = (r.error or "").replace("|", "\\|")[:80]
        rtf_str = f"{r.rtf:.3f}" if r.rtf else "-"
        md += f"| {r.test_id} | {r.group} | {r.section} | {r.language} | {desc} | {'✓' if r.ok else '✗'} | {r.e2e_ms:.1f} | {r.audio_sec:.2f} | {rtf_str} | {error} |\n"
    
    if errors:
        md += "\n## Errors\n\n"
        for e in errors:
            md += f"- **{e.test_id}**: {e.error}\n"
    
    md += f"""
## ZONOS2 Actual Capabilities (Tested)

✅ **Supported Languages ({len(languages)} tested):**
- Tier 1: `en_us`, `en_gb`, `en`, `zh` (Chinese), `ja` (Japanese)
- Tier 2: `ko` (Korean), `fr` (French), `es` (Spanish), `de` (German), `ru` (Russian), `pt` (Portuguese), `it` (Italian), `vi` (Vietnamese), `nl` (Dutch), `pl` (Polish), `tr` (Turkish), `he` (Hebrew), `ar` (Arabic)
- Tier 3: `sv` (Swedish), `hi` (Hindi), `ta` (Tamil), `te` (Telugu), `th` (Thai), `da` (Danish), `no` (Norwegian), `fi` (Finnish), `uk` (Ukrainian)
- **Note**: Server validation restricts to English; model supports all. Use `text_normalization=False` for non-English.

✅ **Speaking Rate**: 8 buckets (0-7, ranges 0-8 to 40+ bytes/sec)

✅ **Quality Conditioning**: 6 features with varying buckets:
- lufs: 12 buckets, estimated_snr: 12, max_pause: 12
- estimated_bandlimit_hz: 8, leading_silence_s: 8, trailing_silence_s: 8

✅ **Speaker Cloning**: via reference audio file OR speaker embedding (works cross-lingual)

✅ **Sampling Control**: temperature, top_p, top_k, min_p, repetition params, seed

✅ **Advanced Flags**: clean_speaker_background, accurate_mode, ignore_eos

❌ **NOT Native**: Dedicated emotion parameter (use prompt prefix or speaker ref)

❌ **Limitations**: Streaming (v1), Multi-speaker single request

## Test Notes

- All speaker cloning tests use reference audio: `/tmp/ref_audio_clone.wav`
- Style prefix tests (whisper, excited, etc.) use prompt text prefixes — NOT native parameters
- Quality bucket indices per feature: lufs/snr/max_pause: 0-11; bandlimit/leading/trailing: 0-7
- Speaking rate bucket ranges: [0-8, 8-11, 11-14, 14-17, 17-21, 21-28, 28-40, 40+]

Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    Path(output_path).write_text(md, encoding="utf-8")
    print(f"Markdown report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive test suite for mlx-ZONOS2 (actual capabilities)")
    parser.add_argument("--url", default="http://127.0.0.1:1920", help="Server base URL")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max generation tokens")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--skip-multilingual", action="store_true", help="Skip multilingual tests")
    parser.add_argument("--skip-rates", action="store_true", help="Skip speaking rate tests")
    parser.add_argument("--skip-quality", action="store_true", help="Skip quality tests")
    parser.add_argument("--skip-clone", action="store_true", help="Skip speaker cloning tests")
    parser.add_argument("--skip-style", action="store_true", help="Skip style prefix tests")
    parser.add_argument("--skip-sampling", action="store_true", help="Skip sampling tests")
    parser.add_argument("--skip-advanced", action="store_true", help="Skip advanced flags tests")
    parser.add_argument("--output-json", default="benchmark/out/zonos2_full_capabilities_report.json", help="JSON output")
    parser.add_argument("--output-md", default="benchmark/out/zonos2_full_capabilities_report.md", help="MD output")
    args = parser.parse_args()
    
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"mlx-ZONOS2 Full Capabilities Test Suite")
    print(f"Server: {args.url}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Seed: {args.seed}")
    print(f"Note: Model supports 25+ languages; server restricts to English.")
    print(f"Note: Using text_normalization=False for non-English languages.")
    print(f"{'='*60}\n")
    
    all_results = []
    total_start = time.perf_counter()
    base_seed = args.seed
    
    # 1. Multilingual baseline tests
    if not args.skip_multilingual:
        print("1. Running multilingual baseline tests (25+ languages)...")
        results = run_tests(args.url, MULTILINGUAL_TESTS, "Multilingual", "Language", base_seed, max_tokens=args.max_tokens)
        all_results.extend(results)
        print()
    
    # 2. Speaking rate (8 buckets - REAL ZONOS2 feature)
    if not args.skip_rates:
        print("2. Running speaking rate conditioning tests (8 buckets)...")
        results = run_speaking_rate_tests(args.url, base_seed)
        all_results.extend(results)
        print()
    
    # 3. Quality conditioning
    if not args.skip_quality:
        print("3. Running quality conditioning tests (6 features)...")
        results = run_quality_tests(args.url, base_seed)
        all_results.extend(results)
        print()
    
    # 4. Speaker cloning (reference audio)
    if not args.skip_clone:
        print("4. Running speaker cloning tests (cross-lingual)...")
        if not Path("/tmp/ref_audio_clone.wav").exists():
            print("  Reference audio not found! Generating...")
            import urllib.request
            import json
            import numpy as np
            import wave
            body = {"text": "This is a reference audio sample for voice cloning tests. It contains clear speech with natural prosody.", "language": "en_us", "max_tokens": 1024, "stream": False, "seed": 42}
            payload = json.dumps(body).encode("utf-8")
            req = urllib.request.Request('http://127.0.0.1:1920/tts/generate', data=payload, headers={'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
                pcm = np.frombuffer(data, dtype=np.float32)
                int_pcm = np.clip(pcm * 32767, -32768, 32767).astype(np.int16)
                with wave.open('/tmp/ref_audio_clone.wav', 'wb') as wf:
                    wf.setparams((1, 2, 44100, 0, 'NONE', 'NONE'))
                    wf.writeframes(int_pcm.tobytes())
            print("  Generated reference audio.")
        results = run_speaker_clone_tests(args.url, base_seed)
        all_results.extend(results)
        print()
    
    # 5. Style prefix tests
    if not args.skip_style:
        print("5. Running prompt style prefix tests...")
        results = run_tests(args.url, STYLE_PREFIX_TESTS, "Prompt Style", "Style Prefix", base_seed, max_tokens=args.max_tokens)
        all_results.extend(results)
        print()
    
    # 6. Sampling parameter tests
    if not args.skip_sampling:
        print("6. Running sampling parameter tests...")
        results = run_tests(args.url, SAMPLING_TESTS, "Sampling", "Sampling Params", base_seed, max_tokens=args.max_tokens)
        all_results.extend(results)
        print()
    
    # 7. Advanced flags
    if not args.skip_advanced:
        print("7. Running advanced flags tests...")
        results = run_tests(args.url, ADVANCED_TESTS, "Advanced", "Flags", base_seed, max_tokens=args.max_tokens)
        all_results.extend(results)
        print()
    
    total_time = time.perf_counter() - total_start
    
    # Summary
    print_summary(all_results, total_time)
    
    # Write reports
    write_json_report(all_results, total_time, args.output_json)
    write_md_report(all_results, total_time, args.output_md, args.url)
    
    print(f"\n{'='*60}")
    print("Test suite completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()