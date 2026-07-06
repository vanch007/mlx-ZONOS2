#!/usr/bin/env python3
"""
Generate audio files for listening test and create listening report.
Re-runs key tests and saves PCM audio as WAV files.
"""

import json
import urllib.request
import wave
import struct
from pathlib import Path


# Key test cases to save audio for listening
LISTENING_TESTS = [
    # Baseline tests
    {
        "id": "01_en_us_news_baseline",
        "language": "en_us",
        "text": "Breaking news: Apple Silicon achieves record-breaking TTS performance with MLX native inference.",
        "desc": "English US news announcer - baseline"
    },
    {
        "id": "02_en_us_conversational",
        "language": "en_us",
        "text": "Hey there! Welcome to our podcast. Today we're talking about how AI is changing everything.",
        "desc": "English US conversational - baseline"
    },
    {
        "id": "03_en_us_emotional",
        "language": "en_us",
        "text": "I can't believe we actually did it! After all these years, we finally made it happen!",
        "desc": "English US emotional/excited"
    },
    {
        "id": "04_en_us_whisper_style",
        "language": "en_us",
        "text": "(Whisper) Shh... this is a secret. Listen very carefully now.",
        "desc": "English US whisper style"
    },
    # Creative voice designs
    {
        "id": "05_en_us_raspy_old_man",
        "language": "en_us",
        "text": "(Raspy old man) The world has changed, son. It ain't what it used to be.",
        "desc": "Creative: raspy old man voice design"
    },
    {
        "id": "06_en_us_pirate",
        "language": "en_us",
        "text": "(Pirate captain) All hands on deck! Secure the mainsail! We ride this storm or we die trying!",
        "desc": "Creative: pirate captain voice design"
    },
    {
        "id": "07_en_us_asmr_female",
        "language": "en_us",
        "text": "(Soft-spoken, breathy female voice with ASMR quality) Close your eyes and imagine you're lying on a warm beach.",
        "desc": "Creative: ASMR female voice design"
    },
    {
        "id": "08_en_us_sports_commentator",
        "language": "en_us",
        "text": "(Confident, energetic male sports commentator) He receives the ball in midfield, advances with speed, dribbles past one, past two — he's alone in front of the goalkeeper — he shoots! GOAL!",
        "desc": "Creative: sports commentator voice design"
    },
    # Controllable styles
    {
        "id": "09_en_us_cheerful",
        "language": "en_us",
        "text": "(Cheerful and laughing) I just got the best news — you won't believe what happened today! Everything worked out perfectly!",
        "desc": "Controlled: cheerful style"
    },
    {
        "id": "10_en_us_angry",
        "language": "en_us",
        "text": "(Angry tone, volume gradually increased) Today is the moment of our final confrontation. I will make you suffer here!",
        "desc": "Controlled: angry style"
    },
    # English GB
    {
        "id": "11_en_gb_news",
        "language": "en_gb",
        "text": "Good evening. The Prime Minister has announced new measures to address the cost of living crisis.",
        "desc": "English GB news style"
    },
    {
        "id": "12_en_gb_conversational",
        "language": "en_gb",
        "text": "Hello there! Fancy a cuppa? It's a brilliant day for a walk in the park, isn't it?",
        "desc": "English GB conversational"
    },
    # Speaking rate conditioning
    {
        "id": "13_rate_bucket_0_very_slow",
        "language": "en_us",
        "text": "The quick brown fox jumps over the lazy dog. This tests speaking rate conditioning across eight buckets.",
        "desc": "Speaking rate: very slow (bucket 0: 0-8 bytes/sec)",
        "speaking_rate_enabled": True,
        "speaking_rate_bucket": 0
    },
    {
        "id": "14_rate_bucket_2_normal",
        "language": "en_us",
        "text": "The quick brown fox jumps over the lazy dog. This tests speaking rate conditioning across eight buckets.",
        "desc": "Speaking rate: normal (bucket 2: 11-14 bytes/sec)",
        "speaking_rate_enabled": True,
        "speaking_rate_bucket": 2
    },
    {
        "id": "15_rate_bucket_5_rapid",
        "language": "en_us",
        "text": "The quick brown fox jumps over the lazy dog. This tests speaking rate conditioning across eight buckets.",
        "desc": "Speaking rate: rapid (bucket 5: 21-28 bytes/sec)",
        "speaking_rate_enabled": True,
        "speaking_rate_bucket": 5
    },
    {
        "id": "16_rate_bucket_7_extreme",
        "language": "en_us",
        "text": "The quick brown fox jumps over the lazy dog. This tests speaking rate conditioning across eight buckets.",
        "desc": "Speaking rate: extreme (bucket 7: 40+ bytes/sec)",
        "speaking_rate_enabled": True,
        "speaking_rate_bucket": 7
    },
    # Quality conditioning
    {
        "id": "17_quality_trailing_silence_3s",
        "language": "en_us",
        "text": "Quality conditioning test with different trailing silence and quality feature settings.",
        "desc": "Quality: trailing silence 3 seconds",
        "quality_enabled": True,
        "quality_buckets": {"trailing_silence_s": 6}
    },
    {
        "id": "18_quality_trailing_silence_0s",
        "language": "en_us",
        "text": "Quality conditioning test with different trailing silence and quality feature settings.",
        "desc": "Quality: no trailing silence",
        "quality_enabled": True,
        "quality_buckets": {"trailing_silence_s": 0}
    },
    {
        "id": "19_quality_balanced",
        "language": "en_us",
        "text": "Quality conditioning test with different trailing silence and quality feature settings.",
        "desc": "Quality: balanced all features",
        "quality_enabled": True,
        "quality_buckets": {
            "lufs": 3,
            "estimated_snr": 3,
            "max_pause": 3,
            "estimated_bandlimit_hz": 3,
            "leading_silence_s": 3,
            "trailing_silence_s": 3,
        }
    },
]


def save_audio_as_wav(pcm_data: bytes, filepath: Path, sample_rate: int = 44100):
    """Save raw PCM float32 data as WAV file."""
    # Convert float32 to int16 for WAV
    num_samples = len(pcm_data) // 4
    float_samples = struct.unpack(f'<{num_samples}f', pcm_data)
    # Scale to int16 range
    int_samples = [int(max(-32768, min(32767, s * 32767))) for s in float_samples]
    
    with wave.open(str(filepath), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack(f'<{num_samples}h', *int_samples))


def fetch_tts_audio(url: str, test: dict, seed: int = 42) -> bytes:
    """Fetch TTS audio from server."""
    body = {
        "text": test["text"],
        "language": test["language"],
        "text_normalization": True,
        "temperature": 1.15,
        "topk": 106,
        "top_p": 0.0,
        "min_p": 0.18,
        "max_tokens": 1024,
        "stream": False,
        "fade_out_ms": 0.0,
        "accurate_mode": True,
        "seed": seed,
    }
    
    if test.get("speaking_rate_enabled"):
        body["speaking_rate_enabled"] = True
        body["speaking_rate_bucket"] = test["speaking_rate_bucket"]
    
    if test.get("quality_enabled"):
        body["quality_enabled"] = True
        body["quality_buckets"] = test["quality_buckets"]
    
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/tts/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main():
    base_url = "http://127.0.0.1:1920"
    output_dir = Path("benchmark/out/listening_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("mlx-ZONOS2 Listening Test Audio Generation")
    print(f"Output directory: {output_dir}")
    print(f"Server: {base_url}")
    print("=" * 60)
    
    report_lines = [
        "# mlx-ZONOS2 Listening Test Report",
        "",
        f"**Generated**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Model**: Zyphra-ZONOS2-mlx (MLX, Apple Silicon)",
        f"**Server**: {base_url}",
        f"**Sample Rate**: 44.1 kHz, 16-bit mono WAV",
        "",
        "## Test Audio Files",
        "",
        "| # | File | Description | Category |",
        "|---|------|-------------|----------|",
    ]
    
    for i, test in enumerate(LISTENING_TESTS, 1):
        test_id = test["id"]
        desc = test["desc"]
        
        # Determine category
        if "rate_" in test_id:
            category = "Speaking Rate"
        elif "quality_" in test_id:
            category = "Quality"
        elif "en_gb" in test_id:
            category = "English (GB)"
        elif any(x in test_id for x in ["raspy", "pirate", "asmr", "sports"]):
            category = "Creative Voice Design"
        elif any(x in test_id for x in ["cheerful", "angry", "whisper"]):
            category = "Controllable Style"
        else:
            category = "Baseline"
        
        wav_filename = f"{test_id}.wav"
        wav_path = output_dir / wav_filename
        
        print(f"\n[{i}/{len(LISTENING_TESTS)}] {test_id}")
        print(f"    {desc}")
        print(f"    Language: {test['language']}")
        
        try:
            pcm_data = fetch_tts_audio(base_url, test)
            save_audio_as_wav(pcm_data, wav_path)
            duration = len(pcm_data) / (44100 * 4.0)
            print(f"    ✓ Saved: {wav_filename} ({len(pcm_data)} bytes, {duration:.2f}s)")
            
            report_lines.append(f"| {i} | `{wav_filename}` | {desc} | {category} |")
            
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            report_lines.append(f"| {i} | **FAILED** | {desc} | {category} |")
    
    report_lines.extend([
        "",
        "## Listening Notes",
        "",
        "### Categories",
        "",
        "1. **Baseline** (Tests 1-3): Standard speech synthesis quality check",
        "2. **Controllable Style** (Tests 4, 9-10): Whisper, cheerful, angry styles via style prefixes",
        "3. **Creative Voice Design** (Tests 5-8): Voice design from natural language descriptions",
        "4. **English GB** (Tests 11-12): UK English variant",
        "5. **Speaking Rate** (Tests 13-16): 8-bucket conditioning (very slow → extreme)",
        "6. **Quality** (Tests 17-19): Trailing silence and multi-feature quality conditioning",
        "",
        "### What to Listen For",
        "",
        "- **Baseline**: Naturalness, clarity, prosody, absence of artifacts",
        "- **Controllable**: Does whisper sound whispered? Does angry sound angry? Is cheerful actually cheerful?",
        "- **Creative**: Does raspy old man sound raspy/old? Does pirate sound like a pirate? Does ASMR have breathy quality? Does sports commentator sound energetic?",
        "- **English GB**: British accent markers (RP or regional), vocabulary differences",
        "- **Speaking Rate**: Clear progression from very slow to extreme speed, intelligibility maintained",
        "- **Quality**: Trailing silence presence/absence, overall audio quality consistency",
        "",
        "### Technical Specs",
        "",
        f"- **Total files**: {len(LISTENING_TESTS)}",
        f"- **Sample rate**: 44.1 kHz",
        f"- **Bit depth**: 16-bit PCM",
        f"- **Channels**: Mono",
        f"- **Format**: WAV (RIFF)",
        "",
        "---",
        f"*Generated by mlx-ZONOS2 listening test script*",
    ])
    
    report_path = output_dir / "LISTENING_REPORT.md"
    report_path.write_text("\n".join(report_lines))
    print(f"\n{'='*60}")
    print(f"Listening report saved to: {report_path}")
    print(f"Audio files in: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()