#!/usr/bin/env python3
"""
Generate audio files for listening test of ZONOS2 capabilities.
"""

import json
import urllib.request
import wave
import struct
import numpy as np
from pathlib import Path


LISTENING_TESTS = [
    # Baseline
    {
        "id": "01_baseline_en_us_news",
        "language": "en_us",
        "text": "Breaking news: Apple Silicon achieves record-breaking TTS performance with MLX native inference.",
        "desc": "Baseline: English US News"
    },
    {
        "id": "02_baseline_en_us_conversational",
        "language": "en_us",
        "text": "Hey there! Welcome to our podcast. Today we're talking about how AI is changing everything.",
        "desc": "Baseline: English US Conversational"
    },
    {
        "id": "03_baseline_en_gb_news",
        "language": "en_gb",
        "text": "Good evening. The Prime Minister has announced new measures to address the cost of living crisis.",
        "desc": "Baseline: English GB News"
    },
    # Speaking Rate (key buckets)
    {
        "id": "04_rate_very_slow",
        "language": "en_us",
        "text": "The quick brown fox jumps over the lazy dog. This tests Speaking Rate conditioning.",
        "desc": "Speaking Rate: Very Slow (bucket 0)",
        "speaking_rate_bucket": 0
    },
    {
        "id": "05_rate_normal",
        "language": "en_us",
        "text": "The quick brown fox jumps over the lazy dog. This tests Speaking Rate conditioning.",
        "desc": "Speaking Rate: Normal (bucket 2)",
        "speaking_rate_bucket": 2
    },
    {
        "id": "06_rate_rapid",
        "language": "en_us",
        "text": "The quick brown fox jumps over the lazy dog. This tests Speaking Rate conditioning.",
        "desc": "Speaking Rate: Rapid (bucket 5)",
        "speaking_rate_bucket": 5
    },
    {
        "id": "07_rate_extreme",
        "language": "en_us",
        "text": "The quick brown fox jumps over the lazy dog. This tests Speaking Rate conditioning.",
        "desc": "Speaking Rate: Extreme (bucket 7)",
        "speaking_rate_bucket": 7
    },
    # Quality
    {
        "id": "08_quality_trailing_max",
        "language": "en_us",
        "text": "Quality conditioning test with different audio quality feature settings.",
        "desc": "Quality: Trailing Silence Max",
        "quality_buckets": {"trailing_silence_s": 7}
    },
    {
        "id": "09_quality_trailing_zero",
        "language": "en_us",
        "text": "Quality conditioning test with different audio quality feature settings.",
        "desc": "Quality: No Trailing Silence",
        "quality_buckets": {"trailing_silence_s": 0}
    },
    {
        "id": "10_quality_leading_max",
        "language": "en_us",
        "text": "Quality conditioning test with different audio quality feature settings.",
        "desc": "Quality: Leading Silence Max",
        "quality_buckets": {"leading_silence_s": 7}
    },
    {
        "id": "11_quality_balanced",
        "language": "en_us",
        "text": "Quality conditioning test with different audio quality feature settings.",
        "desc": "Quality: Balanced All Features",
        "quality_buckets": {
            "lufs": 5, "estimated_snr": 5, "max_pause": 5,
            "estimated_bandlimit_hz": 3, "leading_silence_s": 3, "trailing_silence_s": 3
        }
    },
    # Speaker Cloning
    {
        "id": "12_clone_basic",
        "language": "en_us",
        "text": "This is a voice cloning test using a reference audio sample. The model should replicate the speaker characteristics.",
        "desc": "Cloning: Basic Voice Cloning",
        "ref_audio": "/tmp/ref_audio_clone.wav"
    },
    {
        "id": "13_clone_en_gb",
        "language": "en_gb",
        "text": "Voice cloning with British English variant. The cloned speaker should maintain identity across language codes.",
        "desc": "Cloning: Cross-Language (en_gb)",
        "ref_audio": "/tmp/ref_audio_clone.wav"
    },
    {
        "id": "14_clone_with_rate",
        "language": "en_us",
        "text": "Voice cloning combined with speaking rate control. The speaker identity should persist at different speeds.",
        "desc": "Cloning + Speaking Rate (bucket 5)",
        "ref_audio": "/tmp/ref_audio_clone.wav",
        "speaking_rate_bucket": 5
    },
    # Style Prefix (Prompt Engineering)
    {
        "id": "15_style_whisper",
        "language": "en_us",
        "text": "(Whisper) Shh... this is a secret. Listen very carefully now.",
        "desc": "Style: Whisper via Prompt Prefix"
    },
    {
        "id": "16_style_shouting",
        "language": "en_us",
        "text": "(Shouting) HEY! CAN YOU HEAR ME FROM OVER HERE?! THIS IS LOUD!",
        "desc": "Style: Shouting via Prompt Prefix"
    },
    {
        "id": "17_style_sad",
        "language": "en_us",
        "text": "(Sad, trembling voice) I don't know what to say anymore. Everything feels so heavy.",
        "desc": "Style: Sad via Prompt Prefix"
    },
    {
        "id": "18_style_excited",
        "language": "en_us",
        "text": "(Excited, energetic) Oh my goodness! This is absolutely amazing! I can't believe it!",
        "desc": "Style: Excited via Prompt Prefix"
    },
    {
        "id": "19_style_pirate",
        "language": "en_us",
        "text": "(Pirate captain) Arrr matey! Welcome aboard me ship! Hoist the sails and prepare for adventure!",
        "desc": "Style: Pirate via Prompt Prefix"
    },
    # Sampling
    {
        "id": "20_sampling_low_temp",
        "language": "en_us",
        "text": "Testing deterministic generation with low temperature for consistent output.",
        "desc": "Sampling: Low Temperature (0.5)",
        "temperature": 0.5
    },
    {
        "id": "21_sampling_high_temp",
        "language": "en_us",
        "text": "Testing creative generation with high temperature for more variation.",
        "desc": "Sampling: High Temperature (1.5)",
        "temperature": 1.5
    },
]


def save_audio_as_wav(pcm_data: bytes, filepath: Path, sample_rate: int = 44100):
    """Save raw PCM float32 data as WAV file."""
    num_samples = len(pcm_data) // 4
    float_samples = struct.unpack(f'<{num_samples}f', pcm_data)
    int_samples = [int(max(-32768, min(32767, s * 32767))) for s in float_samples]
    
    with wave.open(str(filepath), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack(f'<{num_samples}h', *int_samples))


def fetch_tts_audio(url: str, test: dict, seed: int = 42) -> bytes:
    """Fetch TTS audio from server."""
    body = {
        "text": test["text"],
        "language": test["language"],
        "text_normalization": True,
        "temperature": test.get("temperature", 1.15),
        "topk": 106,
        "top_p": 0.0,
        "min_p": 0.18,
        "max_tokens": 1024,
        "stream": False,
        "fade_out_ms": 0.0,
        "accurate_mode": True,
        "seed": seed,
    }
    
    if test.get("speaking_rate_bucket") is not None:
        body["speaking_rate_enabled"] = True
        body["speaking_rate_bucket"] = test["speaking_rate_bucket"]
    
    if test.get("quality_buckets") is not None:
        body["quality_enabled"] = True
        body["quality_buckets"] = test["quality_buckets"]
    
    if test.get("ref_audio"):
        import base64
        with open(test["ref_audio"], "rb") as f:
            audio_data = f.read()
        body["speaker_audio_base64"] = base64.b64encode(audio_data).decode()
    
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
    output_dir = Path("benchmark/out/zonos2_listening_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("mlx-ZONOS2 Capabilities Listening Test Audio Generation")
    print(f"Output directory: {output_dir}")
    print(f"Server: {base_url}")
    print("=" * 60)
    
    report_lines = [
        "# mlx-ZONOS2 Capabilities Listening Test Report",
        "",
        f"**Generated**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Model**: Zyphra-ZONOS2-mlx (BF16, MLX, Apple Silicon)",
        f"**Server**: {base_url}",
        f"**Sample Rate**: 44.1 kHz, 16-bit mono WAV",
        "",
        "## Test Audio Files",
        "",
        "| # | File | Description | Category |",
        "|---|------|-------------|----------|",
    ]
    
    categories = {
        "Baseline": list(range(1, 4)),
        "Speaking Rate": list(range(4, 8)),
        "Quality": list(range(8, 12)),
        "Speaker Cloning": list(range(12, 15)),
        "Style Prefix": list(range(15, 20)),
        "Sampling": [20, 21],
    }
    
    for i, test in enumerate(LISTENING_TESTS, 1):
        test_id = test["id"]
        desc = test["desc"]
        
        # Determine category
        category = "Other"
        for cat, indices in categories.items():
            if i in indices:
                category = cat
                break
        
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
        "1. **Baseline** (Tests 1-3): Standard speech synthesis quality check for en_us and en_gb",
        "2. **Speaking Rate** (Tests 4-7): 8-bucket conditioning - clear progression from very slow to extreme",
        "3. **Quality** (Tests 8-11): Quality feature conditioning - trailing/leading silence, balanced features",
        "4. **Speaker Cloning** (Tests 12-14): Reference audio based cloning (en_us, en_gb, with rate)",
        "5. **Style Prefix** (Tests 15-19): Prompt engineering style control (whisper, shout, sad, excited, pirate)",
        "6. **Sampling** (Tests 20-21): Temperature control for deterministic vs creative output",
        "",
        "### What to Listen For",
        "",
        "- **Baseline**: Naturalness, clarity, prosody, accent differences (US vs GB)",
        "- **Speaking Rate**: Clear speed progression, intelligibility maintained at all speeds",
        "- **Quality**: Trailing/leading silence presence, audio quality differences",
        "- **Speaker Cloning**: Speaker identity preservation across language codes and rate changes",
        "- **Style Prefix**: Does whisper sound whispered? Does shouting sound loud? Emotional authenticity?",
        "- **Sampling**: Low temp = more deterministic/consistent; High temp = more variation",
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
        f"*Generated by mlx-ZONOS2 capabilities listening test script*",
    ])
    
    report_path = output_dir / "LISTENING_REPORT.md"
    report_path.write_text("\n".join(report_lines))
    print(f"\n{'='*60}")
    print(f"Listening report saved to: {report_path}")
    print(f"Audio files in: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()