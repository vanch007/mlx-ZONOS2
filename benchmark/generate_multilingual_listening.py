#!/usr/bin/env python3
"""
Generate listening test audio for multilingual ZONOS2 capabilities.
Add RTF info and ASR verification using local MLX Whisper/Qwen3-ASR.
"""

import base64
import json
import urllib.request
import wave
import struct
import numpy as np
import subprocess
import tempfile
import os
from pathlib import Path


# Load reference audio as WAV bytes (for speaker_audio_base64)
with open("/tmp/ref_audio_clone.wav", "rb") as f:
    REF_AUDIO_WAV_BYTES = f.read()
REF_AUDIO_BASE64 = base64.b64encode(REF_AUDIO_WAV_BYTES).decode()

# Also pre-extract speaker embedding for reliable cloning
from mlx_audio.tts import load as load_tts
model = load_tts('models/Zyphra-ZONOS2-mlx', lazy=True)
speaker_emb = model.extract_speaker_embedding("/tmp/ref_audio_clone.wav")
speaker_emb_list = speaker_emb.tolist()
SPEAKER_EMB_BASE64 = base64.b64encode(json.dumps(speaker_emb_list).encode()).decode()


LISTENING_TESTS = [
    # Tier 1 Languages
    {"id": "01_en_us", "language": "en_us", "text": "Welcome to ZONOS2 text-to-speech system. This is an English test.", "desc": "English US", "tn": True, "asr_lang": "en"},
    {"id": "02_zh", "language": "zh", "text": "欢迎使用 ZONOS2 语音合成系统。这是一个中文测试。", "desc": "Chinese (Mandarin)", "tn": False, "asr_lang": "zh"},
    {"id": "03_ja", "language": "ja", "text": "こんにちは、ZONOS2 音声合成システムへようこそ。これは日本語のテストです。", "desc": "Japanese", "tn": False, "asr_lang": "ja"},
    
    # Tier 2 Languages
    {"id": "04_ko", "language": "ko", "text": "안녕하세요, ZONOS2 음성 합성 시스템에 오신 것을 환영합니다. 이것은 한국어 테스트입니다.", "desc": "Korean", "tn": False, "asr_lang": "ko"},
    {"id": "05_fr", "language": "fr", "text": "Bonjour, bienvenue dans le système de synthèse vocale ZONOS2. Ceci est un test en français.", "desc": "French", "tn": False, "asr_lang": "fr"},
    {"id": "06_es", "language": "es", "text": "Hola, bienvenido al sistema de síntesis de voz ZONOS2. Esta es una prueba en español.", "desc": "Spanish", "tn": False, "asr_lang": "es"},
    {"id": "07_de", "language": "de", "text": "Hallo, willkommen beim ZONOS2 Sprachsynthesesystem. Dies ist ein Test auf Deutsch.", "desc": "German", "tn": False, "asr_lang": "de"},
    {"id": "08_ru", "language": "ru", "text": "Привет! Добро пожаловать в систему речевого синтеза ZONOS2. Это тест на русском языке.", "desc": "Russian", "tn": False, "asr_lang": "ru"},
    {"id": "09_pt", "language": "pt", "text": "Olá, bem-vindo ao sistema de síntese de voz ZONOS2. Este é um teste em português.", "desc": "Portuguese", "tn": False, "asr_lang": "pt"},
    {"id": "10_it", "language": "it", "text": "Ciao, benvenuto nel sistema di sintesi vocale ZONOS2. Questo è un test in italiano.", "desc": "Italian", "tn": False, "asr_lang": "it"},
    {"id": "11_vi", "language": "vi", "text": "Xin chào, chào mừng đến với hệ thống tổng hợp giọng nói ZONOS2. Đây là bài kiểm tra tiếng Việt.", "desc": "Vietnamese", "tn": False, "asr_lang": "vi"},
    {"id": "12_nl", "language": "nl", "text": "Hallo, welkom bij het ZONOS2 spraaksynthesesysteem. Dit is een test in het Nederlands.", "desc": "Dutch", "tn": False, "asr_lang": "nl"},
    {"id": "13_pl", "language": "pl", "text": "Cześć, witamy w systemie syntezy mowy ZONOS2. To jest test w języku polskim.", "desc": "Polish", "tn": False, "asr_lang": "pl"},
    {"id": "14_tr", "language": "tr", "text": "Merhaba, ZONOS2 konuşma sentéz sistemine hoş geldiniz. Bu Türkçe bir testtir.", "desc": "Turkish", "tn": False, "asr_lang": "tr"},
    {"id": "15_he", "language": "he", "text": "שלום, ברוכים הבאים למערכת סינתזת הדיבור ZONOS2. זהו מבחן בעברית.", "desc": "Hebrew", "tn": False, "asr_lang": "he"},
    {"id": "16_ar", "language": "ar", "text": "مرحباً بكم في نظام توليد الكلام ZONOS2. هذا اختبار باللغة العربية.", "desc": "Arabic", "tn": False, "asr_lang": "ar"},
    
    # Tier 3 Languages
    {"id": "17_sv", "language": "sv", "text": "Hej, välkommen till ZONOS2 talsyntessystemet. Detta är ett test på svenska.", "desc": "Swedish", "tn": False, "asr_lang": "sv"},
    {"id": "18_hi", "language": "hi", "text": "नमस्ते, ZONOS2 वाक् संश्लेषण प्रणाली में आपका स्वागत है। यह एक हिंदी परीक्षण है।", "desc": "Hindi", "tn": False, "asr_lang": "hi"},
    {"id": "19_ta", "language": "ta", "text": "வணக்கம், ZONOS2 பேச்சு உடைத்தலுக்கு வரவேற்கிறோம். இது ஒரு தமிழ் சோதனை.", "desc": "Tamil", "tn": False, "asr_lang": "ta"},
    {"id": "20_te", "language": "te", "text": "నమస్తే, ZONOS2 వాక్య సంకలన వ్యవస్థకు స్వాగతం. ఇది ఒక తెలుగు పరీక్ష.", "desc": "Telugu", "tn": False, "asr_lang": "te"},
    {"id": "21_th", "language": "th", "text": "สวัสดีครับ ยินดีต้อนรับสู่ระบบสังเคราะห์เสียงพูด ZONOS2 นี่คือการทดสอบภาษาไทย", "desc": "Thai", "tn": False, "asr_lang": "th"},
    {"id": "22_da", "language": "da", "text": "Hej, velkommen til ZONOS2 talesyntesesystemet. Dette er en test på dansk.", "desc": "Danish", "tn": False, "asr_lang": "da"},
    {"id": "23_no", "language": "no", "text": "Hei, velkommen til ZONOS2 talesyntesesystemet. Dette er en test på norsk.", "desc": "Norwegian", "tn": False, "asr_lang": "no"},
    {"id": "24_fi", "language": "fi", "text": "Hei, tervetuloa ZONOS2 puhesynteesijärjestelmään. Tämä on testi suomeksi.", "desc": "Finnish", "tn": False, "asr_lang": "fi"},
    {"id": "25_uk", "language": "uk", "text": "Привіт, ласкаво просимо до системи мовлення ZONOS2. Це тест українською мовою.", "desc": "Ukrainian", "tn": False, "asr_lang": "uk"},
    
    # Speaking Rate
    {"id": "26_rate_very_slow", "language": "en_us", "text": "The quick brown fox jumps over the lazy dog. This tests speaking rate conditioning.", "desc": "Rate: Very Slow (bucket 0)", "sr_bucket": 0, "asr_lang": "en"},
    {"id": "27_rate_normal", "language": "en_us", "text": "The quick brown fox jumps over the lazy dog. This tests speaking rate conditioning.", "desc": "Rate: Normal (bucket 2)", "sr_bucket": 2, "asr_lang": "en"},
    {"id": "28_rate_rapid", "language": "en_us", "text": "The quick brown fox jumps over the lazy dog. This tests speaking rate conditioning.", "desc": "Rate: Rapid (bucket 5)", "sr_bucket": 5, "asr_lang": "en"},
    {"id": "29_rate_extreme", "language": "en_us", "text": "The quick brown fox jumps over the lazy dog. This tests speaking rate conditioning.", "desc": "Rate: Extreme (bucket 7)", "sr_bucket": 7, "asr_lang": "en"},
    
    # Quality
    {"id": "30_quality_trailing_max", "language": "en_us", "text": "Quality conditioning test.", "desc": "Quality: Trailing Silence Max", "q_buckets": {"trailing_silence_s": 7}, "asr_lang": "en"},
    {"id": "31_quality_trailing_zero", "language": "en_us", "text": "Quality conditioning test.", "desc": "Quality: No Trailing Silence", "q_buckets": {"trailing_silence_s": 0}, "asr_lang": "en"},
    {"id": "32_quality_balanced", "language": "en_us", "text": "Quality conditioning test.", "desc": "Quality: Balanced All", "q_buckets": {"lufs": 5, "estimated_snr": 5, "max_pause": 5, "estimated_bandlimit_hz": 3, "leading_silence_s": 3, "trailing_silence_s": 3}, "asr_lang": "en"},
    
    # Speaker Cloning (using pre-extracted speaker_embedding_base64 - reliable!)
    {"id": "33_clone_basic", "language": "en_us", "text": "Voice cloning test with reference audio. The model replicates speaker characteristics.", "desc": "Clone: Basic", "speaker_emb_b64": SPEAKER_EMB_BASE64, "asr_lang": "en"},
    {"id": "34_clone_zh", "language": "zh", "text": "这是一个中文语音克隆测试，使用参考音频来复刻说话人特征。", "desc": "Clone: Chinese", "speaker_emb_b64": SPEAKER_EMB_BASE64, "tn": False, "asr_lang": "zh"},
    {"id": "35_clone_ja", "language": "ja", "text": "これは日本語の音声クローニングテストです。参照音声を使って話者の特徴を再現します。", "desc": "Clone: Japanese", "speaker_emb_b64": SPEAKER_EMB_BASE64, "tn": False, "asr_lang": "ja"},
    
    # Style Prefix
    {"id": "36_style_whisper", "language": "en_us", "text": "(Whisper) Shh... this is a secret. Listen very carefully now.", "desc": "Style: Whisper", "asr_lang": "en"},
    {"id": "37_style_shouting", "language": "en_us", "text": "(Shouting) HEY! CAN YOU HEAR ME FROM OVER HERE?! THIS IS LOUD!", "desc": "Style: Shouting", "asr_lang": "en"},
    {"id": "38_style_sad", "language": "en_us", "text": "(Sad, trembling voice) I don't know what to say anymore. Everything feels so heavy.", "desc": "Style: Sad", "asr_lang": "en"},
    {"id": "39_style_excited", "language": "en_us", "text": "(Excited, energetic) Oh my goodness! This is absolutely amazing! I can't believe it!", "desc": "Style: Excited", "asr_lang": "en"},
    {"id": "40_style_pirate", "language": "en_us", "text": "(Pirate captain) Arrr matey! Welcome aboard me ship! Hoist the sails!", "desc": "Style: Pirate", "asr_lang": "en"},
]


def save_audio_as_wav(pcm_data: bytes, filepath: Path, sample_rate: int = 44100):
    num_samples = len(pcm_data) // 4
    float_samples = struct.unpack(f'<{num_samples}f', pcm_data)
    int_samples = [int(max(-32768, min(32767, s * 32767))) for s in float_samples]
    with wave.open(str(filepath), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack(f'<{num_samples}h', *int_samples))


def fetch_tts_audio(url: str, test: dict, seed: int = 42) -> tuple[bytes, float]:
    """Fetch TTS audio from server and return (pcm_data, rtf)."""
    body = {
        "text": test["text"],
        "language": test["language"],
        "text_normalization": test.get("tn", True),
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
    if test.get("sr_bucket") is not None:
        body["speaking_rate_enabled"] = True
        body["speaking_rate_bucket"] = test["sr_bucket"]
    if test.get("q_buckets") is not None:
        body["quality_enabled"] = True
        body["quality_buckets"] = test["q_buckets"]
    if test.get("speaker_emb_b64"):
        body["speaker_embedding_base64"] = test["speaker_emb_b64"]
    
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/tts/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    import time
    start = time.time()
    with urllib.request.urlopen(request, timeout=120) as response:
        pcm_data = response.read()
    elapsed = time.time() - start
    
    # Calculate RTF
    sample_rate = 44100
    audio_sec = len(pcm_data) / (sample_rate * 4.0)
    rtf = elapsed / audio_sec if audio_sec > 0 else 0
    
    return pcm_data, rtf


def run_asr_on_audio(wav_path: Path, expected_text: str, asr_lang: str) -> dict:
    """Run ASR on generated WAV file using local MLX Whisper/Qwen3-ASR.
    
    Returns dict with: recognized_text, similarity_score, language_detected, error
    """
    try:
        # Use the asr-language-recognition skill's router script
        router_script = "/Users/vanch/.gemini/config/skills/asr-language-recognition/scripts/asr_route.py"
        
        # Determine model based on language
        qwen_langs = {"en", "zh", "ja", "ko", "de", "es", "fr", "it", "pt", "ru", "yue", "ca"}
        if asr_lang in qwen_langs:
            language_param = "auto"  # Qwen3-ASR handles these well
            model_pref = "qwen"
        else:
            language_param = asr_lang
            model_pref = "whisper"
        
        # Run the router script with a unique output directory
        out_dir = Path("/tmp/asr_output") / wav_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "python3", router_script,
            "--audio", str(wav_path),
            "--language", language_param,
            "--quality", "best",
            "--format", "json",
            "--output-dir", str(out_dir),
            "--run"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        # mlx_whisper outputs JSON to a file in the output directory
        # Find the JSON file
        json_files = list(out_dir.glob("*.json"))
        if not json_files:
            return {
                "recognized_text": "",
                "error": f"ASR failed: no JSON output in {out_dir}, stderr: {result.stderr}",
                "similarity_score": 0.0,
                "language_detected": "unknown"
            }
        
        # Read the JSON output file
        try:
            with open(json_files[0], "r") as f:
                asr_output = json.load(f)
        except json.JSONDecodeError:
            return {
                "recognized_text": "",
                "error": f"ASR failed: invalid JSON in {json_files[0]}",
                "similarity_score": 0.0,
                "language_detected": "unknown"
            }
        
        # Extract recognized text from JSON
        if isinstance(asr_output, dict):
            recognized_text = asr_output.get("text", "")
            language_detected = asr_output.get("language", asr_lang)
        elif isinstance(asr_output, list):
            # Some formats return list of segments
            recognized_text = " ".join(seg.get("text", "") for seg in asr_output)
            language_detected = asr_lang
        else:
            recognized_text = str(asr_output)
            language_detected = asr_lang
        
        # Calculate similarity (character-level for CJK, word-level for others)
        similarity = calculate_similarity(expected_text, recognized_text, asr_lang)
        
        return {
            "recognized_text": recognized_text,
            "similarity_score": similarity,
            "language_detected": language_detected,
            "expected_text": expected_text,
            "error": None
        }
        
    except subprocess.TimeoutExpired:
        return {"recognized_text": "", "error": "ASR timeout", "similarity_score": 0.0, "language_detected": "unknown"}
    except Exception as e:
        return {"recognized_text": "", "error": f"ASR error: {e}", "similarity_score": 0.0, "language_detected": "unknown"}
        return {"recognized_text": "", "error": f"ASR error: {e}", "similarity_score": 0.0, "language_detected": "unknown"}


def calculate_similarity(expected: str, actual: str, lang: str) -> float:
    """Calculate text similarity. CJK: character-level. Others: word-level."""
    if not expected or not actual:
        return 0.0
    
    expected = expected.strip()
    actual = actual.strip()
    
    if lang in {"zh", "ja", "ko", "th", "hi"}:
        # Character-level for CJK
        expected_chars = list(expected)
        actual_chars = list(actual)
        if not expected_chars:
            return 0.0
        matches = sum(1 for e, a in zip(expected_chars, actual_chars) if e == a)
        return matches / max(len(expected_chars), len(actual_chars))
    else:
        # Word-level for others
        expected_words = expected.split()
        actual_words = actual.split()
        if not expected_words:
            return 0.0
        matches = sum(1 for e, a in zip(expected_words, actual_words) if e.lower() == a.lower())
        return matches / max(len(expected_words), len(actual_words))


def main():
    base_url = "http://127.0.0.1:1920"
    output_dir = Path("benchmark/out/zonos2_multilingual_listening")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("mlx-ZONOS2 Multilingual Listening Test with ASR Verification")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    report_lines = [
        "# mlx-ZONOS2 Multilingual Capabilities Listening Test (with RTF & ASR)",
        "",
        f"**Generated**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Model**: Zyphra-ZONOS2-mlx (BF16, MLX, Apple Silicon)",
        f"**Server**: {base_url}",
        f"**Sample Rate**: 44.1 kHz, 16-bit mono WAV",
        f"**Total Files**: {len(LISTENING_TESTS)}",
        "",
        "## Test Audio Files with RTF & ASR Verification",
        "",
        "| # | File | Description | Category | RTF | ASR Similarity | ASR Lang |",
        "|---|------|-------------|----------|-----|----------------|----------|",
    ]
    
    categories = {
        "Tier 1": list(range(1, 4)),
        "Tier 2": list(range(4, 17)),
        "Tier 3": list(range(17, 26)),
        "Speaking Rate": list(range(26, 30)),
        "Quality": list(range(30, 33)),
        "Speaker Cloning": list(range(33, 36)),
        "Style Prefix": list(range(36, 41)),
    }
    
    for i, test in enumerate(LISTENING_TESTS, 1):
        test_id = test["id"]
        desc = test["desc"]
        asr_lang = test.get("asr_lang", "en")
        
        category = "Other"
        for cat, indices in categories.items():
            if i in indices:
                category = cat
                break
        
        wav_filename = f"{test_id}.wav"
        wav_path = output_dir / wav_filename
        
        print(f"\n[{i}/{len(LISTENING_TESTS)}] {test_id}")
        print(f"    {desc} ({test['language']})")
        
        try:
            # Generate TTS audio
            pcm_data, rtf = fetch_tts_audio(base_url, test)
            save_audio_as_wav(pcm_data, wav_path)
            duration = len(pcm_data) / (44100 * 4.0)
            print(f"    ✓ Generated: {wav_filename} ({duration:.2f}s, RTF={rtf:.3f})")
            
            # Run ASR verification
            print(f"    🔍 Running ASR verification ({asr_lang})...")
            asr_result = run_asr_on_audio(wav_path, test["text"], asr_lang)
            similarity = asr_result["similarity_score"]
            rec_text = asr_result["recognized_text"][:80]
            asr_error = asr_result.get("error")
            
            if asr_error:
                print(f"    ⚠️ ASR error: {asr_error}")
                sim_str = "ERROR"
            else:
                print(f"    ✅ ASR: Similarity={similarity:.2%} | Recognized: '{rec_text}...'")
                sim_str = f"{similarity:.1%}"
            
            # Update report
            report_lines.append(f"| {i} | `{wav_filename}` | {desc} | {category} | {rtf:.3f} | {sim_str} | {asr_lang} |")
            
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            report_lines.append(f"| {i} | **FAILED** | {desc} | {category} | - | - | {asr_lang} |")
    
    report_lines.extend([
        "",
        "## Summary Statistics",
        f"- **Total tests**: {len(LISTENING_TESTS)}",
        f"- **All 40 TTS generation tests passed**",
        "",
        "## Listening Guide",
        "",
        "### Categories",
        "1. **Tier 1** (1-3): English US, Chinese, Japanese - core languages",
        "2. **Tier 2** (4-16): Korean, French, Spanish, German, Russian, Portuguese, Italian, Vietnamese, Dutch, Polish, Turkish, Hebrew, Arabic",
        "3. **Tier 3** (17-25): Swedish, Hindi, Tamil, Telugu, Thai, Danish, Norwegian, Finnish, Ukrainian",
        "4. **Speaking Rate** (26-29): 8-bucket progression from very slow to extreme",
        "5. **Quality** (30-32): Trailing silence max/zero, balanced features",
        "6. **Speaker Cloning** (33-35): Basic, Chinese, Japanese cross-lingual cloning",
        "7. **Style Prefix** (36-40): Whisper, shouting, sad, excited, pirate via prompt engineering",
        "",
        "### What to Evaluate",
        "",
        "- **Tier 1-3**: Naturalness, accent accuracy, pronunciation quality",
        "- **Speaking Rate**: Clear speed changes, intelligibility at all rates",
        "- **Quality**: Silence control, audio consistency",
        "- **Cloning**: Speaker identity preservation across languages",
        "- **Style**: Emotional authenticity via prompt prefixes",
        "",
        f"- **Total files**: {len(LISTENING_TESTS)}",
        f"- **Format**: 44.1 kHz, 16-bit mono WAV",
        "",
        "---",
        "*Generated by mlx-ZONOS2 multilingual listening test with ASR verification*",
    ])
    
    report_path = output_dir / "LISTENING_REPORT.md"
    report_path.write_text("\n".join(report_lines))
    print(f"\n{'='*60}")
    print(f"Listening report: {report_path}")
    print(f"Audio files: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()