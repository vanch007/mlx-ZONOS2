#!/usr/bin/env python3
"""
Generate HTML listening report from ZONOS2 × VoxCPM2 test results.

Reads benchmark/out/zonos2_voxcpm2_listening/report.json and produces
benchmark/out/zonos2_voxcpm2_listening/index.html — a dark-theme HTML
listening comparison report similar to voxcpm2's format.

Output columns: Official Demo | Reference Audio | MLX ZONOS2 BF16
"""

import html as html_module
import json
from pathlib import Path

OUTPUT_DIR = Path("benchmark/out/zonos2_voxcpm2_listening")

# VoxCPM2 section_title → display category mapping
# Both English and Chinese labels from voxcpm2 manifest
SECTION_MAP = {
    "Multilingual: 30-Language & Chinese Dialect Support": {
        "zh": "多语言: 30种语言 & 中文方言",
        "en": "Multilingual: 30-Language & Chinese Dialect Support",
    },
    "Expressive: Cross-Lingual Voice Transfer": {
        "zh": "表现力: 跨语言声音迁移",
        "en": "Expressive: Cross-Lingual Voice Transfer",
    },
    "Creative: Voice Design from Natural-Language Description": {
        "zh": "创意: 自然语言描述的声音设计",
        "en": "Creative: Voice Design from Natural-Language Description",
    },
    "Controllable Voice Cloning: Same Voice, Different Styles": {
        "zh": "可控语音克隆: 相同声音, 不同风格",
        "en": "Controllable Voice Cloning: Same Voice, Different Styles",
    },
}

# voxcpm2 display_group (Chinese) → English display category
DISPLAY_GROUP_MAP = {
    "中文与中文方言": "多语言: 30种语言 & 中文方言",
    "英文与亚洲语言": "英文与亚洲语言",
    "西方语言": "西方语言",
    "其他语言与风格": "其他语言与风格",
}

# VoxCPM2 category ordering (Chinese names to match manifest)
CATEGORY_ORDER = [
    "多语言: 30种语言 & 中文方言",
    "英文与亚洲语言",
    "西方语言",
    "其他语言与风格",
    "表现力: 跨语言声音迁移",
    "创意: 自然语言描述的声音设计",
    "可控语音克隆: 相同声音, 不同风格",
    "ZONOS2: Speaking Rate",
    "ZONOS2: Quality",
    "ZONOS2: Style Prefix",
    "ZONOS2: Speaker Cloning",
]

CATEGORY_ICONS = {
    "多语言: 30种语言 & 中文方言": "🌐",
    "英文与亚洲语言": "🌏",
    "西方语言": "🌍",
    "其他语言与风格": "🗺",
    "表现力: 跨语言声音迁移": "🌍",
    "创意: 自然语言描述的声音设计": "🎭",
    "可控语音克隆: 相同声音, 不同风格": "👤",
    "ZONOS2: Speaking Rate": "⚡",
    "ZONOS2: Quality": "🎵",
    "ZONOS2: Style Prefix": "🎤",
    "ZONOS2: Speaker Cloning": "🔊",
}

# ZONOS2-specific groupings (for ZONOS2-only tests)
ZONOS2_CATEGORIES = [
    "ZONOS2: Speaking Rate",
    "ZONOS2: Quality",
    "ZONOS2: Style Prefix",
    "ZONOS2: Speaker Cloning",
]

# If a result has no voxcpm2 display_group, assign it to ZONOS2 groups
def get_category(result: dict) -> str:
    """Get the display category for a result, with Chinese→English mapping."""
    cat = result.get("category", "")
    # Map Chinese display_group names to English category names
    if cat in DISPLAY_GROUP_MAP:
        cat = DISPLAY_GROUP_MAP[cat]
    if cat:
        return cat
    if result.get("voxcpm2_display_group"):
        vg = result["voxcpm2_display_group"]
        if vg in DISPLAY_GROUP_MAP:
            vg = DISPLAY_GROUP_MAP[vg]
        return vg
    return "Other"

def category_sort_key(cat: str) -> int:
    """Sort key for categories: voxcpm2 first, ZONOS2 last."""
    for i, c in enumerate(CATEGORY_ORDER):
        if cat == c:
            return i
    # ZONOS2 categories after voxcpm2
    for i, c in enumerate(CATEGORY_ORDER):
        if c.startswith("ZONOS2:") and cat.startswith("ZONOS2:"):
            return len(CATEGORY_ORDER) + i
    return len(CATEGORY_ORDER) + 100

def escape(text: str | None) -> str:
    """Escape HTML special characters."""
    if text is None:
        return ""
    return html_module.escape(text)

def format_number(n: float | None, digits: int = 2) -> str:
    """Format a number, return '-' if None."""
    if n is None:
        return "-"
    return f"{n:.{digits}f}"

def audio_icon(rtf: float | None) -> str:
    """Return emoji based on RTF quality."""
    if rtf is None:
        return "⚠️"
    if rtf < 0.7:
        return "🚀"
    if rtf < 1.0:
        return "⚡"
    return "🐢"

def main():
    # Load report
    report_path = OUTPUT_DIR / "report.json"
    with open(report_path, "r") as f:
        report = json.load(f)

    results = report["rows"]
    # Sort: voxcpm2 first by category order, then ZONOS2 groups
    results.sort(key=lambda r: (category_sort_key(get_category(r)), r.get("voxcpm2_idx", 999)))

    # Group by category
    groups: dict[str, list[dict]] = {}
    for result in results:
        cat = get_category(result)
        if cat not in groups:
            groups[cat] = []
        groups[cat].append(result)

    # Summary stats
    successful = [r for r in results if r.get("rtf") is not None]
    failed = [r for r in results if r.get("rtf") is None]
    avg_rtf = sum(r["rtf"] for r in successful) / len(successful) if successful else 0
    avg_duration = sum(r["duration_s"] for r in successful) / len(successful) if successful else 0

    # Build HTML
    html_parts = []
    html_parts.append("<!doctype html>")
    html_parts.append("<html lang=\"zh-CN\">")
    html_parts.append("<head>")
    html_parts.append('<meta charset="utf-8">')
    html_parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    html_parts.append("<title>ZONOS2 × VoxCPM2 试听对比</title>")
    html_parts.append("""
<style>
:root { color-scheme: dark; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
body { margin: 0; background: #101112; color: #eee; }
header { padding: 34px 24px 24px; text-align: center; border-bottom: 1px solid #2a2c30; }
h1 { margin: 0 0 10px; font-size: 32px; }
.subtitle { color: #cbd0d7; font-size: 16px; }
.summary { display: flex; justify-content: center; flex-wrap: wrap; gap: 12px; margin-top: 18px; }
.summary span { background: #202328; border: 1px solid #31343a; border-radius: 999px; padding: 7px 12px; color: #dfe3ea; font-size: 14px; }
main { max-width: 1320px; margin: 0 auto; padding: 24px; }
.group-title { margin: 34px 0 8px; padding-top: 12px; font-size: 22px; border-top: 1px solid #2a2c30; }
.case { border: 1px solid #2a2c30; border-radius: 8px; background: #151719; margin: 18px 0; padding: 18px; }
.meta, .stats { display: flex; flex-wrap: wrap; gap: 10px; color: #aeb4bd; font-size: 13px; }
.meta span, .stats span { background: #22252a; border-radius: 999px; padding: 4px 9px; }
h3 { margin: 14px 0 8px; font-size: 19px; }
h3 .vox-num { color: #666; font-size: 14px; font-weight: normal; }
p { color: #cbd0d7; line-height: 1.45; margin: 0 0 8px 0; }
.players { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 18px; margin-top: 16px; }
.player h4 { margin: 0 0 8px; color: #aeb4bd; font-size: 13px; text-transform: uppercase; letter-spacing: .08em; }
audio { width: 100%; }
details { margin-top: 14px; color: #aeb4bd; }
pre { white-space: pre-wrap; background: #0f1012; padding: 12px; border-radius: 6px; color: #d9dde3; font-size: 13px; }
.failed .player { grid-column: 1 / -1; text-align: center; color: #e85d5d; padding: 20px; }
.ref-audio { display: inline-block; width: 100%; }
</style>
""")
    html_parts.append("</head>")
    html_parts.append("<body>")

    # Header
    html_parts.append("  <header>")
    html_parts.append("    <h1>ZONOS2 × VoxCPM2 试听对比</h1>")
    html_parts.append(f'    <div class="subtitle">测试设备：M3 Max MacBook Pro | Model: {escape(report.get("model", ""))}</div>')
    html_parts.append("    <div class=\"summary\">")
    html_parts.append(f"      <span>{report.get('total_tests', 0)} 条测试</span>")
    html_parts.append(f"      <span>✓ {report.get('successful', 0)} 成功</span>")
    html_parts.append(f"      <span>✗ {report.get('failed', 0)} 失败</span>")
    html_parts.append(f"      <span>平均 RTF {format_number(report.get('avg_rtf'))}</span>")
    html_parts.append("    </div>")
    html_parts.append("  </header>")
    html_parts.append("  <main>")

    # Group by category
    for cat in CATEGORY_ORDER:
        if cat not in groups:
            # Check ZONOS2 categories
            for zo_cat in ZONOS2_CATEGORIES:
                if zo_cat in groups:
                    cat = zo_cat
                    break

        if cat not in groups:
            continue

        cat_results = groups[cat]
        icon = CATEGORY_ICONS.get(cat, "")
        html_parts.append(f'<h2 class="group-title">{icon} {cat}</h2>')

        for result in cat_results:
            html_parts.append(_build_case(result))

    # ZONOS2-only groups not in CATEGORY_ORDER
    for cat in ZONOS2_CATEGORIES:
        if cat in groups:
            continue  # Already added above

    html_parts.append("  </main>")
    html_parts.append("</body>")
    html_parts.append("</html>")

    html = "\n".join(html_parts)
    html_path = OUTPUT_DIR / "index.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"HTML report: {html_path}")
    print(f"  Total: {report.get('total_tests', 0)} tests")
    print(f"  Successful: {report.get('successful', 0)}, Failed: {report.get('failed', 0)}")

def _build_case(result: dict) -> str:
    """Build HTML for a single test case."""
    parts = []
    parts.append('<article class="case">')

    # Meta
    parts.append('  <div class="meta">')
    if result.get("voxcpm2_idx"):
        parts.append(f'    <span>#{"{:03d}".format(result["voxcpm2_idx"])} 📦</span>')

    # Language tag
    if result.get("zon_lang"):
        lang_tag = result["zon_lang"]
        if result.get("style_prefix"):
            lang_tag += f" + {result['style_prefix']}"
        if result.get("speaker_embedding_base64"):
            lang_tag += " 👤"
        if result.get("speaking_rate_bucket") is not None:
            lang_tag += f" ⚡ Bucket {result['speaking_rate_bucket']}"
        parts.append(f'    <span>{escape(lang_tag)}</span>')

    # Display label
    parts.append(f'    <span>{escape(result.get("display_label", ""))}</span>')

    parts.append('  </div>')

    # H3
    parts.append(f'  <h3>{escape(result.get("display_label", ""))}</h3>')

    # Target text
    if result.get("text"):
        parts.append(f'  <p>{escape(result["text"])}</p>')

    # Stats
    parts.append('  <div class="stats">')
    if result.get("official_duration_s") is not None:
        parts.append(f'    <span>官方 {format_number(result["official_duration_s"])}s</span>')
    if result.get("rtf") is not None:
        icon = audio_icon(result["rtf"])
        parts.append(f'    <span>{icon} ZONOS2 BF16 RTF {format_number(result["rtf"])} | {format_number(result["duration_s"])}s</span>')
    if result.get("error"):
        parts.append(f'    <span>✗ {escape(result["error"])}</span>')
    parts.append('  </div>')

    # Audio players
    parts.append('  <div class="players">')

    # Official Demo
    if result.get("official_audio"):
        # Map official_audio path to local relative path
        # e.g., "official/002_official.wav" → just the filename part
        # But we need to find the actual file — voxcpm2 uses specific naming
        # e.g., 002_official.wav
        official_basename = Path(result["official_audio"]).name
        official_local = f"official/{official_basename}"
        parts.append(f'    <div class="player"><h4>官方 Demo</h4><audio controls src="{official_local}"></audio></div>')

    # Reference Audio
    if result.get("ref_audio"):
        ref_basename = Path(result["ref_audio"]).name
        ref_local = f"refs/{ref_basename}"
        parts.append(f'    <div class="player"><h4>参考音频</h4><audio controls src="{ref_local}"></audio></div>')

    # ZONOS2 BF16
    if result.get("wav_filename"):
        parts.append(f'    <div class="player"><h4>MLX ZONOS2 BF16</h4><audio controls src="zonos2_bf16/{escape(result["wav_filename"])}"></audio></div>')
    elif result.get("error"):
        parts.append('    <div class="player" style="color:#e85d5d">— 生成失败 —</div>')

    parts.append('  </div>')

    # Details > prompt text
    if result.get("api_text"):
        parts.append(f'  <details><summary>ZONOS2 API Request</summary><pre>{escape(result["api_text"])}</pre></details>')

    parts.append('</article>')
    return "\n".join(parts)

if __name__ == "__main__":
    main()
