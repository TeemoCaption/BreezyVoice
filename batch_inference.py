import os
import argparse
import csv
import re
from single_inference import single_inference, CustomCosyVoice
from g2pw import G2PWConverter

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_SINGLE_INFERENCE_PATH = os.path.join(ROOT_DIR, "single_inference.py")

_INVISIBLE_CHARS = dict.fromkeys(
    map(
        ord,
        [
            "\u200b",
            "\u200c",
            "\u200d",
            "\ufeff",
            "\u2060",
            "\u00a0",
        ],
    ),
    None,
)
_PAREN_CHARS = dict.fromkeys(map(ord, ["(", ")", "（", "）"]), None)
_WHITESPACE_RE = re.compile(r"\s+")
_CJK_CHAR_RE = re.compile(r"[\u3400-\u9fff]")
_CJK_SPACE_RE = re.compile(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])")
_CJK_PUNCT_SPACE_RE = re.compile(r"(?<=[\u3400-\u9fff])\s+(?=[，。！？；：、])")
_PUNCT_CJK_SPACE_RE = re.compile(r"(?<=[。！？；：，、.!?;:,])\s+(?=[\u3400-\u9fff])")
_CJK_SPACED_TEXT_RE = re.compile(r"(?:[\u3400-\u9fff]\s+){2,}[\u3400-\u9fff]")
_STRUCTURAL_BREAK_RE = re.compile(r"(?:\r\n|\r|\n|\t|\||｜)+")
_PLACEHOLDER_TAG_RE = re.compile(r"_*\[:[^\]]+\]_*\s*")
_REPEATED_SYMBOL_RE = re.compile(r"[_~`*=]{2,}")
_REPEATED_HYPHEN_RE = re.compile(r"-{3,}")
_REPEATED_WEAK_PUNCT_RE = re.compile(r"([，、,；;：:])(?:\s*\1)+")
_REPEATED_STRONG_PUNCT_RE = re.compile(r"([。！？.!?；;])(?:\s*\1)+")
_COMMON_INSTRUCTION_PREFIX_RE = re.compile(
    r"^\s*(?:請回答(?:這個)?問題|請直接回答|回答(?:下列)?問題|問題|請問|答案)\s*[:：]\s*"
)
_COLON_BEFORE_STRONG_PUNCT_RE = re.compile(r"[：:]\s*([。！？?!])")
_NOISY_UPPER_TOKEN_RE = re.compile(r"\b[A-Z]{5,}\b")
_ASCII_NOISE_RUN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_VALID_TTS_CONTENT_RE = re.compile(r"[A-Za-z0-9\u3400-\u9fff]")
_BOPOMOFO_TEXT_RE = re.compile(r"[\u3105-\u3129\u02d9\u02ca\u02c7\u02cb\u02ea\u02eb]")
_CJK_ASCII_PUNCT_TRANSLATION = str.maketrans(
    {
        ",": "，",
        ";": "；",
        ":": "：",
        "?": "？",
        "!": "！",
    }
)


def _clean_batch_text(
    text: str,
    preserve_bopomofo_markup: bool = False,
    drop_ascii_noise: bool = False,
) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = text.translate(_INVISIBLE_CHARS)
    cleaned = cleaned.translate(_PAREN_CHARS)
    if not preserve_bopomofo_markup:
        cleaned = _PLACEHOLDER_TAG_RE.sub(" ", cleaned)
    cleaned = _REPEATED_SYMBOL_RE.sub(" ", cleaned)
    cleaned = _REPEATED_HYPHEN_RE.sub(" ", cleaned)
    cleaned = _STRUCTURAL_BREAK_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    cleaned = _COMMON_INSTRUCTION_PREFIX_RE.sub("", cleaned)
    cleaned = _COLON_BEFORE_STRONG_PUNCT_RE.sub(r"\1", cleaned)
    if _CJK_CHAR_RE.search(cleaned):
        # Drop embedded ASCII runs inside Chinese text to avoid English-like
        # prompt leakage such as FIREFOXONANDROID/CREDIT affecting synthesis.
        if drop_ascii_noise:
            cleaned = _ASCII_NOISE_RUN_RE.sub(" ", cleaned)
        cleaned = cleaned.translate(_CJK_ASCII_PUNCT_TRANSLATION)
        cleaned = _CJK_SPACE_RE.sub("", cleaned)
        cleaned = _CJK_PUNCT_SPACE_RE.sub("", cleaned)
        cleaned = _PUNCT_CJK_SPACE_RE.sub("", cleaned)
        if _CJK_SPACED_TEXT_RE.search(cleaned):
            cleaned = cleaned.replace(" ", "")
    elif drop_ascii_noise:
        cleaned = _NOISY_UPPER_TOKEN_RE.sub(" ", cleaned)
    cleaned = _REPEATED_WEAK_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = _REPEATED_STRONG_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip(" ，、；：,.!?;:-_")
    return cleaned


def _clean_batch_bopomofo_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = text.translate(_INVISIBLE_CHARS)
    cleaned = _STRUCTURAL_BREAK_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    if not cleaned:
        return ""
    if not _BOPOMOFO_TEXT_RE.search(cleaned):
        return ""
    return cleaned


def _should_drop_prompt_text(text: str) -> bool:
    cleaned = _clean_batch_text(text, drop_ascii_noise=True)
    if not cleaned:
        return True
    if not _VALID_TTS_CONTENT_RE.search(cleaned):
        return True
    upper_tokens = _NOISY_UPPER_TOKEN_RE.findall(cleaned)
    if not upper_tokens:
        return False
    cjk_count = len(_CJK_CHAR_RE.findall(cleaned))
    return cjk_count < max(10, int(len(cleaned) * 0.6))


def _preprocess_csv_rows(rows):
    cleaned_rows = []
    dropped_rows = 0
    for row in rows:
        prompt_text = _clean_batch_text(
            str(row.get("speaker_prompt_text_transcription", "") or ""),
            drop_ascii_noise=True,
        )
        if _should_drop_prompt_text(prompt_text):
            prompt_text = ""

        content_text = _clean_batch_text(
            str(row.get("content_to_synthesize", "") or "")
        )
        content_bopomofo = _clean_batch_bopomofo_text(
            str(row.get("content_bopomofo", "") or "")
        )
        content_bopomofo_inline_markup = _clean_batch_text(
            str(row.get("content_bopomofo_inline_markup", "") or "")
            or str(row.get("content_inline_bopomofo_markup", "") or ""),
            preserve_bopomofo_markup=True,
            drop_ascii_noise=False,
        )
        enable_auto_bopomofo = str(
            row.get("enable_auto_bopomofo", "1") or "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
        try:
            speed_scale = float(row.get("speed_scale", 1.0) or 1.0)
        except (TypeError, ValueError):
            speed_scale = 1.0
        if not content_text or (not _VALID_TTS_CONTENT_RE.search(content_text)):
            dropped_rows += 1
            continue

        cleaned_rows.append(
            {
                "speaker_prompt_audio_filename": str(
                    row.get("speaker_prompt_audio_filename", "") or ""
                ).strip(),
                "speaker_prompt_text_transcription": prompt_text,
                "content_to_synthesize": content_text,
                "content_bopomofo": content_bopomofo,
                "content_bopomofo_inline_markup": content_bopomofo_inline_markup,
                "enable_auto_bopomofo": enable_auto_bopomofo,
                "speed_scale": speed_scale,
                "output_audio_filename": str(
                    row.get("output_audio_filename", "") or ""
                ).strip(),
            }
        )
    if dropped_rows:
        print(f"preprocess csv: dropped {dropped_rows} invalid rows")
    return cleaned_rows


def _is_output_up_to_date(output_audio_path, dependency_paths):
    if not os.path.exists(output_audio_path):
        return False

    output_mtime = os.path.getmtime(output_audio_path)
    for dependency_path in dependency_paths:
        if not dependency_path or not os.path.exists(dependency_path):
            continue
        if os.path.getmtime(dependency_path) > output_mtime:
            return False
    return True


def process_batch(
    csv_file,
    speaker_prompt_audio_folder,
    output_audio_folder,
    model,
):
    with open(csv_file, "r", encoding="utf-8-sig", newline="") as f:
        rows = _preprocess_csv_rows(list(csv.DictReader(f)))

    cosyvoice, bopomofo_converter = model

    total_rows = len(rows)
    processed_count = 0
    for row_index, row in enumerate(rows):
        speaker_prompt_audio_path = os.path.join(
            speaker_prompt_audio_folder, f"{row['speaker_prompt_audio_filename']}.wav"
        )
        speaker_prompt_text_transcription = str(
            row.get("speaker_prompt_text_transcription", "") or ""
        )
        content_to_synthesize = str(row.get("content_to_synthesize", "") or "")
        content_bopomofo = str(row.get("content_bopomofo", "") or "")
        content_bopomofo_inline_markup = str(
            row.get("content_bopomofo_inline_markup", "") or ""
        )
        enable_auto_bopomofo = bool(row.get("enable_auto_bopomofo", True))
        speed_scale = float(row.get("speed_scale", 1.0) or 1.0)
        output_audio_path = os.path.join(
            output_audio_folder, f"{row['output_audio_filename']}.wav"
        )

        if not os.path.exists(speaker_prompt_audio_path):
            processed_count += 1
            print(f"\r[{processed_count}/{total_rows}]", end="", flush=True)
            continue

        dependency_paths = [
            csv_file,
            speaker_prompt_audio_path,
            __file__,
            _SINGLE_INFERENCE_PATH,
        ]
        if _is_output_up_to_date(output_audio_path, dependency_paths):
            processed_count += 1
            print(f"\r[{processed_count}/{total_rows}]", end="", flush=True)
            continue

        single_inference(
            speaker_prompt_audio_path,
            content_to_synthesize,
            output_audio_path,
            cosyvoice,
            bopomofo_converter,
            speaker_prompt_text_transcription,
            content_bopomofo,
            content_bopomofo_inline_markup,
            enable_auto_bopomofo=enable_auto_bopomofo,
            speed_scale=speed_scale,
        )
        processed_count += 1
        print(f"\r[{processed_count}/{total_rows}]", end="", flush=True)

    if total_rows:
        print()


def main():
    parser = argparse.ArgumentParser(description="Batch process audio generation.")
    parser.add_argument(
        "--csv_file", required=True, help="Path to the CSV file containing input data."
    )
    parser.add_argument(
        "--speaker_prompt_audio_folder",
        required=True,
        help="Path to the folder containing speaker prompt audio files.",
    )
    parser.add_argument(
        "--output_audio_folder",
        required=True,
        help="Path to the folder where results will be stored.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default="MediaTek-Research/BreezyVoice-300M",
        help="Specifies the model used for speech synthesis.",
    )
    parser.add_argument(
        "--ttsfrd_resource_dir",
        type=str,
        required=False,
        default="",
        help="Optional path to CosyVoice-ttsfrd/resource.",
    )
    args = parser.parse_args()

    cosyvoice = CustomCosyVoice(args.model_path, args.ttsfrd_resource_dir)
    bopomofo_converter = G2PWConverter()

    os.makedirs(args.output_audio_folder, exist_ok=True)

    process_batch(
        csv_file=args.csv_file,
        speaker_prompt_audio_folder=args.speaker_prompt_audio_folder,
        output_audio_folder=args.output_audio_folder,
        model=(cosyvoice, bopomofo_converter),
    )


if __name__ == "__main__":
    main()
