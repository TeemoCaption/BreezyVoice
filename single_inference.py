import argparse
import os
import sys
import re
from functools import partial

import torch

torch.set_num_threads(1)
import torchaudio
import torchaudio.functional as F
import whisper
import opencc
from hyperpyyaml import load_hyperpyyaml
from huggingface_hub import snapshot_download
from g2pw import G2PWConverter

from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.frontend_utils import (
    contains_chinese,
    replace_blank,
    replace_corner_mark,
    remove_bracket,
    spell_out_number,
    split_paragraph,
)
from utils.word_utils import word_to_dataset_frequency, char2phn, always_augment_chars

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/third_party/Matcha-TTS".format(ROOT_DIR))


####new normalize
class CustomCosyVoiceFrontEnd(CosyVoiceFrontEnd):
    def text_normalize_new(self, text, split=False):
        text = text.strip()
        if not text:
            return text

        def split_by_brackets(input_string):
            # Use regex to find text inside and outside the brackets
            inside_brackets = re.findall(r"\[(.*?)\]", input_string)
            outside_brackets = re.split(r"\[.*?\]", input_string)

            # Filter out empty strings from the outside list (result of consecutive brackets)
            outside_brackets = [part for part in outside_brackets if part]

            return inside_brackets, outside_brackets

        def normalize_with_fallback(input_text, is_chinese):
            normalized = ""
            if self.use_ttsfrd:
                try:
                    normalized = self.frd.get_frd_extra_info(input_text, "input") or ""
                    normalized = normalized.strip()
                except Exception as exc:
                    print(
                        f"ttsfrd normalize failed, fallback to WeTextProcessing: {exc}"
                    )
                    normalized = ""

            if normalized:
                return normalized

            if self.use_ttsfrd:
                print(
                    "ttsfrd normalization returned empty, fallback to WeTextProcessing"
                )

            if is_chinese:
                return self.zh_tn_model.normalize(input_text)
            return self.en_tn_model.normalize(input_text)

        def text_normalize_no_split(text, is_last=False):
            text = text.strip()
            if not text:
                return text
            text_is_terminated = text[-1] == "。"
            if contains_chinese(text):
                # print(text)
                text = normalize_with_fallback(text, is_chinese=True)
                if text and not text_is_terminated and not is_last:
                    text = text[:-1]
                # print(text)
                text = text.replace("\n", "")
                text = replace_blank(text)
                text = replace_corner_mark(text)
                text = text.replace(".", "、")
                # print(text)
                text = text.replace(" - ", "，")
                # print(text)
                text = remove_bracket(text)
                # print(text)
                text = re.sub(r"[，,]+$", "。", text)
                # print(text)
            else:
                text = normalize_with_fallback(text, is_chinese=False)
                text = spell_out_number(text, self.inflect_parser)
            return text

        def join_interleaved(outside, inside):
            # Ensure the number of parts match between outside and inside
            result = []

            # Iterate and combine alternating parts
            for o, i in zip(outside, inside):
                result.append(o + "[" + i + "]")

            # Append any remaining part (if outside is longer than inside)
            if len(outside) > len(inside):
                result.append(outside[-1])

            return "".join(result)

        inside_brackets, outside_brackets = split_by_brackets(text)
        # print("io",inside_brackets, outside_brackets)
        # text = re.sub(r'(\[[^\]]*\])(.*?)', normalize_outside_brackets, text)
        # print(text)
        for n in range(len(outside_brackets)):
            e_out = text_normalize_no_split(
                outside_brackets[n], is_last=n == len(outside_brackets) - 1
            )
            outside_brackets[n] = e_out

        text = join_interleaved(outside_brackets, inside_brackets)
        # print()

        # if contains_chinese(text):
        #     texts = [i for i in split_paragraph(
        #         text, partial(self.tokenizer.encode, allowed_special=self.allowed_special),
        #         "zh", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False
        #     )]
        # else:
        #     texts = [i for i in split_paragraph(
        #         text, partial(self.tokenizer.encode, allowed_special=self.allowed_special),
        #         "en", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False
        #     )]

        if split is False:
            return text
        return texts

    def frontend_zero_shot(self, tts_text, prompt_text, prompt_speech_16k):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
        prompt_speech_22050 = torchaudio.transforms.Resample(
            orig_freq=16000, new_freq=22050
        )(prompt_speech_16k)
        speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_22050)
        speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        model_input = {
            "text": tts_text_token,
            "text_len": tts_text_token_len,
            "prompt_text": prompt_text_token,
            "prompt_text_len": prompt_text_token_len,
            "llm_prompt_speech_token": speech_token,
            "llm_prompt_speech_token_len": speech_token_len,
            "flow_prompt_speech_token": speech_token,
            "flow_prompt_speech_token_len": speech_token_len,
            "prompt_speech_feat": speech_feat,
            "prompt_speech_feat_len": speech_feat_len,
            "llm_embedding": embedding,
            "flow_embedding": embedding,
        }
        return model_input

    def frontend_zero_shot_dual(
        self,
        tts_text,
        prompt_text,
        prompt_speech_16k,
        flow_prompt_text,
        flow_prompt_speech_16k,
    ):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
        flow_prompt_text_token, flow_prompt_text_token_len = self._extract_text_token(
            flow_prompt_text
        )
        flow_prompt_speech_22050 = torchaudio.transforms.Resample(
            orig_freq=16000, new_freq=22050
        )(flow_prompt_speech_16k)
        speech_feat, speech_feat_len = self._extract_speech_feat(
            flow_prompt_speech_22050
        )

        flow_speech_token, flow_speech_token_len = self._extract_speech_token(
            flow_prompt_speech_16k
        )
        # speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
        speech_token = flow_speech_token.clone()
        speech_token_len = flow_speech_token_len.clone()
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        # flow_embedding = self._extract_spk_embedding(flow_prompt_speech_16k)
        flow_embedding = embedding.clone()
        model_input = {
            "text": tts_text_token,
            "text_len": tts_text_token_len,
            "prompt_text": prompt_text_token,
            "prompt_text_len": prompt_text_token_len,
            "llm_prompt_speech_token": speech_token,
            "llm_prompt_speech_token_len": speech_token_len,
            "flow_prompt_speech_token": flow_speech_token,
            "flow_prompt_speech_token_len": flow_speech_token_len,
            "prompt_speech_feat": speech_feat,
            "prompt_speech_feat_len": speech_feat_len,
            "llm_embedding": embedding,
            "flow_embedding": flow_embedding,
        }
        return model_input


####model
def _is_cudnn_not_initialized_error(exc):
    return "CUDNN_STATUS_NOT_INITIALIZED" in str(exc).upper()


class CustomCosyVoiceModel(CosyVoiceModel):
    def _run_inference_once(
        self,
        text,
        text_len,
        flow_embedding,
        llm_embedding=torch.zeros(0, 192),
        prompt_text=torch.zeros(1, 0, dtype=torch.int32),
        prompt_text_len=torch.zeros(1, dtype=torch.int32),
        llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        llm_prompt_speech_token_len=torch.zeros(1, dtype=torch.int32),
        flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        flow_prompt_speech_token_len=torch.zeros(1, dtype=torch.int32),
        prompt_speech_feat=torch.zeros(1, 0, 80),
        prompt_speech_feat_len=torch.zeros(1, dtype=torch.int32),
        ):
        tts_speech_token = self.llm.inference(
            text=text.to(self.device),
            text_len=text_len.to(self.device),
            prompt_text=prompt_text.to(self.device),
            prompt_text_len=prompt_text_len.to(self.device),
            prompt_speech_token=llm_prompt_speech_token.to(self.device),
            prompt_speech_token_len=llm_prompt_speech_token_len.to(self.device),
            embedding=llm_embedding.to(self.device),
            beam_size=1,
            sampling=25,
        )

        # input()

        tts_mel = self.flow.inference(
            token=tts_speech_token,
            token_len=torch.tensor([tts_speech_token.size(1)], dtype=torch.int32).to(
                self.device
            ),
            prompt_token=flow_prompt_speech_token.to(self.device),
            prompt_token_len=flow_prompt_speech_token_len.to(self.device),
            prompt_feat=prompt_speech_feat.to(self.device),
            prompt_feat_len=prompt_speech_feat_len.to(self.device),
            embedding=flow_embedding.to(self.device),
        )
        tts_speech = self.hift.inference(mel=tts_mel).cpu()
        torch.cuda.empty_cache()
        return {"tts_speech": tts_speech}

    def inference(
        self,
        text,
        text_len,
        flow_embedding,
        llm_embedding=torch.zeros(0, 192),
        prompt_text=torch.zeros(1, 0, dtype=torch.int32),
        prompt_text_len=torch.zeros(1, dtype=torch.int32),
        llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        llm_prompt_speech_token_len=torch.zeros(1, dtype=torch.int32),
        flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        flow_prompt_speech_token_len=torch.zeros(1, dtype=torch.int32),
        prompt_speech_feat=torch.zeros(1, 0, 80),
        prompt_speech_feat_len=torch.zeros(1, dtype=torch.int32),
    ):
        try:
            return self._run_inference_once(
                text,
                text_len,
                flow_embedding,
                llm_embedding=llm_embedding,
                prompt_text=prompt_text,
                prompt_text_len=prompt_text_len,
                llm_prompt_speech_token=llm_prompt_speech_token,
                llm_prompt_speech_token_len=llm_prompt_speech_token_len,
                flow_prompt_speech_token=flow_prompt_speech_token,
                flow_prompt_speech_token_len=flow_prompt_speech_token_len,
                prompt_speech_feat=prompt_speech_feat,
                prompt_speech_feat_len=prompt_speech_feat_len,
            )
        except RuntimeError as exc:
            if not _is_cudnn_not_initialized_error(exc) or self.device.type != "cuda":
                raise
            cudnn_backend = getattr(torch.backends, "cudnn", None)
            if cudnn_backend is None or not hasattr(cudnn_backend, "enabled"):
                raise
            print("偵測到顯示卡卷積加速初始化失敗，暫時關閉後重試。")
            original_enabled = cudnn_backend.enabled
            try:
                # 只在這一次重試關閉卷積加速，避免單筆資料直接被跳過。
                cudnn_backend.enabled = False
                torch.cuda.empty_cache()
                return self._run_inference_once(
                    text,
                    text_len,
                    flow_embedding,
                    llm_embedding=llm_embedding,
                    prompt_text=prompt_text,
                    prompt_text_len=prompt_text_len,
                    llm_prompt_speech_token=llm_prompt_speech_token,
                    llm_prompt_speech_token_len=llm_prompt_speech_token_len,
                    flow_prompt_speech_token=flow_prompt_speech_token,
                    flow_prompt_speech_token_len=flow_prompt_speech_token_len,
                    prompt_speech_feat=prompt_speech_feat,
                    prompt_speech_feat_len=prompt_speech_feat_len,
                )
            finally:
                cudnn_backend.enabled = original_enabled


###CosyVoice
class CustomCosyVoice:

    def __init__(self, model_dir, ttsfrd_resource_dir: str = ""):
        # assert os.path.exists(model_dir), f"model path '{model_dir}' not exist, please check the path: pretrained_models/CosyVoice-300M-zhtw"
        instruct = False

        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        print("model", model_dir)
        self.model_dir = model_dir

        with open("{}/cosyvoice.yaml".format(model_dir), "r") as f:
            configs = load_hyperpyyaml(f)
        self.frontend = CustomCosyVoiceFrontEnd(
            configs["get_tokenizer"],
            configs["feat_extractor"],
            model_dir,
            "{}/campplus.onnx".format(model_dir),
            "{}/speech_tokenizer_v1.onnx".format(model_dir),
            ttsfrd_resource_dir,
            "{}/spk2info.pt".format(model_dir),
            instruct,
            configs["allowed_special"],
        )
        self.model = CustomCosyVoiceModel(
            configs["llm"], configs["flow"], configs["hift"]
        )
        self.model.load(
            "{}/llm.pt".format(model_dir),
            "{}/flow.pt".format(model_dir),
            "{}/hift.pt".format(model_dir),
        )
        del configs

    def list_avaliable_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def inference_sft(self, tts_text, spk_id):
        tts_speeches = []
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_sft(i, spk_id)
            model_output = self.model.inference(**model_input)
            tts_speeches.append(model_output["tts_speech"])
        return {"tts_speech": torch.concat(tts_speeches, dim=1)}

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False)
        tts_speeches = []
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_zero_shot(
                i, prompt_text, prompt_speech_16k
            )
            model_output = self.model.inference(**model_input)
            tts_speeches.append(model_output["tts_speech"])
        return {"tts_speech": torch.concat(tts_speeches, dim=1)}

    def inference_zero_shot_no_unit_condition_no_normalize(
        self,
        tts_text,
        prompt_text,
        prompt_speech_16k,
        flow_prompt_text=None,
        flow_prompt_speech_16k=None,
    ):
        if flow_prompt_text == None:
            flow_prompt_text = prompt_text
        if flow_prompt_speech_16k == None:
            flow_prompt_speech_16k = prompt_speech_16k
        prompt_text = prompt_text
        model_input = self.frontend.frontend_zero_shot_dual(
            tts_text,
            prompt_text,
            prompt_speech_16k,
            flow_prompt_text,
            flow_prompt_speech_16k,
        )
        model_input["llm_prompt_speech_token"] = model_input["llm_prompt_speech_token"][
            :, :0
        ]
        model_input["llm_prompt_speech_token_len"][0] = 0
        model_output = self.model.inference(**model_input)
        return {"tts_speech": model_output["tts_speech"]}

    def inference_zero_shot_no_normalize(
        self, tts_text, prompt_text, prompt_speech_16k
    ):
        prompt_text = prompt_text
        model_input = self.frontend.frontend_zero_shot(
            tts_text, prompt_text, prompt_speech_16k
        )
        model_output = self.model.inference(**model_input)
        return {"tts_speech": model_output["tts_speech"]}


####wav2text
def transcribe_audio(audio_file):
    # model = whisper.load_model("base")
    # result = model.transcribe(audio_file)
    from transformers import pipeline

    # Load Whisper model
    whisper_asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")

    # Perform ASR on an audio file
    result = whisper_asr(audio_file)

    converter = opencc.OpenCC("s2t")
    traditional_text = converter.convert(result["text"])
    return traditional_text


CONNECTED_CJK_RARE_AUGMENT_THRESHOLD = 500
CONNECTED_CJK_POLYPHONE_AUGMENT_THRESHOLD = 2000
_CONNECTED_CJK_SINGLE_PRONUNCIATION_AUGMENT_THRESHOLD = 20

_CONNECTED_CJK_SHORT_PHRASE_LEN = 12
_QUESTION_CONNECTED_CJK_SHORT_PHRASE_LEN = 20
_QUESTION_TAIL_FALLBACK_MAX_CANDIDATES = 2
_SINGLE_PRONUNCIATION_FALLBACK_THRESHOLD = 100


def _is_cjk_char(char):
    return isinstance(char, str) and len(char) == 1 and "\u3400" <= char <= "\u9fff"


def _get_connected_cjk_span_length(text_w_bopomofo, index):
    start = index
    while start > 0 and _is_cjk_char(text_w_bopomofo[start - 1][0]):
        start -= 1

    end = index
    while end + 1 < len(text_w_bopomofo) and _is_cjk_char(text_w_bopomofo[end + 1][0]):
        end += 1

    return end - start + 1


def _is_question_tail_char(text_w_bopomofo, index, max_cjk_chars=2):
    cjk_count = 0
    for lookahead_index in range(index, len(text_w_bopomofo)):
        lookahead_char = text_w_bopomofo[lookahead_index][0]
        if lookahead_char in "？?":
            return 0 < cjk_count <= max_cjk_chars
        if _is_cjk_char(lookahead_char):
            cjk_count += 1
            continue
        if lookahead_char in "，、,:：":
            return False
    return False


def _is_question_connected_phrase(text_w_bopomofo, index):
    if index < 0 or index >= len(text_w_bopomofo):
        return False
    if not _is_cjk_char(text_w_bopomofo[index][0]):
        return False

    start = index
    while start > 0 and _is_cjk_char(text_w_bopomofo[start - 1][0]):
        start -= 1

    end = index
    while end + 1 < len(text_w_bopomofo) and _is_cjk_char(text_w_bopomofo[end + 1][0]):
        end += 1

    if (end - start + 1) > _QUESTION_CONNECTED_CJK_SHORT_PHRASE_LEN:
        return False

    lookahead_index = end + 1
    while lookahead_index < len(text_w_bopomofo):
        lookahead_char = text_w_bopomofo[lookahead_index][0]
        if lookahead_char in "？?":
            return True
        if lookahead_char in " \t\r\n'\"）)]】〉》」』":
            lookahead_index += 1
            continue
        return False
    return False


def get_bopomofo_rare(text, converter):
    res = converter(text)
    text_w_bopomofo = [x for x in zip(list(text), res[0])]
    reconstructed_text = ""

    for i in range(len(text_w_bopomofo)):
        t = text_w_bopomofo[i]
        char = t[0]
        pronunciation = t[1]
        char_frequency = word_to_dataset_frequency.get(char, 0)
        candidate_pronunciations = char2phn.get(char, [])
        try:
            prev_t_char = text_w_bopomofo[i - 1][0] if i > 0 else None
        except:
            prev_t_char = None
        try:
            next_t_char = text_w_bopomofo[i + 1][0]
        except:
            next_t_char = None
        prev_is_cjk = _is_cjk_char(prev_t_char)
        next_is_cjk = _is_cjk_char(next_t_char)
        in_connected_cjk_phrase = prev_is_cjk or next_is_cjk
        in_embedded_cjk_phrase = prev_is_cjk and next_is_cjk
        is_question_tail_char = _is_question_tail_char(text_w_bopomofo, i)
        in_question_connected_phrase = _is_question_connected_phrase(text_w_bopomofo, i)
        if pronunciation is None and candidate_pronunciations:
            should_use_single_pronunciation_fallback = (
                len(candidate_pronunciations) == 1
                and char_frequency < _SINGLE_PRONUNCIATION_FALLBACK_THRESHOLD
            )
            should_use_question_tail_fallback = (
                is_question_tail_char
                and len(candidate_pronunciations)
                <= _QUESTION_TAIL_FALLBACK_MAX_CANDIDATES
            )
            if (
                should_use_single_pronunciation_fallback
                or should_use_question_tail_fallback
            ):
                pronunciation = candidate_pronunciations[0]
        connected_cjk_span_length = _get_connected_cjk_span_length(text_w_bopomofo, i)
        connected_phrase_protect_limit = (
            _QUESTION_CONNECTED_CJK_SHORT_PHRASE_LEN
            if in_question_connected_phrase
            else _CONNECTED_CJK_SHORT_PHRASE_LEN
        )
        preserve_short_connected_phrase = (
            in_connected_cjk_phrase
            and connected_cjk_span_length <= connected_phrase_protect_limit
        )
        suppress_question_phrase_markup = (
            in_question_connected_phrase and char not in always_augment_chars
        )
        # print(t[0], word_to_dataset_frequency[t[0]], t[1])

        should_force_rare_connected_markup = (
            in_connected_cjk_phrase
            and not preserve_short_connected_phrase
            and pronunciation is not None
            and not suppress_question_phrase_markup
            and char_frequency < CONNECTED_CJK_RARE_AUGMENT_THRESHOLD
            and (
                char in always_augment_chars
                or len(candidate_pronunciations) >= 2
                or char_frequency
                < _CONNECTED_CJK_SINGLE_PRONUNCIATION_AUGMENT_THRESHOLD
            )
        )
        should_force_single_pronunciation_markup = (
            len(candidate_pronunciations) == 1
            and char_frequency < _SINGLE_PRONUNCIATION_FALLBACK_THRESHOLD
            and pronunciation is not None
            and next_t_char != "["
            and not suppress_question_phrase_markup
            and (
                not in_connected_cjk_phrase
                or char in always_augment_chars
                or is_question_tail_char
                or char_frequency
                < _CONNECTED_CJK_SINGLE_PRONUNCIATION_AUGMENT_THRESHOLD
            )
        )
        should_force_question_tail_markup = (
            is_question_tail_char
            and pronunciation is not None
            and next_t_char != "["
            and not suppress_question_phrase_markup
            and len(candidate_pronunciations) <= _QUESTION_TAIL_FALLBACK_MAX_CANDIDATES
        )

        if (
            should_force_single_pronunciation_markup
            or should_force_question_tail_markup
        ):
            reconstructed_text += char + f"[:{pronunciation}]"

        elif (
            char_frequency < CONNECTED_CJK_RARE_AUGMENT_THRESHOLD
            and pronunciation is not None
            and next_t_char != "["
            and not preserve_short_connected_phrase
            and not suppress_question_phrase_markup
            and (not in_connected_cjk_phrase or should_force_rare_connected_markup)
        ):
            # Add the char and the pronunciation
            reconstructed_text += char + f"[:{pronunciation}]"

        elif len(candidate_pronunciations) >= 2:
            should_augment_polyphone = (
                pronunciation != candidate_pronunciations[0]
                and (char_frequency < 10000 or char in always_augment_chars)
                and next_t_char != "["
            )
            should_force_connected_polyphone_markup = (
                in_connected_cjk_phrase
                and not preserve_short_connected_phrase
                and pronunciation is not None
                and char_frequency < CONNECTED_CJK_POLYPHONE_AUGMENT_THRESHOLD
            )
            # Keep phrase-level prosody by avoiding most inline bopomofo markup
            # inside connected Chinese text, but still allow it for explicitly
            # important or unusually rare characters we want to guide.
            if (
                should_augment_polyphone
                and (
                    char in always_augment_chars
                    or should_force_connected_polyphone_markup
                    or not in_embedded_cjk_phrase
                    and not in_connected_cjk_phrase
                )
                and not suppress_question_phrase_markup
            ):
                # Add the char and the pronunciation
                reconstructed_text += char + f"[:{pronunciation}]"
            else:
                reconstructed_text += char
            # print("DEBUG, multiphone char", t[0], char2phn[t[0]])
        else:
            # Add only the char
            reconstructed_text += char

    # print("Reconstructed:", reconstructed_text)
    return reconstructed_text


def _should_augment_chunk_head_char(char, pronunciation, next_char):
    if not _is_cjk_char(char) or pronunciation is None or next_char == "[":
        return False

    char_frequency = word_to_dataset_frequency.get(char, 0)
    candidate_pronunciations = char2phn.get(char, [])
    if char in always_augment_chars:
        return True
    if len(candidate_pronunciations) == 1:
        return char_frequency < _SINGLE_PRONUNCIATION_FALLBACK_THRESHOLD
    if len(candidate_pronunciations) >= 2:
        return (
            pronunciation != candidate_pronunciations[0]
            and char_frequency < CONNECTED_CJK_POLYPHONE_AUGMENT_THRESHOLD
        )
    return char_frequency < CONNECTED_CJK_RARE_AUGMENT_THRESHOLD


def _augment_chunk_head_pronunciation(
    text,
    converter,
    max_cjk_chars=4,
    max_markup_chars=1,
):
    if not text:
        return text

    res = converter(text)
    pronunciations = res[0]
    head_cjk_count = 0
    head_markup_count = 0
    reconstructed = []

    for index, char in enumerate(text):
        pronunciation = pronunciations[index]
        next_char = text[index + 1] if index + 1 < len(text) else None

        if _is_cjk_char(char):
            if (
                head_cjk_count < max_cjk_chars
                and head_markup_count < max_markup_chars
                and _should_augment_chunk_head_char(char, pronunciation, next_char)
            ):
                reconstructed.append(char + f"[:{pronunciation}]")
                head_markup_count += 1
            else:
                reconstructed.append(char)
            head_cjk_count += 1
            continue

        reconstructed.append(char)
        if head_cjk_count > 0:
            break

    if len(reconstructed) < len(text):
        reconstructed.append(text[len(reconstructed) :])
    return "".join(reconstructed)


def _has_inline_bopomofo_markup(text):
    return bool(_BOPOMOFO_MARK_RE.search(str(text or "")))


def _contains_bopomofo_text(text):
    return any(
        "\u3105" <= char <= "\u3129" or char in "˙ˊˇˋ˪˫" for char in str(text or "")
    )


def _longest_connected_cjk_span(text):
    longest = 0
    current = 0
    for char in str(text or ""):
        if _is_cjk_char(char):
            current += 1
            if current > longest:
                longest = current
            continue
        current = 0
    return longest


def _should_preserve_phrase_prosody_for_external_bopomofo(text):
    normalized_text = _normalize_chunk_text(text)
    if not normalized_text or not contains_chinese(normalized_text):
        return False

    # 逐字外掛注音會讓連續中文詞組的韻律變得很碎。
    # 對大多數中文句子，寧可保留原句，交給 auto bopomofo / 稀有字標記
    # 做較少量的引導，也不要把整句轉成每字一個 markup。
    longest_connected_span = _longest_connected_cjk_span(normalized_text)
    if longest_connected_span >= 4:
        return True
    return len(normalized_text) >= 10


def _build_text_with_external_bopomofo(text, bopomofo):
    normalized_text = _normalize_chunk_text(text)
    normalized_bopomofo = _MULTISPACE_RE.sub(" ", str(bopomofo or "")).strip()
    if not normalized_text or not normalized_bopomofo:
        return normalized_text
    if _has_inline_bopomofo_markup(normalized_bopomofo):
        return normalized_bopomofo
    if _should_preserve_phrase_prosody_for_external_bopomofo(normalized_text):
        return normalized_text

    bopomofo_tokens = [token for token in normalized_bopomofo.split(" ") if token]
    if not bopomofo_tokens:
        return normalized_text

    reconstructed = []
    bopomofo_index = 0
    attached_count = 0
    for char in normalized_text:
        if _is_cjk_char(char):
            while bopomofo_index < len(bopomofo_tokens) and not _contains_bopomofo_text(
                bopomofo_tokens[bopomofo_index]
            ):
                bopomofo_index += 1
            if bopomofo_index < len(bopomofo_tokens):
                reconstructed.append(char + f"[:{bopomofo_tokens[bopomofo_index]}]")
                bopomofo_index += 1
                attached_count += 1
            else:
                reconstructed.append(char)
            continue

        if (
            bopomofo_index < len(bopomofo_tokens)
            and bopomofo_tokens[bopomofo_index] == char
        ):
            bopomofo_index += 1
        reconstructed.append(char)

    if attached_count == 0:
        return normalized_text
    return "".join(reconstructed)


def parse_transcript(text, end):
    pattern = r"<\|(\d+\.\d+)\|>([^<]+)<\|(\d+\.\d+)\|>"
    matches = re.findall(pattern, text)

    parsed_output = [
        (float(start), float(end), content.strip()) for start, content, end in matches
    ]
    count0 = 0
    for i in range(len(parsed_output)):
        if parsed_output[i][0] == 0:
            count0 += 1
        if count0 >= 2:
            parsed_output = parsed_output[:i]
            break
    # print("a", parsed_output)
    for i in range(len(parsed_output)):
        if parsed_output[i][0] >= end:
            parsed_output = parsed_output[:i]
            break
    # print("b", parsed_output)
    for i in range(len(parsed_output)):
        if parsed_output[i][0] < end - 15:
            continue
        else:
            parsed_output = parsed_output[i:]
            break
    # print("c", parsed_output)
    start = parsed_output[0][0]
    parsed_output = "".join([p[2] for p in parsed_output])
    return parsed_output, start


_STRONG_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？?!；;])\s*")
_WEAK_SENTENCE_SPLIT_RE = re.compile(r"(?<=[，、,：:])\s*")
_MULTISPACE_RE = re.compile(r"\s+")
_ASCII_WORD_RE = re.compile(r"[A-Za-z]+")
_DIGIT_RE = re.compile(r"\d")
_BOPOMOFO_MARK_RE = re.compile(r"\[:[^\]]+\]")
_CLOSING_QUOTE_BOUNDARY_RE = re.compile(r'(?<=[”」』"])\s*(?=[\u3400-\u9fffA-Za-z0-9])')
_CLOSING_QUOTE_RE = re.compile(r'[”」』"]$')
_LEADING_CONNECTIVE_RE = re.compile(
    r"^(比喻|例如|比如|所以|因此|也就是|換句話說|意思是)"
)
_LEADING_ENUMERATION_RE = re.compile(r"^[零一二三四五六七八九十百\d]+、")
_CJK_PUNCT_TRANSLATION = str.maketrans(
    {
        ",": "，",
        ";": "；",
        ":": "：",
        "?": "？",
        "!": "！",
    }
)
_MAX_TTS_TEXT_TOKEN_LEN = 600
_MAX_PROMPT_TEXT_TOKEN_LEN = 400
_MAX_AUGMENTED_TOKEN_RATIO = 3.0
_SHORT_CHINESE_SENTENCE_CHARS = 80
_DEFAULT_TTS_CHUNK_MAX_CHARS = 120
_UNPUNCTUATED_TTS_CHUNK_MAX_CHARS = 180
_CONSERVATIVE_TTS_CHUNK_MAX_CHARS = 120
_CONSERVATIVE_TTS_CHUNK_MIN_CHARS = 14
_LONG_FORM_TTS_CHUNK_MAX_CHARS = 150
_LONG_FORM_CONSERVATIVE_CHUNK_MAX_CHARS = 140
_LONG_FORM_CONSERVATIVE_CHUNK_MIN_CHARS = 20


def _normalize_chunk_text(text):
    if not isinstance(text, str):
        return ""
    normalized = _MULTISPACE_RE.sub(" ", text).strip()
    if contains_chinese(normalized):
        normalized = normalized.translate(_CJK_PUNCT_TRANSLATION)
    normalized = normalized.replace(" ,", "，").replace(" .", "。")
    normalized = normalized.replace(" ?", "？").replace(" !", "！")
    return normalized


def _has_pause_punctuation(text):
    normalized = _normalize_chunk_text(text)
    if not normalized:
        return False
    return any(char in normalized for char in "。！？?!；;，、,:：")


def _is_long_form_dense_text(text):
    normalized = _normalize_chunk_text(text)
    if not normalized or not contains_chinese(normalized):
        return False

    text_len = len(normalized)
    if text_len < 110:
        return False

    strong_punct_count = sum(normalized.count(char) for char in "。！？?!；;")
    weak_punct_count = sum(normalized.count(char) for char in "，、,:：")
    quote_count = sum(normalized.count(char) for char in '「」『』“”"')
    return (
        strong_punct_count >= 2
        or weak_punct_count >= 6
        or (quote_count >= 4 and weak_punct_count >= 3)
    )


def _resolve_tts_chunk_limits(text):
    if _is_long_form_dense_text(text):
        return (
            _LONG_FORM_TTS_CHUNK_MAX_CHARS,
            _LONG_FORM_CONSERVATIVE_CHUNK_MAX_CHARS,
            _LONG_FORM_CONSERVATIVE_CHUNK_MIN_CHARS,
        )
    return (
        _DEFAULT_TTS_CHUNK_MAX_CHARS,
        _CONSERVATIVE_TTS_CHUNK_MAX_CHARS,
        _CONSERVATIVE_TTS_CHUNK_MIN_CHARS,
    )


def _tighten_long_form_chunk_limit(max_chars):
    try:
        resolved = int(max_chars)
    except (TypeError, ValueError):
        return None
    if resolved <= 0:
        return None

    # 只做小幅收緊，避免把本來能完整念完的中長句切得太碎。
    tightened = int(round(resolved * 0.95))
    return max(_LONG_FORM_CONSERVATIVE_CHUNK_MIN_CHARS, min(resolved, tightened))


def _is_short_fragment_chunk(text):
    normalized = _normalize_chunk_text(text)
    if not normalized:
        return False
    return (
        contains_chinese(normalized)
        and len(normalized) <= 34
        and not _is_enumeration_item_chunk(normalized)
    )


def _is_enumeration_item_chunk(text):
    normalized = _normalize_chunk_text(text)
    if not normalized or not contains_chinese(normalized):
        return False

    body = normalized[:-1] if normalized[-1] in "。！？?!；;" else normalized
    if not _LEADING_ENUMERATION_RE.match(body):
        return False

    return _effective_chunk_text_length(body) > 0


def _is_brief_final_sentence_chunk(text):
    normalized = _normalize_chunk_text(text)
    if not normalized or not contains_chinese(normalized):
        return False
    return len(normalized) <= 14 and normalized[-1] in "。！？?!；;"


def _is_short_chinese_question_chunk(text):
    normalized = _normalize_chunk_text(text)
    if not normalized or not contains_chinese(normalized):
        return False
    if normalized[-1] not in "？?":
        return False

    text_body = normalized[:-1].strip()
    if not text_body:
        return False
    if len(text_body) > 9:
        return False
    if any(ch in text_body for ch in "。！!？?；;，、,:："):
        return False

    cjk_chars = sum(1 for char in text_body if _is_cjk_char(char))
    return cjk_chars >= 4


def _has_question_followup_clause(text, *, chunk_limit=None):
    normalized = _normalize_chunk_text(text)
    if not normalized:
        return False
    if chunk_limit is not None:
        try:
            if len(normalized) > int(chunk_limit):
                return False
        except (TypeError, ValueError):
            return False

    body = normalized[:-1] if normalized[-1] in "。！？.!?；;" else normalized
    if not body:
        return False
    if not any(ch in body for ch in "？?"):
        return False

    strong_punct = "。！？.!?；;"
    if any((ch in strong_punct) and ch not in "？?" for ch in body):
        return False

    question_pos = max(body.rfind("？"), body.rfind("?"))
    if question_pos < 0:
        return False

    tail = body[question_pos + 1 :].strip()
    return bool(tail)


def _should_preserve_final_tail_clause(text):
    normalized = _normalize_chunk_text(text)
    if not normalized or not contains_chinese(normalized):
        return False

    trailing = normalized[-1]
    text_body = normalized[:-1] if trailing in "。！？?!；;，、,:：" else normalized
    if not text_body:
        return False
    if len(normalized) > 24:
        return False

    # 對沒有句內停頓的短中文單句，句尾常會出現較輕的收尾音。
    # 這類尾音若再做最終尾端裁切，很容易把最後 2 到 4 個字直接削掉。
    if _has_brief_final_weak_clause(normalized):
        return True
    return not any(ch in text_body for ch in "。！？?!；;，、,:：")


def _get_text_token_length(frontend, text):
    if not text:
        return 0
    return len(
        frontend.tokenizer.encode(text, allowed_special=frontend.allowed_special)
    )


def _select_safe_text_variant(
    frontend,
    preferred_text,
    fallback_text,
    *,
    max_token_len,
    label,
):
    preferred_len = _get_text_token_length(frontend, preferred_text)
    fallback_len = _get_text_token_length(frontend, fallback_text)
    safe_ratio_limit = max(32, int(max(fallback_len, 1) * _MAX_AUGMENTED_TOKEN_RATIO))

    if preferred_len <= max_token_len and preferred_len <= safe_ratio_limit:
        return preferred_text

    print(
        f"fallback to plain {label}: token_len={preferred_len}, "
        f"plain_token_len={fallback_len}"
    )
    return fallback_text


def _is_attention_length_mismatch(exc):
    message = str(exc)
    return (
        "must match the size of tensor" in message
        and "non-singleton dimension 3" in message
    )


def _estimate_tts_cost(text):
    text = _normalize_chunk_text(text)
    if not text:
        return 0.0

    cost = 0.0
    index = 0
    while index < len(text):
        bopomofo_match = _BOPOMOFO_MARK_RE.match(text, index)
        if bopomofo_match:
            cost += 0.8
            index = bopomofo_match.end()
            continue

        char = text[index]
        if "\u3400" <= char <= "\u9fff":
            cost += 1.0
        elif char in "。！？?!；;":
            cost += 0.2
        elif char in "，、,:：":
            cost += 0.12
        elif char.isdigit():
            cost += 1.3
        elif char.isalpha():
            cost += 0.45
        else:
            cost += 0.15
        index += 1

    ascii_words = len(_ASCII_WORD_RE.findall(text))
    digits = len(_DIGIT_RE.findall(text))
    cost += ascii_words * 0.7 + digits * 0.15
    return cost


def _should_keep_strong_chunk(text, max_chars=120):
    text = _normalize_chunk_text(text)
    if not text:
        return False
    if _should_force_conservative_subsplit(text, max_chars=max_chars):
        return False

    text_len = len(text)
    if text_len <= max_chars:
        return True

    strong_punct_count = sum(text.count(char) for char in "。！？?!；;")
    weak_punct_count = sum(text.count(char) for char in "，、,:：")
    ascii_words = len(_ASCII_WORD_RE.findall(text))
    digits = len(_DIGIT_RE.findall(text))
    estimated_cost = _estimate_tts_cost(text)

    return (
        strong_punct_count <= 1
        and weak_punct_count <= 3
        and ascii_words <= 4
        and digits <= 10
        and text_len <= max_chars + 24
        and estimated_cost <= 180
    )


def _split_long_chunk(text, max_chars=120):
    text = _normalize_chunk_text(text)
    if len(text) <= max_chars:
        return [text] if text else []

    pieces = []
    remainder = text
    while len(remainder) > max_chars:
        split_at = -1
        window = remainder[: max_chars + 1]
        for delimiter in ["；", "：", "，", "、", ";", ":", ",", " "]:
            candidate = window.rfind(delimiter)
            if candidate > split_at:
                split_at = candidate

        if split_at <= 0:
            split_at = max_chars
        else:
            split_at += 1

        piece = _normalize_chunk_text(remainder[:split_at])
        if piece:
            pieces.append(piece)
        remainder = _normalize_chunk_text(remainder[split_at:])

    if remainder:
        pieces.append(remainder)
    return pieces


def _is_punctuation_only_chunk(text):
    normalized = _normalize_chunk_text(text)
    if not normalized:
        return False
    return not any(_is_cjk_char(char) or char.isalnum() for char in normalized)


def _merge_tiny_chunks(chunks, max_chars=60, min_chars=14):
    normalized_chunks = []
    for chunk in chunks:
        chunk = _normalize_chunk_text(chunk)
        if not chunk:
            continue
        normalized_chunks.append(chunk)

    merged_chunks = []
    index = 0
    while index < len(normalized_chunks):
        chunk = normalized_chunks[index]
        if _is_punctuation_only_chunk(chunk):
            if merged_chunks and len(merged_chunks[-1] + chunk) <= max_chars + 2:
                merged_chunks[-1] += chunk
                index += 1
                continue
            if index + 1 < len(normalized_chunks):
                normalized_chunks[index + 1] = chunk + normalized_chunks[index + 1]
                index += 1
                continue
        if len(chunk) >= min_chars:
            merged_chunks.append(chunk)
            index += 1
            continue

        if merged_chunks and len(merged_chunks[-1] + chunk) <= max_chars:
            merged_chunks[-1] += chunk
            index += 1
            continue

        if index + 1 < len(normalized_chunks):
            next_chunk = normalized_chunks[index + 1]
            if len(chunk + next_chunk) <= max_chars:
                merged_chunks.append(chunk + next_chunk)
                index += 2
                continue

        merged_chunks.append(chunk)
        index += 1

    return merged_chunks


def _ensure_sentence_tail(text):
    text = _normalize_chunk_text(text)
    if not text:
        return text
    if text[-1] in "。！？?!；;":
        return text
    if text[-1] in "，、,:：":
        return text
    if contains_chinese(text):
        return text + "。"
    return text + "."


def _merge_split_chunks(chunks, max_chars=120):
    merged_chunks = []
    for chunk in chunks:
        chunk = _normalize_chunk_text(chunk)
        if not chunk:
            continue
        candidate = merged_chunks[-1] + chunk if merged_chunks else chunk
        if (
            merged_chunks
            and len(chunk) < 28
            and len(candidate) <= max_chars
            and _estimate_tts_cost(candidate) <= 160
        ):
            merged_chunks[-1] += chunk
        else:
            merged_chunks.append(chunk)

    connective_merged_chunks = []
    for chunk in merged_chunks:
        chunk = _normalize_chunk_text(chunk)
        if not chunk:
            continue

        if connective_merged_chunks:
            candidate = connective_merged_chunks[-1] + chunk
            if (
                _LEADING_CONNECTIVE_RE.match(chunk)
                and len(candidate) <= max_chars + 20
                and _estimate_tts_cost(candidate) <= 165
            ):
                connective_merged_chunks[-1] += chunk
                continue

        connective_merged_chunks.append(chunk)

    return connective_merged_chunks


def _merge_connective_sentences(chunks, max_chars=120):
    merged_chunks = []
    for chunk in chunks:
        chunk = _normalize_chunk_text(chunk)
        if not chunk:
            continue

        if merged_chunks:
            previous_chunk = merged_chunks[-1]
            candidate = previous_chunk + chunk
            if (
                previous_chunk[-1] in "。！？?!；;"
                and _LEADING_CONNECTIVE_RE.match(chunk)
                and len(candidate) <= max_chars
                and _estimate_tts_cost(candidate) <= 170
            ):
                merged_chunks[-1] = candidate
                continue

        merged_chunks.append(chunk)

    return merged_chunks


def _split_tts_content_with_frontend(
    text,
    frontend,
    token_max_n=80,
    token_min_n=60,
    merge_len=20,
):
    normalized = _normalize_chunk_text(text)
    if not normalized:
        return []

    raw_chunks = split_paragraph(
        normalized,
        partial(frontend.tokenizer.encode, allowed_special=frontend.allowed_special),
        "zh",
        token_max_n=token_max_n,
        token_min_n=token_min_n,
        merge_len=merge_len,
        comma_split=False,
    )

    chunks = []
    for raw_chunk in raw_chunks:
        chunk = _normalize_chunk_text(raw_chunk)
        if chunk:
            chunks.append(chunk)

    if not chunks:
        return []

    return [_ensure_sentence_tail(chunk) for chunk in chunks]


def _split_chunk_conservatively(text, max_chars=60, min_chars=14):
    normalized = _normalize_chunk_text(text)
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    chunks = [normalized]
    split_patterns = (
        _STRONG_SENTENCE_SPLIT_RE,
        _WEAK_SENTENCE_SPLIT_RE,
    )
    for split_pattern in split_patterns:
        next_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                next_chunks.append(chunk)
                continue
            split_parts = [
                _normalize_chunk_text(part)
                for part in split_pattern.split(chunk)
                if _normalize_chunk_text(part)
            ]
            if len(split_parts) <= 1:
                next_chunks.append(chunk)
                continue
            next_chunks.extend(split_parts)
        chunks = next_chunks

    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
            continue
        final_chunks.extend(_split_long_chunk(chunk, max_chars=max_chars))

    return _merge_tiny_chunks(
        final_chunks,
        max_chars=max_chars,
        min_chars=min_chars,
    )


def _stabilize_frontend_content_chunks(chunks, max_chars=120, min_chars=14):
    stabilized_chunks = []
    for chunk in chunks:
        chunk = _normalize_chunk_text(chunk)
        if not chunk:
            continue
        if contains_chinese(chunk) and len(chunk) > max_chars:
            stabilized_chunks.extend(
                _split_chunk_conservatively(
                    chunk,
                    max_chars=max_chars,
                    min_chars=min_chars,
                )
            )
            continue
        stabilized_chunks.append(chunk)
    return stabilized_chunks


def _resolve_content_chunks(frontend, preferred_tts_text, max_chunk_chars=None):
    if _has_inline_bopomofo_markup(preferred_tts_text):
        normalized_text = preferred_tts_text
    else:
        normalized_text = frontend.text_normalize(preferred_tts_text, split=False)
    normalized_text = _normalize_chunk_text(normalized_text)
    if not normalized_text:
        return []

    (
        chunk_max_chars,
        conservative_max_chars,
        conservative_min_chars,
    ) = _resolve_tts_chunk_limits(normalized_text)

    requested_max_chunk_chars = _resolve_requested_max_chunk_chars(max_chunk_chars)
    if requested_max_chunk_chars is not None:
        chunk_max_chars = min(chunk_max_chars, requested_max_chunk_chars)
        conservative_max_chars = min(conservative_max_chars, requested_max_chunk_chars)
        if _is_long_form_dense_text(normalized_text):
            chunk_max_chars = _tighten_long_form_chunk_limit(chunk_max_chars)
            conservative_max_chars = _tighten_long_form_chunk_limit(
                conservative_max_chars
            )

    content_chunks = _split_tts_content(
        normalized_text,
        max_chars=chunk_max_chars,
    )
    if not content_chunks:
        content_chunks = [normalized_text]

    return _stabilize_frontend_content_chunks(
        content_chunks,
        max_chars=conservative_max_chars,
        min_chars=conservative_min_chars,
    )


def _split_quote_tail_chunks(text):
    text = _normalize_chunk_text(text)
    if not text:
        return []

    raw_chunks = [chunk for chunk in _CLOSING_QUOTE_BOUNDARY_RE.split(text) if chunk]
    if len(raw_chunks) <= 1:
        return [text]

    chunks = []
    for raw_chunk in raw_chunks:
        chunk = _normalize_chunk_text(raw_chunk)
        if not chunk:
            continue
        if chunks and _CLOSING_QUOTE_RE.search(chunks[-1]):
            prev_chunk = chunks[-1]
            prev_cjk_chars = sum(1 for char in prev_chunk if _is_cjk_char(char))
            prev_has_internal_pause = any(
                punct in prev_chunk for punct in "，、,:：；;。！？?!"
            )
            if prev_cjk_chars < 6 and not prev_has_internal_pause:
                chunks[-1] = prev_chunk + chunk
                continue
        chunks.append(chunk)

    return chunks or [text]


def _split_tts_content(text, max_chars=120):
    normalized = _normalize_chunk_text(text)
    if not normalized:
        return []

    # 問號後面如果還有接續內容，且整句仍在上限內，就先保留整句。
    # 這樣可以避免強標點先切成兩段，後面再疊出過長停頓。
    if max_chars is not None:
        try:
            resolved_max_chars = int(max_chars)
        except (TypeError, ValueError):
            resolved_max_chars = None
        else:
            if resolved_max_chars > 0 and len(normalized) <= resolved_max_chars:
                if _has_question_followup_clause(
                    normalized,
                    chunk_limit=resolved_max_chars,
                ):
                    return [normalized]

    rough_chunks = []
    for quote_chunk in _split_quote_tail_chunks(normalized):
        rough_chunks.extend(
            chunk for chunk in _STRONG_SENTENCE_SPLIT_RE.split(quote_chunk) if chunk
        )
    if not rough_chunks:
        rough_chunks = [normalized]

    chunks = []
    for rough_chunk in rough_chunks:
        rough_chunk = _normalize_chunk_text(rough_chunk)
        if not rough_chunk:
            continue

        rough_cost = _estimate_tts_cost(rough_chunk)
        ascii_words = len(_ASCII_WORD_RE.findall(rough_chunk))
        digits = len(_DIGIT_RE.findall(rough_chunk))
        force_conservative_subsplit = _should_force_conservative_subsplit(
            rough_chunk,
            max_chars=max_chars,
        )

        if (
            len(rough_chunk) <= max_chars
            and rough_cost <= 165
            and ascii_words <= 4
            and digits <= 10
            and not force_conservative_subsplit
        ):
            chunks.append(rough_chunk)
            continue

        sentence_chunks = []
        if _should_keep_strong_chunk(rough_chunk, max_chars=max_chars):
            sentence_chunks.append(rough_chunk)
            chunks.extend(_merge_split_chunks(sentence_chunks, max_chars=max_chars))
            continue

        weak_chunks = [
            chunk for chunk in _WEAK_SENTENCE_SPLIT_RE.split(rough_chunk) if chunk
        ]
        if len(weak_chunks) <= 1:
            chunks.extend(
                _merge_split_chunks(
                    _split_long_chunk(rough_chunk, max_chars=max_chars),
                    max_chars=max_chars,
                )
            )
            continue

        preserve_final_tail_chunk = _has_brief_final_weak_clause(rough_chunk)
        current = ""
        current_cost = 0.0
        sentence_target_chars = min(
            max_chars,
            max(22, len(rough_chunk) // 2 + 6),
        )
        if force_conservative_subsplit:
            sentence_target_chars = min(
                sentence_target_chars,
                max(14, int(round(len(rough_chunk) * 0.7))),
            )
        for weak_index, weak_chunk in enumerate(weak_chunks):
            weak_chunk = _normalize_chunk_text(weak_chunk)
            if not weak_chunk:
                continue

            weak_cost = _estimate_tts_cost(weak_chunk)
            is_final_weak_chunk = weak_index == len(weak_chunks) - 1
            if preserve_final_tail_chunk and is_final_weak_chunk and current:
                sentence_chunks.append(current)
                current = weak_chunk
                current_cost = weak_cost
                continue

            candidate = weak_chunk if not current else current + weak_chunk
            candidate_cost = _estimate_tts_cost(candidate)
            if len(candidate) <= sentence_target_chars and candidate_cost <= 150:
                current = candidate
                current_cost = candidate_cost
                continue

            # 短片段（< 6 字）併入目前區塊，避免產生孤兒片段造成不自然斷句
            if (
                current
                and len(weak_chunk) < 6
                and len(candidate) <= max_chars
                and candidate_cost <= 165
            ):
                current = candidate
                current_cost = candidate_cost
                continue

            if current:
                sentence_chunks.append(current)
            if len(weak_chunk) <= max_chars and weak_cost <= 150:
                current = weak_chunk
                current_cost = weak_cost
            else:
                sentence_chunks.extend(
                    _split_long_chunk(weak_chunk, max_chars=max_chars)
                )
                current = ""
                current_cost = 0.0

        if current:
            sentence_chunks.append(current)

        merge_max_chars = max(sentence_target_chars + 10, 32)
        if force_conservative_subsplit:
            merge_max_chars = max(12, sentence_target_chars)
        if preserve_final_tail_chunk and len(sentence_chunks) >= 2:
            chunks.extend(
                _merge_split_chunks(
                    sentence_chunks[:-1],
                    max_chars=merge_max_chars,
                )
            )
            chunks.append(sentence_chunks[-1])
            continue
        chunks.extend(
            _merge_split_chunks(
                sentence_chunks,
                max_chars=merge_max_chars,
            )
        )

    return [
        _ensure_sentence_tail(chunk)
        for chunk in _merge_connective_sentences(chunks, max_chars=max_chars)
    ]


def _pause_samples_for_chunk(text, sample_rate=22050):
    trailing = text[-1] if text else ""
    if _is_short_fragment_chunk(text):
        if trailing in "。！？?!；;":
            return int(sample_rate * 0.055)
        if trailing in "，、,:：":
            return int(sample_rate * 0.02)
        return int(sample_rate * 0.012)
    if trailing in "。！？?!；;":
        return int(sample_rate * 0.09)
    if trailing in "，、,:：":
        return int(sample_rate * 0.035)
    return int(sample_rate * 0.015)


def _chunk_keep_trailing_sec(text, is_last_chunk):
    trailing = text[-1] if text else ""
    if _is_brief_final_sentence_chunk(text) and not is_last_chunk:
        if trailing in "。！？?!；;":
            return 0.16
        if trailing in "，、,:：":
            return 0.1
        return 0.08
    if _is_short_fragment_chunk(text):
        if is_last_chunk:
            if _is_brief_final_sentence_chunk(text):
                if trailing in "。！？?!；;":
                    return 0.2
                if trailing in "，、,:：":
                    return 0.12
                return 0.1
            if trailing in "。！？?!；;":
                return 0.08
            if trailing in "，、,:：":
                return 0.055
            return 0.045
        if trailing in "。！？?!；;":
            return 0.08
        if trailing in "，、,:：":
            return 0.055
        return 0.045
    if is_last_chunk:
        if trailing in "。！？?!；;":
            return 0.075
        if trailing in "，、,:：":
            return 0.055
        return 0.045
    if trailing in "。！？?!；;":
        return 0.12
    if trailing in "，、,:：":
        return 0.08
    return 0.05


def _resolve_speed_scale(speed_scale):
    try:
        resolved = float(speed_scale)
    except (TypeError, ValueError):
        return 1.0
    return min(max(resolved, 0.7), 1.3)


def _apply_speed_scale(waveform, sample_rate, speed_scale):
    resolved_speed = _resolve_speed_scale(speed_scale)
    if waveform.numel() == 0 or abs(resolved_speed - 1.0) < 1e-4:
        return waveform

    target_sample_rate = max(8000, int(round(sample_rate / resolved_speed)))
    if target_sample_rate == sample_rate:
        return waveform
    return F.resample(waveform, sample_rate, target_sample_rate)


def _trim_waveform_silence(
    waveform,
    sample_rate=22050,
    threshold=0.003,
    keep_leading_sec=0.02,
    keep_trailing_sec=0.08,
    trim_leading=True,
    trim_trailing=True,
):
    if waveform.numel() == 0:
        return waveform

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    amplitude = waveform.abs().amax(dim=0)
    peak_amplitude = float(amplitude.max().item()) if amplitude.numel() else 0.0
    effective_threshold = min(threshold, max(0.0008, peak_amplitude * 0.08))
    voiced_indices = torch.nonzero(
        amplitude > effective_threshold, as_tuple=False
    ).flatten()
    if voiced_indices.numel() == 0:
        return waveform

    keep_leading = int(sample_rate * keep_leading_sec)
    keep_trailing = int(sample_rate * keep_trailing_sec)
    if trim_leading:
        start = max(0, int(voiced_indices[0].item()) - keep_leading)
    else:
        start = 0

    if trim_trailing:
        end = min(waveform.shape[1], int(voiced_indices[-1].item()) + keep_trailing + 1)
    else:
        end = waveform.shape[1]

    return waveform[:, start:end]


def _resolve_internal_silence_compression_config(text, *, chunk_count):
    normalized = _normalize_chunk_text(text)
    if not normalized:
        return {
            "enabled": False,
        }

    config = {
        "enabled": True,
        "threshold_ratio": 0.006,
        "smoothing_window_sec": 0.012,
        "max_internal_silence_sec": 0.28,
        "target_internal_silence_sec": 0.09,
        "min_region_sec": 0.08,
        "min_side_region_sec": 0.12,
        "max_compressions": 1,
    }

    text_body = (
        normalized[:-1] if normalized[-1] in "。！？?!；;，、,:：" else normalized
    )
    has_weak_pause = any(ch in text_body for ch in "，、,:：")
    has_question_followup_clause = _has_question_followup_clause(normalized)

    if has_weak_pause:
        return {
            "enabled": False,
        }

    if has_question_followup_clause:
        config.update(
            {
                # 問號後還有接續內容時，保留一點比較像人在說話的停頓。
                # 仍保留壓縮機制，避免再次出現過長空白。
                "max_internal_silence_sec": max(
                    config["max_internal_silence_sec"], 0.45
                ),
                "target_internal_silence_sec": max(
                    config["target_internal_silence_sec"], 0.16
                ),
                "min_side_region_sec": min(config["min_side_region_sec"], 0.1),
            }
        )

    if normalized[-1] in "？?":
        config.update(
            {
                "max_internal_silence_sec": min(
                    config["max_internal_silence_sec"], 0.22
                ),
                "target_internal_silence_sec": 0.08,
                "min_side_region_sec": max(config["min_side_region_sec"], 0.14),
            }
        )

    if _is_short_fragment_chunk(normalized) and not _is_enumeration_item_chunk(
        normalized
    ):
        config.update(
            {
                "max_internal_silence_sec": min(
                    config["max_internal_silence_sec"], 0.2
                ),
                "target_internal_silence_sec": min(
                    config["target_internal_silence_sec"], 0.075
                ),
                "min_region_sec": max(config["min_region_sec"], 0.1),
                "min_side_region_sec": max(config["min_side_region_sec"], 0.16),
            }
        )

    if chunk_count == 1 and _should_preserve_final_tail_clause(normalized):
        config.update(
            {
                "enabled": False,
            }
        )

    return config


def _compress_internal_silence(
    waveform,
    sample_rate=22050,
    *,
    threshold_ratio=0.006,
    smoothing_window_sec=0.012,
    max_internal_silence_sec=0.16,
    target_internal_silence_sec=0.055,
    min_region_sec=0.05,
    min_side_region_sec=0.08,
    max_compressions=1,
):
    if waveform.numel() == 0:
        return waveform
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    amplitude = waveform.abs().amax(dim=0)
    if amplitude.numel() == 0:
        return waveform

    peak = float(amplitude.max().item())
    if peak <= 1e-8:
        return waveform

    frame_samples = max(1, int(round(float(smoothing_window_sec) * sample_rate)))
    if frame_samples > 1:
        smoothed = torch.nn.functional.avg_pool1d(
            amplitude.view(1, 1, -1),
            kernel_size=frame_samples,
            stride=1,
            padding=frame_samples // 2,
        ).view(-1)
        if smoothed.shape[0] > amplitude.shape[0]:
            smoothed = smoothed[: amplitude.shape[0]]
    else:
        smoothed = amplitude

    threshold = max(7e-5, peak * float(threshold_ratio))
    active_indices = torch.nonzero(smoothed >= threshold, as_tuple=False).flatten()
    if active_indices.numel() < 2:
        return waveform

    regions = []
    region_start = int(active_indices[0].item())
    region_end = int(active_indices[0].item())
    for idx in active_indices[1:]:
        idx = int(idx.item())
        if idx <= region_end + 1:
            region_end = idx
            continue
        regions.append((region_start, region_end))
        region_start = idx
        region_end = idx
    regions.append((region_start, region_end))

    if len(regions) < 2:
        return waveform

    max_gap_samples = max(1, int(round(float(max_internal_silence_sec) * sample_rate)))
    target_gap_samples = max(
        0,
        min(
            max_gap_samples,
            int(round(float(target_internal_silence_sec) * sample_rate)),
        ),
    )
    min_region_samples = max(1, int(round(float(min_region_sec) * sample_rate)))
    min_side_region_samples = max(
        1, int(round(float(min_side_region_sec) * sample_rate))
    )

    compression_budget = max(0, int(max_compressions))
    if compression_budget == 0:
        return waveform

    candidate_gap_lengths = []
    for region_index in range(1, len(regions)):
        prev_start, prev_end = regions[region_index - 1]
        current_start, current_end = regions[region_index]
        gap_length = current_start - prev_end - 1
        prev_region_length = prev_end - prev_start + 1
        current_region_length = current_end - current_start + 1
        if (
            gap_length > max_gap_samples
            and gap_length > target_gap_samples
            and prev_region_length >= min_region_samples
            and current_region_length >= min_region_samples
            and prev_region_length >= min_side_region_samples
            and current_region_length >= min_side_region_samples
        ):
            candidate_gap_lengths.append((gap_length, region_index))

    if not candidate_gap_lengths:
        return waveform

    compress_region_indices = {
        region_index
        for _, region_index in sorted(candidate_gap_lengths, reverse=True)[
            :compression_budget
        ]
    }

    changed = False
    rebuilt_segments = [waveform[:, : regions[0][1] + 1]]

    for region_index in range(1, len(regions)):
        prev_start, prev_end = regions[region_index - 1]
        current_start, current_end = regions[region_index]
        gap_start = prev_end + 1
        gap_end = current_start
        gap_length = current_start - prev_end - 1
        prev_region_length = prev_end - prev_start + 1
        current_region_length = current_end - current_start + 1

        should_compress = region_index in compress_region_indices

        if should_compress:
            silence = torch.zeros(
                (waveform.shape[0], target_gap_samples),
                dtype=waveform.dtype,
                device=waveform.device,
            )
            rebuilt_segments.append(silence)
            changed = True
        else:
            rebuilt_segments.append(waveform[:, gap_start:gap_end])

        rebuilt_segments.append(waveform[:, current_start : current_end + 1])

    rebuilt_segments.append(waveform[:, regions[-1][1] + 1 :])
    if not changed:
        return waveform
    return torch.cat(rebuilt_segments, dim=1)


def _append_tail_silence(
    waveform,
    sample_rate=22050,
    tail_sec=0.28,
    fade_out_sec=0.02,
):
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if waveform.numel() == 0:
        return waveform

    fade_samples = min(int(sample_rate * fade_out_sec), waveform.shape[1])
    if fade_samples > 1:
        fade = torch.linspace(
            1.0,
            0.0,
            steps=fade_samples,
            dtype=waveform.dtype,
            device=waveform.device,
        ).unsqueeze(0)
        waveform = waveform.clone()
        waveform[:, -fade_samples:] = waveform[:, -fade_samples:] * fade

    tail = torch.zeros(
        (waveform.shape[0], int(sample_rate * tail_sec)),
        dtype=waveform.dtype,
        device=waveform.device,
    )
    return torch.cat([waveform, tail], dim=1)


def _trim_final_tail_artifact(
    waveform,
    sample_rate=22050,
    max_trim_sec=0.35,
    keep_padding_sec=0.01,
    threshold_ratio=0.01,
):
    if waveform.numel() == 0:
        return waveform
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    amplitude = waveform.abs().amax(dim=0)
    if amplitude.numel() == 0:
        return waveform

    peak = float(amplitude.max().item())
    if peak <= 1e-8:
        return waveform

    frame_samples = max(1, int(round(sample_rate * 0.012)))
    if frame_samples > 1:
        smoothed = torch.nn.functional.avg_pool1d(
            amplitude.view(1, 1, -1),
            kernel_size=frame_samples,
            stride=1,
            padding=frame_samples // 2,
        ).view(-1)
        if smoothed.shape[0] > amplitude.shape[0]:
            smoothed = smoothed[: amplitude.shape[0]]
    else:
        smoothed = amplitude

    threshold = max(7e-5, peak * float(threshold_ratio))
    active_indices = torch.nonzero(smoothed >= threshold, as_tuple=False).flatten()
    if active_indices.numel() == 0:
        return waveform

    last_active = int(active_indices[-1].item())
    keep_padding = int(round(float(keep_padding_sec) * sample_rate))
    max_trim = int(round(float(max_trim_sec) * sample_rate))
    min_end_index = max(0, waveform.shape[1] - max_trim - 1)
    target_end_index = min(waveform.shape[1] - 1, last_active + keep_padding)
    final_end_index = max(min_end_index, target_end_index)
    if final_end_index >= waveform.shape[1] - 1:
        return waveform
    return waveform[:, : final_end_index + 1]


def _chunk_trim_threshold(text, is_last_chunk):
    if _is_brief_final_sentence_chunk(text):
        trailing = text[-1] if text else ""
        if not is_last_chunk:
            if trailing in "。！？?!；;":
                return 0.0011
            if trailing in "，、,:：":
                return 0.00105
            return 0.00105
        if trailing in "。！？?!；;":
            return 0.0015
        if trailing in "，、,:：":
            return 0.0014
        return 0.0014
    if not is_last_chunk:
        return 0.0018
    trailing = text[-1] if text else ""
    if trailing in "。！？?!；;":
        return 0.0028
    if trailing in "，、,:：":
        return 0.0025
    return 0.0023


def _resolve_chunk_trim_threshold(text, *, chunk_index, chunk_count):
    is_last_chunk = chunk_index == chunk_count
    if chunk_count == 1 and _is_brief_final_sentence_chunk(text):
        return 0.0009
    if chunk_index == 1:
        return 0.0012
    return _chunk_trim_threshold(text, is_last_chunk)


def _resolve_final_tail_trim_config(text):
    if _is_brief_final_sentence_chunk(text):
        return {
            "enabled": False,
        }
    if _should_preserve_final_tail_clause(text):
        return {
            "enabled": False,
        }
    if _is_short_fragment_chunk(text):
        return {
            "enabled": True,
            "max_trim_sec": 0.32,
            "keep_padding_sec": 0.02,
            "threshold_ratio": 0.011,
        }
    return {
        "enabled": True,
        "max_trim_sec": 0.38,
        "keep_padding_sec": 0.01,
        "threshold_ratio": 0.014,
    }


def _resolve_tail_append_config(text):
    if _is_brief_final_sentence_chunk(text):
        return {
            "tail_sec": 0.36,
            "fade_out_sec": 0.0,
        }
    if _should_preserve_final_tail_clause(text):
        return {
            "tail_sec": 0.36,
            "fade_out_sec": 0.0,
        }
    if _is_short_fragment_chunk(text):
        return {
            "tail_sec": 0.34,
            "fade_out_sec": 0.01,
        }
    return {
        "tail_sec": 0.3,
        "fade_out_sec": 0.018,
    }


def _get_detached_brief_sentence_tail_region(
    waveform,
    sample_rate=22050,
    *,
    threshold_ratio=0.012,
    min_gap_sec=0.02,
    max_gap_sec=0.12,
    min_tail_region_sec=0.02,
    max_tail_region_sec=0.1,
    min_primary_region_sec=0.4,
    min_tail_start_ratio=0.82,
):
    if waveform.numel() == 0:
        return None
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    amplitude = waveform.abs().amax(dim=0)
    if amplitude.numel() == 0:
        return None

    peak = float(amplitude.max().item())
    if peak <= 1e-8:
        return None

    frame_samples = max(1, int(round(sample_rate * 0.01)))
    if frame_samples > 1:
        smoothed = torch.nn.functional.avg_pool1d(
            amplitude.view(1, 1, -1),
            kernel_size=frame_samples,
            stride=1,
            padding=frame_samples // 2,
        ).view(-1)
        if smoothed.shape[0] > amplitude.shape[0]:
            smoothed = smoothed[: amplitude.shape[0]]
    else:
        smoothed = amplitude

    threshold = max(7e-5, peak * float(threshold_ratio))
    active_indices = torch.nonzero(smoothed >= threshold, as_tuple=False).flatten()
    if active_indices.numel() < 2:
        return None

    regions = []
    region_start = int(active_indices[0].item())
    region_end = int(active_indices[0].item())
    for idx in active_indices[1:]:
        idx = int(idx.item())
        if idx <= region_end + 1:
            region_end = idx
            continue
        regions.append((region_start, region_end))
        region_start = idx
        region_end = idx
    regions.append((region_start, region_end))
    if len(regions) < 2:
        return None

    primary_start, primary_end = regions[-2]
    tail_start, tail_end = regions[-1]
    primary_region_sec = float(primary_end - primary_start + 1) / float(sample_rate)
    tail_region_sec = float(tail_end - tail_start + 1) / float(sample_rate)
    gap_sec = float(tail_start - primary_end - 1) / float(sample_rate)
    tail_start_ratio = float(tail_start) / float(waveform.shape[1])

    if primary_region_sec < float(min_primary_region_sec):
        return None
    if gap_sec < float(min_gap_sec) or gap_sec > float(max_gap_sec):
        return None
    if tail_region_sec < float(min_tail_region_sec):
        return None
    if tail_region_sec > float(max_tail_region_sec):
        return None
    if tail_start_ratio < float(min_tail_start_ratio):
        return None
    return tail_start, tail_end


def _has_detached_brief_sentence_tail(
    waveform,
    sample_rate=22050,
    **kwargs,
):
    return (
        _get_detached_brief_sentence_tail_region(
            waveform,
            sample_rate=sample_rate,
            **kwargs,
        )
        is not None
    )


def _append_detached_brief_sentence_tail_release(
    waveform,
    sample_rate=22050,
    *,
    peak_lead_in_sec=0.01,
    release_sec=0.04,
):
    region = _get_detached_brief_sentence_tail_region(
        waveform,
        sample_rate=sample_rate,
    )
    if region is None:
        return waveform
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    tail_start, tail_end = region
    tail_region = waveform[:, tail_start : tail_end + 1]
    if tail_region.shape[1] < 8:
        return waveform

    tail_amplitude = tail_region.abs().amax(dim=0)
    peak_offset = int(torch.argmax(tail_amplitude).item())
    peak_lead_in_samples = max(4, int(round(float(peak_lead_in_sec) * sample_rate)))
    source_start_offset = max(0, peak_offset - peak_lead_in_samples)
    source = tail_region[:, source_start_offset:]
    if source.shape[1] < 8:
        return waveform

    release_samples = max(source.shape[1], int(round(float(release_sec) * sample_rate)))
    release = torch.nn.functional.interpolate(
        source.unsqueeze(0),
        size=release_samples,
        mode="linear",
        align_corners=False,
    ).squeeze(0)
    fade = torch.linspace(
        0.92,
        0.0,
        steps=release_samples,
        dtype=release.dtype,
        device=release.device,
    ).unsqueeze(0)
    release = release * fade
    return torch.cat([waveform, release], dim=1)


def _resolve_effective_chunk_speed_scale(
    text,
    speed_scale,
    *,
    chunk_count,
    waveform=None,
):
    resolved_speed = _resolve_speed_scale(speed_scale)
    if chunk_count == 1 and _has_question_followup_clause(text) and resolved_speed > 1.0:
        return 1.0
    if _is_short_chinese_question_chunk(text) and resolved_speed > 1.0:
        return 1.0
    if chunk_count != 1 or not _is_brief_final_sentence_chunk(text):
        return resolved_speed
    if resolved_speed <= 1.0:
        return resolved_speed
    if waveform is not None and _has_detached_brief_sentence_tail(waveform):
        return 1.0
    return min(resolved_speed, 1.02)


def _resolve_requested_max_chunk_chars(max_chunk_chars):
    try:
        resolved = int(max_chunk_chars)
    except (TypeError, ValueError):
        return None
    if resolved <= 0:
        return None
    return max(24, resolved)


def _effective_chunk_text_length(text):
    normalized = _normalize_chunk_text(text)
    if not normalized:
        return 0
    normalized = _BOPOMOFO_MARK_RE.sub("", normalized)
    return sum(1 for char in normalized if char not in "。！？?!；;，、,:： ")


def _split_internal_weak_clause_spans(text):
    normalized = _normalize_chunk_text(text)
    if not normalized:
        return []
    body = normalized[:-1] if normalized[-1] in "。！？?!；;" else normalized
    return [
        segment.strip()
        for segment in _WEAK_SENTENCE_SPLIT_RE.split(body)
        if segment and segment.strip()
    ]


def _has_brief_final_weak_clause(text):
    normalized = _normalize_chunk_text(text)
    if not normalized or not contains_chinese(normalized):
        return False

    spans = _split_internal_weak_clause_spans(normalized)
    if len(spans) < 2:
        return False

    weak_punct_count = sum(normalized.count(char) for char in "，、,:：")
    if weak_punct_count <= 0:
        return False

    if weak_punct_count < 2 and len(spans) < 3:
        return False

    total_len = _effective_chunk_text_length(normalized)
    if total_len < 18 or total_len > 42:
        return False

    tail_len = _effective_chunk_text_length(spans[-1])
    head_len = sum(_effective_chunk_text_length(span) for span in spans[:-1])
    tail_limit = max(3, min(6, int(round(total_len * 0.28))))
    return tail_len > 0 and tail_len <= tail_limit and head_len >= max(10, tail_len + 3)


def _is_weak_punctuation_dense_sentence(text):
    normalized = _normalize_chunk_text(text)
    if not normalized or not contains_chinese(normalized):
        return False

    body = normalized[:-1] if normalized[-1] in "。！？?!；;" else normalized
    if not body:
        return False

    weak_punct_count = sum(body.count(char) for char in "，、,:：")
    effective_len = _effective_chunk_text_length(body)
    if effective_len < 18:
        return False
    if weak_punct_count >= 5:
        return True
    if weak_punct_count >= 4 and effective_len >= 22:
        return True
    return (
        weak_punct_count >= 3
        and effective_len >= 18
        and bool(_LEADING_ENUMERATION_RE.match(body))
    )


def _should_force_conservative_subsplit(text, max_chars=120):
    normalized = _normalize_chunk_text(text)
    if not normalized or not contains_chinese(normalized):
        return False
    effective_len = _effective_chunk_text_length(normalized)
    if effective_len <= 14:
        return False
    conservative_trigger_len = max(48, int(round(max_chars * 0.9)))
    if effective_len < conservative_trigger_len:
        return False
    return _is_weak_punctuation_dense_sentence(
        normalized
    ) or _has_brief_final_weak_clause(normalized) or _is_long_form_dense_text(normalized)


def single_inference(
    speaker_prompt_audio_path,
    content_to_synthesize,
    output_path,
    cosyvoice,
    bopomofo_converter,
    speaker_prompt_text_transcription=None,
    content_bopomofo=None,
    content_bopomofo_inline_markup=None,
    enable_auto_bopomofo=True,
    speed_scale=1.0,
    max_chunk_chars=None,
):
    prompt_speech_16k = load_wav(speaker_prompt_audio_path, 16000)
    content_to_synthesize = content_to_synthesize
    output_path = output_path.strip()

    # Respect an explicitly provided empty prompt text from batch preprocessing.
    # Falling back to ASR here can re-introduce noisy prompt text and destabilize
    # prosody, even when batch_inference intentionally dropped it.
    if speaker_prompt_text_transcription is None:
        speaker_prompt_text_transcription = transcribe_audio(speaker_prompt_audio_path)

    ###normalization
    if speaker_prompt_text_transcription:
        speaker_prompt_text_transcription = cosyvoice.frontend.text_normalize(
            speaker_prompt_text_transcription, split=False
        )
    else:
        speaker_prompt_text_transcription = ""
    preferred_tts_text = _normalize_chunk_text(
        str(content_bopomofo_inline_markup or "")
    )
    if not preferred_tts_text:
        preferred_tts_text = _build_text_with_external_bopomofo(
            content_to_synthesize,
            content_bopomofo,
        )
    if not preferred_tts_text:
        preferred_tts_text = content_to_synthesize
    content_chunks = _resolve_content_chunks(
        cosyvoice.frontend,
        preferred_tts_text,
        max_chunk_chars=max_chunk_chars,
    )
    speaker_prompt_text_transcription_bopomo = get_bopomofo_rare(
        speaker_prompt_text_transcription, bopomofo_converter
    )
    prompt_text_for_inference = _select_safe_text_variant(
        cosyvoice.frontend,
        speaker_prompt_text_transcription_bopomo,
        speaker_prompt_text_transcription,
        max_token_len=_MAX_PROMPT_TEXT_TOKEN_LEN,
        label="prompt_text",
    )

    if not content_chunks:
        raise ValueError("content_to_synthesize is empty after normalization")

    chunk_audios = []
    for chunk_index, content_chunk in enumerate(content_chunks, start=1):
        is_last_chunk = chunk_index == len(content_chunks)
        force_question_chunk_guidance = _is_short_chinese_question_chunk(content_chunk)
        if _has_inline_bopomofo_markup(content_chunk):
            tts_text_for_inference = content_chunk
        elif enable_auto_bopomofo or force_question_chunk_guidance:
            content_to_synthesize_bopomo = get_bopomofo_rare(
                content_chunk, bopomofo_converter
            )
            if not any(char in content_chunk for char in "？?"):
                content_to_synthesize_bopomo = _augment_chunk_head_pronunciation(
                    content_to_synthesize_bopomo,
                    bopomofo_converter,
                )
            tts_text_for_inference = _select_safe_text_variant(
                cosyvoice.frontend,
                content_to_synthesize_bopomo,
                content_chunk,
                max_token_len=_MAX_TTS_TEXT_TOKEN_LEN,
                label=f"tts_chunk_{chunk_index}",
            )
        else:
            tts_text_for_inference = content_chunk
        try:
            output = cosyvoice.inference_zero_shot_no_normalize(
                tts_text_for_inference,
                prompt_text_for_inference,
                prompt_speech_16k,
            )
        except RuntimeError as exc:
            if not _is_attention_length_mismatch(exc):
                raise
            if (
                tts_text_for_inference == content_chunk
                and prompt_text_for_inference == speaker_prompt_text_transcription
            ):
                raise
            print(
                f"retry plain text for chunk {chunk_index} after attention "
                f"length mismatch: {exc}"
            )
            output = cosyvoice.inference_zero_shot_no_normalize(
                content_chunk,
                speaker_prompt_text_transcription,
                prompt_speech_16k,
            )
        trim_threshold = _resolve_chunk_trim_threshold(
            content_chunk,
            chunk_index=chunk_index,
            chunk_count=len(content_chunks),
        )
        keep_leading_sec = 0.08 if chunk_index == 1 else 0.01
        trimmed_chunk = _trim_waveform_silence(
            output["tts_speech"],
            threshold=trim_threshold,
            keep_leading_sec=keep_leading_sec,
            keep_trailing_sec=_chunk_keep_trailing_sec(content_chunk, is_last_chunk),
            trim_trailing=True,
        )
        effective_chunk_speed_scale = _resolve_effective_chunk_speed_scale(
            content_chunk,
            speed_scale,
            chunk_count=len(content_chunks),
            waveform=trimmed_chunk,
        )
        trimmed_chunk = _apply_speed_scale(
            trimmed_chunk,
            22050,
            effective_chunk_speed_scale,
        )
        internal_silence_cfg = _resolve_internal_silence_compression_config(
            content_chunk,
            chunk_count=len(content_chunks),
        )
        if internal_silence_cfg.get("enabled", True):
            trimmed_chunk = _compress_internal_silence(
                trimmed_chunk,
                sample_rate=22050,
                threshold_ratio=internal_silence_cfg["threshold_ratio"],
                smoothing_window_sec=internal_silence_cfg["smoothing_window_sec"],
                max_internal_silence_sec=internal_silence_cfg[
                    "max_internal_silence_sec"
                ],
                target_internal_silence_sec=internal_silence_cfg[
                    "target_internal_silence_sec"
                ],
                min_region_sec=internal_silence_cfg["min_region_sec"],
                min_side_region_sec=internal_silence_cfg["min_side_region_sec"],
                max_compressions=internal_silence_cfg["max_compressions"],
            )
        chunk_audios.append(trimmed_chunk)
        if not is_last_chunk:
            pause = torch.zeros(
                (1, _pause_samples_for_chunk(content_chunk)),
                dtype=trimmed_chunk.dtype,
            )
            chunk_audios.append(pause)

    final_waveform = torch.cat(chunk_audios, dim=1)
    if (
        len(content_chunks) == 1
        and _is_brief_final_sentence_chunk(content_chunks[-1])
        and (not _is_short_fragment_chunk(content_chunks[-1]))
    ):
        final_waveform = _append_detached_brief_sentence_tail_release(
            final_waveform,
            sample_rate=22050,
        )
    final_tail_trim_cfg = _resolve_final_tail_trim_config(content_chunks[-1])
    if final_tail_trim_cfg.get("enabled", True):
        final_waveform = _trim_final_tail_artifact(
            final_waveform,
            max_trim_sec=final_tail_trim_cfg["max_trim_sec"],
            keep_padding_sec=final_tail_trim_cfg["keep_padding_sec"],
            threshold_ratio=final_tail_trim_cfg["threshold_ratio"],
        )
    tail_append_cfg = _resolve_tail_append_config(content_chunks[-1])
    output = {
        "tts_speech": _append_tail_silence(
            final_waveform,
            tail_sec=tail_append_cfg["tail_sec"],
            fade_out_sec=tail_append_cfg["fade_out_sec"],
        )
    }
    torchaudio.save(output_path, output["tts_speech"], 22050)


def main():
    ####args
    parser = argparse.ArgumentParser(
        description="Run BreezyVoice text-to-speech with custom inputs"
    )
    parser.add_argument(
        "--content_to_synthesize",
        type=str,
        required=True,
        help="Specifies the content that will be synthesized into speech.",
    )
    parser.add_argument(
        "--speaker_prompt_audio_path",
        type=str,
        required=True,
        help="Specifies the path to the prompt speech audio file of the speaker.",
    )
    parser.add_argument(
        "--speaker_prompt_text_transcription",
        type=str,
        required=False,
        help="Specifies the transcription of the speaker prompt audio (Highly Recommended, if not provided, the system will fall back to transcribing with Whisper.)",
    )
    parser.add_argument(
        "--content_bopomofo",
        type=str,
        required=False,
        help="Optional bopomofo sequence aligned to the synthesis text.",
    )
    parser.add_argument(
        "--content_bopomofo_inline_markup",
        type=str,
        required=False,
        help="Optional inline bopomofo markup text such as 字[:ㄗˋ].",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default="results/output.wav",
        help="Specifies the name and path for the output .wav file.",
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
    parser.add_argument(
        "--enable_auto_bopomofo",
        type=str,
        required=False,
        default="1",
        help="Whether to auto-generate bopomofo guidance when external guidance is absent.",
    )
    parser.add_argument(
        "--speed_scale",
        type=float,
        required=False,
        default=1.0,
        help="Speech speed scaling factor.",
    )
    parser.add_argument(
        "--max_chunk_chars",
        type=int,
        required=False,
        default=0,
        help="Optional hard cap for per-chunk text length.",
    )
    args = parser.parse_args()

    cosyvoice = CustomCosyVoice(args.model_path, args.ttsfrd_resource_dir)

    bopomofo_converter = G2PWConverter()

    speaker_prompt_audio_path = args.speaker_prompt_audio_path
    content_to_synthesize = args.content_to_synthesize
    output_path = args.output_path.strip()
    single_inference(
        speaker_prompt_audio_path,
        content_to_synthesize,
        output_path,
        cosyvoice,
        bopomofo_converter,
        args.speaker_prompt_text_transcription,
        args.content_bopomofo,
        args.content_bopomofo_inline_markup,
        enable_auto_bopomofo=(
            str(args.enable_auto_bopomofo).strip().lower()
            not in {"0", "false", "no", "off"}
        ),
        speed_scale=float(args.speed_scale),
        max_chunk_chars=args.max_chunk_chars,
    )


if __name__ == "__main__":
    main()
