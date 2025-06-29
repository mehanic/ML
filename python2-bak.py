import re
import os
import glob
from pathlib import Path
from datasets import Dataset, Audio
from langdetect import detect
import srt
import json

AUDIO_DIR = "audio/wav"  # путь к аудио, если нужен
SRT_DIR = "audio/srt"    # папка с srt

# Функция для извлечения только арабских слов из текста

def extract_arabic_words(text):
    return [w for w in text.split() if re.fullmatch(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+', w)]


def parse_srt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    subs = list(srt.parse(content))
    # Собираем весь текст субтитров
    full_text = " ".join(sub.content.strip() for sub in subs)
    return full_text

def collect_arabic_data():
    data = []
    for srt_path in glob.glob(f"{SRT_DIR}/*_whisper.srt"):
        base_name = Path(srt_path).stem.replace("_whisper", "")
        audio_path = os.path.join(AUDIO_DIR, base_name + ".wav")
        if not os.path.exists(audio_path):
            continue

        transcript = parse_srt(srt_path)

        arabic_words = extract_arabic_words(transcript)
        if arabic_words:
            # Составляем строку из арабских слов, разделенных пробелом
            arabic_text = " ".join(arabic_words)

            # Можно проверить язык (если хочешь)
            try:
                lang = detect(arabic_text)
            except:
                lang = "unknown"

            if lang == "ar" or True:  # Можно убрать проверку, если уверен
                print(f"{audio_path} → Arabic detected: {arabic_text[:30]}...")
                data.append({"audio": audio_path, "text": arabic_text, "lang": lang})

    return data

arabic_data = collect_arabic_data()

# Если хочешь, создадим датасет HuggingFace (если нужна аудио-колонка)
arabic_dataset = Dataset.from_list(arabic_data)
arabic_dataset = arabic_dataset.cast_column("audio", Audio(sampling_rate=16_000))

# Сохраняем в JSON файл
arabic_dataset.to_json("arabic_dataset.json")

with open("arabic_dataset_clean.jsonl", "w", encoding="utf-8") as f:
    json.dump(arabic_data, f, ensure_ascii=False, indent=2)


# Если хочешь, просто сохранить чистый текст в отдельный файл для обучения
with open("arabic_text_only.txt", "w", encoding="utf-8") as f:
    for item in arabic_data:
        f.write(item["text"] + "\n")
