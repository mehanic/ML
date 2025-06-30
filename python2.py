import re
import os
import glob
from pathlib import Path
from datasets import Dataset, Audio
from langdetect import detect
import srt
import torchaudio

AUDIO_DIR = "audio/wav"
SRT_DIR = "audio/srt"

def extract_arabic_words(text):
    words = re.findall(r'[\u0600-\u06FF0-9ـًٌٍَُِّْٰٓٔ]+', text)
    return words

def parse_srt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    subs = list(srt.parse(content))
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
            arabic_text = " ".join(arabic_words)
            try:
                lang = detect(arabic_text)
            except:
                lang = "unknown"

            if lang == "ar":
                try:
                    info = torchaudio.info(audio_path)
                    duration = info.num_frames / info.sample_rate
                    sampling_rate = info.sample_rate
                except Exception as e:
                    print(f"Ошибка чтения аудио {audio_path}: {e}")
                    continue

                print(f"{audio_path} → Arabic detected: {arabic_text[:30]}...")
                data.append({
                    "audio": audio_path,
                    "text": arabic_text,
                    "lang": lang,
                    "duration": duration,
                    "sampling_rate": sampling_rate
                })

    return data

arabic_data = collect_arabic_data()

arabic_dataset = Dataset.from_list(arabic_data)
arabic_dataset = arabic_dataset.cast_column("audio", Audio(sampling_rate=16000))

split = arabic_dataset.train_test_split(test_size=0.1)
train_dataset = split["train"]
test_dataset = split["test"]

train_dataset.to_json("arabic_train.jsonl", lines=True, force_ascii=False)
test_dataset.to_json("arabic_test.jsonl", lines=True, force_ascii=False)
arabic_dataset.to_json("arabic_dataset.jsonl", orient="records", lines=True, force_ascii=False)

with open("arabic_text_only.txt", "w", encoding="utf-8") as f:
    for item in arabic_data:
        f.write(item["text"] + "\n")

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
