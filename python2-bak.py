import re
import os
import glob
from pathlib import Path
from datasets import Dataset, Audio
from langdetect import detect
import srt
import json

AUDIO_DIR = "audio/wav"  
SRT_DIR = "audio/srt"    


def extract_arabic_words(text):
    return [w for w in text.split() if re.fullmatch(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+', w)]


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

            if lang == "ar" or True:  
                print(f"{audio_path} → Arabic detected: {arabic_text[:30]}...")
                data.append({"audio": audio_path, "text": arabic_text, "lang": lang})

    return data

arabic_data = collect_arabic_data()

arabic_dataset = Dataset.from_list(arabic_data)
arabic_dataset = arabic_dataset.cast_column("audio", Audio(sampling_rate=16_000))

arabic_dataset.to_json("arabic_dataset.json")

with open("arabic_dataset_clean.jsonl", "w", encoding="utf-8") as f:
    json.dump(arabic_data, f, ensure_ascii=False, indent=2)


with open("arabic_text_only.txt", "w", encoding="utf-8") as f:
    for item in arabic_data:
        f.write(item["text"] + "\n")
