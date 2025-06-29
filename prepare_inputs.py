import json
from datasets import load_dataset, Audio, Dataset
from transformers import Wav2Vec2Processor
import os

# === Конфигурация ===
INPUT_JSONL = "train_clean.jsonl"
MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
OUTPUT_PATH = "data/arabic_prepared"
SAMPLING_RATE = 16000

# === Шаг 1: Загрузка и обработка аудио ===
dataset = load_dataset("json", data_files=INPUT_JSONL, split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

# === Шаг 2: Загрузка процессора модели ===
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)


# === Шаг 3: Функция подготовки примеров ===
def prepare(example):
    audio = example["audio"]
    input_values = processor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_values[0]

    with processor.as_target_processor():
        labels = processor(example["text"]).input_ids

    return {
        "input_values": input_values,
        "labels": labels[0],
    }


# === Шаг 4: Применение подготовки ко всем примерам ===
prepared_dataset = dataset.map(prepare, remove_columns=dataset.column_names)

# === Шаг 5: Сохранение на диск ===
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
prepared_dataset.save_to_disk(OUTPUT_PATH)

print(f"[✓] Prepared dataset saved to: {OUTPUT_PATH}")
