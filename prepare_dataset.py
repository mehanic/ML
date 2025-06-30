import json
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor

input_file = "arabic_train.jsonl"
output_file = "train_clean.jsonl"

with open(input_file, "r", encoding="utf-8") as f_in, open(
    output_file, "w", encoding="utf-8"
) as f_out:
    for line in f_in:
        sample = json.loads(line)
        sample["audio"] = sample["audio"]["path"]
        f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"[✓] Saved cleaned JSONL: {output_file}")

dataset = load_dataset("json", data_files=output_file, split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

print(f"[✓] Loaded dataset with {len(dataset)} samples")

model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
processor = Wav2Vec2Processor.from_pretrained(model_name)

print(f"[✓] Loaded processor from: {model_name}")

print("\n[Preview Sample]:")
print(dataset[0])
