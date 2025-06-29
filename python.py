import os
import subprocess
import re
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import whisper

# Настройки
input_dir = "/home/mehanic/Downloads/MLProject/audio"
wav_output_dir = os.path.join(input_dir, "wav")
txt_output_dir = os.path.join(input_dir, "txt")
srt_output_dir = os.path.join(input_dir, "srt")

os.makedirs(wav_output_dir, exist_ok=True)
os.makedirs(txt_output_dir, exist_ok=True)
os.makedirs(srt_output_dir, exist_ok=True)

# Загрузка моделей
print("Загружаем модели...")
wav2vec_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_name)
model = Wav2Vec2ForCTC.from_pretrained(wav2vec_model_name)
whisper_model = whisper.load_model("base", device="cpu")

def seconds_to_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

# Обработка всех .mp3 файлов
for filename in sorted(os.listdir(input_dir)):
    if not filename.lower().endswith(".mp3"):
        continue

    print(f"\n=== Обработка: {filename} ===")
    base = os.path.splitext(filename)[0]

    mp3_path = os.path.join(input_dir, filename)
    wav_path = os.path.join(wav_output_dir, base + ".wav")
    txt_path = os.path.join(txt_output_dir, base + ".txt")
    srt_w2v_path = os.path.join(srt_output_dir, base + "_w2v.srt")
    srt_whisper_path = os.path.join(srt_output_dir, base + "_whisper.srt")

    # MP3 → WAV
    if not os.path.exists(wav_path):
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", wav_path],
            check=True,
        )

    # ✅ Проверка WAV-файла
    info = torchaudio.info(wav_path)
    assert info.sample_rate == 16000, f"{filename} has wrong sample rate!"
    assert info.num_channels == 1, f"{filename} is not mono!"

    # Wav2Vec2 Транскрипция
    waveform, sample_rate = torchaudio.load(wav_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform / waveform.abs().max()

    input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0]).strip()

    # Wav2Vec2 → SRT
    sentences = re.split(r"(?<=[\.!\؟\؟])\s+", transcription)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcription)

    duration = waveform.shape[1] / 16000
    step = duration / max(len(sentences), 1)

    with open(srt_w2v_path, "w", encoding="utf-8") as f:
        for idx, sentence in enumerate(sentences):
            start = idx * step
            end = (idx + 1) * step
            f.write(f"{idx + 1}\n")
            f.write(f"{seconds_to_timestamp(start)} --> {seconds_to_timestamp(end)}\n")
            f.write(f"{sentence.strip()}\n\n")

    print(f"Wav2Vec2 SRT: {srt_w2v_path}")

    # Whisper → SRT
    result = whisper_model.transcribe(mp3_path, word_timestamps=True)

    with open(srt_whisper_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"], 1):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            f.write(f"{i}\n")
            f.write(f"{seconds_to_timestamp(start)} --> {seconds_to_timestamp(end)}\n")
            f.write(f"{text}\n\n")

    print(f"✅ Whisper SRT: {srt_whisper_path}")
