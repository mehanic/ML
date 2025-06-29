import subprocess
import os
import re
import torch
import torchaudio
import whisper
import wandb
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# --- Weights & Biases ---
wandb.init(project="audio-transcription", name="wav2vec2_vs_whisper")

# Пути к файлам
mp3_path = "2-24_Leccion_24.mp3"
wav_path = "2-24_Leccion_24.wav"

# Логгируем исходный MP3
wandb.log(
    {"audio_sample": wandb.Audio(mp3_path, caption="first MP3", sample_rate=16000)}
)

# Конвертация MP3 в WAV (если WAV ещё не существует)
if not os.path.exists(wav_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", wav_path],
        check=True,
    )

# --- Wav2Vec2 распознавание ---
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

waveform, sample_rate = torchaudio.load(wav_path)

if sample_rate != 16000:
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(
        waveform
    )

if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

waveform = waveform / waveform.abs().max()

input_values = processor(
    waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000
).input_values
with torch.no_grad():
    logits = model(input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])

sentences = re.split(r"(?<=[\.!\؟\؟])\s+", transcription.strip())

print("Text sentenses (Wav2Vec2):")
for sentence in sentences:
    print(sentence.strip())

# --- SRT (Wav2Vec2) ---
duration = waveform.shape[1] / 16000
step = duration / max(len(sentences), 1)


def seconds_to_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


srt_path = "2-24_Leccion_24_auto.srt"
with open(srt_path, "w", encoding="utf-8") as f:
    for idx, sentence in enumerate(sentences):
        start = idx * step
        end = (idx + 1) * step
        f.write(f"{idx + 1}\n")
        f.write(f"{seconds_to_timestamp(start)} --> {seconds_to_timestamp(end)}\n")
        f.write(f"{sentence.strip()}\n\n")

print(f"\nSRT-file  Wav2Vec2 path: {srt_path}")

wandb.log(
    {
        "wav2vec2_transcription": transcription,
        "wav2vec2_sentences": "\n".join(sentences),
        "wav2vec2_sentence_count": len(sentences),
        "duration_sec": duration,
    }
)

print("\nЗапускаем Whisper for subtiter with times")
whisper_model = whisper.load_model("base", device="cpu")
result = whisper_model.transcribe(mp3_path, word_timestamps=True)

print("\n Whisper:")
print(result["text"])


def to_srt_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


whisper_srt_path = "2-24_Leccion_24_whisper.srt"
with open(whisper_srt_path, "w", encoding="utf-8") as f:
    for i, segment in enumerate(result["segments"], 1):
        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()
        f.write(f"{i}\n")
        f.write(f"{to_srt_timestamp(start)} --> {to_srt_timestamp(end)}\n")
        f.write(f"{text}\n\n")

print(f"\nSRT-path: {whisper_srt_path}")

wandb.log(
    {
        "whisper_transcription": result["text"],
        "whisper_segment_count": len(result["segments"]),
    }
)

wandb.save(srt_path)
wandb.save(whisper_srt_path)

wandb.finish()
