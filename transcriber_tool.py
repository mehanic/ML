from langchain.tools import tool
import subprocess, os, re, torch, torchaudio, whisper, wandb
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


@tool
def transcribe_with_models(mp3_path: str) -> str:
    """
    two models: Wav2Vec2 и Whisper
    """
    wav_path = mp3_path.replace(".mp3", ".wav")
    wandb.init(project="audio-transcription", name="wav2vec2_vs_whisper")

    wandb.log(
        {
            "audio_sample": wandb.Audio(
                mp3_path, caption="Исходный MP3", sample_rate=16000
            )
        }
    )

    if not os.path.exists(wav_path):
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", wav_path]
        )

    # Wav2Vec2
    processor = Wav2Vec2Processor.from_pretrained(
        "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
    )
    waveform, sample_rate = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform / waveform.abs().max()

    input_values = processor(
        waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000
    ).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    wav2vec_text = processor.decode(predicted_ids[0])

    # Whisper
    whisper_model = whisper.load_model("base", device="cpu")
    result = whisper_model.transcribe(mp3_path, word_timestamps=True)
    whisper_text = result["text"]

    wandb.log(
        {
            "wav2vec2_transcription": wav2vec_text,
            "whisper_transcription": whisper_text,
            "whisper_segment_count": len(result["segments"]),
        }
    )

    wandb.finish()
    return f"---Wav2Vec2---\n{wav2vec_text}\n\n---Whisper---\n{whisper_text}"
