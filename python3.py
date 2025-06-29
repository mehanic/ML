from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition", model="facebook/wav2vec2-large-xlsr-53-arabic"
)

sample = dataset[0]
prediction = asr(sample["audio"]["array"], sampling_rate=16000)

print("Предсказание:", prediction["text"])
print("Ожидаемый текст:", sample["text"])
