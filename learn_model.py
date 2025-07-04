from transformers import (
    TrainingArguments,
    Wav2Vec2ForCTC,
    Trainer,
    Wav2Vec2Processor,
)
from datasets import load_from_disk
import torch
import transformers
import inspect
import gc
from dataclasses import dataclass
from typing import List, Dict, Union
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("Transformers version:", transformers.__version__)
print("TrainingArguments file path:", inspect.getfile(TrainingArguments))

dataset = load_from_disk("data/arabic_prepared")

# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xls-r-300m")

processor = Wav2Vec2Processor.from_pretrained(
    "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
)

# model = Wav2Vec2ForCTC.from_pretrained(
# "facebook/wav2vec2-xls-r-300m",
# ctc_loss_reduction="mean",
# pad_token_id=processor.tokenizer.pad_token_id,
# )


model = Wav2Vec2ForCTC.from_pretrained(
    "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

model.freeze_feature_encoder()
model.gradient_checkpointing_enable()

training_args = TrainingArguments(
    output_dir="checkpoints",
    group_by_length=True,  
    per_device_train_batch_size=1,  
    gradient_accumulation_steps=4,  
    eval_strategy="no",
    num_train_epochs=3,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    learning_rate=3e-4,
    warmup_steps=500,
)


@dataclass
class DataCollatorCTCWithPaddingCustom:
    processor: Wav2Vec2Processor
    padding: bool = True

    def __call__(
        self, features: List[Dict[str, Union[List[float], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_values = [f["input_values"] for f in features]
        labels = [f["labels"] for f in features]

        batch = self.processor.pad(
            {"input_values": input_values}, padding=self.padding, return_tensors="pt"
        )

        cleaned_labels = []
        for label in labels:
            if isinstance(label, torch.Tensor):
                label = label.tolist()
            elif isinstance(label, int):
                label = [label]
            cleaned_labels.append(label)

        max_label_len = max(len(label) for label in cleaned_labels)
        padded_labels = [
            label + [-100] * (max_label_len - len(label)) for label in cleaned_labels
        ]
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


#  collator
data_collator = DataCollatorCTCWithPaddingCustom(processor=processor, padding=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor,
    data_collator=data_collator,
)

print(torch.cuda.memory_summary(device=0, abbreviated=False))

try:
    trainer.train()
finally:
    torch.cuda.empty_cache()
    gc.collect()
