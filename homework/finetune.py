# finetune.py (modified)
from pathlib import Path
import os

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Trainer, TrainingArguments

from .base_vlm import BaseVLM
from .data import VQADataset, benchmark

# Device detection
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Use the same processor as before
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")


def load(model_name: str = "vlm_model") -> BaseVLM:
    from pathlib import Path
    from peft import PeftModel

    model_path = Path(__file__).parent / model_name

    vlm = BaseVLM()
    vlm.model = PeftModel.from_pretrained(vlm.model, model_path).to(vlm.device)
    vlm.model.eval()

    return vlm


def custom_data_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Pad input_ids/attention_mask/labels to max length in batch and stack pixel_values.
    Note: labels are token ids with -100 outside the answer (keeps same contract as before).
    """
    # Get max sequence length
    max_length = max(f["input_ids"].shape[0] for f in features)

    def pad_tensor(tensor, pad_value):
        return torch.cat([tensor, torch.full((max_length - tensor.shape[0],), pad_value, dtype=tensor.dtype)])

    input_ids = torch.stack([pad_tensor(f["input_ids"], pad_value=processor.tokenizer.eos_token_id) for f in features])
    attention_mask = torch.stack([pad_tensor(f["attention_mask"], pad_value=0) for f in features])
    labels = torch.stack([pad_tensor(f["labels"], pad_value=-100) for f in features])
    pixel_values = torch.stack([f["pixel_values"] for f in features])  # assume all are same shape

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
    }


class VQADatasetForTraining(Dataset):
    """
    Wrap VQADataset into a training dataset that:
      - loads images
      - resizes them to a smaller fixed size to save GPU memory
      - tokenizes with processor
      - constructs labels masked to answer tokens (same contract as before)
    """

    def __init__(self, dataset: VQADataset, processor: AutoProcessor, resize_image: int = 224):
        self.dataset = dataset
        self.processor = processor
        self.features = ["image", "question", "answer"]
        # image token id (possibly used elsewhere)
        try:
            self.image_token_id = self.processor.tokenizer.additional_special_tokens_ids[
                self.processor.tokenizer.additional_special_tokens.index("<image>")
            ]
        except Exception:
            self.image_token_id = None
        # Ensure pad token
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        # Resize to reduce memory usage (small but reasonable)
        self.resize_image = resize_image

    def __len__(self):
        return len(self.dataset)

    def _open_and_resize(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        # Resize to square thumbnail to reduce memory. Use ANTIALIAS (Pillow auto)
        if self.resize_image:
            img = img.resize((self.resize_image, self.resize_image), Image.BICUBIC)
        return img

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        image = self._open_and_resize(item["image_path"])

        # Prepare input text in chat format
        input_message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": item["question"]}]}]
        prompt = self.processor.apply_chat_template(input_message, add_generation_prompt=True)
        full_text = prompt + item["answer"]  # append the answer to the prompt

        # Use processor to prepare tensors
        inputs = self.processor(
            images=image,
            text=full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Get answer length (tokenized separately)
        answer_ids = self.processor(images=None, text=item["answer"], return_tensors="pt", truncation=True).input_ids.squeeze(0)
        answer_len = len(answer_ids)

        # Prepare labels: mask everything except the answer tokens
        labels = input_ids.clone()
        if answer_len > 0:
            labels[:-answer_len] = -100  # only keep loss on answer
        else:
            labels[:] = -100

        # Ensure EOS token is at the end of the sequence
        eos_id = self.processor.tokenizer.eos_token_id
        if input_ids.shape[0] == 0 or input_ids[-1] != eos_id:
            input_ids = torch.cat([input_ids, torch.tensor([eos_id], dtype=input_ids.dtype)])
            attention_mask = torch.cat([attention_mask, torch.tensor([1], dtype=attention_mask.dtype)])
            labels = torch.cat([labels, torch.tensor([eos_id], dtype=labels.dtype)])

        return {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.long(),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": labels.long(),
        }


def train(
    data_dir: Path | None = None,
    train_dataset_name: str = "train",
    output_dir: str = "vlm_model",
    # Tuned defaults for single-GPU training:
    num_train_epochs: float = 0.05,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-4,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    num_workers: int = 2,
    resize_image: int = 224,
):
    """
    Fine-tune a VLM model using LoRA with memory-saving defaults and better training hyperparams.
    - Reduced image resolution (resize_image)
    - Mixed precision (fp16) when running on CUDA
    - Gradient checkpointing enabled
    - Lower dataloader workers
    - Slightly higher num_train_epochs (default 1.0) — reduce if you want faster runs
    """

    vlm = BaseVLM()

    # Create output directory
    output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard writer
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Initialize model and processor
    processor_local = vlm.processor
    model = vlm.model

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
        bias="none",
    )

    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # Memory & training flags
    try:
        model.config.use_cache = False
    except Exception:
        pass

    # Enable input grads for PEFT
    try:
        model.enable_input_require_grads()
    except Exception:
        pass

    # Try enabling gradient checkpointing (saves memory)
    try:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    except Exception:
        pass

    model.train()

    # Prepare datasets
    train_dataset = VQADataset(train_dataset_name, data_dir)
    train_dataset = VQADatasetForTraining(train_dataset, processor_local, resize_image=resize_image)

    # Optional validation dataset (if exists) so Trainer can evaluate
    val_dataset = None
    try:
        vd = VQADataset("valid", data_dir)
        val_dataset = VQADatasetForTraining(vd, processor_local, resize_image=resize_image)
    except Exception:
        val_dataset = None

    if processor_local.tokenizer.pad_token is None:
        processor_local.tokenizer.pad_token = processor_local.tokenizer.eos_token

    # Configure training arguments (compatible with older transformers)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(output_dir),
        report_to="tensorboard",

        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,

        bf16=True if DEVICE == "cuda" else False,

        # OLD-VERSION COMPATIBLE FIELDS
        logging_steps=10,
        save_steps=200,
        eval_steps=200,

        # these must be strings in old versions
        save_total_limit=2,
    )


    # free cache before training
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if val_dataset is not None else None,
        data_collator=custom_data_collator,
    )

    # Run training
    trainer.train()

    # Save the model
    trainer.save_model(output_dir)

    # Close TensorBoard writer
    writer.close()

    return model, processor_local


def evaluate(model: nn.Module, val_loader: DataLoader) -> float:
    """
    Evaluate the model on the validation set.

    Args:
        model: Model to evaluate
        val_loader: Validation data loader

    Returns:
        Average validation loss
    """
    model.eval()
    val_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            # Some model outputs may not contain .loss; be defensive
            if hasattr(outputs, "loss"):
                val_loss += outputs.loss.item()
                n += 1

    model.train()
    return (val_loss / n) if n > 0 else 0.0


def demo_train():
    train(
        train_dataset_name="train_demo",
        output_dir="demo_train",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        num_workers=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-8,
        resize_image=128,
    )


def test_model(ckpt_path: str, val_dataset: str = "valid_grader"):
    testset = VQADataset(val_dataset)

    llm = load(ckpt_path)

    benchmark_result = benchmark(llm, testset, 128)
    print(benchmark_result.accuracy)


if __name__ == "__main__":
    from fire import Fire

    Fire({"demo_train": demo_train, "train": train, "test": test_model})
