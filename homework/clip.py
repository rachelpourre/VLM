from pathlib import Path
from typing import Any

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Trainer, TrainingArguments

from .base_vlm import BaseVLM
from .data import CaptionDataset, MultiChoiceQADataset

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load(model_name: str = "clip_model"):
    from pathlib import Path

    from peft import PeftModel

    model_path = Path(__file__).parent / model_name

    vlm = BaseVLM()
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model
    clip = CLIP(vision_encoder, text_encoder)
    clip = PeftModel.from_pretrained(clip, model_path).to(device)

    clip.model.load_pretrained(model_path)
    clip.model.eval()
    if device == "cuda":
        clip = clip.to(dtype=torch.bfloat16)

    return clip


def clip_data_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Custom data collator for CLIP training.
    """
    # Get max sequence length
    max_length = max(f["input_ids"].shape[0] for f in features)

    def pad_tensor(tensor, pad_value):
        return torch.cat([tensor, torch.full((max_length - tensor.shape[0],), pad_value, dtype=tensor.dtype)])

    input_ids = torch.stack([pad_tensor(f["input_ids"], pad_value=processor.tokenizer.eos_token_id) for f in features])
    attention_mask = torch.stack([pad_tensor(f["attention_mask"], pad_value=0) for f in features])
    pixel_values = torch.stack([f["pixel_values"] for f in features])  # assume all are same shape
    labels = torch.stack([pad_tensor(f["labels"], pad_value=-100) for f in features])

    return {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.long(),
        "pixel_values": pixel_values.float(),
        "labels": labels.long(),
    }


class CaptionDatasetForTraining(Dataset):
    def __init__(self, dataset: CaptionDataset, processor: AutoProcessor):
        self.dataset = dataset
        self.image_processor = tv.transforms.Compose(
            [
                tv.transforms.Resize(192),
                tv.transforms.RandomResizedCrop(192, scale=(0.5, 1.0)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.image_processor(image)
        text = item["caption"] + self.processor.tokenizer.eos_token
        text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        input_ids = text_inputs["input_ids"].squeeze(0).long()
        attention_mask = text_inputs["attention_mask"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,  # placeholder to fit the collator
        }


class CLIP(nn.Module):
    def __init__(
        self, vision_encoder: nn.Module, text_encoder: nn.Module, proj_dim: int = 64, temperature: float = 0.07
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        # try to read hidden dims from encoder configs, fall back to common defaults
        v_dim =  getattr(getattr(self.vision_encoder, "config", None), "hidden_size", None) or \
                 getattr(getattr(self.vision_encoder, "config", None), "projection_dim", None) or 768
        t_dim =  getattr(getattr(self.text_encoder, "config", None), "hidden_size", None) or \
                 getattr(getattr(self.text_encoder, "config", None), "projection_dim", None) or 768

        # projection heads
        self.image_proj = nn.Linear(int(v_dim), proj_dim)
        self.text_proj = nn.Linear(int(t_dim), proj_dim)

        # logit scale parameter (stored as log to keep positive)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / temperature), dtype=torch.float32))

        # small initializer for projection layers
        nn.init.normal_(self.image_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.text_proj.weight, mean=0.0, std=0.02)
        if self.image_proj.bias is not None:
            nn.init.zeros_(self.image_proj.bias)
        if self.text_proj.bias is not None:
            nn.init.zeros_(self.text_proj.bias)

    def _encode_image_backbone(self, pixel_values: torch.Tensor):
        """
        Run the vision encoder and return a pooled vector (B, vision_dim).
        Robust to backbones that return last_hidden_state or pooler_output.
        """
        # vision backbones typically accept pixel_values keyword and return a dict-like object
        out = None
        try:
            out = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
        except TypeError:
            # fallback: try positional
            out = self.vision_encoder(pixel_values)
        # attempt to extract pooled representation
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            # try CLS token first
            last = out.last_hidden_state
            feat = last[:, 0, :] if last.shape[1] > 0 else last.mean(dim=1)
        elif isinstance(out, (tuple, list)) and len(out) > 0:
            cand = out[0]
            if hasattr(cand, "shape") and cand.ndim == 3:
                feat = cand[:, 0, :]
            else:
                feat = torch.mean(torch.stack([torch.as_tensor(cand)]), dim=0)
        else:
            # as a last resort try to call the module directly on pixel_values and hope for (B, D)
            feat = self.vision_encoder(pixel_values)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
        return feat

    def _encode_text_backbone(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Run the text encoder and return a pooled vector (B, text_dim).
        Use attention_mask-aware mean pooling if pooled output not available.
        """
        out = None
        try:
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        except TypeError:
            out = self.text_encoder(input_ids, attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            last = out.last_hidden_state  # (B, L, D)
            if attention_mask is None:
                feat = last.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1).type_as(last)  # (B, L, 1)
                summed = (last * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1e-9)
                feat = summed / denom
        elif isinstance(out, (tuple, list)) and len(out) > 0:
            cand = out[0]
            if cand.ndim == 3:
                feat = cand[:, 0, :]
            else:
                feat = cand.mean(dim=1)
        else:
            feat = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
        return feat

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        feat = self._encode_image_backbone(image)
        proj = self.image_proj(feat)
        norm = F.normalize(proj, dim=-1)
        return norm

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        feat = self._encode_text_backbone(input_ids, attention_mask)
        proj = self.text_proj(feat)
        norm = F.normalize(proj, dim=-1)
        return norm

    def save_pretrained(self, save_directory: str, **kwargs):
        """Customize save method, save additional parameters"""

        additional_state_dict = {}
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            additional_state_dict[name] = param.data

        torch.save(additional_state_dict, Path(save_directory) / "additional_weights.pt")

    def load_pretrained(self, load_directory: str, **kwargs):
        """Customize load method, load projection additional parameters"""

        additional_weights_path = Path(load_directory) / "additional_weights.pt"
        if additional_weights_path.exists():
            additional_state_dict = torch.load(additional_weights_path, map_location="cpu")

            for name, param in self.named_parameters():
                if "vision_encoder." in name or "text_encoder." in name:
                    continue
                param.data = additional_state_dict[name]

    def set_trainable_parameters(self):
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            param.requires_grad = True

    def gradient_checkpointing_enable(self, **kwargs):
        """
        Enable gradient checkpointing for the vision and text backbones.
        (You don't need to touch this method)
        """
        try:
            self.vision_encoder.gradient_checkpointing_enable(**kwargs)
        except Exception:
            pass
        try:
            self.text_encoder.gradient_checkpointing_enable(**kwargs)
        except Exception:
            pass

    def enable_input_require_grads(self):
        """
        Enable input require grads for the vision and text backbones.
        (You don't need to touch this method)
        """

        # Reference: https://discuss.huggingface.co/t/peft-lora-gpt-neox-backward-pass-failing/35641
        def make_inputs_require_grads(module, input, output):  # noqa: A002
            output.requires_grad_(True)

        try:
            self.vision_encoder.embeddings.register_forward_hook(make_inputs_require_grads)
        except Exception:
            pass
        try:
            self.text_encoder.get_input_embeddings().register_forward_hook(make_inputs_require_grads)
        except Exception:
            pass

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the CLIP model.

        Returns:
            image_features (B, proj_dim), text_features (B, proj_dim), logits_per_image (B, B)
        """
        # encode
        pixel_values = pixel_values.to(next(self.parameters()).device)
        input_ids = input_ids.to(next(self.parameters()).device)
        attention_mask = attention_mask.to(next(self.parameters()).device) if attention_mask is not None else None

        image_features = self.encode_image(pixel_values)  # (B, D)
        text_features = self.encode_text(input_ids=input_ids, attention_mask=attention_mask)  # (B, D)

        # compute logits scaled by learnable temperature
        logit_scale = self.logit_scale.exp()
        logits_per_image = torch.matmul(image_features, text_features.T) * logit_scale  # (B, B)

        return image_features, text_features, logits_per_image


def compute_clip_loss(*args, **kwargs):
    """
    Flexible compute function:
    - If called as compute_clip_loss(model, inputs) (Trainer style), run forward and compute symmetric CE.
    - If called as compute_clip_loss(outputs, labels, ...) compute loss from precomputed outputs.
    """
    # Trainer style: (model, inputs)
    if len(args) >= 2 and hasattr(args[0], "parameters"):
        model = args[0]
        inputs = args[1]
        pixel_values = inputs["pixel_values"].to(next(model.parameters()).device)
        input_ids = inputs["input_ids"].to(next(model.parameters()).device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(next(model.parameters()).device)
        outputs = model(pixel_values, input_ids, attention_mask)
        # outputs: image_features, text_features, logits_per_image
        _, _, logits_per_image = outputs
        b = logits_per_image.shape[0]
        labels = torch.arange(b, device=logits_per_image.device)
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_image.T, labels)
        return (loss_i2t + loss_t2i) / 2.0

    # legacy style: outputs, labels, num_items_in_batch
    outputs = args[0]
    if isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
        _, _, logits_per_image = outputs
        b = logits_per_image.shape[0]
        labels = torch.arange(b, device=logits_per_image.device)
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_image.T, labels)
        return (loss_i2t + loss_t2i) / 2.0

    raise RuntimeError("compute_clip_loss received unexpected arguments")


def get_target_modules_for_lora(model: nn.Module) -> list[str]:
    target_modules = []
    for name, module in model.named_modules():
        # capture nn.Linear modules inside the encoders (but not projection layers)
        if (
            isinstance(module, nn.Linear)
            and ("vision_encoder" in name or "text_encoder" in name)
            and "projection" not in name
        ):
            target_modules.append(name)
    return target_modules


def train(
    data_dir: Path | None = None,
    output_dir: str = "clip",
    num_train_epochs: float = 0.05,  # for debugging purpose, increase this once the dry run works
    per_device_train_batch_size: int = 1024,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 5e-4,
    num_workers: int = 16,
):
    vlm = BaseVLM()

    output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard writer
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Initialize model and processor
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model
    model = CLIP(vision_encoder, text_encoder).to(device).bfloat16()
    model.set_trainable_parameters()

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
        # target_modules="all-linear",
        target_modules=get_target_modules_for_lora(model),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(device)
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # load dataset
    train_dataset = CaptionDataset("train", data_dir)
    train_dataset = CaptionDatasetForTraining(train_dataset, processor)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        bf16=True if device == "cuda" else False,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        label_names=["labels"],
        dataloader_num_workers=num_workers,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=clip_data_collator,
        compute_loss=compute_clip_loss,  # Trainer expects a function (model, inputs) -> loss
    )

    trainer.train()

    # save model
    trainer.save_model(output_dir)
    model.model.save_pretrained(output_dir)

    writer.close()

    return model, processor


def demo_train():
    train(
        train_dataset_name="train_demo",
        output_dir="demo_clip",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        num_workers=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-8,
    )


def test(ckpt_path: str, val_dataset: str = "valid_grader"):
    import tqdm

    testset = MultiChoiceQADataset(val_dataset)

    clip = load(ckpt_path)
    clip = clip.model.to(device)

    image_processor = tv.transforms.Compose(
        [
            tv.transforms.Resize(192),
            tv.transforms.CenterCrop(192),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    correct_count = 0
    total_count = 0

    for pair in tqdm.tqdm(testset):
        image = Image.open(pair["image_path"]).convert("RGB")
        pixel_values = image_processor(image).unsqueeze(0).to(device).bfloat16()
        text_inputs = processor(
            text=[s + processor.tokenizer.eos_token for s in pair["candidates"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = text_inputs["input_ids"].long().to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        vision_feature, text_feature, _ = clip(pixel_values, input_ids, attention_mask)
        prediction = torch.matmul(vision_feature, text_feature.T).argmax(dim=-1)
        if prediction == pair["correct_index"]:
            correct_count += 1
        total_count += 1

    print(f"Accuracy: {correct_count / total_count}")


def main():
    from fire import Fire

    Fire({"train": train, "test": test})


if __name__ == "__main__":
    main()
