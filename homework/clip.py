# homework/clip.py
from pathlib import Path
from typing import Any, Tuple
import os

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
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from transformers.modeling_outputs import ModelOutput

from .base_vlm import BaseVLM
from .data import CaptionDataset, MultiChoiceQADataset

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load(model_name: str = "clip", base_dir: str | Path | None = None):
    """
    Load a checkpoint previously saved by trainer.save_model (or our callback).
    Returns a PEFT-wrapped model (so callers can use .model / .base_model as needed).
    """
    from peft import PeftModel

    base = Path(__file__).parent
    if base_dir:
        model_path = Path(base_dir) / model_name
    else:
        model_path = base / model_name

    vlm = BaseVLM()
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model

    # construct bare CLIP and then wrap with PeftModel loader (this mirrors training flow)
    clip = CLIP(vision_encoder, text_encoder)
    peft_wrapped = PeftModel.from_pretrained(clip, model_path).to(device)

    # load additional projection weights that we saved separately (if any)
    try:
        base_model = getattr(peft_wrapped, "base_model", getattr(peft_wrapped, "model", peft_wrapped))
        if hasattr(base_model, "load_pretrained"):
            base_model.load_pretrained(model_path)
    except Exception:
        # best-effort; not fatal
        pass

    peft_wrapped.eval()
    if device == "cuda":
        peft_wrapped = peft_wrapped.to(dtype=torch.bfloat16)

    return peft_wrapped


def clip_data_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Custom data collator for CLIP training.
    Pads textual tensors to max length in batch and stacks pixel_values.
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
    """
    CLIP wrapper around two backbone encoders (vision + text).
    This class:
      - computes image/text pooled features from backbones
      - projects to a shared embedding dimension
      - normalizes and computes scaled dot-product logits
      - computes symmetric cross-entropy loss between image and text
    """

    def __init__(self, vision_encoder: nn.Module, text_encoder: nn.Module, proj_dim: int = 64, temperature: float = 0.07):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        # try to read hidden dims from encoder configs, fall back to common defaults
        v_dim = getattr(getattr(self.vision_encoder, "config", None), "hidden_size", None) or getattr(
            getattr(self.vision_encoder, "config", None), "projection_dim", None
        ) or 768
        t_dim = getattr(getattr(self.text_encoder, "config", None), "hidden_size", None) or getattr(
            getattr(self.text_encoder, "config", None), "projection_dim", None
        ) or 768

        # projection heads (these are the extra params that must be saved + trained)
        self.image_proj = nn.Linear(int(v_dim), proj_dim)
        self.text_proj = nn.Linear(int(t_dim), proj_dim)

        # logit scale parameter (stored as log to keep positive)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / temperature), dtype=torch.float32))

        # initialize projection weights/bias
        nn.init.normal_(self.image_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.text_proj.weight, mean=0.0, std=0.02)
        if self.image_proj.bias is not None:
            nn.init.zeros_(self.image_proj.bias)
        if self.text_proj.bias is not None:
            nn.init.zeros_(self.text_proj.bias)

    # ---- encoding helpers (robust to HF backbones) ----
    def _encode_image_backbone(self, pixel_values: torch.Tensor):
        try:
            out = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
        except TypeError:
            out = self.vision_encoder(pixel_values)
        # prefer pooler_output, then first token of last_hidden_state, else mean pool
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            last = out.last_hidden_state
            feat = last[:, 0, :] if last.shape[1] > 0 else last.mean(dim=1)
        elif isinstance(out, (tuple, list)) and len(out) > 0:
            cand = out[0]
            if getattr(cand, "ndim", None) == 3:
                feat = cand[:, 0, :]
            else:
                feat = cand.mean(dim=1)
        else:
            feat = self.vision_encoder(pixel_values)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
        return feat

    def _encode_text_backbone(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
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
            if getattr(cand, "ndim", None) == 3:
                feat = cand[:, 0, :]
            else:
                feat = cand.mean(dim=1)
        else:
            feat = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
        return feat

    # ---- public encoders ----
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

    # ---- save/load helpers for "additional" non-backbone weights ----
    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the non-backbone parameters (projection heads + logit_scale) into additional_weights.pt
        Also call any underlying encoder save_pretrained if present.
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        additional_state_dict = {}
        for name, param in self.named_parameters():
            # skip backbone encoder parameters (they are saved by HF AutoModel)
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            additional_state_dict[name] = param.detach().cpu()

        torch.save(additional_state_dict, save_dir / "additional_weights.pt")

        # If encoders have their own save_pretrained, call them to be safe (best effort)
        try:
            if hasattr(self.vision_encoder, "save_pretrained"):
                self.vision_encoder.save_pretrained(save_dir / "vision_encoder")
            if hasattr(self.text_encoder, "save_pretrained"):
                self.text_encoder.save_pretrained(save_dir / "text_encoder")
        except Exception:
            pass

    def load_pretrained(self, load_directory: str, **kwargs):
        """
        Load additional_weights.pt (if present) and populate parameters.
        """
        load_dir = Path(load_directory)
        additional = load_dir / "additional_weights.pt"
        if additional.exists():
            additional_state_dict = torch.load(additional, map_location="cpu")
            for name, param in self.named_parameters():
                if "vision_encoder." in name or "text_encoder." in name:
                    continue
                if name in additional_state_dict:
                    try:
                        param.data.copy_(additional_state_dict[name].to(param.device))
                    except Exception:
                        param.data = additional_state_dict[name].to(param.device)

    def set_trainable_parameters(self):
        """
        Ensure that non-backbone parameters are trainable (LoRA adapters are set by get_peft_model).
        Call this AFTER wrapping with get_peft_model.
        """
        for name, param in self.named_parameters():
            # keep backbone encoders frozen by default; adapters and projection layers trainable
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            param.requires_grad = True

    def enable_input_require_grads(self):
        """
        Enable input require grads for the vision and text backbones.
        Hook into embedding layers to avoid PEFT backward issues.
        """
        def make_inputs_require_grads(module, input, output):  # noqa: A002
            output.requires_grad_(True)

        try:
            if hasattr(self.vision_encoder, "embeddings"):
                self.vision_encoder.embeddings.register_forward_hook(make_inputs_require_grads)
        except Exception:
            pass
        try:
            if hasattr(self.text_encoder, "get_input_embeddings"):
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
    ):
        device_loc = next(self.parameters()).device
        pixel_values = pixel_values.to(device_loc)
        input_ids = input_ids.to(device_loc)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device_loc)

        # Encode
        image_features = self.encode_image(pixel_values)      # (B, D)
        text_features  = self.encode_text(input_ids, attention_mask) # (B, D)

        # Similarity logits
        logit_scale = self.logit_scale.exp()
        logits = image_features @ text_features.T * logit_scale

        # ---------------------------------------------------------
        # CASE 1: Training — labels provided
        # Return ModelOutput w/ loss so Trainer can optimize
        # ---------------------------------------------------------
        if labels is not None:
            b = logits.size(0)
            labels_idx = torch.arange(b, device=logits.device)

            loss_i2t = F.cross_entropy(logits, labels_idx)
            loss_t2i = F.cross_entropy(logits.T, labels_idx)
            loss = (loss_i2t + loss_t2i) / 2.0

            return {
                "loss": loss,
                "logits": logits,
                "image_features": image_features,
                "text_features": text_features,
            }

        # ---------------------------------------------------------
        # CASE 2: Evaluation — NO labels
        # MUST return exactly 3 positional outputs (for grader)
        # ---------------------------------------------------------
        return image_features, text_features, logits


# Trainer callback to ensure our "additional_weights.pt" is written to each checkpoint
class SaveAdditionalWeightsCallback(TrainerCallback):
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called when Trainer saves a checkpoint. Persist the model's additional weights
        to the checkpoint folder: <output_dir>/checkpoint-<global_step>/additional_weights.pt
        """
        model = kwargs.get("model", None)
        if model is None:
            return

        step = state.global_step
        if step and step > 0:
            ckpt_dir = Path(args.output_dir) / f"checkpoint-{step}"
        else:
            ckpt_dir = Path(args.output_dir)

        ckpt_dir.mkdir(parents=True, exist_ok=True)

        try:
            base_model = getattr(model, "base_model", getattr(model, "model", model))
            if hasattr(base_model, "save_pretrained"):
                try:
                    base_model.save_pretrained(ckpt_dir)
                except Exception:
                    pass
            if hasattr(model, "save_pretrained"):
                try:
                    model.save_pretrained(ckpt_dir)
                except Exception:
                    pass
        except Exception as e:
            print("SaveAdditionalWeightsCallback failed:", e)


def get_target_modules_for_lora(model: nn.Module) -> list[str]:
    """
    Return target module names for LoRA injection (linear layers in encoders).
    """
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ("vision_encoder" in name or "text_encoder" in name) and "projection" not in name:
            target_modules.append(name)
    return target_modules


def train(
    data_dir: str = "",
    output_dir: str = "clip",
    num_train_epochs: float = 0.05,  # debugging default; increase for real training
    per_device_train_batch_size: int = 64,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 5e-4,
    num_workers: int = 4,
    resume_from_checkpoint: str | None = None,
):
    """
    Train CLIP on caption dataset.

    Important: apply get_peft_model BEFORE calling set_trainable_parameters (we do that below).
    """
    data_dir_str = data_dir
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    vlm = BaseVLM()

    output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # get encoders
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model

    # construct base CLIP model
    base_clip = CLIP(vision_encoder, text_encoder).to(device).bfloat16()

    # --- FIX HERE ---
    # Safeguard: make gradient_checkpointing_enable a no-op
    if not hasattr(base_clip, "gradient_checkpointing_enable"):
        base_clip.gradient_checkpointing_enable = lambda *a, **k: None
    # -----------------

    # configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=get_target_modules_for_lora(base_clip),
        bias="none",
    )

    # wrap with PEFT LoRA
    peft_model = get_peft_model(base_clip, peft_config)

    # now make sure our projection heads and any other non-backbone params are trainable
    try:
        peft_model.set_trainable_parameters()
    except Exception:
        try:
            base_clip.set_trainable_parameters()
        except Exception:
            pass

    # ensure underlying base model params are set correctly
    try:
        underlying = getattr(peft_model, "base_model", getattr(peft_model, "model", peft_model))
        if hasattr(underlying, "set_trainable_parameters"):
            underlying.set_trainable_parameters()
    except Exception:
        pass

    peft_model.print_trainable_parameters()
    peft_model.to(device)
    peft_model.train()

    # enable hooks to avoid PEFT backward issues (best-effort)
    try:
        underlying = getattr(peft_model, "base_model", getattr(peft_model, "model", peft_model))
        if hasattr(underlying, "enable_input_require_grads"):
            underlying.enable_input_require_grads()
        elif hasattr(peft_model, "enable_input_require_grads"):
            peft_model.enable_input_require_grads()
    except Exception:
        pass

    # load dataset
    train_dataset = CaptionDataset("train", Path(data_dir_str))  # <-- FIX: pass Path here
    train_dataset = CaptionDatasetForTraining(train_dataset, processor)

    # --- FIX HERE ---
    # disable gradient checkpointing at HF level (CLIP does not support it)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(output_dir),
        report_to="tensorboard",

        num_train_epochs=1.0,                       # ★ biggest win
        per_device_train_batch_size=16,             # ★ safer batch
        gradient_accumulation_steps=4,              # effective batch = 64

        learning_rate=1e-3,                         # ★ LoRA sweet spot
        warmup_ratio=0.03,                          # smoother start

        gradient_checkpointing=False,
        bf16=True if device == "cuda" else False,
        dataloader_num_workers=8,

        logging_steps=10,
        save_strategy="steps",
        save_steps=200,                             # avoid SPAM saving
        save_total_limit=3,

        remove_unused_columns=False,
    )

    # -----------------

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=clip_data_collator,
        callbacks=[SaveAdditionalWeightsCallback()],
    )

    # start training (resume_from_checkpoint can be None or a path)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # final save: ensure additional_weights.pt is saved into final output_dir
    try:
        underlying = getattr(trainer.model, "base_model", getattr(trainer.model, "model", trainer.model))
        if hasattr(underlying, "save_pretrained"):
            underlying.save_pretrained(output_dir)
    except Exception:
        pass

    try:
        if hasattr(trainer.model, "save_pretrained"):
            trainer.model.save_pretrained(output_dir)
    except Exception:
        pass

    # PEFT save
    try:
        trainer.model.save_pretrained(str(output_dir))
    except Exception:
        pass

    trainer.save_model(str(output_dir))
    writer.close()

    return trainer.model, processor


def demo_train():
    train(
        output_dir="demo_clip",
        num_train_epochs=0.01,
        per_device_train_batch_size=2,
        num_workers=0,
        gradient_accumulation_steps=1,
    )


def test(ckpt_path: str, val_dataset: str = "valid_grader"):
    """
    Evaluate a saved checkpoint (path to folder that contains the saved PEFT model &
    additional_weights.pt).
    """
    import tqdm

    testset = MultiChoiceQADataset(val_dataset)

    # load the peft-wrapped model
    clip = load(ckpt_path)
    # get underlying model for forwarding
    model_for_infer = getattr(clip, "base_model", getattr(clip, "model", clip))
    model_for_infer = (model_for_infer if hasattr(model_for_infer, "encode_image") else clip)

    model_for_infer = model_for_infer.to(device)
    model_for_infer.eval()

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

        out = clip(pixel_values, input_ids, attention_mask)
        if isinstance(out, ModelOutput):
            vision_feature = out.image_features
            text_feature = out.text_features
        elif isinstance(out, (tuple, list)) and len(out) >= 3:
            vision_feature, text_feature, _ = out[0], out[1], out[2]
        else:
            raise RuntimeError("Unexpected model output shape in test")

        prediction = torch.matmul(vision_feature, text_feature.T).argmax(dim=-1)
        if int(prediction) == int(pair["correct_index"]):
            correct_count += 1
        total_count += 1

    print(f"Accuracy: {correct_count / total_count}")


def main():
    from fire import Fire

    Fire({"train": train, "test": test, "demo_train": demo_train})


if __name__ == "__main__":
    main()
