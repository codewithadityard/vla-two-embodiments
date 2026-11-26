"""Frozen SmolVLM2 backbone: first N language-model layers, images first then instruction.

Input per frame: main image, wrist image (HxWx3 uint8), instruction string.
Prompt: "<image><image>" + instruction, standard causal attention (plain HF forward).
Because images come first and attention is causal, the image tokens never see the text:
the first `prefix_len` tokens are the same whatever the instruction is (checked in M2).
Output: prefix tokens [B, prefix_len, D] and text tokens [B, T_txt, D] + mask, fp16.
Runs on Kaggle only (needs torch + transformers).
"""
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

PROMPT_IMAGES = "<image><image>"


def load_vlm(name, revision, device, num_layers=None):
    """SmolVLM2 in fp16, frozen, eval mode. If num_layers is given, keep only the first num_layers LM layers."""
    vlm = AutoModelForImageTextToText.from_pretrained(name, revision=revision, dtype=torch.float16)
    text = vlm.model.text_model
    if num_layers is not None:
        text.layers = text.layers[:num_layers]
        text.config.num_hidden_layers = num_layers
        if getattr(text.config, "layer_types", None):
            text.config.layer_types = text.config.layer_types[:num_layers]
    return vlm.model.to(device).eval().requires_grad_(False)


class Backbone:
    def __init__(self, name, revision, num_layers, device="cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(name, revision=revision)
        self.processor.image_processor.do_image_splitting = False  # one 512x512 tile per image, as SmolVLA
        self.processor.tokenizer.padding_side = "right"  # pads after the text: earlier tokens are untouched
        self.vlm = load_vlm(name, revision, device, num_layers)
        self.image_token_id = self.vlm.config.image_token_id
        # Number of tokens before the instruction starts (BOS, image wrappers, 2 x image tokens).
        blank = np.zeros((256, 256, 3), dtype=np.uint8)
        self.prefix_len = self.tokenize([blank], [blank], [""])["input_ids"].shape[1]

    def tokenize(self, main, wrist, instructions):
        """Processor output for a batch (images as uint8 HxWx3 arrays)."""
        texts = [PROMPT_IMAGES + s for s in instructions]
        images = [[Image.fromarray(m), Image.fromarray(w)] for m, w in zip(main, wrist)]
        return self.processor(text=texts, images=images, return_tensors="pt", padding=True)

    @torch.no_grad()
    def forward(self, batch):
        """Processor batch -> last hidden state [B, T, D] (fp16) after the final RMSNorm."""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        batch["pixel_values"] = batch["pixel_values"].half()
        return self.vlm(**batch).last_hidden_state

    def encode(self, main, wrist, instructions):
        """-> (prefix [B, P, D], text [B, T_txt, D], text_mask [B, T_txt] bool), all on device."""
        batch = self.tokenize(main, wrist, instructions)
        h = self.forward(batch)
        P = self.prefix_len
        mask = batch["attention_mask"][:, P:].to(self.device).bool()
        return h[:, :P], h[:, P:], mask
