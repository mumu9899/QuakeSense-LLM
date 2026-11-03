import os
import json
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


DEFAULT_LABEL_MAPPING = {
    "0": "No obvious shaking / irrelevant",
    "1": "Slight shaking",
    "2": "Moderate shaking",
    "3": "Strong shaking / affected"
}


def load_label_mapping(model_dir: str) -> Dict[str, str]:
    """Load label mapping from pretrained/label_mapping.json."""
    mapping_path = os.path.join(model_dir, "label_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            label_mapping = json.load(f)
        return {str(k): v for k, v in label_mapping.items()}
    return DEFAULT_LABEL_MAPPING


class QuakeSenseModel:
    """
    Inference wrapper for post-earthquake social media text classification.

    The model supports:
    1. A full Hugging Face sequence classification model in model_dir.
    2. A LoRA adapter in model_dir with a specified base model.
    """

    def __init__(
        self,
        model_dir: str = "pretrained",
        base_model: Optional[str] = None,
        max_length: int = 256,
        device: Optional[str] = None,
        num_labels: int = 4,
        trust_remote_code: bool = True
    ):
        self.model_dir = model_dir
        self.base_model = base_model
        self.max_length = max_length
        self.num_labels = num_labels
        self.trust_remote_code = trust_remote_code

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.label_mapping = load_label_mapping(model_dir)
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.model.eval()

    def _has_lora_adapter(self) -> bool:
        adapter_config = os.path.join(self.model_dir, "adapter_config.json")
        return os.path.exists(adapter_config)

    def _load_tokenizer(self):
        """Load tokenizer from model_dir if available; otherwise use base_model."""
        tokenizer_source = self.model_dir
        tokenizer_files = ["tokenizer.json", "tokenizer.model", "vocab.txt", "spiece.model"]
        has_tokenizer = any(os.path.exists(os.path.join(self.model_dir, f)) for f in tokenizer_files)
        if not has_tokenizer:
            if self.base_model is None:
                raise ValueError(
                    "Tokenizer files were not found in model_dir. Please provide --base_model to load tokenizer."
                )
            tokenizer_source = self.base_model
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=self.trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self):
        """Load either a full sequence classification model or a LoRA adapter."""
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if self._has_lora_adapter():
            if not PEFT_AVAILABLE:
                raise ImportError("PEFT is required for loading LoRA adapters. Please install it with: pip install peft")
            if self.base_model is None:
                raise ValueError("LoRA adapter detected. Please provide --base_model.")
            base = AutoModelForSequenceClassification.from_pretrained(
                self.base_model,
                num_labels=self.num_labels,
                torch_dtype=dtype,
                trust_remote_code=self.trust_remote_code
            )
            model = PeftModel.from_pretrained(base, self.model_dir)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir,
                num_labels=self.num_labels,
                torch_dtype=dtype,
                trust_remote_code=self.trust_remote_code
            )
        model.to(self.device)
        return model

    @torch.no_grad()
    def predict_batch(self, texts: List[str], batch_size: int = 8) -> List[Dict[str, Any]]:
        """Predict labels for a list of texts."""
        all_results = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            pred_labels = np.argmax(probs, axis=-1)
            for text, pred_label, prob in zip(batch_texts, pred_labels, probs):
                pred_label_str = str(int(pred_label))
                pred_name = self.label_mapping.get(pred_label_str, pred_label_str)
                confidence = float(prob[int(pred_label)])
                result = {
                    "text": text,
                    "pred_label": int(pred_label),
                    "pred_name": pred_name,
                    "confidence": confidence
                }
                for i, p in enumerate(prob):
                    result[f"prob_{i}"] = float(p)
                all_results.append(result)
        return all_results

    def predict_one(self, text: str) -> Dict[str, Any]:
        """Predict label for a single text."""
        return self.predict_batch([text], batch_size=1)[0]