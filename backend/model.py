"""Cat vs Dog classifier — Muhammad Ubaid."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

CLASS_LABELS = ("cat", "dog")

# ImageNet synsets that map cleanly to cats / dogs
_CAT_INDICES = {281, 282, 283, 284, 285}  # tabby, tiger cat, persian, siamese, egyptian
_DOG_INDICES = set(range(151, 269))  # ImageNet dog breeds

_CUSTOM_WEIGHTS = Path(__file__).resolve().parent / "weights" / "image_classifier_v01.pth"


class ImageClassifier(nn.Module):
    """CNN architecture from the CatvsDogDetection notebook."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 56 * 56, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pool1(self.relu1(self.conv1(x)))
        out = self.pool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        return self.fc2(out)


class Predictor:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = "imagenet"
        self.model: nn.Module

        if _CUSTOM_WEIGHTS.exists():
            self.model = ImageClassifier()
            state = torch.load(_CUSTOM_WEIGHTS, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self.mode = "custom"
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            weights = models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights)
            self.transform = weights.transforms()
            self.mode = "imagenet"

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        if self.mode == "custom":
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            idx = int(torch.argmax(probs).item())
            confidence = float(probs[idx].item())
            label = CLASS_LABELS[idx]
            return {
                "label": label,
                "confidence": round(confidence * 100, 2),
                "scores": {
                    "cat": round(float(probs[0].item()) * 100, 2),
                    "dog": round(float(probs[1].item()) * 100, 2),
                },
                "engine": "custom-cnn",
            }

        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        cat_idx = list(_CAT_INDICES)
        dog_idx = list(_DOG_INDICES)
        cat_score = float(probs[cat_idx].sum().item())
        dog_score = float(probs[dog_idx].sum().item())
        total = cat_score + dog_score

        if total < 1e-6:
            top_prob, top_idx = torch.max(probs, dim=0)
            idx = int(top_idx.item())
            if idx in _CAT_INDICES:
                label, confidence = "cat", float(top_prob.item())
                cat_score, dog_score = confidence, 1.0 - confidence
            elif idx in _DOG_INDICES:
                label, confidence = "dog", float(top_prob.item())
                dog_score, cat_score = confidence, 1.0 - confidence
            else:
                label = "cat" if cat_score >= dog_score else "dog"
                confidence = max(cat_score, dog_score)
        else:
            cat_n = cat_score / total
            dog_n = dog_score / total
            if cat_n >= dog_n:
                label, confidence = "cat", cat_n
            else:
                label, confidence = "dog", dog_n
            cat_score, dog_score = cat_n, dog_n

        return {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "scores": {
                "cat": round(cat_score * 100, 2),
                "dog": round(dog_score * 100, 2),
            },
            "engine": "resnet18-imagenet",
        }


_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor
