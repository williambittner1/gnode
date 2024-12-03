import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from typing import List

import clip

# Placeholder for SAM and DINO imports
# Assuming you have appropriate modules or use pretrained models

class FeatureExtractor(nn.Module):
    def __init__(self, model_name: str = "clip", device: str = "cuda"):
        super(FeatureExtractor, self).__init__()
        self.device = device
        self.model_name = model_name.lower()

        if self.model_name == "clip":
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        elif self.model_name == "sam":
            # Initialize SAM model and preprocess here
            pass
        elif self.model_name == "dino":
            # Initialize DINO model and preprocess here
            pass
        else:
            raise ValueError(f"Model {self.model_name} not supported.")

    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        if self.model_name == "clip":
            images_preprocessed = torch.stack([self.preprocess(img) for img in images]).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(images_preprocessed)
            return features.cpu()
        elif self.model_name == "sam":
            # Implement SAM feature extraction
            pass
        elif self.model_name == "dino":
            # Implement DINO feature extraction
            pass

    def extract_and_save_features(self, image_paths: List[str], output_paths: List[str]):
        images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
        features = self.forward(images)
        for feature, output_path in zip(features, output_paths):
            torch.save(feature, output_path)
