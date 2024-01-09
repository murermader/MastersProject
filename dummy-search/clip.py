import torch
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from PIL import Image
import torch.nn.functional as F


class Clip:
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        print("CLIP initialized")

    def get_text_embedding(self, text: str):
        inputs = self.tokenizer([text], padding=True, return_tensors="pt")
        text_features = self.model.get_text_features(**inputs)
        return text_features

    def get_image_embedding(self, image_path: str):
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        image_features = self.model.get_image_features(**inputs)
        return image_features

    @staticmethod
    def calc_cosine_similarity(a, b):
        # normalize
        a = F.normalize(a, p=2, dim=1)
        b = F.normalize(b, p=2, dim=1)
        cosine_similarity = F.cosine_similarity(
            a, b
        )
        return round(cosine_similarity.item(), 3)
