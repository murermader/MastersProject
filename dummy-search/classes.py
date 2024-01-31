import os
import re


class Dataset:
    def __init__(self, name: str, ids: list[str]):
        self.name = name
        self.ids = ids
        self.is_active = False
        self.images: [Image] = []


class Image:
    def __init__(self, image_path: str):
        self.image_path = image_path

        # Get ID from filename
        filename = os.path.basename(image_path)
        self.id = re.sub(r"\(\d+\)", "", filename).lstrip("P").rstrip(".jpg").strip()

        # Will be computed later
        self.label = ""
        self.from_dataset = []
        self.embedding = None
        self.image_similarity = None
        self.rank = None

        if not os.path.isfile(self.image_path):
            raise ValueError(f"File does not exist: {self.image_path}")

    def from_dataset_as_str(self):
        return ", ".join(self.from_dataset)
