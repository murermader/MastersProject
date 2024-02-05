import os
import re


class Dataset:
    def __init__(self, name: str, keywords: list[str], allow_list: list[str], block_list: list[str]):
        self.name = name
        self.allow_list = set(allow_list)
        self.block_list = set(block_list)
        self.is_active = False
        self.images: [Image] = []
        self.keywords = [keyword.lower() for keyword in keywords]


class Image:
    def __init__(self, image_path: str):
        self.image_path = image_path

        # Get ID from filename
        filename = os.path.basename(image_path)
        self.id = re.sub(r"\(\d+\)", "", filename).lstrip("P").rstrip(".jpg").strip()

        # Will be computed later
        self.label = ""
        self.from_dataset = set()
        self.embedding = None
        self.image_similarity = None
        self.rank = None
        self.is_relevant = False

        if not os.path.isfile(self.image_path):
            raise ValueError(f"File does not exist: {self.image_path}")

