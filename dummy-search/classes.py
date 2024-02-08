import os
import re


class Dataset:
    def __init__(
        self,
        name: str,
        keyword_allow_list: list[str] = None,
        keyword_block_list: list[str] = None,
        allow_list: list[str] = None,
        block_list: list[str] = None,
    ):
        # Initialize possible None values
        if not keyword_allow_list:
            keyword_allow_list = []
        if not keyword_block_list:
            keyword_block_list = []
        if not allow_list:
            allow_list = []
        if not block_list:
            block_list = []

        self.name = name
        self.allow_list = set(allow_list)
        self.block_list = set(block_list)
        self.is_active = False
        self.images: set[Image] = set()
        self.keywords_allow_list = [keyword.lower() for keyword in keyword_allow_list]
        self.keywords_block_list = [keyword.lower() for keyword in keyword_block_list]


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
