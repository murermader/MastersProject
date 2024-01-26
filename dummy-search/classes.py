import os


class Dataset:
    def __init__(self, name: str, ids: list[str]):
        self.name = name
        self.ids = ids
        self.is_active = False

        # Will be computed at runtime
        self.images: list[Image] = []


class Image:
    def __init__(self, id: str, label: str):
        self.id = id
        self.label = label
        self.image_path = os.path.join(os.environ["IMAGE_FOLDER"], f"P{id}.jpg")

        # Will be computed later
        self.embedding = None
        self.image_similarity = None
        self.label_similarity = None
        self.rank_by_image = None
        self.rank_by_label = None

        if not os.path.isfile(self.image_path):
            raise ValueError(f"File does not exist: {self.image_path}")
