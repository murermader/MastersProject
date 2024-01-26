import os


class Dataset:
    def __init__(self, name: str, ids: list[str]):
        self.name = name
        self.ids = ids
        self.is_active = False

        # Will be computed at runtime
        self.id_to_image: dict[str, Image] = []


class Image:
    def __init__(self, id: str, label: str):
        self.id = id
        self.label = label
        self.image_path = os.path.join(os.environ["IMAGE_FOLDER"], f"P{id}.jpg")

        # Will be computed later
        self.from_dataset = ""
        self.embedding = None
        self.image_similarity = None
        self.rank = None

        if not os.path.isfile(self.image_path):
            raise ValueError(f"File does not exist: {self.image_path}")
