from clip import Clip
from diskcache import Cache
from glob import glob
from classes import Image

if __name__ == "__main__":
    clip = Clip()
    counter = 0
    with Cache("diskcache") as cache:
        for image_path in glob("images/*.jpg"):
            image = Image(image_path)

            if image.id not in cache:
                counter += 1
                print(f"Create embedding for [{image.id}]")
                cache.add(image.id, clip.get_image_embedding(image_path))
    print(f"Done (added {counter} embeddings)")