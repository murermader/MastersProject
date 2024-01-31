import os
import re
from clip import Clip
from diskcache import Cache
from glob import glob

if __name__ == "__main__":
    clip = Clip()
    with Cache("diskcache") as cache:
        for image_path in glob("images/*.jpg"):
            filename = os.path.basename(image_path)
            accesion_number = (
                re.sub(r"\(\d+\)", "", filename).lstrip("P").rstrip(".jpg")
            )
            
            if accesion_number not in cache:
                print(f"Create embedding for [{accesion_number}]")
                embedding = clip.get_image_embedding(image_path)
                cache.add(accesion_number, embedding)
