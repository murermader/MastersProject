from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_path_dog = "/Users/rb/Desktop/puppy.jpg"
image_path_cat_and_dog = "/Users/rb/Desktop/cat_and_dog.jpg"

paths = [
    "/Users/rb/Desktop/native_and_white_19754011059.jpg",
    "/Users/rb/Desktop/native_P199110763540.jpg",
    "/Users/rb/Desktop/white_19754002002.jpg",
]

for path in paths:
    print(f"Path: {path}")
    image = Image.open(path)
    # image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(text=["a photo of a caucasian person", "a photo of native person"], images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    print(f"Probabilities: {probs}")
