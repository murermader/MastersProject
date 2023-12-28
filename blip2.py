from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor
import torch
from glob import glob


def answer_questions():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    model.to(device)

    paths = [
        # Answer: from the culture of the people of the island of
        "/Users/rb/Desktop/white_19754002002.jpg",
        # Answer: person
        "/Users/rb/Desktop/native_P199110763540.jpg",
        # Answer: The people in the photo are from the native americ
        "/Users/rb/Desktop/native_and_white_19754011059.jpg",
    ]

    prompts = [
        "Question: From which culture are these people? Answer:",
        "Question: From which culture are these people? Answer:",
        "Question: From which culture are these people? Answer:",
    ]

    for prompt, path in zip(prompts, paths):
        print(f"Path: {path}")
        print(f"Prompt: {prompt}")

        image = Image.open(path)
        inputs = processor(image, text=prompt, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=10)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(generated_text)


def generate_captions():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load model
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        # Requires packages accelerate, bitsandbytes and scipy + cuda
        # load_in_8bit=True,
        # device_map={"": 0},
        # torch_dtype=torch.float16,
    )
    model.to(device)
    for image_path in glob("galt/person/images/*.jpg"):
        print("Generating caption for image:", image_path)
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        print("Caption:", generated_text)


if __name__ == "__main__":
    # generate_captions()
    answer_questions()
