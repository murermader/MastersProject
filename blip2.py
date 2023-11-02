from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from glob import glob


def main():
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
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ].strip()
        print("Caption:", generated_text)


if __name__ == "__main__":
    main()
