from transformers import CLIPProcessor, CLIPModel, AutoTokenizer

if __name__ == '__main__':
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    text = "Photograph of Indians and French hunters in Lethbridge. The hunters came from France to shoot upland game birds in southern Alberta and were feted by Lethbridge boosters. Upon arrival a private reception was held for the group sponsored by the Alberta Fish and Game Association where they were presented with honorary memberships in the Lethbridge Fish and game Association by president Joe Balla. In addition to shopping"

    print(f"Length: {len(text)}")
    inputs = tokenizer([text], padding=True, return_tensors="pt")

    print(f"Length Inputs: {len(inputs)}")

    text_features = model.get_text_features(**inputs)


