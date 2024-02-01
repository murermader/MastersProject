import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


if __name__ == "__main__":
    # Each input text should start with "query: " or "passage: ", even for non-English texts.
    # For tasks other than retrieval, you can simply use the "query: " prefix.
    input_texts = [
        "query: native",
        "passage: Photograph of unidentified Native Chief in traditional dress.",
        "passage: No information available.",
        "passage: Photograph of a piece of art work",
    ]

    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")

    # Tokenize the input texts
    batch_dict = tokenizer(
        input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:2] @ embeddings[2:].T) * 100
    print(scores.tolist())
    # Result: [[76.78286743164062, 75.78587341308594], [76.5135498046875, 79.4857406616211]]
    # ???
