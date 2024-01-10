import os
from pathlib import Path
import shutil
from dotenv import load_dotenv
import pandas as pd
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


def extract_nouns_and_adjectives(sentence):
    # Tokenize the sentence
    tokenizer = RegexpTokenizer(r"\w+")
    words = tokenizer.tokenize(sentence)

    # Remove stopwords
    words = [
        word.lower() for word in words if word.lower() not in stopwords.words("english")
    ]

    # Perform Part-of-Speech tagging
    tagged_words = pos_tag(words)

    # Extract nouns and adjectives
    nouns = [word for word, pos in tagged_words if pos.startswith("N")]
    adjectives = [word for word, pos in tagged_words if pos.startswith("J")]

    return nouns, adjectives


if __name__ == "__main__":
    load_dotenv()

    labels_df = pd.read_excel(os.environ["LABELS_FILE"])
    # labels_df = labels_df.dropna(columns=["Scope and Content"])

    keywords_to_filter = [
        # A
        "basketball",
        "sport",
        "canadian",
        "american",
        "european",
        
        # B
        "first nation",
        "indian",
        "native",
        "blackfoot",
        "tribe",
    ]

    for keyword in keywords_to_filter:
        filtered_labels = labels_df[
            ~labels_df["Scope and Content"].isna()
            & labels_df["Scope and Content"].str.contains(keyword, case=False)
        ]
        print(f"Found {len(filtered_labels)} labels that contain [{keyword}]")

        for label, id in zip(
            filtered_labels["Scope and Content"], filtered_labels["Accession No."]
        ):
            print(f"{id}: {label}")
            # nouns, adjectives = extract_nouns_and_adjectives(label)
            # print("Nouns:", nouns)
            # print("Adjectives:", adjectives)
            # continue

            keyword_directory = os.path.join(os.environ["IMAGE_FOLDER"], keyword)
            Path(keyword_directory).mkdir(parents=True, exist_ok=True)
            file_path = os.path.join(os.environ["IMAGE_FOLDER"], f"P{id}.jpg")

            if not os.path.isfile(file_path):
                print(
                    f"File with Accession No. [{id}] could not be found at {file_path}"
                )
                continue

            # sanitize label (windows)
            label = label.replace("\n", " ")
            label = label.replace("\t", " ")
            for char in ["\\", "/", "*", "?", "<", ">", "|", '"', "'", ":"]:
                label = label.replace(char, "")

            if len(label) + len(keyword_directory) >= 255:
                chars_to_remove = (len(label) + len(keyword_directory)) - (
                    255 + len(" - ") + len(".jpg") + len(id) + 1
                )
                label = label[:-chars_to_remove]

            ext = "jpg" if label.endswith(".") else ".jpg"
            output_file = os.path.join(keyword_directory, str(id) + " - " + label + ext)
            shutil.copyfile(file_path, output_file)
