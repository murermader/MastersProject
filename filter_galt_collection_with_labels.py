import os
from pathlib import Path
import shutil
from dotenv import load_dotenv
import pandas as pd

if __name__ == "__main__":
    load_dotenv()

    labels_df = pd.read_excel(os.environ["LABELS_FILE"])
    # labels_df = labels_df.dropna(columns=["Scope and Content"])

    keywords_to_filter = ["first nation", "indian", "native", "blackfoot", "tribe"]

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

            keyword_directory = os.path.join(os.environ["IMAGE_FOLDER"], keyword)
            Path(keyword_directory).mkdir(parents=True, exist_ok=True)
            file_path = os.path.join(os.environ["IMAGE_FOLDER"], f"P{id}.jpg")

            if not os.path.isfile(file_path):
                print(f"File with Accession No. [{id}] could not be found at {file_path}")
                continue

            shutil.copy(file_path, keyword_directory)
