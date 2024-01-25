import os
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, request, send_from_directory
from clip import Clip
from data import datasets
from classes import Dataset, Image

load_dotenv(dotenv_path="../.env")
dotenv_error_message = (
    "Create file .env in the git repo root, with variable {key_value}"
)
if "IMAGE_FOLDER" not in os.environ:
    raise ValueError(
        dotenv_error_message.replace("{path}", "IMAGE_FOLDER=<path_to_galt_images>")
    )
if "LABELS_FILE" not in os.environ:
    raise ValueError(
        dotenv_error_message.replace(
            "{path}", "LABELS_FILE=<path_to_excel_file_with_labels>"
        )
    )

app = Flask(__name__)


def load_labels_for_datasets(datasets: list[Dataset]):
    """
    Go thou
    :param datasets:
    :return:
    """
    labels = pd.read_excel(os.environ["LABELS_FILE"])

    for dataset in datasets:
        images_with_labels = []
        for id in dataset.ids:
            label = labels[labels["Accession No."] == id]["Scope and Content"].item()

            if not label:
                print(f"No label for image with id [{id}]")
                label = ""

            image = Image(id, label)
            images_with_labels.append(image)

        dataset.images = images_with_labels


@app.route("/")
def index():
    return render_template("index.html", datasets=datasets)


# Serve images route
@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(os.environ["IMAGE_FOLDER"], filename)


# Add the necessary functions/variables to the Jinja2 environment
app.jinja_env.globals.update(os=os, serve_image=serve_image)


@app.route("/results", methods=["POST"])
def results():
    use_cosine_similarity = False

    if request.method == "POST":
        search_query = request.form["search_query"]
        search_query_embedding = clip.get_text_embedding(search_query)

        for idx, image in enumerate(images):
            print(f"Calculating similarity for image {idx + 1}")
            image_embedding = clip.get_image_embedding(image.image_path)
            label_embedding = clip.get_text_embedding(image.label)

            if use_cosine_similarity:
                image.image_similarity = clip.calc_cosine_similarity(
                    search_query_embedding, image_embedding
                )
                image.label_similarity = clip.calc_cosine_similarity(
                    search_query_embedding, label_embedding
                )
            else:
                image.image_similarity = clip.calc_l2_distance(
                    search_query_embedding, image_embedding
                )
                image.label_similarity = clip.calc_l2_distance(
                    search_query_embedding, label_embedding
                )

            # Shorten label
            max_length = 100
            if len(image.label) > max_length:
                image.label = image.label[:max_length] + "..."

        print("Image Similarity")
        if use_cosine_similarity:
            images_desc_by_cosine = sorted(
                images[:], key=lambda x: x.image_similarity, reverse=True
            )
            labels_desc_by_cosine = sorted(
                images[:], key=lambda x: x.label_similarity, reverse=True
            )

            for idx, image in enumerate(images_desc_by_cosine):
                image.rank_by_image = idx + 1
                print(
                    f"Rank: {idx + 1} Label: {image.label} Cosine Similarity: {image.image_similarity}"
                )

            print("Label Similarity")
            for idx, image in enumerate(labels_desc_by_cosine):
                image.rank_by_label = idx + 1
                print(
                    f"Rank: {idx + 1} Label: {image.label} Cosine Similarity: {image.label_similarity}"
                )

            return render_template(
                "results.html",
                search_query=search_query,
                images_by_image_similarity=images_desc_by_cosine,
                images_by_label_similarity=labels_desc_by_cosine,
            )
        else:
            images_asc_by_l2 = sorted(
                images[:], key=lambda x: x.image_similarity
            )
            labels_asc_by_l2 = sorted(
                images[:], key=lambda x: x.label_similarity
            )

            for idx, image in enumerate(images_asc_by_l2):
                image.rank_by_image = idx + 1
                print(
                    f"Rank: {idx + 1} Label: {image.label} L2 Distance: {image.image_similarity}"
                )

            print("Label Similarity")
            for idx, image in enumerate(labels_asc_by_l2):
                image.rank_by_label = idx + 1
                print(
                    f"Rank: {idx + 1} Label: {image.label} L2 Distance: {image.label_similarity}"
                )

            return render_template(
                "results.html",
                search_query=search_query,
                images_by_image_similarity=images_asc_by_l2,
                images_by_label_similarity=labels_asc_by_l2,
            )


if __name__ == "__main__":
    print("Loading CLIP...")
    clip = Clip()
    print("Done")

    print("Loading labels...")
    load_labels_for_datasets(datasets)
    print("Done")

    app.run(debug=True)
