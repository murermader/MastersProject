import os
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, request, send_from_directory
from clip import Clip
from data import datasets
from classes import Dataset, Image
from diskcache import Cache
from itertools import chain

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


def load_or_create_image_embeddings(datasets: list[Dataset]):
    with Cache("diskcache") as cache:
        for dataset in datasets:
            for image in dataset.images:
                if image.id in cache:
                    image.embedding = cache[image.id]
                else:
                    image.embedding = clip.get_image_embedding(image.image_path)
                    cache.add(image.id, image.embedding)


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
    if not request.method == "POST":
        return

    search_query = request.form["search_query"]
    similarity_measurement = request.form["similarity_measurement"]
    print(similarity_measurement)

    search_query_embedding = clip.get_text_embedding(search_query)
    images = list(
        chain.from_iterable(
            [[image for image in dataset.images] for dataset in datasets]
        )
    )

    for idx, image in enumerate(images):
        if similarity_measurement == "Cosine Similarity":
            image.image_similarity = clip.calc_cosine_similarity(
                search_query_embedding, image.embedding
            )
        else:
            image.image_similarity = clip.calc_l2_distance(
                search_query_embedding, image.embedding
            )

        # Shorten label
        max_length = 100
        if len(image.label) > max_length:
            image.label = image.label[:max_length] + "..."

    if similarity_measurement == "Cosine Similarity":
        images = sorted(images[:], key=lambda x: x.image_similarity, reverse=True)
    else:
        images = sorted(images[:], key=lambda x: x.image_similarity)

    for idx, image in enumerate(images):
        image.rank_by_image = idx + 1
        print(
            f"Rank: {idx + 1} Label: {image.label} Cosine Similarity: {image.image_similarity}"
        )
    return render_template(
        "results.html",
        search_query=search_query,
        similarity_measurement=similarity_measurement,
        images_by_image_similarity=images,
    )


if __name__ == "__main__":
    print("Loading CLIP...")
    clip = Clip()
    print("Done")

    print("Loading labels...")
    load_labels_for_datasets(datasets)
    print("Done")

    print("Loading image embeddings...")
    load_or_create_image_embeddings(datasets)
    print("Done")

    app.run(debug=True)
