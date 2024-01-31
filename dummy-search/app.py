import os
from glob import glob

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, request, send_from_directory
from clip import Clip
from data import datasets
from classes import Image
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
images = []


def load_all_images():
    global images

    # Load labels and convert the pandas datatable to a dictionary, because the lookup in a pandas datatable
    # is very slow.
    labels = pd.read_excel(os.environ["LABELS_FILE"])
    df_unique = labels.drop_duplicates(subset="Accession No.")
    labels_dict = {
        row["Accession No."]: row["Scope and Content"]
        for index, row in df_unique.iterrows()
    }

    image_glob = os.path.join(os.environ["IMAGE_FOLDER"], "*.jpg")
    with Cache("diskcache") as cache:

        image_paths = list(glob(image_glob))
        for idx, image_path in enumerate(image_paths):
            print(f"{idx} / {len(image_paths)}")
            image = Image(image_path)

            # Load label
            image.label = labels_dict.get(image.id, "")

            # Load embeddingn from cache if possible, otherwhise compute it.
            if image.id in cache:
                image.embedding = cache[image.id]
            else:
                print(f"Loading embedding for {image.image_path} and iamge id {image.id}")
                image.embedding = clip.get_image_embedding(image.image_path)
                cache.add(image.id, image.embedding)

            # To determine relevancy, we need to know in which dataset the images is in.
            for dataset in datasets:
                if image.id in dataset.ids:
                    image.from_dataset.append(dataset.name)
                    dataset.images.append(image)
            images.append(image)

    print(f"Loaded {len(images)} images")


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
    global images

    if not request.method == "POST":
        return

    search_query = request.form["search_query"]
    similarity_measurement = request.form["similarity_measurement"]
    print(similarity_measurement)
    search_query_embedding = clip.get_text_embedding(search_query)

    for image in images:
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
        image.rank = idx + 1
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

    print("Loading images...")
    load_all_images()
    print("Done")

    app.run(debug=True)
