import os
from glob import glob
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, request, send_from_directory
from clip import Clip
from data import datasets
from classes import Image
from diskcache import Cache
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


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
    labels = labels.replace(np.nan, "", regex=True)
    df_unique = labels.drop_duplicates(subset="Accession No.")
    labels_dict = {
        row["Accession No."]: row["Scope and Content"]
        for index, row in df_unique.iterrows()
    }

    image_glob = os.path.join(os.environ["IMAGE_FOLDER"], "*.jpg")
    with Cache("diskcache") as cache:
        image_paths = list(glob(image_glob))
        for idx, image_path in enumerate(image_paths):
            image = Image(image_path)

            # Load label
            image.label = labels_dict.get(image.id, "")

            # Load embeddingn from cache if possible, otherwhise compute it.
            if image.id in cache:
                image.embedding = cache[image.id]
            else:
                print(
                    f"Loading embedding for {image.image_path} and iamge id {image.id}"
                )
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
        max_length = 500
        if len(image.label) > max_length:
            image.label = image.label[:max_length] + "..."

    if similarity_measurement == "Cosine Similarity":
        sorted_images = sorted(
            images[:], key=lambda x: x.image_similarity, reverse=True
        )
    else:
        sorted_images = sorted(images[:], key=lambda x: x.image_similarity)

    all_relevant_keywords = set()

    # Figure out which datasets are relevant for the search
    relevant_datasets = set()
    for dataset in datasets:
        # If one keyword matches, extract all others
        if search_query.lower() in dataset.keywords:
            for keyword in dataset.keywords:
                all_relevant_keywords.add(keyword)

        # for keyword in dataset.keywords:
        #     # Instead of checking for equality, we could also check for
        #     # an embedding distance between the search term and the keyword?
        #     # Either using clip, or something else like multilingual-e5-large
        #     if search_query.lower() == keyword.lower():
        #         relevant_datasets.add(dataset.name)
        #         break

    print(f"Relevant Keywords [{all_relevant_keywords}]")
    print(f"Relevant datasets [{', '.join(relevant_datasets)}]")

    for idx, image in enumerate(sorted_images):
        # Determine relevancy:

        # Relevancy based only on keyword occurence in label
        for keyword in all_relevant_keywords:
            if keyword in image.label:
                image.is_relevant = True
                break

        # Based on affiliation with a dataset
        for dataset in image.from_dataset:
            if dataset in relevant_datasets:
                image.is_relevant = True
                break

        # Add ranking information
        image.rank = idx + 1

    create_roc_curve(images, search_query)

    max_images = 100
    if len(sorted_images) > max_images:
        print(f"Limit results to top {max_images} images.")
        sorted_images = sorted_images[:max_images]

    print(
        f"{sum([1 for image in sorted_images if image.is_relevant])} out of the top {max_images} images are relevant"
    )

    return render_template(
        "results.html",
        search_query=search_query,
        similarity_measurement=similarity_measurement,
        images_by_image_similarity=sorted_images,
    )


def create_roc_curve(images: list[Image], query: str):
    ranks = np.array([image.rank for image in images])
    labels = np.array([1 if image.is_relevant else 0 for image in images])

    fpr, tpr, thresholds = roc_curve(labels, ranks)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic for [{query}]")
    plt.legend(loc="lower right")

    # Specify the SVG file path
    svg_file_path = f"roc_curve_{query}.svg"  # Adjust path as needed

    # Save the plot as an SVG file
    plt.savefig(svg_file_path, format="svg")

    # Clear the figure to free memory
    plt.clf()


if __name__ == "__main__":
    print("Loading CLIP...")
    clip = Clip()
    print("Done")

    print("Loading images...")
    load_all_images()
    print("Done")

    app.run(debug=False)
