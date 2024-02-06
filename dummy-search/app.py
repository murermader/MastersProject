import copy
import os
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, request, send_from_directory
from clip import Clip
from data import datasets
from classes import Image
from diskcache import Cache
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize

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

    lenghts = []
    wordcount_label = []
    with open("labels.txt", mode="w", encoding="utf-8-sig") as f:
        for label in labels_dict.values():
            if label == "No information available.":
                continue

            label = label.replace("\n", "")
            f.write(label + "\n")
            wordcount_label.append(len(word_tokenize(label)))
            lenghts.append(len(label))

    print(f"Average Label Length: {sum(lenghts) / len(lenghts)}")
    print(f"Average Label Words : {sum(wordcount_label) / len(wordcount_label)}")

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
                # Do not add image based on block list
                if image.id in dataset.block_list:
                    continue

                # Add image based on allow list
                if image.id in dataset.allow_list:
                    image.from_dataset.add(dataset.name)
                    dataset.images.append(image)
                    continue

                # Add image based on keyword in label
                for keyword in dataset.keywords:
                    # Keywords needs to be surrounded by space or end of string, otherwise it would
                    # be possible to match part of another word
                    if re.search(rf"(\s+|\Z|\.|,|;|:){keyword.lower()}(\s+|\Z|\.|,|;|:)", image.label.lower()):
                        image.from_dataset.add(dataset.name)
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

    images_copy = copy.deepcopy(images)

    if not request.method == "POST":
        return

    search_query = request.form["search_query"]
    similarity_measurement = request.form["similarity_measurement"]
    print(similarity_measurement)
    search_query_embedding = clip.get_text_embedding(search_query)

    for image in images_copy:
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
            images_copy[:], key=lambda x: x.image_similarity, reverse=True
        )
    else:
        sorted_images = sorted(images_copy[:], key=lambda x: x.image_similarity)

    # Figure out which datasets are relevant for the search
    relevant_datasets = set()
    for dataset in datasets:
        for search_term in search_query.lower().split(" "):
            if search_term in dataset.keywords:
                relevant_datasets.add(dataset.name)
                continue

    print(f"[{search_query}] Relevant datasets [{', '.join(relevant_datasets)}]")

    for idx, image in enumerate(sorted_images):
        # Determine relevancy based
        image.is_relevant = len(image.from_dataset.intersection(relevant_datasets)) > 0

        # Add ranking information
        image.rank = idx + 1

    for k in [5, 10, 25, 50, 100, 200, 500]:
        p_at_k = precision_at_k(sorted_images, k)
        print(f"P@{k}: {p_at_k}")

    create_roc_curve_manually(sorted_images, search_query)

    max_images = 100
    if len(sorted_images) > max_images:
        print(f"Limit results to top {max_images} images.")
        sorted_images = sorted_images[:max_images]

    info_text = f"{sum([1 for image in sorted_images if image.is_relevant])} out of the top {max_images} images are relevant"
    print(info_text)

    return render_template(
        "results.html",
        search_query=search_query,
        similarity_measurement=similarity_measurement,
        images_by_image_similarity=sorted_images,
        info_text=info_text,
    )


@app.route("/broken")
def broken():
    global images
    images_copy = copy.deepcopy(images)

    for image in images_copy:
        if not image.label:
            continue

        label_embedding = clip.get_text_embedding(image.label)
        image.image_similarity = clip.calc_cosine_similarity(
            label_embedding, image.embedding
        )

    sorted_images = sorted(
        images_copy[:], key=lambda x: x.image_similarity, reverse=True
    )

    for idx, image in enumerate(sorted_images):
        # Add ranking information
        image.rank = idx + 1

    max_images = 100
    if len(sorted_images) > max_images:
        print(f"Limit results to top {max_images} images.")
        sorted_images = sorted_images[:max_images]

    return render_template(
        "results.html",
        similarity_measurement="Cosine Similarity",
        images_by_image_similarity=sorted_images
    )


def precision_at_k(images: list[Image], k: int):
    images = images[:k]

    correct = 0
    for image in images:
        if image.is_relevant:
            correct += 1

    return correct / k


def create_roc_curve_manually(images: list[Image], query: str):
    # Limit graph to top x images
    # images = images[:500]

    step_size = 10
    tresholds = [t + step_size for t in range(0, len(images), step_size)]
    y = [0]  # tpr
    x = [0]  # fpr

    positives = sum([1 for image in images if image.is_relevant])
    negatives = sum([1 for image in images if not image.is_relevant])

    for t in tresholds:
        true_positives = 0
        false_positives = 0

        for image in images[:t]:
            if image.is_relevant:
                true_positives += 1
            else:
                false_positives += 1

        if positives == 0:
            tpr = 0
        else:
            tpr = true_positives / positives

        if negatives == 0:
            fpr = 0
        else:
            fpr = false_positives / negatives

        y.append(tpr)
        x.append(fpr)

    roc_auc = auc(x, y)

    # plt.rcParams["svg.fonttype"] = "none"
    plt.figure()
    plt.plot(x, y, color="darkorange", lw=2, label=f"Area = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    # plt.title(f"Receiver Operator Characteristic for [{query}]")
    plt.legend(loc="lower right")

    Path("plots/").mkdir(exist_ok=True)

    # Specify the SVG file path
    svg_file_path = f"plots/roc_plot_{query}.svg"  # Adjust path as needed

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
