import copy
import os
import random
import re
from collections import Counter
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, request, send_from_directory
from matplotlib import rcParams

from clip import Clip
from data import datasets
from classes import Image, Dataset
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
    for label in labels_dict.values():
        if label == "No information available.":
            continue

        label = label.replace("\n", "")
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
                    dataset.images.add(image)
                    continue

                # Do not add image based on keyword in label
                is_blocked = False
                for keyword in dataset.keywords_block_list:
                    # Keywords needs to be surrounded by space or end of string, otherwise it would
                    # be possible to match part of another word
                    if re.search(
                        rf"(\s+|\Z|\.|,|;|:){keyword.lower()}(\s+|\Z|\.|,|;|:)",
                        image.label.lower(),
                    ):
                        is_blocked = True
                        break
                if is_blocked:
                    continue

                # Add image based on keyword in label
                for keyword in dataset.keywords_allow_list:
                    # Keywords needs to be surrounded by space or end of string, otherwise it would
                    # be possible to match part of another word
                    if re.search(
                        rf"(\s+|\Z|\.|,|;|:){keyword.lower()}(\s+|\Z|\.|,|;|:)",
                        image.label.lower(),
                    ):
                        # if len(dataset.images) < 233:
                        #     ...
                        image.from_dataset.add(dataset.name)
                        dataset.images.add(image)

            images.append(image)

    print(f"Loaded {len(images)} images")
    return images


@app.route("/")
def index():
    return render_template("index.html", datasets=datasets)


# Serve images route
@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(os.environ["IMAGE_FOLDER"], filename)


# Add the necessary functions/variables to the Jinja2 environment
app.jinja_env.globals.update(os=os, serve_image=serve_image)


def rank_images(
    images: list[Image], search_query: str, similarity_measurement: str, clip: Clip
):
    print(f"Rank images for query [{search_query}] with [{similarity_measurement}]")
    search_query_embedding = clip.get_text_embedding(search_query)

    for image in images:
        if similarity_measurement == "Cosine Similarity":
            image.image_similarity = clip.calc_cosine_similarity(
                search_query_embedding, image.embedding
            )
        elif similarity_measurement == "Random":
            image.image_similarity = random.random()
        else:
            image.image_similarity = clip.calc_l2_distance(
                search_query_embedding, image.embedding
            )

        # Shorten label
        max_length = 500
        if len(image.label) > max_length:
            image.label = image.label[:max_length] + "..."

    if (
        similarity_measurement == "Cosine Similarity"
        or similarity_measurement == "Random"
    ):
        sorted_images = sorted(
            images[:], key=lambda x: x.image_similarity, reverse=True
        )
    else:
        sorted_images = sorted(images[:], key=lambda x: x.image_similarity)

    # Figure out which datasets are relevant for the search
    relevant_datasets = set()
    for dataset in datasets:
        for search_term in search_query.lower().split(" "):
            if search_term in dataset.keywords_allow_list:
                relevant_datasets.add(dataset.name)
                continue

    print(f"Relevant datasets [{', '.join(relevant_datasets)}]")

    for idx, image in enumerate(sorted_images):
        # Determine relevancy based
        image.is_relevant = len(image.from_dataset.intersection(relevant_datasets)) > 0

        # Add ranking information
        image.rank = idx + 1

    return sorted_images, ", ".join(relevant_datasets)


@app.route("/results", methods=["POST"])
def results():
    if not request.method == "POST":
        return

    search_query = request.form["search_query"]
    if not search_query:
        return

    queries = [q.strip() for q in search_query.split(";")]
    similarity_measurement = request.form["similarity_measurement"]
    options = set()

    # Sanity Checks
    include_correct = request.form.get("correct-system") is not None
    include_incorrect = request.form.get("incorrect-system") is not None
    include_half_correct = request.form.get("half_correct-system") is not None
    random = request.form.get("random-system") is not None
    normalize = request.form.get("normalize-system") is not None

    data = {}
    result_query, result_images = None, None

    print(f"Queries: {queries}")
    for q in queries:
        images_copy: list[Image] = copy.deepcopy(images)
        datasets_copy: list[Dataset] = copy.copy(datasets)

        if normalize:
            options.add("normalized")

            # 1. Get relevant datasets
            datasets_to_check = set()
            for dataset in datasets_copy:
                for query in queries:
                    for search_term in query.lower().split(" "):
                        if search_term in dataset.keywords_allow_list:
                            datasets_to_check.add(dataset)

            # 2. Find out which has the least amount of images
            least_image_count = min([len(d.images) for d in datasets_to_check])

            # 3. Reduce the size of the other datasets
            for dataset in datasets_to_check:
                images_to_remove = len(dataset.images) - least_image_count
                if images_to_remove > 0:
                    removed_counter = 0
                    for image in images_copy:
                        if removed_counter >= images_to_remove:
                            # We have removed enough images, we are done!
                            break
                        if dataset.name in image.from_dataset:
                            image.from_dataset.remove(dataset.name)
                            removed_counter += 1

            # 4. Check if successful
            relevant_image_counter = {}
            for rd in datasets_to_check:
                print(f"Init [{rd.name}] to 0.")
                relevant_image_counter[rd.name] = 0

            for image in images_copy:
                for fd in image.from_dataset:
                    for rd in datasets_to_check:
                        if fd == rd.name:
                            relevant_image_counter[fd] += 1

            print(f"Normalized Dataset Sizes: {relevant_image_counter}")

        if random:
            options.add("random")
            similarity_measurement = "Random"

        sorted_images, label = rank_images(
            images_copy, q, similarity_measurement, clip
        )

        # Calculate P@K
        for k in [5, 10, 25, 50, 100, 200, 500]:
            p_at_k = precision_at_k(sorted_images, k)
            print(f"P@{k}: {p_at_k}")

        if result_query is None:
            result_query = label
            result_images = sorted_images
        print(f"data[{label}] = {len(sorted_images)} images")
        data[label] = sorted_images

    print(f"Keys={list(data.keys())}")

    # Create plots
    create_histogram(data, options)
    create_roc_curve(data, options)

    # Limit number of images shown in results
    max_images = 100
    if len(result_images) > max_images:
        print(f"Limit results to top {max_images} images.")
        result_images = result_images[:max_images]

    return render_template(
        "results.html",
        search_query=result_query,
        similarity_measurement=similarity_measurement,
        images_by_image_similarity=result_images,
        info_text=f"{sum([1 for image in result_images if image.is_relevant])} out of the top {max_images} images are relevant",
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
        images_by_image_similarity=sorted_images,
    )


def precision_at_k(images: list[Image], k: int):
    images = images[:k]

    correct = 0
    for image in images:
        if image.is_relevant:
            correct += 1

    return correct / k


def save_plot(plt, plot: str, query: str, options: set[str]):
    Path("plots/").mkdir(exist_ok=True)
    # Specify the SVG file path
    svg_file_path = f"plots/{plot}_{query}"

    if options:
        svg_file_path += f"_{'-'.join(list(options))}"

    # Save the plot as an SVG file
    plt.savefig(svg_file_path + ".svg", format="svg")
    # Clear the figure to free memory
    plt.clf()


def create_histogram(data: dict[str, list[Image]], options: set[str]):
    total_bins = 20

    max_rank = 0
    all_ranks = []

    for query, images in data.items():
        if max_rank == 0:
            # Only calculate once, it is the same for all queries
            max_rank = max(img.rank for img in images)

        relevant_images = [img for img in images if img.is_relevant]
        ranks = [img.rank for img in relevant_images]
        all_ranks.append(ranks)

    bins = np.linspace(1, max_rank, total_bins + 1)

    labels = []
    for query, ranks in zip(data.keys(), all_ranks):
        labels.append(f"{query} ( Median = {int(np.median(ranks))})")

    default_width, default_height = rcParams["figure.figsize"]
    plt.figure(figsize=(default_width * 2, default_height * 1.5))
    plt.hist(
        all_ranks, bins=bins, alpha=0.7, histtype="bar", edgecolor="black", label=labels
    )

    plt.ylabel("Relevant Images Per Bin")
    plt.legend()  # Add a legend to show query names

    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

    save_plot(plt, "histogram", "-".join(data.keys()), options)


def create_roc_curve(data: dict[str, list[Image]], options: set[str]):
    for query, images in data.items():
        step_size = 10
        thresholds = [t + step_size for t in range(0, len(images), step_size)]
        y = [0]  # tpr
        x = [0]  # fpr

        positives = sum(1 for image in images if image.is_relevant)
        negatives = sum(1 for image in images if not image.is_relevant)

        for t in thresholds:
            true_positives = 0
            false_positives = 0

            for image in images[:t]:
                if image.is_relevant:
                    true_positives += 1
                else:
                    false_positives += 1

            tpr = true_positives / positives if positives else 0
            fpr = false_positives / negatives if negatives else 0

            y.append(tpr)
            x.append(fpr)

        roc_auc = auc(x, y)

        # Plot the ROC curve for the current query
        plt.plot(x, y, lw=2, label=f"{query} (area = {roc_auc:.2f})")

    # Plot the diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    save_plot(plt, "roc", "-".join(data.keys()), options)


if __name__ == "__main__":
    print("Loading CLIP...")
    clip = Clip()
    print("Done")

    print("Loading images...")
    load_all_images()
    print("Done")

    app.run(debug=False, port=3000)
