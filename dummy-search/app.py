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
                    dataset.images.append(image)
                    continue

                # Add image based on keyword in label
                for keyword in dataset.keywords:
                    # Keywords needs to be surrounded by space or end of string, otherwise it would
                    # be possible to match part of another word
                    if re.search(
                        rf"(\s+|\Z|\.|,|;|:){keyword.lower()}(\s+|\Z|\.|,|;|:)",
                        image.label.lower(),
                    ):
                        image.from_dataset.add(dataset.name)
                        dataset.images.append(image)

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
    images_copy = copy.deepcopy(images)
    print(f"Rank images for query [{search_query}] with [{similarity_measurement}]")
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

    print(f"Relevant datasets [{', '.join(relevant_datasets)}]")

    for idx, image in enumerate(sorted_images):
        # Determine relevancy based
        image.is_relevant = len(image.from_dataset.intersection(relevant_datasets)) > 0

        # Add ranking information
        image.rank = idx + 1

    return sorted_images


@app.route("/results", methods=["POST"])
def results():
    if not request.method == "POST":
        return

    search_query = request.form["search_query"]
    similarity_measurement = request.form["similarity_measurement"]
    sorted_images = rank_images(images, search_query, similarity_measurement, clip)

    # Calculate P@K
    for k in [5, 10, 25, 50, 100, 200, 500]:
        p_at_k = precision_at_k(sorted_images, k)
        print(f"P@{k}: {p_at_k}")

    # Create plots
    create_cumulative_distribution_function(sorted_images, search_query)
    # create_box_plot(sorted_images, search_query)  # looks bad when there are outlies
    create_histogram(sorted_images, search_query)
    # create_scatterplot(sorted_images, search_query)  # not useful
    create_roc_curve(sorted_images, search_query)

    # Limit number of images shown in results
    max_images = 100
    if len(sorted_images) > max_images:
        print(f"Limit results to top {max_images} images.")
        sorted_images = sorted_images[:max_images]

    return render_template(
        "results.html",
        search_query=search_query,
        similarity_measurement=similarity_measurement,
        images_by_image_similarity=sorted_images,
        info_text=f"{sum([1 for image in sorted_images if image.is_relevant])} out of the top {max_images} images are relevant",
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


def save_plot(plt, plot: str, query: str):
    Path("plots/").mkdir(exist_ok=True)
    # Specify the SVG file path
    svg_file_path = f"plots/{plot}_{query}.svg"  # Adjust path as needed
    # Save the plot as an SVG file
    plt.savefig(svg_file_path, format="svg")
    # Clear the figure to free memory
    plt.clf()


def create_box_plot(images: list[Image], query: str):
    # Filter the list to include only relevant images
    relevant_images = [img for img in images if img.is_relevant]

    # Extract the ranks of the relevant images
    ranks = [img.rank for img in relevant_images]

    # Generate the box plot
    plt.figure(figsize=(8, 6))
    plt.boxplot(
        ranks, vert=True, patch_artist=True, showfliers=False
    )  # `vert=True` makes the box plot vertical

    plt.ylabel("Rank")
    # plt.ylim([1, len(images)])

    # Customize the x-axis
    plt.xticks(
        [1], ["Relevant Images"]
    )  # You can adjust this if you have multiple categories

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    save_plot(plt, "box", query)


def create_cumulative_distribution_function(images: list[Image], query: str):
    # Filter the list to include only relevant images
    relevant_images = [img for img in images if img.is_relevant]

    # Extract the ranks of the relevant images
    ranks = [img.rank for img in relevant_images]

    # Sort the ranks in ascending order
    sorted_ranks = np.sort(ranks)

    # Calculate the CDF values
    cdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)

    # Generate the CDF plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        sorted_ranks, cdf, linestyle="-", linewidth=2
    )  # Adjusted for a smooth line

    plt.ylabel("CDF (Proportion of Images)")

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    save_plot(plt, "cdf", query)


def create_histogram(images: list[Image], query: str):
    total_bins = 20

    # Assuming ranks are from 1 to 20000
    max_rank = images[-1].rank

    # Filter the list to include only relevant images
    relevant_images = [img for img in images if img.is_relevant]

    # Extract the ranks of the relevant images
    ranks = [img.rank for img in relevant_images]

    # Create bins. Since ranks are from 1 to max_rank, we divide this range into `total_bins` bins
    bins = np.linspace(1, max_rank, total_bins + 1)

    # Generate the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(ranks, bins=bins, alpha=0.7, edgecolor="black")
    plt.ylim([0, 250])

    # Calculate the median rank of the relevant images
    median_rank = np.median(ranks)
    # Add a vertical line for the median rank
    plt.axvline(median_rank, color="red", linestyle="dashed", linewidth=1)
    plt.text(
        median_rank,
        plt.gca().get_ylim()[1] * 0.95,
        f" Median = {round(median_rank, 2)}",
        color="red",
    )

    plt.ylabel("Relevant Images Per Bin")
    plt.xticks(bins)

    # plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

    save_plot(plt, "histogram", query)


def create_scatterplot(images: list[Image], query: str):
    # Filter the list to include only relevant images
    relevant_images = [img for img in images if img.is_relevant]

    # Extract the ranks of the relevant images
    ranks = [img.rank for img in relevant_images]

    # Create an index list for the x-axis
    indices = list(range(1, len(relevant_images) + 1))

    # Generate the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(indices, ranks, alpha=0.6)

    plt.title("Distribution of Relevant Image Ranks")
    plt.xlabel("Relevant Image Index")
    plt.ylabel("Rank")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Optional: adjust the y-axis to show the rank in reverse order (if lower ranks are considered better)
    plt.gca().invert_yaxis()

    save_plot(plt, "scatter", query)


def create_roc_curve(images: list[Image], query: str):
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

    plt.figure()
    plt.plot(x, y, color="darkorange", lw=2, label=f"Area = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")

    save_plot(plt, "roc", query)


if __name__ == "__main__":
    print("Loading CLIP...")
    clip = Clip()
    print("Done")

    print("Loading images...")
    load_all_images()
    print("Done")

    app.run(debug=False, port=3000)
