import os
import time
from dotenv import load_dotenv
from flask import Flask, render_template, request, send_from_directory
from clip import Clip

load_dotenv(dotenv_path="../.env")
if "IMAGE_FOLDER" not in os.environ:
    raise ValueError("Create file .env in the git repo root, wich variable IMAGE_FOLDER=<path_to_galt_images>")

clip = Clip()
app = Flask(__name__)


class Image:
    def __init__(self, id: str, label: str):
        self.id = id
        self.label = label
        self.image_path = os.path.join(os.environ["IMAGE_FOLDER"], f"P{id}.jpg")

        # Will be computed later
        self.image_similarity = None
        self.label_similarity = None
        self.rank_by_image = None
        self.rank_by_label = None

        if not os.path.isfile(self.image_path):
            raise ValueError(f"File does not exist: {self.image_path}")


# fmt: off
images: list[Image] = [
    # Basketbal Images
    Image("199110767087", "Photograph of Central Catholic High School basketball team."),
    Image("1991107619078", "Unidentified Lethbridge Community College Kodiak basketball player."),
    Image("199110766232", "Photograph of  a two day annual Christmas Senior Men’s Basketball Tournament held at the Civic Centre."),

    # First Nation Images
    Image("199110766930", "Photograph of an two unidentified students in First Nations costume serving tea at a tea and bake sale that was part of Brotherhood Week activities at Hamilton Junior High School."),
    Image("19752990002", "Photograph of a group of First Nations people sitting by a wagon and some horses.  There is a horse drawn buggy in the background."),
    Image("199110765331", "Photograph  of Indians and French hunters in Lethbridge. The hunters came from France to shoot upland game birds in southern Alberta and were feted by Lethbridge boosters.  Upon arrival a private reception was held for the group sponsored by the Alberta Fish and Game Association where they were presented with honorary memberships in the Lethbridge Fish and game Association by president Joe Balla.  In addition to shopping, the hunters toured the city,  and attended a civic reception as well as curling demonstrations.  The group also saw  traditional Indian dancing, branding and rodeo demonstrations before returning to Paris.")
]
# fmt: on


@app.route("/")
def index():
    return render_template("index.html", images=images)


# Serve images route
@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(os.environ["IMAGE_FOLDER"], filename)


# Add the necessary functions/variables to the Jinja2 environment
app.jinja_env.globals.update(os=os, serve_image=serve_image)


@app.route("/results", methods=["POST"])
def results():
    if request.method == "POST":
        search_query = request.form["search_query"]
        search_query_embedding = clip.get_text_embedding(search_query)

        for idx, image in enumerate(images):
            print(f"Calculating similarity for image {idx+1}")
            image_embedding = clip.get_image_embedding(image.image_path)
            label_embedding = clip.get_text_embedding(image.label)
            image.image_similarity = clip.calc_cosine_similarity(
                search_query_embedding, image_embedding
            )
            image.label_similarity = clip.calc_cosine_similarity(
                search_query_embedding, label_embedding
            )

        images_by_image_similarity = sorted(
            images[:], key=lambda x: x.image_similarity, reverse=True
        )
        images_by_label_similarity = sorted(
            images[:], key=lambda x: x.label_similarity, reverse=True
        )

        # images.sort(key=lambda x: x.image_similarity, reverse=True)
        print("Image Similarity")
        for idx, image in enumerate(images_by_image_similarity):
            image.rank_by_image = idx + 1
            print(
                f"Rank: {idx +1} Label: {image.label} Cosine Similarity: {image.image_similarity}"
            )

        print("Label Similarity")
        for idx, image in enumerate(images_by_label_similarity):
            image.rank_by_label = idx + 1
            print(
                f"Rank: {idx +1} Label: {image.label} Cosine Similarity: {image.label_similarity}"
            )

        # Perform any processing or fetching results based on the search query
        # For simplicity, let's just pass the search query to the results template
        return render_template(
            "results.html",
            search_query=search_query,
            images_by_image_similarity=images_by_image_similarity,
            images_by_label_similarity=images_by_label_similarity,
        )


if __name__ == "__main__":
    app.run(debug=True)
