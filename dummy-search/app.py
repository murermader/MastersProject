import os
import time
from dotenv import load_dotenv
from flask import Flask, render_template, request, send_from_directory

load_dotenv()
app = Flask(__name__)


class Image:
    def __init__(self, id: str, label: str):
        self.id = id
        self.label = label
        self.image_path = os.path.join(os.environ["IMAGE_FOLDER"], f"P{id}.jpg")

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
    Image("199110765331", "Photographs of  French bird hunters  and Indians.")
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
        time.sleep(3)
        # Perform any processing or fetching results based on the search query
        # For simplicity, let's just pass the search query to the results template
        return render_template("results.html", search_query=search_query)


if __name__ == "__main__":
    app.run(debug=True)
