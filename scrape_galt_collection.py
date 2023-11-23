import argparse
import os
import re
import requests
from glob import glob


def scrape(folder_path: str):
    glob_path = os.path.join(folder_path, "P*.jpg")

    for file in glob(glob_path):
        file_output_path = file[:-4] + ".html"

        if os.path.exists(file_output_path):
            # File has already been scraped.
            continue

        # Remove numbering like (2)
        filename = os.path.basename(file)
        accesion_number = re.sub(r"\(\d+\)", "", filename)
        accesion_number = accesion_number.lstrip("P")
        accesion_number = accesion_number.rstrip(".jpg")

        if not accesion_number:
            raise ValueError("Something went wrong getting the accession number...")

        print(f"GET: {accesion_number} ({filename})")

        url = f"https://collections.galtmuseum.com/en/list?q={accesion_number}&p=1&ps=20"
        response = requests.get(url)

        if response.status_code == 200:
            # Save scraped info with same filename as input with different extension
            with open(file_output_path, 'wb') as f:
                f.write(response.content)
        else:
            raise ValueError(f"ERR: {response.status_code}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("geocode_addresses")
    parser.add_argument(
        "--input",
        metavar="PATH_TO_FOLDER",
        help="Path to folder that contains JPG files with the format P<ID>.jpg",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    scrape(args.input)
