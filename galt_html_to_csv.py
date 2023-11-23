import argparse
import os
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from glob import glob


def hmtl_to_csv(folder_path: str):
    glob_path = os.path.join(folder_path, "*.html")
    details_list = []

    for file in glob(glob_path):
        filename = os.path.basename(file)
        with open(file, encoding="utf-8-sig", mode="r") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")

        not_found_span = soup.find("span", class_="and-text-notfound")
        if not_found_span:
            print(f"No info for {filename}")
            continue

        more_details_section = soup.find('div', class_='and-citation-detail')
        if not more_details_section:
            raise ValueError(f"The file [{filename}] does not contain a details section?")

        details = {}
        for dl in more_details_section.find_all('dl'):
            dt = dl.find('dt')
            dd = dl.find('dd')
            if dt and dd:
                key = dt.get_text().strip()
                value = dd.get_text().strip()
                details[key] = value

        details_list.append(details)

    df = pd.DataFrame(details_list)

    output_dir = Path(folder_path).parent
    input_folder_name = os.path.basename(folder_path)
    output_path = os.path.join(output_dir, f"{input_folder_name}.csv")
    df.to_csv(output_path, encoding="utf-8-sig")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("galt_html_to_csv")
    parser.add_argument(
        "--input",
        metavar="PATH_TO_FOLDER",
        help="Path to folder that contains HTML files from the GALT Collection Results page",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    hmtl_to_csv(args.input)
    print("Done")
