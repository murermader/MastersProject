from glob import glob
from bs4 import BeautifulSoup
import re
from pathlib import Path
import os
import requests


def main():
    output_path = "./galt/person/images"
    Path(output_path).mkdir(parents=True, exist_ok=True)

    html_files = list(glob("./galt/person/*.html"))
    
    print("Getting image links from", len(html_files), "HTML files")
    
    relative_image_paths = []

    for html_file in html_files:
        with open (html_file, encoding="utf-8-sig", mode="r") as f:
            html = f.read()
        soup = BeautifulSoup(html, 'html.parser')

        gallery_divs = soup.find_all('div', class_='gallery-canvas')
        for div in gallery_divs:
            a_tag = div.find('a')

            if not a_tag:
                continue

            href = a_tag.get('href')
            relative_image_paths.append(href)
    

    print("Collected", len(relative_image_paths), "image paths")

    for relative_path in relative_image_paths:
        base_path = "https://collections.galtmuseum.com"
        full_path = base_path + relative_path

        # Remove watermark
        full_path = full_path.replace("&watermark=wmk", "")

        if match := re.search(r"JPEGS/(.*?\.jpg)?", full_path):
            image_name = match.group(1)
            save_at = os.path.join(output_path, image_name)

            print("Image Name:", image_name)
            
            if os.path.exists(save_at):
                print("Skip (already exists)")
                continue
            
            
            # Send an HTTP GET request to the image URL
            print("Downloading...", full_path)
            response = requests.get(full_path)

            if response.status_code == 200:
                with open(save_at, 'wb') as file:
                    file.write(response.content)
                print(f"Image saved at {save_at}")
            else:
                print(f"Error for [{full_path}]: {response.status_code}")






if __name__ == "__main__":
    main()
