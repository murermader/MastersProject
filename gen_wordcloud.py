import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random

load_dotenv()
labels = pd.read_excel(os.environ["LABELS_FILE"])
labels = labels.replace(np.nan, "", regex=True)
df_unique = labels.drop_duplicates(subset="Accession No.")

text = df_unique["Scope and Content"].values


def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(30, 70)


wordcloud = WordCloud(margin=10, background_color="#FFFFFF", width=800, height=500).generate(str(text))

default_colors = wordcloud.to_array()
plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=3),
           interpolation="bilinear")

plt.imshow(wordcloud)
plt.axis("off")
wordcloud.to_file("wordcloud.png")
plt.show()

