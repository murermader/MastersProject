import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn


def get_wordnet_pos(treebank_tag):
    """
    Return the WordNet POS tag from the Penn Treebank tag
    """
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN  # Default to noun if unknown


def extract_keywords(sentence):
    """
    Simplify the sentence to extract key nouns and adjectives
    """
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)

    # Extract nouns and adjectives
    keywords = [word for word, pos in tagged if pos.startswith('NN') or pos.startswith('JJ')]
    return ' '.join(keywords)


if __name__ == '__main__':
    with open(r"C:\Users\rb\Desktop\labels.txt", encoding="utf-8-sig", mode="r") as f:
        labels = [l.strip() for l in f.readlines()]

    for label in labels:
        print("Label     :", label)
        print("Simplified:", extract_keywords(label))
        print("")
