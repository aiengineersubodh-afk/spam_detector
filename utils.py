import re
import nltk

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))


def remove_html(text):

    return re.sub(r"<.*?>", "", text)


def remove_url(text):

    return re.sub(r"http?://\S+|www\.\S+", "", text)


def remove_punctuation(text):

    return re.sub(r"[^\w\s]", "", text)


def get_wordnet_pos(tag):

    if tag.startswith("J"):
        return wordnet.ADJ

    elif tag.startswith("V"):
        return wordnet.VERB

    elif tag.startswith("N"):
        return wordnet.NOUN

    elif tag.startswith("R"):
        return wordnet.ADV

    return wordnet.NOUN


def preprocess_text(text):

    text = text.lower()

    text = remove_html(text)

    text = remove_url(text)

    text = remove_punctuation(text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    words = [
    lemmatizer.lemmatize(word)
    for word in words
    ]

    return " ".join(words)