import nltk

packages = [
    "punkt",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng"
]

for p in packages:
    nltk.download(p)


import nltk

nltk.download("averaged_perceptron_tagger_eng")