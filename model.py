import pickle
import os
import gdown

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import preprocess_text

MAX_LEN = 100

MODEL_PATH = "email_spam_detection.keras"

FILE_ID = "1klLTDj9ghspnmsYs5QU090b81iaA_Aut"

URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):

    print("=" * 50)
    print("Downloading TensorFlow model...")
    print("=" * 50)

    gdown.download(URL, MODEL_PATH, quiet=False)

    print("=" * 50)
    print("Model downloaded successfully.")
    print("=" * 50)

model = load_model(MODEL_PATH)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


def predict_email(text):

    cleaned = preprocess_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    probability = float(model.predict(padded, verbose=0)[0][0])

    if probability >= 0.5:
        return "Spam", probability
    else:
        return "Ham", 1 - probability