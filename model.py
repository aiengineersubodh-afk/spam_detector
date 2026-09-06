import os

# ============================================================
# TensorFlow memory optimization for Railway (1 GB RAM)
# ============================================================

os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pickle
import gdown

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import preprocess_text


MAX_LEN = 100

MODEL_PATH = "email_spam_detection.keras"

FILE_ID = "1klLTDj9ghspnmsYs5QU090b81iaA_Aut"

URL = f"https://drive.google.com/uc?id={FILE_ID}"


# ============================================================
# Download model if it doesn't exist
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("=" * 50)
    print("Downloading TensorFlow model...")
    print("=" * 50)

    gdown.download(URL, MODEL_PATH, quiet=False)

    print("=" * 50)
    print("Model downloaded successfully.")
    print("=" * 50)


# ============================================================
# Load model for inference only
# compile=False saves memory
# ============================================================

print("Loading TensorFlow model...")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("TensorFlow model loaded successfully.")


# ============================================================
# Load tokenizer
# ============================================================

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


# ============================================================
# Prediction function
# ============================================================

def predict_email(text):

    cleaned = preprocess_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    # Direct model call instead of model.predict()
    output = model(padded, training=False)

    probability = float(output.numpy()[0][0])

    if probability >= 0.5:
        return "Spam", probability
    else:
        return "Ham", 1 - probability