from flask import Flask, render_template, request, jsonify
from model import predict_email

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    email = data.get("email", "")

    if email.strip() == "":
        return jsonify({
            "error": "Please enter an email."
        })

    prediction, confidence = predict_email(email)

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence * 100, 2)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)