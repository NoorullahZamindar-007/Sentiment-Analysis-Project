from pathlib import Path
import re

import joblib
import nltk
from flask import Flask, jsonify, render_template, request
from nltk.corpus import stopwords
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"


def load_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


STOP_WORDS = load_stopwords()
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

app = Flask(__name__)


def validate_artifacts():
    issues = []

    if not MODEL_PATH.exists():
        issues.append(f"Missing model file: {MODEL_PATH.name}")
    if not VECTORIZER_PATH.exists():
        issues.append(f"Missing vectorizer file: {VECTORIZER_PATH.name}")

    try:
        check_is_fitted(vectorizer, attributes=["vocabulary_"])
        check_is_fitted(vectorizer._tfidf, attributes=["idf_"])
    except (AttributeError, NotFittedError):
        issues.append(
            "The saved TF-IDF vectorizer is not fitted. Re-export `tfidf_vectorizer.pkl` "
            "from the notebook after `vectorizer.fit_transform(...)` runs."
        )

    if not hasattr(model, "predict"):
        issues.append("The saved model does not expose a `predict()` method.")

    return issues


ARTIFACT_ISSUES = validate_artifacts()


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    return " ".join(words)


def get_confidence(features):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        return round(float(max(probabilities)) * 100, 2)
    return None


def analyze_sentiment(text: str):
    if ARTIFACT_ISSUES:
        return {
            "ok": False,
            "error": "Model files are not ready for prediction.",
            "details": ARTIFACT_ISSUES,
        }

    if text is None or not text.strip():
        return {
            "ok": False,
            "error": "Please enter some text before analyzing.",
        }

    cleaned_text = clean_text(text)
    if not cleaned_text:
        return {
            "ok": False,
            "error": "The text became empty after cleaning. Try adding more meaningful words.",
        }

    try:
        features = vectorizer.transform([cleaned_text])
    except NotFittedError:
        return {
            "ok": False,
            "error": "The saved TF-IDF vectorizer is not fitted.",
            "details": [
                "Open the training notebook and save the fitted `vectorizer` again.",
            ],
        }

    prediction = int(model.predict(features)[0])
    confidence = get_confidence(features)

    return {
        "ok": True,
        "sentiment": "Positive" if prediction == 1 else "Negative",
        "label": prediction,
        "confidence": confidence,
        "cleaned_text": cleaned_text,
        "original_text": text,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    text = ""

    if request.method == "POST":
        text = request.form.get("text", "")
        result = analyze_sentiment(text)

    return render_template(
        "index.html",
        result=result,
        text=text,
        artifact_issues=ARTIFACT_ISSUES,
    )


@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    result = analyze_sentiment(text)
    status_code = 200 if result["ok"] else 400
    return jsonify(result), status_code


if __name__ == "__main__":
    app.run(debug=True)
