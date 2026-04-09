"""
predict.py
----------
Load saved model artefacts and predict on new text.
Supports selecting from 4 models: lr, dt, rf, gb.

Note: GB uses a separate smaller vectorizer (vectorizer_gb.joblib)
      trained on 5K features to avoid memory issues during training.

Usage (CLI):
  python predict.py --title "Vaccines cause autism" --text "Scientists say..." --model lr
  python predict.py --text "Full article body here..." --model rf
"""

import argparse
import joblib
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

MODEL_DIR        = "model"
VECTORIZER_PATH  = os.path.join(MODEL_DIR, "vectorizer.joblib")       # lr / dt / rf
VECTORIZER_GB    = os.path.join(MODEL_DIR, "vectorizer_gb.joblib")    # gb only

VALID_MODELS = ("lr", "dt", "rf", "gb")
DISPLAY_NAMES = {
    "lr": "Logistic Regression",
    "dt": "Decision Tree",
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
}


# -- Preprocessing ------------------------------------------------------------

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)


# -- Model Loader -------------------------------------------------------------

class FakeNewsDetector:
    def __init__(self, model_key: str = "lr"):
        if model_key not in VALID_MODELS:
            raise ValueError(f"model must be one of {VALID_MODELS}, got '{model_key}'")

        clf_path = os.path.join(MODEL_DIR, f"classifier_{model_key}.joblib")

        # GB uses its own smaller vectorizer
        vec_path = VECTORIZER_GB if model_key == "gb" else VECTORIZER_PATH

        if not os.path.exists(vec_path):
            raise FileNotFoundError(f"Vectorizer not found at {vec_path}. Run train.py first.")
        if not os.path.exists(clf_path):
            raise FileNotFoundError(f"Classifier not found at {clf_path}. Run train.py first.")

        self.model_key  = model_key
        self.model_name = DISPLAY_NAMES[model_key]
        self.vectorizer = joblib.load(vec_path)
        self.classifier = joblib.load(clf_path)
        self.is_gb      = (model_key == "gb")

    def predict(self, title: str = "", text: str = "") -> dict:
        """
        Returns:
          {
            "label":      "FAKE" | "REAL",
            "confidence": float (0-100),
            "fake_prob":  float (0-1),
            "real_prob":  float (0-1),
            "model":      str,
            "model_key":  str,
          }
        """
        combined = clean_text(f"{title} {text}".strip())
        if not combined:
            return {"error": "No input text provided."}

        # GB needs dense input; others work with sparse
        if self.is_gb:
            vec = self.vectorizer.transform([combined]).toarray()
        else:
            vec = self.vectorizer.transform([combined])

        proba = self.classifier.predict_proba(vec)[0]
        pred  = self.classifier.predict(vec)[0]

        # Use clf.classes_ to map probabilities correctly (1=fake, 0=real)
        classes   = list(self.classifier.classes_)
        fake_prob = float(proba[classes.index(1)])
        real_prob = float(proba[classes.index(0)])
        label     = "FAKE" if pred == 1 else "REAL"
        confidence = float(max(fake_prob, real_prob)) * 100

        return {
            "label":      label,
            "confidence": round(confidence, 2),
            "fake_prob":  round(fake_prob, 4),
            "real_prob":  round(real_prob, 4),
            "model":      self.model_name,
            "model_key":  self.model_key,
        }


# -- CLI ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict fake vs real news")
    parser.add_argument("--title", default="", help="News headline")
    parser.add_argument("--text",  default="", help="News body text")
    parser.add_argument(
        "--model", default="lr", choices=VALID_MODELS,
        help="Model to use: lr | dt | rf | gb (default: lr)"
    )
    args = parser.parse_args()

    detector = FakeNewsDetector(model_key=args.model)
    result   = detector.predict(title=args.title, text=args.text)

    print("\n" + "="*45)
    if "error" in result:
        print(f"  Error: {result['error']}")
    else:
        symbol = "+" if result["label"] == "REAL" else "!"
        print(f"  [{symbol}]  Model      : {result['model']}")
        print(f"       Verdict    : {result['label']}")
        print(f"       Confidence : {result['confidence']}%")
        print(f"       Fake prob  : {result['fake_prob']}")
        print(f"       Real prob  : {result['real_prob']}")
    print("="*45 + "\n")