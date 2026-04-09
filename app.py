"""
app.py
------
Flask REST API serving the fake news detection model.

Endpoints:
  POST /predict   ->  { title, text, model? }  ->  { label, confidence, fake_prob, real_prob, model, model_key }
  GET  /health    ->  { status: "ok", models: [...] }
  GET  /models    ->  list of available models

Run:
  python app.py
  # Listens on http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import FakeNewsDetector, VALID_MODELS, DISPLAY_NAMES

app = Flask(__name__)
CORS(app)

# Load all models at startup so switching is instant (no per-request I/O)
print("[INFO] Loading all model artefacts ...")
detectors = {}
for key in VALID_MODELS:
    try:
        detectors[key] = FakeNewsDetector(model_key=key)
        print(f"  [+] {DISPLAY_NAMES[key]} loaded")
    except FileNotFoundError as e:
        print(f"  [!] {DISPLAY_NAMES[key]} not found — skipping ({e})")

if not detectors:
    raise RuntimeError("No models could be loaded. Run train.py first.")

print(f"[+] {len(detectors)} model(s) ready.\n")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": list(detectors.keys()),
    })


@app.route("/models", methods=["GET"])
def list_models():
    """Return available models so the frontend can build its selector."""
    return jsonify([
        {"key": k, "name": DISPLAY_NAMES[k]}
        for k in VALID_MODELS
        if k in detectors
    ])


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    title     = data.get("title", "")
    text      = data.get("text",  "")
    model_key = data.get("model", "lr")   # default to LR if not specified

    if not title and not text:
        return jsonify({"error": "Provide at least 'title' or 'text'."}), 400

    if model_key not in detectors:
        available = list(detectors.keys())
        return jsonify({
            "error": f"Model '{model_key}' not available. Available: {available}"
        }), 400

    result = detectors[model_key].predict(title=title, text=text)

    if "error" in result:
        return jsonify(result), 400

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)