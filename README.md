# Fake News Detection

A machine learning project that detects whether a news article is **Fake** or **Real** using four ML models and a simple web interface.

---

## Datasets

- **WELFake Dataset** — 72,134 articles (title, text, label)
  Download: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

- **ISOT Fake News Dataset** — ~44,898 articles (True.csv + Fake.csv)
  Download: https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/

Place the downloaded CSV files in the project root before training.

---

## Project Files

| File | Description |
|------|-------------|
| `frontend.html` | Web UI to test the model |
| `app.py` | Flask API server |
| `predict.py` | Loads models and runs predictions |
| `preprocess.py` | Cleans and prepares the dataset |
| `train.py` | Trains all 4 models |
| `requirements.txt` | Python dependencies |

---

## Models

| Key | Model |
|-----|-------|
| `lr` | Logistic Regression |
| `dt` | Decision Tree |
| `rf` | Random Forest |
| `gb` | Gradient Boosting |

---

## Setup

**1. Install dependencies**
```
pip install -r requirements.txt
```

**2. Train the models**
```
# WELFake only
python train.py

# ISOT only
python train.py --dataset isot

# Both datasets combined
python train.py --dataset combined
```

**3. Start the API server**
```
python app.py
```

**4. Open `frontend.html` in your browser**

---


## Requirements

- Python 3.9+
- See `requirements.txt` for all packages

---
