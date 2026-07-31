# Fake News Detector

An end-to-end machine learning web application that classifies news articles as **Real** or **Fake** using Natural Language Processing (NLP) and Logistic Regression.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fake-news-predictor-ml-lvjbbtnheucxqekyt3bgd4.streamlit.app/)

## Live Demo

### **https://fake-news-predictor-ml-lvjbbtnheucxqekyt3bgd4.streamlit.app/**

---

## Features

- Predict whether a news article is **Real** or **Fake**
- Analyze articles directly from a news article URL
- Automatically extracts article titles and body text from webpages
- Displays prediction confidence
- Save analyzed articles during the current session
- Fast, responsive Streamlit interface

---

## Overview

Fake News Detector combines machine learning with an interactive web interface to make fake news detection simple and accessible.

The application supports two prediction methods:

### Text Input

Users can paste:

- Article headline
- Article body

The model processes the text and predicts whether the article is **Real** or **Fake**.

### URL Input

Instead of copying article text manually, users can simply paste a news article URL.

The application automatically:

- Downloads the webpage
- Extracts the article title
- Extracts the article body
- Runs the prediction
- Displays the prediction confidence

---

## Machine Learning Pipeline

The prediction model follows the following workflow:

1. Load the dataset
2. Clean and preprocess article text
3. Remove stopwords
4. Convert text into TF-IDF feature vectors
5. Train a Logistic Regression classifier
6. Predict whether new articles are Real or Fake

---

## Model Performance

| Metric | Training | Testing |
|---------|---------:|--------:|
| Accuracy | **94.83%** | **91.14%** |

The model demonstrates strong generalization with minimal overfitting on unseen articles.

---

## Technologies

### Machine Learning

- Python
- scikit-learn
- pandas
- NumPy
- NLTK
- TF-IDF Vectorizer
- Logistic Regression

### Web Application

- Streamlit
- Requests
- BeautifulSoup4

---

## Project Structure

```text
fake-news-predictor/
│
├── app.py                 # Streamlit web application
├── main.ipynb             # Original notebook implementation
├── articles_dataset.csv   # Training dataset
├── requirements.txt       # Python dependencies
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Abhi3025/fake-news-predictor-ml.git
cd fake-news-predictor-ml
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## Dataset

- **7,000+ labeled news articles**
- Features:
  - Article title
  - Article text
- Labels:
  - **0 = Real**
  - **1 = Fake**

---

## Technical Highlights

- **91.14% test accuracy**
- TF-IDF feature engineering
- Logistic Regression classifier
- Natural Language Processing (NLP)
- Automatic webpage scraping
- Real-time predictions
- Interactive Streamlit web application
- URL-based article analysis
- Session-based saved article management

---

## Future Improvements

- Fine-tune transformer models (BERT, RoBERTa)
- Train on a larger and more diverse dataset
- Support additional article formats
- Store prediction history in a database
- Add user authentication
- Explain predictions using feature importance

---

## Try It Yourself

Click below to use the application without installing anything locally.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fake-news-predictor-ml-lvjbbtnheucxqekyt3bgd4.streamlit.app/)