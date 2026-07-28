# 📰 Fake News Detector

A machine learning web application that classifies news articles as **Real** or **Fake** using Natural Language Processing (NLP) and Logistic Regression. The application is built with **Streamlit**, allowing users to either paste article text directly or provide a news article URL for automatic extraction and classification. The underlying model achieves **91.14% accuracy** on unseen test data. The application trains a Logistic Regression model on TF-IDF features and provides a simple interface for users to classify articles or URLs.

## Demo Features

* Paste a news headline and article text for instant predictions
* Analyze articles directly from a URL
* Automatic article title and content extraction
* Displays prediction confidence
* Save previously analyzed articles for later reference
* Clean, responsive Streamlit interface

---

## Project Overview

This project was originally developed to explore machine learning techniques for fake news detection. A Logistic Regression classifier is trained on thousands of labeled news articles using TF-IDF vectorization and text preprocessing.

The project has since been expanded into a fully interactive **Streamlit web application**, making the model accessible through a simple user interface without requiring users to run Jupyter notebooks or interact with Python code directly. The app supports both manual text input and automatic article extraction from URLs before performing classification.

---

## Technologies Used

### Machine Learning

* Python 3
* scikit-learn
* pandas
* NumPy
* NLTK
* TF-IDF Vectorizer
* Logistic Regression

### Web Application

* Streamlit
* Requests
* BeautifulSoup4

---

## Dataset

* **7,000 labeled news articles**
* Features:

  * Article Title
  * Article Text
* Labels:

  * **0 = Real**
  * **1 = Fake**

---

## Key Features

### Machine Learning Pipeline

* Text preprocessing

  * Lowercasing
  * Stopword removal
  * Text cleaning
* TF-IDF feature extraction
* Logistic Regression classifier
* Automatic model training when the application launches

### Streamlit Web Application

Users can classify articles in two different ways:

#### 1. Manual Text Input

Enter:

* Article headline
* Article body

The model predicts whether the article is real or fake and displays its confidence score.

#### 2. URL Analysis

Instead of copying an article, users can simply paste a news article URL.

The application automatically:

* Downloads the webpage
* Extracts the title
* Extracts the article body
* Runs the prediction
* Displays the result

This eliminates the need to manually copy and paste article content.

### Saved Sources

After making a prediction, users can save analyzed articles within the current session for future reference. Saved articles include:

* Title
* Prediction
* Confidence score
* Full article text

---

## Model Performance

| Metric   | Training |    Testing |
| -------- | -------: | ---------: |
| Accuracy |   94.83% | **91.14%** |

The model demonstrates strong generalization with minimal overfitting on unseen articles.

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

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## How It Works

1. Load the labeled news dataset
2. Clean and preprocess article text
3. Convert text into TF-IDF feature vectors
4. Train a Logistic Regression classifier
5. Accept user input (text or URL)
6. Predict whether the article is Real or Fake
7. Display prediction confidence
8. Optionally save the analyzed article during the session

---

## Future Improvements

* Deploy the application publicly
* Train on a larger and more diverse dataset
* Experiment with transformer-based models (BERT, RoBERTa)
* Improve article extraction for additional news websites
* Display probability distributions and model explanations
* Add article history and persistent storage

---

## Technical Highlights

* **91.14% classification accuracy**
* TF-IDF feature engineering
* Logistic Regression classifier
* Automatic URL scraping with BeautifulSoup
* Session-based article management
* Responsive Streamlit interface
* End-to-end fake news prediction workflow from raw article or URL
