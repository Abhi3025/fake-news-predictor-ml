# Machine Learning Fake News Detector

A machine learning project that classifies news articles as real or fake using Natural Language Processing (NLP) techniques and Logistic Regression. The model achieves **91.14% accuracy** on test data by analyzing textual content and applying advanced text preprocessing methods.

## Project Overview

This project addresses the critical issue of misinformation by building an automated system to detect fake news. Using a dataset of 7,000 news articles, the model processes text through stemming, stopword removal, and TF-IDF vectorization before applying logistic regression for binary classification.

## Technologies Used

- **Python 3.11**
- **Machine Learning:** scikit-learn
- **Data Processing:** pandas, numpy
- **NLP:** NLTK (Natural Language Toolkit)
- **Text Vectorization:** TF-IDF Vectorizer
- **Model:** Logistic Regression

## Dataset

- **Size:** 7,000 news articles
- **Features:** Title, Text content
- **Labels:** Binary classification (0 = Real, 1 = Fake)
- **Split:** 90% training, 10% testing with stratified sampling

## Key Features

### Advanced Text Preprocessing
- **Stemming:** Reduces words to root forms using Porter Stemmer
- **Stopword Removal:** Eliminates common English words that don't contribute to classification
- **Text Cleaning:** Removes non-alphabetic characters and converts to lowercase
- **Missing Value Handling:** Fills null values with empty strings

### Machine Learning Pipeline
- **TF-IDF Vectorization:** Converts text to numerical features (49,005 features)
- **Logistic Regression:** Binary classification with L2 regularization
- **Stratified Splitting:** Ensures balanced representation in train/test sets

## Model Performance

| Metric | Training Data | Test Data |
|--------|---------------|-----------|
| Accuracy| 94.83%       | **91.14%** |

The model demonstrates strong generalization with minimal overfitting, indicating robust performance on unseen data.

## Project Structure

```
fake-news-detector/
├── fake_news_detector.ipynb    # Main notebook with complete implementation
├── articles_dataset.csv        # Dataset (7,000 articles)
├── README.md                   # Project documentation
└── requirements.txt            # Dependencies
```

## How It Works

1. **Data Loading:** Imports CSV dataset with news articles and labels
2. **Preprocessing:** 
   - Merges title and text content
   - Applies stemming and removes stopwords
   - Handles missing values
3. **Vectorization:** Converts text to TF-IDF numerical features
4. **Training:** Fits Logistic Regression model on processed data
5. **Evaluation:** Tests model performance and provides prediction system


## Future Enhancements

- **Deep Learning Models:** Implement LSTM or BERT for improved accuracy
- **Feature Engineering:** Add sentiment analysis and readability scores
- **Web Interface:** Create Flask/Streamlit app for real-time predictions
- **Larger Dataset:** Expand training data for better generalization
- **Multi-class Classification:** Extend beyond binary to detect specific types of misinformation

## Technical Insights

- **Vocabulary Size:** 49,005 unique terms after preprocessing
- **Feature Selection:** TF-IDF effectively captures important text patterns
- **Model Choice:** Logistic Regression provides interpretable results with strong performance
- **Cross-validation:** Stratified split ensures balanced class representation
