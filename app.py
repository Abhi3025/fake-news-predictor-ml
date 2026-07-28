from pathlib import Path
import csv
import re
import sys

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).with_name("articles_dataset.csv")


st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
)


st.markdown(
    """
<style>
    header, footer, #MainMenu {
        visibility: hidden;
    }
    .stApp {
        background: #ffffff;
    }
    .block-container {
        max-width: 920px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTextInput input,
    .stTextArea textarea {
        border: 2px solid #000000 !important;
        border-radius: 8px !important;
        box-shadow: none !important;
        background: #ffffff !important;
    }
    .stButton button {
        border: 2px solid #000000 !important;
        border-radius: 8px !important;
        background: #ffffff !important;
        color: #000000 !important;
        width: 100%;
    }
    .stButton button:hover {
        background: #f5f5f5 !important;
    }
    .tiny-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #000000;
        margin: 0.15rem 0 0.25rem 0;
    }
    .app-title {
        font-size: 2.2rem;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: #000000;
        margin: 0 0 0.75rem 0;
    }
    .app-description {
        font-size: 1rem;
        line-height: 1.55;
        color: #111111;
        margin-bottom: 1rem;
    }
    .app-details {
        border-left: 4px solid #000000;
        padding-left: 0.9rem;
        margin-top: 0.9rem;
        color: #222222;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .sources-gap {
        height: 0.75rem;
    }
    .source-remove .stButton button {
        width: auto;
        min-height: 0;
        padding: 0.2rem 0.55rem;
        border-width: 1.6px !important;
        font-size: 0.82rem;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.15s ease-in-out;
    }
    div[data-testid="stExpander"]:hover .source-remove .stButton button,
    div[data-testid="stExpander"] details[open] .source-remove .stButton button {
        opacity: 1;
        pointer-events: auto;
    }
    div[data-testid="stExpander"] {
        border: 2px solid #000000;
        border-radius: 12px;
        background: #ffffff;
        box-shadow: 4px 4px 0 #000000;
        margin-bottom: 0.9rem;
    }
    div[data-testid="stExpander"] details summary {
        padding: 0.65rem 0.9rem;
        font-weight: 700;
    }
    div[data-testid="stExpander"] details div[data-testid="stExpanderDetails"] {
        padding: 0 0.9rem 0.9rem 0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z]", " ", str(text)).lower()
    words = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
    return " ".join(words)


@st.cache_resource(show_spinner=False)
def train_model():
    csv.field_size_limit(sys.maxsize)
    dataset = pd.read_csv(DATA_PATH, engine="python", on_bad_lines="skip")

    required_columns = {"title", "text", "label"}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing_columns))}")

    dataset = dataset.dropna(subset=["label"])
    dataset["title"] = dataset["title"].fillna("")
    dataset["text"] = dataset["text"].fillna("")
    dataset["content"] = (dataset["title"] + " " + dataset["text"]).map(clean_text)

    if len(dataset) > 6000:
        dataset = dataset.sample(n=6000, random_state=42)

    features = dataset["content"].values
    labels = dataset["label"].astype(int).values

    vectorizer = TfidfVectorizer(max_df=0.75, min_df=2, max_features=5000)
    transformed_features = vectorizer.fit_transform(features)

    x_train, x_test, y_train, y_test = train_test_split(
        transformed_features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    test_accuracy = accuracy_score(y_test, model.predict(x_test))
    return model, vectorizer, test_accuracy


def predict_article(model, vectorizer, title: str, text: str):
    content = clean_text(f"{title} {text}")
    features = vectorizer.transform([content])
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    fake_index = list(model.classes_).index(1)
    fake_probability = float(probabilities[fake_index])
    return prediction, fake_probability


def label_name(label: int) -> str:
    return "Fake" if label == 1 else "Real"


def confidence_name(confidence: float) -> str:
    return f"{confidence:.1%}"


def source_title(title: str) -> str:
    cleaned_title = title.strip()
    return cleaned_title if cleaned_title else "Untitled article"


def initialize_session_state() -> None:
    if "saved_sources" not in st.session_state:
        st.session_state.saved_sources = []
    if "latest_prediction" not in st.session_state:
        st.session_state.latest_prediction = None
    if "input_mode" not in st.session_state:
        st.session_state.input_mode = None


def add_saved_source(article: dict[str, str | float]) -> None:
    saved_sources = st.session_state.saved_sources
    duplicate = next(
        (
            source
            for source in saved_sources
            if source["title"] == article["title"]
            and source["text"] == article["text"]
            and source["label"] == article["label"]
        ),
        None,
    )
    if duplicate is None:
        saved_sources.append(article)


def render_prediction_card(article: dict[str, str | float]) -> None:
    st.markdown("<div class='tiny-label'>Prediction result</div>", unsafe_allow_html=True)
    label_text = f"{article['title']} · {article['label']} · {confidence_name(float(article['confidence']))}"
    with st.expander(label_text, expanded=True):
        st.write(f"Title: {article['title']}")
        st.write(f"Label: {article['label']}")
        st.write(f"Confidence: {confidence_name(float(article['confidence']))}")
        with st.expander("Article Text", expanded=False):
            st.write(str(article["text"]))

    if st.button("Save as source", key="save_current_source"):
        add_saved_source(article)
        st.success("Saved to sources below.")


def render_saved_sources() -> None:
    saved_sources = st.session_state.saved_sources
    if not saved_sources:
        return

    st.markdown("<div class='tiny-label'>Saved sources</div>", unsafe_allow_html=True)
    for source_index in range(len(saved_sources) - 1, -1, -1):
        source = saved_sources[source_index]
        label_text = f"{source['title']} · {source['label']} · {confidence_name(float(source['confidence']))}"
        with st.expander(label_text, expanded=False):
            st.write(f"Title: {source['title']}")
            st.write(f"Label: {source['label']}")
            st.write(f"Confidence: {confidence_name(float(source['confidence']))}")
            with st.expander("Article Text", expanded=False):
                st.write(source["text"])
            st.markdown("<div class='source-remove'>", unsafe_allow_html=True)
            remove_clicked = st.button("Remove source", key=f"remove_source_{source_index}")
            st.markdown("</div>", unsafe_allow_html=True)
            if remove_clicked:
                st.session_state.saved_sources.pop(source_index)
                st.rerun()


def fetch_article_from_url(url: str):
    response = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    title = ""
    meta_title = soup.find("meta", property="og:title")
    if meta_title and meta_title.get("content"):
        title = meta_title.get("content", "").strip()
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()

    article = soup.find("article")
    scope = article if article else (soup.body if soup.body else soup)
    text_parts = []
    seen_parts = set()
    for tag in scope.find_all(["h1", "h2", "h3", "p", "li", "div"]):
        part = tag.get_text(" ", strip=True)
        part = re.sub(r"\s+", " ", part).strip()
        if len(part) < 25:
            continue
        if part in seen_parts:
            continue
        seen_parts.add(part)
        text_parts.append(part)

    text = "\n\n".join(text_parts)

    if not text:
        fallback_text = soup.get_text(" ", strip=True)
        fallback_text = re.sub(r"\s+", " ", fallback_text).strip()
        if len(fallback_text) >= 100:
            text = fallback_text

    if not text:
        raise ValueError("Could not extract article text from this URL.")

    return title, text

try:
    sources_anchor = None
    prediction_anchor = None
    initialize_session_state()

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    intro_col, form_col = st.columns([1.05, 1.2], gap="large")

    with intro_col:
        st.markdown("<div class='app-title'>Fake News Detector</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='app-description'>"
            "Classify a headline plus article text, or paste a URL and let the app extract the article automatically."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='app-details'>"
            "<strong>Prediction</strong><br>"
            "The model uses logistic regression on TF-IDF text features built from the bundled article dataset. "
            "For responsiveness, training samples up to 6,000 articles from articles_dataset.csv before scoring new inputs."
            "<br><br>"
            "<strong>URL Processing</strong><br>"
            "When a URL is pasted in, the app fetches the page, reads the article title and body content from the HTML, "
            "and extracts the text needed to make a prediction."
            "<br><br>"
            "<strong>Organization</strong><br>"
            "After an article is saved, it is kept in the sources area so you can review saved items later."
            "</div>",
            unsafe_allow_html=True,
        )
        sources_anchor = st.container()

    title = ""
    text = ""
    url = ""

    with form_col:
        mode = st.radio("Input type", ["Text", "URL"], horizontal=True)

        if st.session_state.input_mode != mode:
            st.session_state.latest_prediction = None
            st.session_state.input_mode = mode

        if mode == "Text":
            st.markdown("<div class='tiny-label'>Headline</div>", unsafe_allow_html=True)
            title = st.text_input("", placeholder="Type the headline here", label_visibility="collapsed")

            st.markdown("<div class='tiny-label'>Text content</div>", unsafe_allow_html=True)
            text = st.text_area("", placeholder="Paste the article text here", height=220, label_visibility="collapsed")
        else:
            st.markdown("<div class='tiny-label'>Article URL</div>", unsafe_allow_html=True)
            url = st.text_input("", placeholder="Paste the article link here", label_visibility="collapsed")

        predict_clicked = st.button("Predict")
        prediction_anchor = st.container()

    model, vectorizer, test_accuracy = train_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

if predict_clicked:
    if mode == "URL":
        if not url.strip():
            st.warning("Add an article URL before predicting.")
        else:
            try:
                fetched_title, fetched_text = fetch_article_from_url(url.strip())
                prediction, fake_probability = predict_article(model, vectorizer, fetched_title, fetched_text)
                label = label_name(prediction)
                confidence = fake_probability if prediction == 1 else 1.0 - fake_probability
                st.session_state.latest_prediction = {
                    "title": source_title(fetched_title),
                    "original_title": fetched_title.strip(),
                    "label": label,
                    "confidence": confidence,
                    "text": fetched_text.strip(),
                    "source": "URL",
                }
            except Exception as error:
                st.error(f"Could not read article from URL: {error}")
    else:
        if not title.strip() and not text.strip():
            st.warning("Add a title, text, or both before predicting.")
        else:
            prediction, fake_probability = predict_article(model, vectorizer, title, text)
            label = label_name(prediction)
            confidence = fake_probability if prediction == 1 else 1.0 - fake_probability
            st.session_state.latest_prediction = {
                "title": source_title(title),
                "original_title": "",
                "label": label,
                "confidence": confidence,
                "text": text.strip(),
                "source": "Text",
            }

if prediction_anchor is not None:
    with prediction_anchor:
        if st.session_state.latest_prediction:
            render_prediction_card(st.session_state.latest_prediction)

if sources_anchor is not None:
    with sources_anchor:
        if st.session_state.saved_sources:
            st.markdown("<div class='sources-gap'></div>", unsafe_allow_html=True)
        render_saved_sources()
