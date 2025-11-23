import streamlit as st
import pandas as pd
import nltk
import glob
from nltk.corpus import stopwords
from gensim import corpora, models
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import spacy
import os

# ---------------------------
# INITIAL SETUP
# ---------------------------
st.set_page_config(page_title="Text Analysis App", layout="wide")

# Download NLTK resources if not available
nltk_packages = ["punkt", "stopwords"]
for pkg in nltk_packages:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except:
        nltk.download(pkg)

# Load SpaCy Model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.warning("Downloading SpaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# ---------------------------
# LOAD ALL TEXT FILES
# ---------------------------
def load_datasets():
    text_files = glob.glob("*.txt")
    documents = []

    for file_path in text_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append((os.path.basename(file_path), f.read()))
        except:
            with open(file_path, "r", encoding="latin-1") as f:
                documents.append((os.path.basename(file_path), f.read()))

    return documents


# ---------------------------
# PREPROCESSING
# ---------------------------
def preprocess(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]

    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]

    return tokens


# ---------------------------
# LDA TOPIC MODELING
# ---------------------------
def lda_topics(docs, num_topics=4):
    cleaned_docs = [preprocess(text) for _, text in docs]
    dictionary = corpora.Dictionary(cleaned_docs)
    corpus = [dictionary.doc2bow(doc) for doc in cleaned_docs]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    return lda_model.print_topics()


# ---------------------------
# SENTIMENT ANALYSIS
# ---------------------------
def sentiment_analysis(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        return "Positive 😊"
    elif sentiment < 0:
        return "Negative 😠"
    else:
        return "Neutral 😐"


# ---------------------------
# SUMMARIZATION
# ---------------------------
def summarize_text(text, sentence_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join([str(sentence) for sentence in summary])


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("📝 Text Analysis Application")
st.write("This app performs **Data Preprocessing**, **Topic Modeling (LDA)**, **Sentiment Analysis**, and **Summarization** on uploaded datasets.")

documents = load_datasets()

if not documents:
    st.error("No text files found in the repository! Upload .txt files to GitHub.")
else:
    st.success(f"Loaded {len(documents)} text files successfully.")

    # Display list of files
    st.subheader("📂 Available Documents")
    for i, (name, _) in enumerate(documents):
        st.write(f"**{i+1}. {name}**")

    st.divider()

    # ---------------------------
    # LDA TOPIC MODELING SECTION
    # ---------------------------
    st.header("📌 Topic Modeling (LDA)")
    if st.button("Run LDA Topic Modeling"):
        topics = lda_topics(documents)
        st.subheader("Extracted Topics:")
        for topic in topics:
            st.write(topic)

    st.divider()

    # ---------------------------
    # SENTIMENT + SUMMARY SECTION
    # ---------------------------
    st.header("📌 Sentiment Analysis & Summarization")

    file_names = [name for name, _ in documents]
    selected_file = st.selectbox("Select a document:", file_names)

    # Get selected text
    selected_text = ""
    for name, content in documents:
        if name == selected_file:
            selected_text = content

    st.subheader("📄 Document Preview")
    st.text_area("File Content", selected_text, height=200)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Analyze Sentiment"):
            result = sentiment_analysis(selected_text)
            st.subheader("Sentiment Result:")
            st.success(result)

    with col2:
        if st.button("Generate Summary"):
            summary = summarize_text(selected_text, sentence_count=4)
            st.subheader("Summary:")
            st.info(summary)
