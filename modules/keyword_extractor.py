"""
keyword_extractor.py

Extracts keywords from research papers using KeyBERT.
"""
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import streamlit as st


@st.cache_resource
def load_keyword_model():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return KeyBERT(model=embedding_model)


class KeywordExtractor:

    def __init__(self):
        self.model = load_keyword_model()

    

    # ----------------------------------------------------
    # Extract keywords
    # ----------------------------------------------------

    def extract(
        self,
        text,
        top_n=10
    ):

        if not text.strip():
            return []

        keywords = self.model.extract_keywords(

            text,

            keyphrase_ngram_range=(1, 2),

            stop_words="english",

            top_n=top_n

        )

        return [

            {
                "keyword": keyword,
                "score": round(score, 3)
            }

            for keyword, score in keywords

        ]