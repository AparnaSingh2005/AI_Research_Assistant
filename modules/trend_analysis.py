import re
from collections import Counter
import pandas as pd


class TrendAnalysis:
    def __init__(self):
        self.stop_words = {
            "the", "a", "an", "and", "or", "of", "to", "in", "for",
            "on", "with", "by", "at", "is", "are", "was", "were",
            "be", "been", "this", "that", "these", "those", "it",
            "as", "from", "into", "using", "we", "our", "their"
        }

    def top_keywords(self, documents, top_n=30):
        words = []

        for doc in documents:
            tokens = re.findall(r"[A-Za-z]{3,}", doc.get("text", "").lower())
            tokens = [t for t in tokens if t not in self.stop_words]
            words.extend(tokens)

        return Counter(words).most_common(top_n)

    def dataframe(self, documents, top_n=30):
        return pd.DataFrame(
            self.top_keywords(documents, top_n),
            columns=["Keyword", "Frequency"]
        )