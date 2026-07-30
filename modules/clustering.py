from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd


class PaperClustering:

    def cluster(self, documents, n_clusters=5):

        if not documents:
            return pd.DataFrame()

        texts = [doc.get("text", "") for doc in documents]

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=1000
        )

        X = vectorizer.fit_transform(texts)

        n_clusters = min(n_clusters, len(texts))

        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X)

        return pd.DataFrame({
            "Paper": [
                doc.get("name", f"Paper {i+1}")
                for i, doc in enumerate(documents)
            ],
            "Cluster": labels
        })