"""
similarity.py

Paper Similarity Module
"""

import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


class PaperSimilarity:

    def __init__(self):

        pass

    # ------------------------------------------------
    # Build Similarity Matrix
    # ------------------------------------------------

    def build_matrix(

        self,

        documents,

        embedder

    ):

        names = []

        embeddings = []

        for doc in documents:

            names.append(doc["name"])

            embeddings.append(

                embedder.embed_query(

                    doc["text"]

                )

            )

        similarity_matrix = cosine_similarity(

            embeddings

        )

        dataframe = pd.DataFrame(

            similarity_matrix,

            index=names,

            columns=names

        )

        return dataframe

    # ------------------------------------------------
    # Most Similar Pair
    # ------------------------------------------------

    def most_similar(

        self,

        dataframe

    ):

        best_score = -1

        best_pair = None

        columns = dataframe.columns

        for i in range(len(columns)):

            for j in range(i + 1, len(columns)):

                score = dataframe.iloc[i, j]

                if score > best_score:

                    best_score = score

                    best_pair = (

                        columns[i],

                        columns[j]

                    )

        return best_pair, best_score

    # ------------------------------------------------
    # Least Similar Pair
    # ------------------------------------------------

    def least_similar(

        self,

        dataframe

    ):

        worst_score = 1

        worst_pair = None

        columns = dataframe.columns

        for i in range(len(columns)):

            for j in range(i + 1, len(columns)):

                score = dataframe.iloc[i, j]

                if score < worst_score:

                    worst_score = score

                    worst_pair = (

                        columns[i],

                        columns[j]

                    )

        return worst_pair, worst_score