"""
embedding_model.py

Sentence Transformer embedding model for
AI Research Paper Assistant.
"""

from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2"
    ):
        """
        Initialize SentenceTransformer model.
        """

        try:

            self.model = SentenceTransformer(
                model_name
            )

        except Exception as e:

            raise RuntimeError(
                f"Failed to load embedding model:\n{e}"
            )

    # =====================================================
    # Embed Single Text
    # =====================================================

    def embed_text(
        self,
        text
    ):

        if not text:

            return np.zeros(
                self.embedding_dimension()
            )

        embedding = self.model.encode(

            text,

            convert_to_numpy=True,

            normalize_embeddings=True

        )

        return embedding

    # =====================================================
    # Embed Multiple Documents
    # =====================================================

    def embed_documents(
        self,
        chunks
    ):

        if len(chunks) == 0:

            return np.empty(
                (
                    0,
                    self.embedding_dimension()
                )
            )

        texts = [

            chunk["text"]

            for chunk in chunks

        ]

        embeddings = self.model.encode(

            texts,

            batch_size=32,

            show_progress_bar=True,

            convert_to_numpy=True,

            normalize_embeddings=True

        )

        return embeddings

    # =====================================================
    # Embed Query
    # =====================================================

    def embed_query(
        self,
        query
    ):

        return self.embed_text(
            query
        )

    # =====================================================
    # Embedding Dimension
    # =====================================================

    def embedding_dimension(
        self
    ):

        return self.model.get_sentence_embedding_dimension()

    # =====================================================
    # Similarity
    # =====================================================

    def similarity(
        self,
        embedding1,
        embedding2
    ):

        embedding1 = np.asarray(
            embedding1
        )

        embedding2 = np.asarray(
            embedding2
        )

        similarity = np.dot(
            embedding1,
            embedding2
        )

        return float(
            similarity
        )

    # =====================================================
    # Batch Similarity
    # =====================================================

    def batch_similarity(
        self,
        query_embedding,
        embeddings
    ):

        embeddings = np.asarray(
            embeddings
        )

        scores = np.dot(
            embeddings,
            query_embedding
        )

        return scores