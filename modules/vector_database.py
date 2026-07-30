"""
vector_database.py

ChromaDB wrapper for AI Research Paper Assistant
"""

import os
import chromadb



class VectorDatabase:

    def __init__(self):

        database_path = "database"

        os.makedirs(
            database_path,
            exist_ok=True
        )

        self.client = chromadb.PersistentClient(
            path=database_path
        )

        self.collection = self.client.get_or_create_collection(
            name="research_papers"
        )

    # ======================================================
    # Store one document
    # ======================================================

    def store(
        self,
        id,
        embedding,
        document,
        metadata
    ):

        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        self.collection.add(
            ids=[id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[metadata]
        )

    # ======================================================
    # Store multiple documents
    # ======================================================

    def add_documents(
        self,
        chunks,
        embeddings
    ):

        ids = []
        docs = []
        metas = []
        vectors = []

        for i, chunk in enumerate(chunks):

            ids.append(f"chunk_{i}")

            docs.append(chunk["text"])

            metas.append({

                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"]

            })

            vectors.append(
                embeddings[i].tolist()
            )
        if self.collection.count() > 0:
            self.clear_database()
        self.collection.add(

            ids=ids,
            embeddings=vectors,
            documents=docs,
            metadatas=metas

        )

    # ======================================================
    # Semantic Search
    # ======================================================



    
    def search(
        self,
        embedding,
        top_k=5
):

        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        results = self.collection.query(

            query_embeddings=[embedding],

            n_results=top_k,

            include=[
                "documents",
                "metadatas",
                "distances"
            ]

        )

        formatted = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):

            similarity = round(
                1 - distance,
                4
            )

            formatted.append(
                {   

                    "document": document,

                    "metadata": metadata,

                    "distance": distance,

                    "similarity": similarity

                }
            )

        return formatted
    
# ======================================================
# Global Search
# ======================================================

    def global_search(
        self,
        query_embedding,
        top_k=10
    ):
        return self.search(
            query_embedding,
            top_k
        )

# ======================================================
# Count
# ======================================================

    def count(self):

        return self.collection.count()

    # ======================================================
    # Delete All
    # ======================================================

    def clear_database(self):

        try:

            self.client.delete_collection(
                "research_papers"
            )

        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name="research_papers"
        )