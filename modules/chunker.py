"""
chunker.py

Splits cleaned documents into smaller chunks
using LangChain RecursiveCharacterTextSplitter.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:

    def __init__(
        self,
        chunk_size=1000,
        chunk_overlap=200
    ):
        """
        Initialize the text splitter.
        """

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def create_chunks(self, documents):
        """
        Split page-wise documents into chunks.

        Parameters
        ----------
        documents : list

        Returns
        -------
        list
        """

        chunks = []

        chunk_id = 1

        for document in documents:

            page = document["page"]
            source = document["source"]
            text = document["text"]

            split_chunks = self.splitter.split_text(text)

            for index, chunk in enumerate(split_chunks):

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "chunk_number": index + 1,
                        "page": page,
                        "source": source,
                        "text": chunk
                    }
                )

                chunk_id += 1

        return chunks

    def count_chunks(self, chunks):
        """
        Return total number of chunks.
        """

        return len(chunks)

    def average_chunk_size(self, chunks):
        """
        Calculate average words per chunk.
        """

        if not chunks:
            return 0

        total_words = sum(
            len(chunk["text"].split())
            for chunk in chunks
        )

        return round(total_words / len(chunks), 2)