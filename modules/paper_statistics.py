"""
paper_statistics.py

Generates statistics for research papers and chunks.
"""


class PaperStatistics:

    def statistics(self, text):
        """
        Statistics for a single text.
        """

        if not text:
            return {
                "Words": 0,
                "Characters": 0,
                "Lines": 0,
                "Paragraphs": 0
            }

        return {
            "Words": len(text.split()),
            "Characters": len(text),
            "Lines": len(text.splitlines()),
            "Paragraphs": len(
                [p for p in text.split("\n\n") if p.strip()]
            )
        }

    def document_statistics(self, documents):
        """
        Overall statistics for uploaded documents.
        """

        total_documents = len(documents)

        total_words = 0
        total_characters = 0

        for doc in documents:
            text = doc["text"]

            total_words += len(text.split())
            total_characters += len(text)

        return {
            "Total Documents": total_documents,
            "Total Words": total_words,
            "Total Characters": total_characters
        }

    def chunk_statistics(self, chunks):
        """
        Statistics for generated chunks.
        """

        if not chunks:
            return {
                "Total Chunks": 0,
                "Average Chunk Size": 0,
                "Largest Chunk": 0,
                "Smallest Chunk": 0
            }

        chunk_sizes = [
            len(chunk["text"].split())
            for chunk in chunks
        ]

        return {
            "Total Chunks": len(chunks),
            "Average Chunk Size": round(
                sum(chunk_sizes) / len(chunk_sizes),
                2
            ),
            "Largest Chunk": max(chunk_sizes),
            "Smallest Chunk": min(chunk_sizes)
        }

    def paper_summary(self, documents):
        """
        Returns statistics for every uploaded paper.
        """

        summary = {}

        for doc in documents:

            source = doc["source"]

            if source not in summary:
                summary[source] = {
                    "Pages": 0,
                    "Words": 0
                }

            summary[source]["Pages"] += 1

            summary[source]["Words"] += len(
                doc["text"].split()
            )

        return summary

    def chart_data(self, documents, chunks):
        """
        Returns chart-ready data.
        """

        doc_stats = self.document_statistics(documents)
        chunk_stats = self.chunk_statistics(chunks)

        return {
            "labels": [
                "Documents",
                "Words",
                "Characters",
                "Chunks"
            ],
            "values": [
                doc_stats["Total Documents"],
                doc_stats["Total Words"],
                doc_stats["Total Characters"],
                chunk_stats["Total Chunks"]
            ]
        }