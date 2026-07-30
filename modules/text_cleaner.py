"""
text_cleaner.py

Cleans extracted PDF text and converts it
into page-wise documents.
"""

import re


class TextCleaner:

    @staticmethod
    def clean_text(text):
        """
        Clean extracted text.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """

        if not text:
            return ""

        # Remove multiple spaces
        text = re.sub(r"[ \t]+", " ", text)

        # Remove extra blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove unwanted characters
        text = text.replace("\x00", "")

        return text.strip()

    @staticmethod
    def split_into_pages(text, source):
        """
        Convert cleaned text into page-wise documents.

        Returns
        -------
        [
            {
                "page":1,
                "source":"paper.pdf",
                "text":"..."
            }
        ]
        """

        cleaned = TextCleaner.clean_text(text)

        # Split using the page separator added by PDFLoader
        pages = re.split(
            r"========== PAGE \d+ ==========",
            cleaned
        )

        documents = []

        page_number = 1

        for page in pages:

            page = page.strip()

            if page:

                documents.append(
                    {
                        "page": page_number,
                        "source": source,
                        "text": page
                    }
                )

                page_number += 1

        return documents

    @staticmethod
    def get_full_text(documents):
        """
        Combine all document text.

        Parameters
        ----------
        documents : list

        Returns
        -------
        str
        """

        return "\n\n".join(
            doc["text"] for doc in documents
        )