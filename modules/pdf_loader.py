"""
pdf_loader.py

Loads PDF files and extracts text page by page.
"""
import os
import fitz  # PyMuPDF


class PDFLoader:

    def __init__(self, pdf_path):
        """
        Initialize PDF Loader.

        Args:
            pdf_path (str): Path to the PDF file.
        """
        self.pdf_path = pdf_path

    def extract_text(self):
        """
        Extract text from all pages.

        Returns:
            tuple:
                full_text (str)
                total_pages (int)
        """

        document = fitz.open(self.pdf_path)

        full_text = ""

        total_pages = len(document)

        for page_number in range(total_pages):

            page = document.load_page(page_number)

            text = page.get_text("text")

            full_text += (
                f"\n\n========== PAGE {page_number + 1} ==========\n\n"
            )

            full_text += text

        document.close()

        return full_text, total_pages

    def extract_pages(self):
        """
        Extract page-wise information.

        Returns:
            list of dictionaries
        """

        document = fitz.open(self.pdf_path)

        pages = []

        for page_number in range(len(document)):

            page = document.load_page(page_number)

            pages.append(
                {
                    "page": page_number + 1,
                    "source":os.path.basename(self.pdf_path),
                    "text": page.get_text("text")
                }
            )

        document.close()

        return pages