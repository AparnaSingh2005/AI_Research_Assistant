
from langchain_text_splitters import RecursiveCharacterTextSplitter
class TextSplitter:

    def __init__(self):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def split_pages(self, pages):

        chunks = []

        for page_data in pages:

            page_number = page_data["page"]

            text = page_data["text"]

            split_chunks = self.splitter.split_text(text)

            for i, chunk in enumerate(split_chunks):

                chunks.append(
                    {
                        "page": page_number,
                        "chunk_id": i,
                        "text": chunk
                    }
                )

        return chunks