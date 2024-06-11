import os
from typing import List
import fitz  # PyMuPDF

class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path):
            if self.path.endswith(".txt"):
                self.load_txt_file(self.path)
            elif self.path.endswith(".pdf"):
                self.load_pdf_file(self.path)
            else:
                raise ValueError("Provided path is neither a valid .txt nor a .pdf file.")
        else:
            raise ValueError("Provided path is neither a valid directory nor a file.")

    def load_txt_file(self, file_path: str):
        with open(file_path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_pdf_file(self, file_path: str):
        content = ""
        try:
            document = fitz.open(file_path)
            for page_num in range(len(document)):
                page = document.load_page(page_num)
                content += page.get_text()
        except Exception as e:
            print(f"Error reading PDF file {file_path}: {e}")
        self.documents.append(content)

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    self.load_txt_file(os.path.join(root, file))
                elif file.endswith(".pdf"):
                    self.load_pdf_file(os.path.join(root, file))

    def load_documents(self):
        self.load()
        return self.documents

class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks

if __name__ == "__main__":
    # Example usage with a .txt file
    txt_loader = TextFileLoader("data/KingLear.txt")
    txt_loader.load()
    splitter = CharacterTextSplitter()
    txt_chunks = splitter.split_texts(txt_loader.documents)
    print(f"Number of chunks from txt: {len(txt_chunks)}")
    print(txt_chunks[0])
    print("--------")
    print(txt_chunks[1])
    print("--------")
    print(txt_chunks[-2])
    print("--------")
    print(txt_chunks[-1])

    # Example usage with a .pdf file
    pdf_loader = TextFileLoader("data/paul-graham-essays.pdf")
    pdf_loader.load()
    pdf_chunks = splitter.split_texts(pdf_loader.documents)
    print(f"Number of chunks from pdf: {len(pdf_chunks)}")
    print(pdf_chunks[0])
    print("--------")
    print(pdf_chunks[1])
    print("--------")
    print(pdf_chunks[-2])
    print("--------")
    print(pdf_chunks[-1])
