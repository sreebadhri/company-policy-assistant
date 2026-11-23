"""Ingestion pipeline: turn the files in data/ into a searchable vector database.

Built step by step:
    Step 1 (this version): load documents from data/   <- you are here
    Step 2: clean and split them into chunks
    Step 3: embed the chunks and store them in Chroma

Run:
    .venv/bin/python ingest.py
"""

from pathlib import Path

import docx2txt
from langchain_core.documents import Document
from pypdf import PdfReader

DATA_DIR = Path(__file__).parent / "data"


# --- One small loader per file type --------------------------------------
# Each loader returns a list of LangChain Documents: the text in
# `page_content`, plus `metadata` describing where the text came from.
# That metadata is what will let the chatbot cite its sources later.


def load_pdf(path: Path) -> list[Document]:
    """One Document per PDF page, so we can cite page numbers."""
    reader = PdfReader(path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": path.name, "page": number},
        )
        for number, page in enumerate(reader.pages, start=1)
    ]


def load_text(path: Path) -> list[Document]:
    """Plain text and Markdown: the whole file is one Document."""
    text = path.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": path.name})]


def load_docx(path: Path) -> list[Document]:
    """Word documents: the whole file is one Document (.docx has no fixed pages)."""
    text = docx2txt.process(str(path))
    return [Document(page_content=text, metadata={"source": path.name})]


# File extension -> loader. To support a new format, add one entry here.
LOADERS = {
    ".pdf": load_pdf,
    ".md": load_text,
    ".txt": load_text,
    ".docx": load_docx,
}


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Load every supported file under data_dir (including subfolders)."""
    documents: list[Document] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            print(f"  skipped (unsupported type): {path.relative_to(data_dir)}")
            continue
        documents.extend(loader(path))
    return documents


def main() -> None:
    print(f"Loading documents from {DATA_DIR}/ ...")
    documents = load_documents()
    print(f"\nLoaded {len(documents)} documents:\n")

    for doc in documents:
        page = doc.metadata.get("page", "-")
        print(f"  {doc.metadata['source']:<35} page {page:<3} {len(doc.page_content):>5} chars")

    # Show one Document in full so you can see what the text looks like.
    if documents:
        sample = documents[0]
        print("\n--- Example Document " + "-" * 40)
        print(f"metadata:     {sample.metadata}")
        print(f"page_content:\n{sample.page_content}")


if __name__ == "__main__":
    main()
