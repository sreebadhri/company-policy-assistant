"""Ingestion pipeline: turn the files in data/ into a searchable vector database.

Built step by step:
    Step 1: load documents from data/
    Step 2: clean and split them into chunks
    Step 3 (this version): embed the chunks and store them in Chroma   <- you are here

Run (needs OPENAI_API_KEY):
    .venv/bin/python ingest.py
"""

import re
from pathlib import Path

import docx2txt
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vector_store import CHROMA_DIR, get_vector_store
from pypdf import PdfReader

DATA_DIR = Path(__file__).parent / "data"

CHUNK_SIZE = 400  # maximum characters of policy text per chunk
CHUNK_OVERLAP = 80  # maximum characters repeated between neighbouring chunks


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


# --- Step 2: clean and split ----------------------------------------------


def clean_document(doc: Document) -> Document:
    """Remove the repeated header/footer from an aio PDF page.

    The title and section are kept as metadata. Anything that doesn't match
    the aio page layout is returned unchanged, so we never drop real text.
    """
    lines = doc.page_content.splitlines()
    is_aio_page = (
        doc.metadata["source"].lower().endswith(".pdf")
        and len(lines) > 7
        and lines[:2] == ["aio", "all in one"]
        and "Fictional training document" in lines[3]
        and lines[-2] == "aio employee handbook"
        and re.fullmatch(r"Page \d+ of \d+", lines[-1])
    )
    if not is_aio_page:
        return doc

    # Lines 0-3 are the header, line 4 the section, the last two the footer.
    return Document(
        page_content="\n".join(lines[5:-2]),
        metadata={**doc.metadata, "title": lines[2], "section": lines[4]},
    )


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks; each chunk keeps its document's metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Tried in order: blank line, line ending a sentence, any line, word,
        # character.
        separators=[r"\n\n", r"(?<=\.)\n", r"\n", " ", ""],
        is_separator_regex=True,
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        # Put the title and section in the text so search can match on them.
        if "title" in chunk.metadata:
            heading = f"{chunk.metadata['title']} > {chunk.metadata['section']}"
            chunk.page_content = f"{heading}\n{chunk.page_content}"
    return chunks


# --- Step 3: embed and store ----------------------------------------------


def store_chunks(chunks: list[Document]) -> None:
    """Rebuild the collection from scratch, so reruns leave no old chunks."""
    store = get_vector_store()
    store.reset_collection()
    store.add_documents(chunks)  # embeds each chunk with the OpenAI API


def main() -> None:
    print(f"Loading documents from {DATA_DIR}/ ...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")
    if not documents:
        return

    chunks = split_documents([clean_document(doc) for doc in documents])
    print(f"Split into {len(chunks)} chunks:\n")
    for chunk in chunks:
        page = chunk.metadata.get("page", "-")
        print(f"  {chunk.metadata['source']:<35} page {page:<3} {len(chunk.page_content):>5} chars")

    # Show one page before and after, so you can see what changed.
    first = documents[0].metadata
    print("\n--- Before: first page " + "-" * 40)
    print(documents[0].page_content)
    print("\n--- After: its chunks " + "-" * 41)
    for chunk in chunks:
        if chunk.metadata["source"] == first["source"] and chunk.metadata.get("page") == first.get("page"):
            print(f"{chunk.metadata}\n{chunk.page_content}\n")

    store_chunks(chunks)
    print(f"Stored {len(chunks)} chunks in {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
