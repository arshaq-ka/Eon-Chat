from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from docx import Document as DocxDocument
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from tools.common import trim_output


RAG_INDEX_PATH = Path(__file__).resolve().with_name("rag_index")
EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "eon_uploaded_documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_RELEVANCE_SCORE = 0.35


@lru_cache(maxsize=1)
def _get_vector_store() -> Chroma:
    import os

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(RAG_INDEX_PATH),
        embedding_function=OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", EMBEDDING_MODEL),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ),
    )


def _extract_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(BytesIO(data), strict=False)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix == ".docx":
        document = DocxDocument(BytesIO(data))
        paragraphs = [paragraph.text for paragraph in document.paragraphs]
        tables = [
            " ".join(cell.text for cell in row.cells)
            for table in document.tables
            for row in table.rows
        ]
        text = "\n".join(paragraphs + tables)
    else:
        raise ValueError("Only PDF and DOCX files are supported.")

    if not text.strip():
        raise ValueError("The document contains no extractable text.")
    return text


def index_uploaded_documents(files: list[tuple[str, bytes]]) -> list[str]:
    try:
        store = _get_vector_store()
    except Exception as exc:
        return [f"Failed to initialize document index: {exc}"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    statuses = []

    for uploaded_name, data in files:
        filename = Path(uploaded_name).name
        try:
            text = _extract_text(filename, data)
            chunks = splitter.split_text(text)
            if not chunks:
                raise ValueError("The document contains no indexable text.")

            content_hash = sha256(data).hexdigest()
            existing = store.get(
                where={"source": filename},
                include=["metadatas"],
            )
            existing_ids = existing.get("ids", [])
            existing_metadata = existing.get("metadatas") or []
            if existing_ids and all(
                metadata.get("file_hash") == content_hash
                for metadata in existing_metadata
            ):
                statuses.append(f"Already indexed: {filename}")
                continue

            document_id = sha256(
                f"{filename}:{content_hash}".encode("utf-8")
            ).hexdigest()
            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "file_hash": content_hash,
                        "chunk_index": index,
                    },
                )
                for index, chunk in enumerate(chunks)
            ]
            ids = [f"{document_id}-{index}" for index in range(len(documents))]
            store.add_documents(documents=documents, ids=ids)

            old_ids = [
                item_id
                for item_id, metadata in zip(existing_ids, existing_metadata)
                if metadata.get("file_hash") != content_hash
            ]
            if old_ids:
                store.delete(ids=old_ids)
            statuses.append(f"Indexed {filename} ({len(documents)} chunks)")
        except Exception as exc:
            statuses.append(f"Failed to index {filename}: {exc}")

    return statuses


def retrieve_documents(query: str) -> str:
    try:
        store = _get_vector_store()
        indexed = store.get(include=["metadatas"])
        if not indexed.get("ids"):
            return "No uploaded documents have been indexed yet."

        sources = sorted(
            {
                metadata["source"]
                for metadata in indexed.get("metadatas", [])
                if metadata and metadata.get("source")
            }
        )
        shown_sources = sources[:5]
        source_list = "Indexed files: " + ", ".join(shown_sources)
        if len(sources) > len(shown_sources):
            source_list += f", and {len(sources) - len(shown_sources)} more"

        matches = store.similarity_search_with_relevance_scores(
            query,
            k=4,
            score_threshold=MIN_RELEVANCE_SCORE,
        )
        if not matches:
            return trim_output(
                f"{source_list}\n"
                "No relevant passages were found in the uploaded documents."
            )

        excerpts = [
            f"[Source: {document.metadata.get('source', 'unknown')}]\n"
            f"{document.page_content}"
            for document, _score in matches
        ]
        return trim_output(
            f"{source_list}\nRelevant excerpts:\n" + "\n\n".join(excerpts)
        )
    except Exception as exc:
        return trim_output(f"Document search failed: {exc}")
