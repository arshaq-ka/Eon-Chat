from io import BytesIO

from docx import Document as DocxDocument
from langchain_core.documents import Document

import rag
from tools.document_search import search_uploaded_documents


class FakeVectorStore:
    def __init__(self):
        self.documents = []
        self.ids = []
        self.deleted = []
        self.scores = []

    def get(self, *, where=None, include=None, limit=None):
        if limit is not None:
            return {"ids": self.ids[:limit]}
        source = (where or {}).get("source")
        selected = [
            (item_id, document)
            for item_id, document in zip(self.ids, self.documents)
            if source is None or document.metadata["source"] == source
        ]
        return {
            "ids": [item_id for item_id, _ in selected],
            "metadatas": [document.metadata for _, document in selected],
        }

    def add_documents(self, *, documents, ids):
        self.documents.extend(documents)
        self.ids.extend(ids)

    def delete(self, *, ids):
        self.deleted.extend(ids)
        remaining = [
            (item_id, document)
            for item_id, document in zip(self.ids, self.documents)
            if item_id not in ids
        ]
        self.ids = [item_id for item_id, _ in remaining]
        self.documents = [document for _, document in remaining]

    def similarity_search_with_relevance_scores(
        self,
        query,
        k,
        score_threshold,
    ):
        scores = self.scores or [0.9] * len(self.documents)
        return [
            (document, score)
            for document, score in zip(self.documents[:k], scores[:k])
            if score >= score_threshold
        ]


def _docx_bytes(text):
    document = DocxDocument()
    document.add_paragraph(text)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_index_docx_chunks_and_skips_duplicate(monkeypatch):
    store = FakeVectorStore()
    monkeypatch.setattr(rag, "_get_vector_store", lambda: store)
    file_bytes = _docx_bytes("Deployment steps for the local application.")

    first = rag.index_uploaded_documents([("guide.docx", file_bytes)])
    second = rag.index_uploaded_documents([("guide.docx", file_bytes)])

    assert first == ["Indexed guide.docx (1 chunks)"]
    assert second == ["Already indexed: guide.docx"]
    assert len(store.documents) == 1
    assert store.documents[0].metadata["source"] == "guide.docx"


def test_updated_document_replaces_previous_chunks(monkeypatch):
    store = FakeVectorStore()
    monkeypatch.setattr(rag, "_get_vector_store", lambda: store)

    rag.index_uploaded_documents([("guide.docx", _docx_bytes("Old content."))])
    result = rag.index_uploaded_documents(
        [("guide.docx", _docx_bytes("Updated deployment instructions."))]
    )

    assert result == ["Indexed guide.docx (1 chunks)"]
    assert len(store.documents) == 1
    assert store.documents[0].page_content == "Updated deployment instructions."
    assert len(store.deleted) == 1


def test_extract_pdf_text(monkeypatch):
    class FakePage:
        def extract_text(self):
            return "PDF body"

    class FakePdfReader:
        def __init__(self, stream, strict):
            assert strict is False
            self.pages = [FakePage()]

    monkeypatch.setattr(rag, "PdfReader", FakePdfReader)

    assert rag._extract_text("guide.pdf", b"pdf bytes") == "PDF body"


def test_search_uploaded_documents_returns_source_citation(monkeypatch):
    store = FakeVectorStore()
    store.documents = [
        Document(
            page_content="Run the deploy command after configuration.",
            metadata={"source": "guide.docx"},
        )
    ]
    store.ids = ["chunk-1"]
    monkeypatch.setattr(rag, "_get_vector_store", lambda: store)

    result = search_uploaded_documents.invoke({"query": "deployment steps"})

    assert "Indexed files: guide.docx" in result
    assert "[Source: guide.docx]" in result
    assert "Run the deploy command" in result


def test_search_reports_when_index_is_empty(monkeypatch):
    monkeypatch.setattr(rag, "_get_vector_store", FakeVectorStore)

    result = rag.retrieve_documents("deployment steps")

    assert result == "No uploaded documents have been indexed yet."


def test_search_omits_low_relevance_passages(monkeypatch):
    store = FakeVectorStore()
    store.documents = [
        Document(
            page_content="Brief Pain Inventory questionnaire content.",
            metadata={"source": "pain-survey.pdf"},
        )
    ]
    store.ids = ["chunk-1"]
    store.scores = [0.1]
    monkeypatch.setattr(rag, "_get_vector_store", lambda: store)

    result = rag.retrieve_documents("give me a brief on AI")

    assert "Indexed files: pain-survey.pdf" in result
    assert "No relevant passages" in result
    assert "Brief Pain Inventory" not in result
