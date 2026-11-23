from langchain_core.tools import tool


@tool
def search_uploaded_documents(query: str) -> str:
    """Search the user's uploaded PDF and DOCX documents."""
    from rag import retrieve_documents

    return retrieve_documents(query)
