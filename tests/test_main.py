
from main import embed_file_v6
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Mock streamlit before importing main
mock_st = MagicMock()
# Handle decorator: @st.cache_resource(...) returns a decorator that returns the function


def mock_cache_decorator(*args, **kwargs):
    def decorator(f):
        return f
    return decorator


mock_st.cache_resource = mock_cache_decorator

sys.modules["streamlit"] = mock_st
sys.modules["streamlit.runtime.scriptrunner"] = MagicMock()
sys.modules["streamlit_extras.buy_me_a_coffee"] = MagicMock()
# Mock langchain_groq because it's imported in main but might not be installed in test env or just to be safe
sys.modules["langchain_groq"] = MagicMock()

# Now import the function to test


@pytest.fixture
def mock_file():
    f = MagicMock()
    f.name = "test_document.pdf"
    f.read.return_value = b"Fake PDF Content"
    return f


@patch("main.pdfplumber")
@patch("main.OpenAIEmbeddings")
@patch("main.Chroma")
@patch("main.tempfile")
@patch("main.os")
def test_embed_file_gpt_success(mock_os, mock_temp, mock_chroma, mock_embeddings, mock_pdf, mock_file):
    # Setup Mocks
    mock_os.getcwd.return_value = "/tmp"
    mock_os.path.join.side_effect = os.path.join
    mock_os.unlink = MagicMock()

    # Mock tempfile
    mock_temp_file = MagicMock()
    mock_temp_file.name = "/tmp/temp.pdf"
    mock_temp.NamedTemporaryFile.return_value.__enter__.return_value = mock_temp_file

    # Mock PDF extraction
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "This is a test PDF content."
    mock_pdf_instance = MagicMock()
    mock_pdf_instance.pages = [mock_page]
    mock_pdf.open.return_value.__enter__.return_value = mock_pdf_instance

    # Mock Chroma
    mock_db = MagicMock()
    mock_chroma.from_documents.return_value = mock_db
    mock_db.as_retriever.return_value = "mock_retriever"

    # Execute with GPT Provider
    retriever = embed_file_v6(
        mock_file, "GPT-4o (상용/고품질)", "fake-api-key", "http://fake-url", None)

    # Assertions
    assert retriever == "mock_retriever"
    mock_embeddings.assert_called_with(
        model="text-embedding-3-small", openai_api_key="fake-api-key")
    # Verify collection name
    args, kwargs = mock_chroma.from_documents.call_args
    assert kwargs['collection_name'] == "openai_collection"


@patch("main.pdfplumber")
@patch("main.OllamaEmbeddings")
@patch("main.Chroma")
@patch("main.tempfile")
@patch("main.os")
def test_embed_file_ollama_success(mock_os, mock_temp, mock_chroma, mock_embeddings, mock_pdf, mock_file):
    # Setup Mocks (similar to above)
    mock_os.getcwd.return_value = "/tmp"
    mock_os.path.join.side_effect = os.path.join
    mock_os.unlink = MagicMock()

    mock_temp_file = MagicMock()
    mock_temp_file.name = "/tmp/temp.pdf"
    mock_temp.NamedTemporaryFile.return_value.__enter__.return_value = mock_temp_file

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Ollama test content."
    mock_pdf_instance = MagicMock()
    mock_pdf_instance.pages = [mock_page]
    mock_pdf.open.return_value.__enter__.return_value = mock_pdf_instance

    mock_db = MagicMock()
    mock_chroma.from_documents.return_value = mock_db
    mock_db.as_retriever.return_value = "ollama_retriever"

    # Execute with Ollama Provider
    retriever = embed_file_v6(
        mock_file, "Ollama (무료/로컬)", "", "http://localhost:11434", None)

    # Assertions
    assert retriever == "ollama_retriever"
    mock_embeddings.assert_called_with(
        base_url="http://localhost:11434", model="llama3", headers={})
    # Verify collection name
    args, kwargs = mock_chroma.from_documents.call_args
    assert kwargs['collection_name'] == "ollama_collection"


@patch("main.pdfplumber")
@patch("main.OpenAIEmbeddings")
@patch("main.Chroma")
@patch("main.tempfile")
@patch("main.os")
def test_embed_file_groq_success(mock_os, mock_temp, mock_chroma, mock_embeddings, mock_pdf, mock_file):
    # Setup Mocks
    mock_os.getcwd.return_value = "/tmp"
    mock_os.path.join.side_effect = os.path.join
    mock_os.unlink = MagicMock()

    mock_temp_file = MagicMock()
    mock_temp_file.name = "/tmp/temp.pdf"
    mock_temp.NamedTemporaryFile.return_value.__enter__.return_value = mock_temp_file

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Groq test content."
    mock_pdf_instance = MagicMock()
    mock_pdf_instance.pages = [mock_page]

    mock_db = MagicMock()
    mock_chroma.from_documents.return_value = mock_db
    mock_db.as_retriever.return_value = "groq_retriever"

    # Execute with Groq Provider (uses OpenAI Embedding)
    retriever = embed_file_v6(
        mock_file, "Groq (Fast/Free)", "fake-openai-key", "http://fake", None)

    # Assertions
    assert retriever == "groq_retriever"
    mock_embeddings.assert_called_with(
        model="text-embedding-3-small", openai_api_key="fake-openai-key")
    # Verify collection name
    args, kwargs = mock_chroma.from_documents.call_args
    assert kwargs['collection_name'] == "openai_collection"


@patch("main.pdfplumber")
@patch("main.tempfile")
@patch("main.os")
@patch("main.st")
def test_embed_file_gpt_no_key(mock_streamlit, mock_os, mock_temp, mock_pdf, mock_file):
    # Setup Mocks to avoid crashes
    mock_os.getcwd.return_value = "/tmp"
    mock_os.path.join.side_effect = os.path.join
    mock_os.unlink = MagicMock()

    mock_temp_file = MagicMock()
    mock_temp_file.name = "/tmp/temp.pdf"
    mock_temp.NamedTemporaryFile.return_value.__enter__.return_value = mock_temp_file

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Content"
    mock_pdf.open.return_value.__enter__.return_value.pages = [mock_page]

    # Execute with GPT provider but empty key
    mock_streamlit.stop.side_effect = Exception("Stop called")
    with pytest.raises(Exception, match="Stop called"):
        embed_file_v6(mock_file, "GPT-4o (상용/고품질)", "", "http://fake", None)

    mock_streamlit.error.assert_called_with(
        "OpenAI API Key Required for Embeddings")
    mock_streamlit.stop.assert_called()
