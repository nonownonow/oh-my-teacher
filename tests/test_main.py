
from main import embed_file_v5
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

# Now import the function to test
# We need to make sure the top-level code in main.py doesn't crash.
# Since we mocked streamlit, st.* calls will just be MagicMocks.


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
def test_embed_file_success(mock_os, mock_temp, mock_chroma, mock_embeddings, mock_pdf, mock_file):
    # Setup Mocks
    mock_os.getcwd.return_value = "/tmp"
    mock_os.path.join.side_effect = os.path.join  # Use real join logic
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

    # Execute
    retriever = embed_file_v5(mock_file, "fake-api-key")

    # Assertions
    assert retriever == "mock_retriever"
    mock_pdf.open.assert_called()
    mock_chroma.from_documents.assert_called()
    mock_embeddings.assert_called_with(
        model="text-embedding-3-small", openai_api_key="fake-api-key")


@patch("main.st")
def test_embed_file_no_key(mock_streamlit, mock_file):
    # Reset mocks just in case (though global mock_st is used on import, patch creates a new one for this test scope if targeting main.st)
    # Actually, main.st is the module level import.

    # Execute with empty key
    # embed_file_v5 calls st.error and st.stop which are bound to the 'st' imported in main.

    embed_file_v5(mock_file, "")

    # Assert st.error and st.stop were called
    # Note: mocking main.st via patch might be tricky if main.py does `import streamlit as st`.
    # Let's see if patch("main.st") works.

    mock_streamlit.error.assert_called_with("OpenAI API Key Required")
    mock_streamlit.stop.assert_called()
