# Streamlit Cloud 배포용 (Linux 환경)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
import pdfplumber
import os
import streamlit as st
import tempfile
import chromadb
from langchain.callbacks.base import BaseCallbackHandler
from streamlit_extras.buy_me_a_coffee import button
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 제목 및 스타일
st.set_page_config(page_title="나의 과외 선생님 👨‍🏫", page_icon="👨‍🏫")
st.title("나의 과외 선생님 (Simple RAG)")
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
</style>
""", unsafe_allow_html=True)
st.write("---")

# --------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    openai_key = st.text_input('OPEN_AI_API_KEY', type="password")

    st.divider()
    button(username="{계정 ID}", floating=False, width=221)

# File Upload
uploaded_file = st.file_uploader("PDF 파일을 올려주세요!", type=['pdf'])
st.write("---")

# --------------------------------------------------------------------------------
# Logic
# --------------------------------------------------------------------------------


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


if uploaded_file is not None:
    @st.cache_resource(show_spinner="문서 분석 및 임베딩 중...")
    def embed_file_v5(file, _openai_key):
        file_content = file.read()

        # Use a temporary directory for file storage to avoid clutter
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        documents = []
        # PDF Parsing with pdfplumber (Text only)
        with pdfplumber.open(tmp_file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                documents.append(Document(page_content=text, metadata={
                                 "page": i+1, "source": file.name}))

        # Clean up temp file
        os.unlink(tmp_file_path)

        # Text Splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        # Embedding
        if not _openai_key:
            st.error("OpenAI API Key Required")
            st.stop()

        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small", openai_api_key=_openai_key)

        # Chroma DB - Persistent Client
        # We use a persistent directory rooted in the current execution folder.
        # This creates a real SQLite file, which is accessible across threads/processes,
        # solving the "no such table" error caused by Streamlit caching in-memory DB connections.
        # We use a subfolder based on the filename to isolate data (simple approach).
        safe_name = "".join([c for c in file.name if c.isalnum()])
        persist_dir = os.path.join(os.getcwd(), ".chroma_db", safe_name)

        client = chromadb.PersistentClient(path=persist_dir)

        db = Chroma.from_documents(
            texts,
            embeddings_model,
            client=client,
            collection_name="openai_collection"
        )

        return db.as_retriever(search_kwargs={"k": 5})

    try:
        retriever = embed_file_v5(uploaded_file, openai_key)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    # Session State
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "네, 문서를 확인했습니다. 질문해 주세요."}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Chat Input
    if prompt_message := st.chat_input("질문을 입력하세요"):
        st.chat_message("user").write(prompt_message)
        st.session_state.messages.append(
            {"role": "user", "content": prompt_message})

        with st.chat_message("assistant"):
            status_container = st.empty()

            # Retrieval
            docs = retriever.invoke(prompt_message)
            context_text = "\n\n".join(doc.page_content for doc in docs)

            # Generation
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                openai_api_key=openai_key,
                streaming=True,
                callbacks=[StreamHandler(status_container)]
            )

            # Simple RAG Prompt
            system_prompt = (
                "You are a helpful tutor. Answer the question based ONLY on the following context.\n"
                "If the answer is not in the context, say you don't know.\n\n"
                f"Context:\n{context_text}"
            )

            response = llm.invoke([
                HumanMessage(content=system_prompt),
                HumanMessage(content=prompt_message)
            ])

            response_content = response.content

        st.session_state.messages.append(
            {"role": "assistant", "content": response_content})
