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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from streamlit_extras.buy_me_a_coffee import button
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Privacy Shield Import
try:
    from privacy_shield import HybridSecurityEngine
except ImportError:
    HybridSecurityEngine = None

# 제목 및 스타일
st.set_page_config(page_title="나의 과외 선생님 🛡️", page_icon="🛡️")
st.title("나의 과외 선생님 (HLS Enhanced)")
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .security-log {
        background-color: #1e1e1e;
        color: #00ff00;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 0.85em;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)
st.write("---")

# --------------------------------------------------------------------------------
# Sidebar: HLS Configuration
# --------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 보안 & 인프라 설정")

    # Mode Selection (Simplified for HLS)
    mode_selection = st.radio(
        "운영 모드",
        ["🛡️ Auto (Hybrid Layered Security)",
         "⚡ Standard (OpenAI Direct)", "🔒 Private (Ollama Only)"],
        index=0,
        help="Auto: 민감도에 따라 로컬/클라우드를 자동 전환합니다."
    )

    st.divider()

    # API Keys & URLs
    openai_key = st.text_input('OPEN_AI_API_KEY', type="password")

    st.subheader("Local/Cloud Ollama")
    ollama_url = st.text_input("Ollama Base URL", value="http://localhost:11434",
                               placeholder="https://your-runpod-id.runpod.net")
    ollama_model = st.text_input("Ollama Model Name", value="llama3.3")

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
    @st.cache_resource(show_spinner="문서 보안 분석 및 임베딩 중...")
    def embed_file(file, _openai_key, _ollama_url, _ollama_model, _mode):
        file_content = file.read()
        file_path = f"./.cache/files/{file.name}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(file_content)

        # PDF Parsing
        documents = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                documents.append(Document(page_content=text, metadata={
                                 "page": i+1, "source": file.name}))

        # Text Splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        # Embedding: HLS prefers Local for 'Auto' or 'Private' to keep vectors private
        use_ollama = _mode != "⚡ Standard (OpenAI Direct)"

        if use_ollama and _ollama_url:
            embeddings_model = OllamaEmbeddings(
                base_url=_ollama_url, model=_ollama_model)
        else:
            if not _openai_key:
                st.error("OpenAI API Key Required")
                st.stop()
            embeddings_model = OpenAIEmbeddings(
                model="text-embedding-3-large", openai_api_key=_openai_key)

<<<<<<< Updated upstream
        # Chroma DB
        # Use a temporary directory to avoid tenant/persistence issues
        chroma_persist_dir = os.path.join(
            tempfile.gettempdir(), "chroma_db_" + file.name)
        # Clear existing checking if needed, but for now just use it.
        # Actually safer to make it random or cleared.
        # Let's simple use a new temp dir for this session's cache of this file.
        # Since embed_file is cached, we want a stable path if we want to reuse?
        # But st.cache_resource caches the RETURN value (retriever), so DB object is kept in memory.
        # The persistence on disk matters for initialization stability.
        # Let's use a unique temp dir.

        persist_dir = tempfile.mkdtemp()
        db = Chroma.from_documents(
            texts, embeddings_model, persist_directory=persist_dir)

        # Retriever
=======
        db = Chroma.from_documents(texts, embeddings_model)
>>>>>>> Stashed changes
        return db.as_retriever(search_kwargs={"k": 5})

    try:
        retriever = embed_file(uploaded_file, openai_key,
                               ollama_url, ollama_model, mode_selection)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    # Session State
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "네, 문서를 확인했습니다. 질문해 주세요."}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
        if "log" in msg:
            with st.expander("🔒 HLS Execution Log", expanded=False):
                for line in msg["log"]:
                    st.write(line)

    # Chat Input
    if prompt_message := st.chat_input("질문을 입력하세요"):
        st.chat_message("user").write(prompt_message)
        st.session_state.messages.append(
            {"role": "user", "content": prompt_message})

        with st.chat_message("assistant"):
            status_container = st.empty()
            log_container = st.expander(
                "🔒 HLS Security Pipeline", expanded=True)

            # 1. Retrieval
            # Use appropriate LLM for retrieval refinement
            try:
                retrieval_llm = ChatOllama(base_url=ollama_url, model=ollama_model, temperature=0) if "OpenAI" not in mode_selection else ChatOpenAI(
                    temperature=0, openai_api_key=openai_key)
                retriever_from_llm = MultiQueryRetriever.from_llm(
                    retriever=retriever, llm=retrieval_llm)
                docs = retriever_from_llm.invoke(prompt_message)
            except:
                docs = retriever.invoke(prompt_message)  # Fallback

            context_text = "\n\n".join(doc.page_content for doc in docs)

            # 2. HLS Execution
            response_content = ""
            execution_log = []

            if mode_selection == "🛡️ Auto (Hybrid Layered Security)":
                if not HybridSecurityEngine:
                    st.error("Privacy Module Missing")
                    st.stop()

                # Initialize Engine
                # Local LLM Callable
                def local_llm_func(prompt):
                    return ChatOllama(base_url=ollama_url, model=ollama_model, temperature=0).invoke(prompt).content

                # GPT LLM Callable
                def gpt_llm_func(prompt):
                    return ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_key).invoke(prompt).content

                engine = HybridSecurityEngine(
                    ollama_client=local_llm_func, gpt_client=gpt_llm_func)

                # EXECUTE
                with log_container:
                    st.write("Analyzing Sensitivity...")
                    final_answer, logs = engine.execute_secure_pipeline(
                        prompt_message, context_text)
                    for l in logs:
                        st.write(l)
                    execution_log = logs

                status_container.markdown(final_answer)
                response_content = final_answer

            elif mode_selection == "🔒 Private (Ollama Only)":
                with log_container:
                    st.write("🔒 Forced Local Mode Active")
                llm = ChatOllama(base_url=ollama_url, model=ollama_model, temperature=0, callbacks=[
                                 StreamHandler(status_container)])
                response = llm.invoke(
                    f"Context: {context_text}\n\nQuestion: {prompt_message}")
                response_content = response.content
                execution_log = ["🔒 Mode: Private (Local Only)"]

            else:  # Standard
                with log_container:
                    st.write("⚡ Direct OpenAI Mode")
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_key,
                                 streaming=True, callbacks=[StreamHandler(status_container)])
                response = llm.invoke(
                    [HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {prompt_message}")])
                response_content = response.content
                execution_log = ["⚡ Mode: Standard"]

            st.session_state.messages.append(
                {"role": "assistant", "content": response_content, "log": execution_log})
