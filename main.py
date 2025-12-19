# Streamlit Cloud 배포용 (Linux 환경)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
import pdfplumber
import os
import streamlit as st
import tempfile
import chromadb
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from streamlit_extras.buy_me_a_coffee import button
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# .env 파일에서 환경변수 로드
load_dotenv()

# 환경변수에서 Ollama API 키 읽기
OLLAMA_API_KEY = os.getenv("OLLAMA_AI_API", "")

# 제목 및 스타일
st.set_page_config(page_title="나의 과외 선생님 👨‍🏫", page_icon="👨‍🏫")
st.title("나의 과외 선생님")
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
    # Model Selection
    model_provider = st.radio(
        "모델 선택",
        ["GPT-4o (상용/고품질)", "Ollama (Cloud)"],
        index=0
    )

    openai_key = ""
    ollama_url = "https://ollama.com"
    ollama_key = OLLAMA_API_KEY

    if model_provider == "GPT-4o (상용/고품질)":
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


@st.cache_resource(show_spinner="문서 분석 및 임베딩 중...")
def embed_file_v6(file, provider, _api_key, _ollama_url, _ollama_key):
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

    # Embedding Logic
    if provider == "GPT-4o (상용/고품질)":
        if not _api_key:
            st.error("OpenAI API Key Required")
            st.stop()
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small", openai_api_key=_api_key)
        collection_name = "openai_collection"
    else:
        # Ollama Embeddings
        headers = {}
        if _ollama_key:
            headers["Authorization"] = f"Bearer {_ollama_key}"

        embeddings_model = OllamaEmbeddings(
            base_url=_ollama_url,
            model="gemma3:27b",
            client_kwargs={"headers": headers} if headers else {}
        )
        collection_name = "ollama_collection"

    # Chroma DB - Persistent Client
    safe_name = "".join([c for c in file.name if c.isalnum()])
    provider_prefix = "gpt" if provider == "GPT-4o (상용/고품질)" else "ollama"
    persist_dir = os.path.join(
        os.getcwd(), ".chroma_db", provider_prefix, safe_name)

    client = chromadb.PersistentClient(path=persist_dir)

    db = Chroma.from_documents(
        texts,
        embeddings_model,
        client=client,
        collection_name=collection_name
    )

    return db.as_retriever(search_kwargs={"k": 10})


if uploaded_file is not None:
    try:
        retriever = embed_file_v6(
            uploaded_file, model_provider, openai_key, ollama_url, ollama_key)
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
            if model_provider == "GPT-4o (상용/고품질)":
                llm = ChatOpenAI(
                    model="gpt-4o",  # Upgraded to 4o for "Best Quality"
                    temperature=0,
                    openai_api_key=openai_key,
                    streaming=True,
                    callbacks=[StreamHandler(status_container)]
                )
            else:
                headers = {}
                if ollama_key:
                    headers["Authorization"] = f"Bearer {ollama_key}"

                llm = ChatOllama(
                    base_url=ollama_url,
                    model="gemma3:27b",  # Using llama3 for "Best Free"
                    temperature=0,
                    streaming=True,
                    callbacks=[StreamHandler(status_container)],
                    client_kwargs={"headers": headers} if headers else {}
                )

            # Simple RAG Prompt (한국어)
            system_prompt = (
                "당신은 문서 기반 질문 답변 전문가입니다. "
                "아래 제공된 문맥(Context)을 기반으로 사용자의 질문에 상세하게 답변하세요.\n"
                "문맥에 관련 정보가 있다면 반드시 그 내용을 활용하여 답변하세요.\n"
                "문맥에 정보가 전혀 없는 경우에만 모른다고 답변하세요.\n\n"
                f"[문맥]\n{context_text}\n\n"
                "[지시사항]\n"
                "- 문맥에서 관련 내용을 찾아 구체적으로 답변하세요.\n"
                "- 한국어로 답변하세요."
            )

            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt_message)
            ])

            response_content = response.content

        st.session_state.messages.append(
            {"role": "assistant", "content": response_content})
