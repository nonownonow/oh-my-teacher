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
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# .env 파일에서 환경변수 로드
load_dotenv()

# 환경변수에서 API 키 읽기 (Ollama만)
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
        ["GPT-4o (상용/고품질)", "Ollama (설치형/보안)", "하이브리드 (GPT분석+Ollama답변)"],
        index=2,
        help="하이브리드: GPT에게 질문만 전송하여 추론 프레임워크를 받고, Ollama가 PDF 내용과 결합하여 답변 생성 (PDF 보안 유지)"
    )

    ollama_url = "https://ollama.com"
    ollama_key = OLLAMA_API_KEY

    # GPT/하이브리드 모드일 때 OpenAI 키 외부 입력 (항상)
    openai_key = ""
    if model_provider in ["GPT-4o (상용/고품질)", "하이브리드 (GPT분석+Ollama답변)"]:
        openai_key = st.text_input('OpenAI API Key', type="password", help="GPT 모델 사용을 위한 API 키를 입력하세요")

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


def get_reasoning_framework_from_gpt(question: str, api_key: str) -> str:
    """GPT를 사용하여 질문에 대한 추론 프레임워크/가이드를 생성 (PDF 내용 없이 질문만 전송)"""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=api_key,
    )

    # PDF 내용 없이 질문만 GPT에게 전송
    framework_prompt = f"""당신은 질문 분석 전문가입니다.
아래 질문에 답변하기 위한 체계적인 분석 프레임워크를 제공해주세요.

[질문]
{question}

[지시사항]
1. 이 질문에 답변하기 위해 문서에서 찾아야 할 핵심 요소들을 나열하세요.
2. 답변을 구성할 때 고려해야 할 논리적 단계를 제시하세요.
3. 좋은 답변의 구조와 포함해야 할 내용을 안내하세요.
4. 답변 시 주의해야 할 점이나 흔한 실수를 언급하세요.
5. 한국어로 작성하세요.

[분석 프레임워크]"""

    response = llm.invoke([HumanMessage(content=framework_prompt)])
    return response.content


@st.cache_resource(show_spinner="문서 분석 및 임베딩 중...")
def embed_file(file, provider, _api_key):
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

    # Embedding Logic - GPT 또는 하이브리드는 OpenAI 임베딩 사용
    if provider in ["GPT-4o (상용/고품질)", "하이브리드 (GPT분석+Ollama답변)"]:
        if not _api_key:
            st.error("OpenAI API Key Required")
            st.stop()
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small", openai_api_key=_api_key)
        collection_name = "openai_collection"
    else:
        # HuggingFace Embeddings (무료, 로컬 실행)
        embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        collection_name = "hf_collection"

    # Chroma DB - Persistent Client
    safe_name = "".join([c for c in file.name if c.isalnum()])
    provider_prefix = "gpt" if provider != "Ollama (설치형/보안)" else "ollama"
    persist_dir = os.path.join(
        os.getcwd(), ".chroma_db", provider_prefix, safe_name)

    client = chromadb.PersistentClient(path=persist_dir)

    db = Chroma.from_documents(
        texts,
        embeddings_model,
        client=client,
        collection_name=collection_name
    )

    return db.as_retriever(search_kwargs={"k": 5})


if uploaded_file is not None:
    try:
        retriever = embed_file(uploaded_file, model_provider, openai_key)
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
                    model="gpt-4o",
                    temperature=0,
                    openai_api_key=openai_key,
                    streaming=True,
                    callbacks=[StreamHandler(status_container)]
                )

                system_prompt = (
                    "당신은 문서 기반 질문 답변 전문가입니다. "
                    "아래 제공된 문맥을 기반으로 사용자의 질문에 상세하게 답변하세요.\n\n"
                    f"[문맥]\n{context_text}\n\n"
                    "[지시사항]\n"
                    "- 문맥에서 관련 내용을 찾아 구체적으로 답변하세요.\n"
                    "- 한국어로 답변하세요."
                )

                response = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt_message)
                ])

            elif model_provider == "하이브리드 (GPT분석+Ollama답변)":
                # 1단계: GPT로 추론 프레임워크 생성 (PDF 내용 없이 질문만 전송)
                status_container.markdown("🧠 GPT가 추론 프레임워크를 생성 중... (PDF 내용은 전송되지 않습니다)")

                reasoning_framework = get_reasoning_framework_from_gpt(
                    prompt_message, openai_key
                )

                # 2단계: Ollama가 PDF 내용 + GPT 추론 프레임워크를 결합하여 답변 생성
                status_container.markdown("✍️ Ollama가 문서를 분석하고 답변을 생성 중...")

                headers = {}
                if ollama_key:
                    headers["Authorization"] = f"Bearer {ollama_key}"

                llm = ChatOllama(
                    base_url=ollama_url,
                    model="gemma3:27b",
                    temperature=0,
                    streaming=True,
                    callbacks=[StreamHandler(status_container)],
                    client_kwargs={"headers": headers} if headers else {}
                )

                system_prompt = f"""당신은 친절한 과외 선생님입니다.
아래 제공된 문서 내용과 분석 프레임워크를 활용하여 학생의 질문에 답변하세요.

[문서 내용]
{context_text}

[GPT가 제공한 분석 프레임워크]
{reasoning_framework}

[지시사항]
- 위 분석 프레임워크의 가이드를 따라 문서에서 관련 정보를 찾으세요.
- 프레임워크에서 제시한 논리적 단계에 따라 답변을 구성하세요.
- 문서의 구체적인 내용을 인용하며 답변하세요.
- 한국어로 친절하고 상세하게 답변하세요."""

                response = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt_message)
                ])

            else:  # Ollama (설치형/보안)
                headers = {}
                if ollama_key:
                    headers["Authorization"] = f"Bearer {ollama_key}"

                llm = ChatOllama(
                    base_url=ollama_url,
                    model="gemma3:27b",
                    temperature=0,
                    streaming=True,
                    callbacks=[StreamHandler(status_container)],
                    client_kwargs={"headers": headers} if headers else {}
                )

                system_prompt = (
                    "당신은 문서 기반 질문 답변 전문가입니다. "
                    "아래 제공된 문맥을 기반으로 사용자의 질문에 상세하게 답변하세요.\n\n"
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
