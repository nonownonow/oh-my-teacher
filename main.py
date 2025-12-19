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
        ["GPT-4o (상용/고품질)", "Ollama (설치형/보안)", "하이브리드 (GPT벡터추론+Ollama답변)"],
        index=2,
        help="하이브리드: GPT가 질문만 받아 벡터 추론(의미 확장) 수행 → Ollama가 향상된 벡터 검색으로 고품질 답변 생성 (PDF 내용 보호)"
    )

    ollama_url = "https://ollama.com"
    ollama_key = OLLAMA_API_KEY

    # GPT/하이브리드 모드일 때 OpenAI 키 외부 입력 (항상)
    openai_key = ""
    if model_provider in ["GPT-4o (상용/고품질)", "하이브리드 (GPT벡터추론+Ollama답변)"]:
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


def get_gpt_reasoning_answer(question: str, api_key: str) -> str:
    """
    GPT가 자신의 학습 데이터만으로 질문에 답변 (PDF 내용 없이)
    고품질 추론 답변 생성
    """
    llm = ChatOpenAI(
        model="gpt-4o",  # 고품질 추론
        temperature=0,
        openai_api_key=api_key,
    )

    reasoning_prompt = f"""당신은 지식이 풍부한 전문가입니다.
아래 질문에 대해 당신이 알고 있는 지식을 바탕으로 상세하게 답변해주세요.

[질문]
{question}

[지시사항]
- 질문의 핵심을 파악하고 체계적으로 답변하세요.
- 관련된 배경 지식, 맥락, 중요한 포인트를 포함하세요.
- 정확하고 신뢰할 수 있는 정보만 제공하세요.
- 한국어로 답변하세요.

[답변]"""

    response = llm.invoke([HumanMessage(content=reasoning_prompt)])
    return response.content


def refine_answer_with_gpt(draft_answer: str, question: str, api_key: str) -> str:
    """
    Ollama가 생성한 초안 답변을 GPT가 교정
    더 정확하고 정교하게 다듬음
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=api_key,
    )

    refine_prompt = f"""당신은 전문 편집자입니다.
아래 초안 답변을 검토하고 더 정확하고 정교하게 교정해주세요.

[원본 질문]
{question}

[초안 답변]
{draft_answer}

[교정 지시사항]
1. 사실적 오류가 있으면 수정하세요.
2. 논리적 흐름을 개선하세요.
3. 불명확한 표현을 명확하게 다듬으세요.
4. 중복된 내용은 정리하세요.
5. 교육적으로 부적절한 표현은 순화하세요.
6. 핵심 내용은 유지하면서 품질을 높이세요.
7. 한국어로 자연스럽게 작성하세요.

[교정된 답변]"""

    response = llm.invoke([HumanMessage(content=refine_prompt)])
    return response.content


def verify_with_ollama_pdf(
    gpt_answer: str,
    question: str,
    pdf_context: str,
    ollama_url: str,
    ollama_key: str,
    status_container
) -> str:
    """
    GPT 답변을 PDF 원본 데이터 기반으로 Ollama가 최종 검증
    PDF에 없는 거짓 정보는 삭제하고, PDF 기반 진실만 남김
    """
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

    verify_prompt = f"""당신은 엄격한 팩트체커입니다.
GPT가 작성한 답변을 PDF 원본 문서와 대조하여 철저히 검증하세요.

[PDF 원본 문서 - 유일한 진실의 기준]
{pdf_context}

[GPT가 작성한 답변 - 검증 대상]
{gpt_answer}

[검증 규칙 - 반드시 준수]
1. PDF 문서가 유일한 진실입니다. PDF에 없으면 거짓입니다.
2. 등장인물: PDF에 명시된 이름만 사용하세요. PDF에 없는 인물은 삭제하세요.
3. 관계: PDF에 명시된 관계만 사용하세요. 추측하지 마세요.
4. 사건: PDF에 있는 사건만 포함하세요.
5. GPT가 언급했지만 PDF에 없는 모든 정보는 과감히 삭제하세요.
6. 확신이 없으면 포함하지 마세요.

[콘텐츠 필터링]
- 교육적으로 부적절한 표현은 순화하세요.

[최종 답변]"""

    response = llm.invoke([
        SystemMessage(content=verify_prompt),
        HumanMessage(content=question)
    ])

    return response.content


def get_semantic_expansion_from_gpt(question: str, api_key: str) -> dict:
    """
    GPT를 사용하여 질문의 의미적 확장을 수행 (벡터 추론)
    PDF 내용 없이 질문만 전송하여 관련 개념, 동의어, 하위 질문을 생성
    이를 통해 벡터 검색의 품질을 향상시킴
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,  # 약간의 창의성 허용
        openai_api_key=api_key,
    )

    # GPT에게 질문의 의미적 확장 요청 (벡터 공간에서의 관계 추론)
    expansion_prompt = f"""당신은 의미론적 분석 전문가입니다.
아래 질문을 분석하여 벡터 검색 품질을 높이기 위한 의미적 확장을 수행하세요.

[원본 질문]
{question}

[지시사항]
JSON 형식으로 다음을 제공하세요:
1. "core_concepts": 질문의 핵심 개념 키워드 (3-5개)
2. "synonyms": 각 핵심 개념의 동의어/유사어 (개념당 2-3개)
3. "sub_questions": 원본 질문을 답하기 위해 필요한 하위 질문들 (3-5개)
4. "related_topics": 관련될 수 있는 주제/맥락 (3-5개)
5. "search_queries": 문서에서 검색할 최적화된 쿼리문 (3-5개)

[출력 형식]
```json
{{
  "core_concepts": ["개념1", "개념2", ...],
  "synonyms": {{"개념1": ["동의어1", "동의어2"], ...}},
  "sub_questions": ["하위질문1", "하위질문2", ...],
  "related_topics": ["주제1", "주제2", ...],
  "search_queries": ["쿼리1", "쿼리2", ...]
}}
```"""

    response = llm.invoke([HumanMessage(content=expansion_prompt)])

    # JSON 파싱 시도
    import json
    import re
    try:
        # JSON 블록 추출
        json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        else:
            # JSON 블록 없이 직접 파싱 시도
            return json.loads(response.content)
    except json.JSONDecodeError:
        # 파싱 실패 시 기본값 반환
        return {
            "core_concepts": [question],
            "synonyms": {},
            "sub_questions": [question],
            "related_topics": [],
            "search_queries": [question]
        }


def enhanced_vector_search(retriever, question: str, semantic_expansion: dict, k: int = 5) -> list:
    """
    GPT의 의미적 확장을 활용한 향상된 벡터 검색
    여러 쿼리로 검색 후 중복 제거 및 결과 병합
    """
    all_docs = []
    seen_contents = set()

    # 1. 원본 질문으로 검색
    original_docs = retriever.invoke(question)
    for doc in original_docs:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            all_docs.append(doc)

    # 2. 확장된 검색 쿼리로 추가 검색
    search_queries = semantic_expansion.get("search_queries", [])
    for query in search_queries[:3]:  # 최대 3개 쿼리
        try:
            docs = retriever.invoke(query)
            for doc in docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    all_docs.append(doc)
        except Exception:
            continue

    # 3. 하위 질문으로 추가 검색
    sub_questions = semantic_expansion.get("sub_questions", [])
    for sub_q in sub_questions[:2]:  # 최대 2개 하위 질문
        try:
            docs = retriever.invoke(sub_q)
            for doc in docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    all_docs.append(doc)
        except Exception:
            continue

    # 결과 개수 제한
    return all_docs[:k]


def map_reduce_with_ollama(
    docs: list,
    question: str,
    semantic_expansion: dict,
    ollama_url: str,
    ollama_key: str,
    status_container,
    gpt_reasoning_answer: str = "",
    batch_size: int = 2
) -> str:
    """
    Map-Reduce 패턴으로 문서를 분할 처리 후 합침
    1. Map: 각 문서 배치에서 관련 정보 추출
    2. Reduce: GPT 추론 답변(높은 가중치) + 추출된 정보들을 합쳐서 최종 답변 생성
    """
    headers = {}
    if ollama_key:
        headers["Authorization"] = f"Bearer {ollama_key}"

    # Map 단계용 LLM (스트리밍 없이)
    map_llm = ChatOllama(
        base_url=ollama_url,
        model="gemma3:27b",
        temperature=0,
        streaming=False,
        client_kwargs={"headers": headers} if headers else {}
    )

    # 문서를 배치로 분할
    batches = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]

    # Map 단계: 각 배치에서 관련 정보 추출
    extracted_infos = []
    for idx, batch in enumerate(batches):
        status_container.markdown(f"📄 문서 분석 중... ({idx + 1}/{len(batches)})")

        batch_content = "\n\n".join(doc.page_content for doc in batch)

        map_prompt = f"""다음 문서에서 질문에 답변하기 위해 필요한 핵심 정보만 추출하세요.

[문서]
{batch_content}

[질문]
{question}

[핵심 개념 참고]
{', '.join(semantic_expansion.get('core_concepts', []))}

[지시사항]
- 질문과 관련된 정보만 간결하게 추출하세요.
- 불필요한 정보는 제외하세요.
- 관련 정보가 없으면 "관련 정보 없음"이라고 답하세요.
- 교육적으로 부적절한 표현은 순화된 표현으로 대체하세요.

[추출된 정보]"""

        try:
            response = map_llm.invoke([HumanMessage(content=map_prompt)])
            if "관련 정보 없음" not in response.content:
                extracted_infos.append(response.content)
        except Exception:
            continue

    if not extracted_infos:
        return "문서에서 관련 정보를 찾을 수 없습니다."

    # Reduce 단계: 추출된 정보들을 합쳐서 최종 답변 생성
    status_container.markdown("✍️ 최종 답변 생성 중...")

    combined_info = "\n\n---\n\n".join(extracted_infos)

    # Reduce용 LLM (스트리밍 포함)
    reduce_llm = ChatOllama(
        base_url=ollama_url,
        model="gemma3:27b",
        temperature=0,
        streaming=True,
        callbacks=[StreamHandler(status_container)],
        client_kwargs={"headers": headers} if headers else {}
    )

    expansion_info = f"""[GPT 벡터 추론 결과]
- 핵심 개념: {', '.join(semantic_expansion.get('core_concepts', []))}
- 관련 주제: {', '.join(semantic_expansion.get('related_topics', []))}
- 분석 관점: {', '.join(semantic_expansion.get('sub_questions', [])[:3])}"""

    # GPT 추론 답변 섹션 (참고용, 낮은 가중치)
    gpt_section = ""
    if gpt_reasoning_answer:
        gpt_section = f"""
[GPT 참고 답변 - 구조/표현만 참고]
{gpt_reasoning_answer}

"""

    reduce_prompt = f"""당신은 친절한 과외 선생님입니다.
아래 정보들을 종합하여 학생의 질문에 답변을 작성하세요.

[★★★ PDF 문서 정보 - 가중치 최우선 ★★★]
{combined_info}

{expansion_info}
{gpt_section}
[가중치 적용 규칙 - 매우 중요]
1. PDF 문서 정보에 가장 높은 가중치(70%)를 부여하세요.
2. GPT 답변은 구조와 표현 참고용으로만 사용하세요(30%).
3. PDF 문서와 GPT 답변이 충돌할 경우, PDF 문서를 우선하세요.
4. 등장인물 이름, 관계, 사건은 반드시 PDF 문서 그대로 사용하세요.
5. GPT가 언급했지만 PDF에 없는 정보는 포함하지 마세요.

[지시사항]
- PDF 문서의 내용을 기반으로 답변을 구성하세요.
- GPT 답변의 좋은 구조와 표현만 참고하세요.
- 한국어로 친절하고 상세하게 답변하세요.

[콘텐츠 필터링 - 필수]
- 교육적으로 부적절한 표현(난봉, 바람둥이, 색골 등)은 순화된 표현으로 대체하세요.
- 선정적이거나 폭력적인 묘사는 피하고 교육적으로 적합한 표현을 사용하세요.
- 학생에게 적합한 품위 있는 언어를 사용하세요."""

    response = reduce_llm.invoke([
        SystemMessage(content=reduce_prompt),
        HumanMessage(content=question)
    ])

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
    if provider in ["GPT-4o (상용/고품질)", "하이브리드 (GPT벡터추론+Ollama답변)"]:
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
                    "- 한국어로 답변하세요.\n\n"
                    "[콘텐츠 필터링 - 필수]\n"
                    "- 교육적으로 부적절한 표현(난봉, 바람둥이, 색골 등)은 순화된 표현으로 대체하세요.\n"
                    "- 선정적이거나 폭력적인 묘사는 피하고 교육적으로 적합한 표현을 사용하세요.\n"
                    "- 학생에게 적합한 품위 있는 언어를 사용하세요."
                )

                response = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt_message)
                ])

            elif model_provider == "하이브리드 (GPT벡터추론+Ollama답변)":
                # 진행 상태만 간단히 표시
                progress_placeholder = status_container.empty()
                progress_placeholder.markdown("⏳ 답변 생성 중...")

                # 1단계: GPT 추론 답변 (PDF 없이)
                gpt_reasoning_answer = get_gpt_reasoning_answer(
                    prompt_message, openai_key
                )

                # 2단계: GPT 벡터 추론
                semantic_expansion = get_semantic_expansion_from_gpt(
                    prompt_message, openai_key
                )

                # 3단계: 향상된 벡터 검색
                enhanced_docs = enhanced_vector_search(
                    retriever, prompt_message, semantic_expansion, k=10
                )

                # 4단계: Map-Reduce (중간 출력 숨김)
                hidden_container = st.empty()
                draft_answer = map_reduce_with_ollama(
                    docs=enhanced_docs,
                    question=prompt_message,
                    semantic_expansion=semantic_expansion,
                    ollama_url=ollama_url,
                    ollama_key=ollama_key,
                    status_container=hidden_container,
                    gpt_reasoning_answer=gpt_reasoning_answer,
                    batch_size=2
                )
                hidden_container.empty()  # 중간 출력 삭제

                # 5단계: GPT 교정
                gpt_refined_answer = refine_answer_with_gpt(
                    draft_answer, prompt_message, openai_key
                )

                # 6단계: Ollama 최종 검증 (PDF 기반 진위 판별)
                progress_placeholder.empty()  # 진행 상태 삭제

                # enhanced_docs를 사용하여 더 많은 PDF 컨텍스트로 검증
                enhanced_context = "\n\n".join(doc.page_content for doc in enhanced_docs)

                final_answer = verify_with_ollama_pdf(
                    gpt_answer=gpt_refined_answer,
                    question=prompt_message,
                    pdf_context=enhanced_context,
                    ollama_url=ollama_url,
                    ollama_key=ollama_key,
                    status_container=status_container
                )

                # 세션에 저장하고 종료
                st.session_state.messages.append(
                    {"role": "assistant", "content": final_answer})
                st.stop()

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
                    "- 한국어로 답변하세요.\n\n"
                    "[콘텐츠 필터링 - 필수]\n"
                    "- 교육적으로 부적절한 표현(난봉, 바람둥이, 색골 등)은 순화된 표현으로 대체하세요.\n"
                    "- 선정적이거나 폭력적인 묘사는 피하고 교육적으로 적합한 표현을 사용하세요.\n"
                    "- 학생에게 적합한 품위 있는 언어를 사용하세요."
                )

                response = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt_message)
                ])

            response_content = response.content

        st.session_state.messages.append(
            {"role": "assistant", "content": response_content})
