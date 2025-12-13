# Streamlit Cloud 배포용 (Linux 환경)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # 로컬(macOS/Windows) 환경에서는 기본 install된 sqlite3 사용
    pass

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
import pdfplumber
from io import BytesIO
import base64
from langchain.callbacks.base import BaseCallbackHandler
from streamlit_extras.buy_me_a_coffee import button
import os
import tempfile
import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader

# 제목
st.title("나의 과외 선생님")
st.write("---")

# OpenAI 키 입력받기
openai_key = st.text_input('OPEN_AI_API_KEY', type="password")

# 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요!", type=['pdf'])
st.write("---")

# Buy me a coffee
button(username="{계정 ID}", floating=True, width=221)


# 스트리밍 처리할 Handler 생성
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# 업로드된 파일 처리
if uploaded_file is not None:
    @st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
    def embed_file(file):
        file_content = file.read()
        file_path = f"./.cache/files/{file.name}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(file_content)

        # PDF 처리: 하이브리드 방식 (텍스트 추출 우선 + 필요시 AI 분석을 위해 이미지 저장)
        documents = []
        images_dir = f"./.cache/images/{file.name}"
        os.makedirs(images_dir, exist_ok=True)

        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""

                # 페이지를 이미지로 변환 및 저장 (벡터 검색 시 활용)
                img = page.to_image(resolution=300).original
                img_path = os.path.join(images_dir, f"page_{i+1}.png")
                img.save(img_path, format="PNG")

                documents.append(Document(page_content=text, metadata={
                                 "page": i+1, "source": file.name, "image_path": img_path}))

        # Splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.split_documents(documents)

        # Embedding
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=openai_key
        )

        # Chroma DB
        db = Chroma.from_documents(texts, embeddings_model)

        # Retriever
        return db.as_retriever(search_kwargs={"k": 5})

    # 캐싱된 함수 호출 (apikey가 있어야 실행)
    if openai_key:
        try:
            retriever = embed_file(uploaded_file)
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.stop()

        # 세션 스테이트 초기화
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "content": "PDF 내용을 기반으로 무엇이든 물어보세요!"}]

        # 이전 대화 내용 출력
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        # 채팅 입력 처리
        if prompt_message := st.chat_input("질문을 입력하세요"):
            # 사용자 메시지 표시 및 저장
            st.chat_message("user").write(prompt_message)
            st.session_state.messages.append(
                {"role": "user", "content": prompt_message})

            with st.chat_message("assistant"):
                # 스트리밍 처리할 Handler 생성
                stream_handler = StreamHandler(st.empty())

                # Retriever (MultiQueryRetriever for better recall)
                llm = ChatOpenAI(temperature=0, openai_api_key=openai_key)
                retriever_from_llm = MultiQueryRetriever.from_llm(
                    retriever=retriever,
                    llm=llm
                )

                # Prompt Template
                # Retrieved Documents
                docs = retriever_from_llm.invoke(prompt_message)

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                # Generate Answer with Multimodal Context
                initial_msg = f"""Answer the question based on the following context and the attached images. If the answer is not in the context, use your general knowledge to answer.

Context:
{format_docs(docs)}

Question: {prompt_message}

Answer in Korean:"""

                messages_content = [{"type": "text", "text": initial_msg}]

                # Add images from retrieved documents (top 3 unique images)
                seen_images = set()
                for doc in docs:
                    img_path = doc.metadata.get("image_path")
                    if img_path and img_path not in seen_images:
                        seen_images.add(img_path)
                        try:
                            with open(img_path, "rb") as f:
                                img_base64 = base64.b64encode(
                                    f.read()).decode("utf-8")
                                messages_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                                })
                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")

                        if len(seen_images) >= 3:
                            break

                message = HumanMessage(content=messages_content)

                # Generate
                generate_llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    openai_api_key=openai_key,
                    streaming=True,
                    callbacks=[stream_handler]
                )

                response = generate_llm.invoke([message])
                st.session_state.messages.append(
                    {"role": "assistant", "content": response.content})
