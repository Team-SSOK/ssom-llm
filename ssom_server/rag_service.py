import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from qdrant_client import QdrantClient

from log_summary_prompt import get_prompt_template as get_log_prompt, get_output_schema as get_log_schema
from github_issue_prompt import get_prompt_template as get_github_prompt, get_output_schema as get_github_schema

from logging_utils import logger
from env_config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    RETRIEVER_TOP_K,
    LLM_MODEL,
    LLM_TEMPERATURE
)

# Qdrant client 연결
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

if not client.collection_exists(COLLECTION_NAME):
    from qdrant_client.http.models import VectorParams, Distance
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    logger.info(f"{COLLECTION_NAME} 컬렉션 자동 생성됨")

# 임베딩 모델
embedding_model = OpenAIEmbeddings(
    model=EMBEDDING_MODEL
)

# Qdrant VectorStore
vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embedding_model,
)

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})

def get_chain_and_retriever(prompt_type: str):
    if prompt_type == "log_summary":
        prompt_template = get_log_prompt()
        output_schema = get_log_schema()
    elif prompt_type == "github_issue":
        prompt_template = get_github_prompt()
        output_schema = get_github_schema()
    else:
        raise ValueError("지원하지 않는 프롬프트 타입입니다.")

    # LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE
    ).with_structured_output(output_schema, method="json_mode")

    # 입력을 여러 체인(또는 함수)에 병렬로 전달
    setup_and_retrieval = RunnableParallel(
        context=lambda x: retriever.invoke(x["question"]),
        question=RunnablePassthrough()
    )
    # LCEL 체인 조립: 병렬 결과 → 프롬프트 → LLM
    chain = setup_and_retrieval | prompt_template | llm

    return chain, retriever
