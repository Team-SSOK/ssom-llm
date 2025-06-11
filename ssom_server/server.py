import traceback
import asyncio
import json
from fastapi import FastAPI
from fastapi_health import health
from pydantic import BaseModel
from qdrant_client import QdrantClient
from embedding_service import embed_documents
from rag_service import get_chain_and_retriever
from logging_utils import log_relevant_docs, log_llm_prompt
from exceptions import CustomException, custom_exception_handler
from typing import List, Dict, Any
from settings import settings

# FastAPI 앱 생성
app = FastAPI()
app.add_exception_handler(CustomException, custom_exception_handler)

# request 스키마
class QuestionRequest(BaseModel):
    log: List[Dict[str, Any]]

class EmbeddingRequest(BaseModel):
    github_url: str

# response 스키마
class QuestionResponseItem(BaseModel):
    message: dict

class QuestionResponse(BaseModel):
    isSuccess: bool
    code: str
    message: str
    result: List[QuestionResponseItem]

class EmbeddingResponse(BaseModel):
    isSuccess: bool
    code: str
    message: str

# Qdrant 연결 체크 함수
async def readiness():
    try:
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        client.get_collections()
        return True
    except Exception as e:
        traceback.print_exc()
        return False

async def liveness():
    return True

# Healthcheck 라우터 설정
app.add_api_route("/health/liveness", health([liveness]))
app.add_api_route("/health/readiness", health([readiness]))

# 질문 API
@app.post("/api/logs/summary", response_model=QuestionResponse)
async def analyze_logs(request: QuestionRequest):
    try:
        # 프롬프트 타입 "log_summary" or "github_issue"
        # 체인과 retriever 동시 획득
        chain, retriever = get_chain_and_retriever("log_summary")

        # 1. 각 로그별로 N개씩 유사 코드 검색
        all_relevant_docs = await get_relevant_docs_for_logs(retriever, request.log)

        # 2. context/question 딕셔너리 생성
        inputs = build_chain_inputs(all_relevant_docs, request.log)

        # 로그 출력
        # log_relevant_docs(all_relevant_docs)
        # log_llm_prompt(inputs["context"], inputs["question"])

        # 3. 체인 실행
        result = await asyncio.to_thread(chain.invoke, inputs)

        return QuestionResponse(
            isSuccess=True,
            code="2000",
            message="로그 요약을 완료했습니다.",
            result=[QuestionResponseItem(message=result.model_dump())]
        )

    except ValueError as ve:
        raise CustomException(code="4001", message=f"입력값 오류: {str(ve)}", status_code=400)

    except Exception as e:
        raise CustomException(code="5000", message=f"알 수 없는 오류: {str(e)}", status_code=500)

@app.post("/api/logs/issues", response_model=QuestionResponse)
async def analyze_logs(request: QuestionRequest):
    try:
        # 프롬프트 타입 "log_summary" or "github_issue"
        # 체인과 retriever 동시 획득
        chain, retriever = get_chain_and_retriever("github_issue")

        # 1. 각 로그별로 N개씩 유사 코드 검색
        all_relevant_docs = await get_relevant_docs_for_logs(retriever, request.log)

        # 2. context/question 딕셔너리 생성
        inputs = build_chain_inputs(all_relevant_docs, request.log)

        # 로그 출력
        # log_relevant_docs(all_relevant_docs)
        # log_llm_prompt(inputs["context"], inputs["question"])

        # 3. 체인 실행
        result = await asyncio.to_thread(chain.invoke, inputs)

        return QuestionResponse(
                isSuccess=True,
                code="2000",
                message="이슈 작성을 완료했습니다.",
                result=[QuestionResponseItem(message=result.model_dump())]
        )

    except ValueError as ve:
        raise CustomException(code="4001", message=f"입력값 오류: {str(ve)}", status_code=400)

    except Exception as e:
        raise CustomException(code="5000", message=f"알 수 없는 오류: {str(e)}", status_code=500)

# 임베딩 API
@app.post("/api/codes/embedding", response_model=EmbeddingResponse)
async def embed_codes(request: EmbeddingRequest):
    try:
        embed_documents(request.github_url)

        return EmbeddingResponse(
            isSuccess=True,
            code="2001",
            message=f"{request.github_url} 주소 코드의 임베딩을 완료했습니다."
        )

    except Exception as e:
        traceback.print_exc()
        raise CustomException(code="5000", message=f"알 수 없는 오류: {str(e)}", status_code=500)


async def get_relevant_docs_for_logs(retriever, logs: List[Dict[str, Any]]) -> List[Any]:
    async def async_invoke(log):
        log_str = json.dumps(log, ensure_ascii=False, indent=2)
        # retriever.invoke를 비동기로 실행
        return await asyncio.to_thread(retriever.invoke, log_str)

    # 각 로그에 대해 태스크 생성
    tasks = [async_invoke(log) for log in logs]
    results = await asyncio.gather(*tasks)

    # 결과를 List[List[문서]]로 반환
    return [doc for sublist in results for doc in sublist]


def build_chain_inputs(all_relevant_docs: List[Any], logs: List[Dict[str, Any]]) -> Dict[str, str]:
    # 코드 내용 합치기
    context_str = "\n\n---\n\n".join([doc.page_content for doc in all_relevant_docs])
    # 로그 전체를 JSON 문자열로
    log_str = json.dumps(logs, ensure_ascii=False, indent=2)

    return {
        "context": context_str,
        "question": log_str
    }