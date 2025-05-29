import traceback
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from embedding_service import embed_documents
from rag_service import get_chain_and_retriever
from logging_utils import log_relevant_docs
from typing import List

# FastAPI 앱 생성
app = FastAPI()

# request 스키마
class QuestionRequest(BaseModel):
    log: List[str]

class EmbeddingRequest(BaseModel):
    github_url: str

# response 스키마
class QuestionResponseItem(BaseModel):
    log: str
    message: dict

class QuestionResponse(BaseModel):
    isSuccess: bool
    code: int
    message: str
    result: List[QuestionResponseItem]

class EmbeddingResponse(BaseModel):
    isSuccess: bool
    code: int
    message: str


# 질문 API
@app.post("/api/logs/summary", response_model=QuestionResponse)
async def analyze_logs(request: QuestionRequest):
    try:
        # 프롬프트 타입 "log_summary" or "github_issue"
        # 체인과 retriever 동시 획득
        chain, retriever = get_chain_and_retriever("log_summary")

        tasks = [
            process_log(chain, retriever, log)
            for log in request.log
        ]

        results = await asyncio.gather(*tasks)

        return QuestionResponse(
            isSuccess=True,
            code=2000,
            message="로그 요약 성공했음 ㅋ",
            result=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/logs/github-issues", response_model=QuestionResponse)
async def analyze_logs(request: QuestionRequest):
    try:
        # 프롬프트 타입 "log_summary" or "github_issue"
        # 체인과 retriever 동시 획득
        chain, retriever = get_chain_and_retriever("github_issue")

        tasks = [
            process_log(chain, retriever, log)
            for log in request.log
        ]

        results = await asyncio.gather(*tasks)

        return QuestionResponse(
            isSuccess=True,
            code=2000,
            message="깃허브 이슈 작성 성공했음 ㅋ",
            result=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 임베딩 API
@app.post("/api/codes/embedding", response_model=EmbeddingResponse)
async def embed_codes(request: EmbeddingRequest):
    try:
        embed_documents(request.github_url)

        return EmbeddingResponse(
            isSuccess=True,
            code=2001,
            message=f"{request.github_url} 주소 코드의 임베딩을 완료했습니다."
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

async def process_log(chain, retriever, log):
    # 관련 문서 검색 및 로깅
    # relevant_docs = await asyncio.to_thread(retriever.invoke, log)
    # log_relevant_docs(relevant_docs)

    # 체인 실행
    result = await asyncio.to_thread(chain.invoke, log)

    return QuestionResponseItem(log=log, message=result.model_dump())
