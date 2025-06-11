import os
import subprocess
import shutil
from pathlib import Path
from urllib.parse import urlparse
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.docstore.document import Document
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from logging_utils import logger
from env_config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    EMBEDDING_MODEL
)

# 깃허브 URL에서 레포 이름 추출
def get_repo_name(github_url: str) -> str:
    path = urlparse(github_url).path
    repo_name_with_git = os.path.basename(path)
    repo_name = repo_name_with_git.removesuffix(".git")
    return repo_name

# github_repo 내부 모든 내용 삭제 함수
def clean_github_repo_dir():
    github_repo_root = Path("./github_repo")
    if github_repo_root.exists() and github_repo_root.is_dir():
        for item in github_repo_root.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        logger.info(f"{github_repo_root} 내부 파일/폴더 삭제 완료")
    else:
        logger.info(f"{github_repo_root} 폴더가 존재하지 않아 삭제과정 스킵")

# 벡터 임베딩 처리 함수
def embed_documents(github_url: str):
    # github_repo 내부 정리
    clean_github_repo_dir()

    # 경로 생성
    repo_name = get_repo_name(github_url)
    clone_dir = f"./github_repo/{repo_name}"

    # 깃허브 레포 클론
    subprocess.run(["git", "clone", github_url, clone_dir], check=True)
    logger.info(f"GitHub 레포지토리 클론 완료: {github_url}")

    # .java 파일 탐색
    java_files = list(Path(clone_dir).rglob("*.java"))
    if not java_files:
        logger.error(".java 파일이 없음")
        for path in Path(clone_dir).rglob("*"):
            logger.info(f"디렉토리 탐색 중: {path}")
        raise ValueError(".java 파일이 없습니다.")
    logger.info(f"{len(java_files)}개의 .java 파일을 찾음")

    # Qdrant 연결
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # 기존 컬렉션 삭제 및 재생성
    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(collection_name=COLLECTION_NAME)
        logger.info(f"기존 컬렉션 삭제: {COLLECTION_NAME}")

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    logger.info(f"새 컬렉션 생성: {COLLECTION_NAME}")

    # 임베딩 모델 초기화
    embedding_model = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    # Document 객체 생성
    docs = []
    for file_path in java_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        docs.append(Document(
            page_content=content,
            metadata={"source": str(file_path)}
        ))

    # Qdrant에 저장
    vectorstore = QdrantVectorStore(
        client=qdrant,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME
    )
    vectorstore.add_documents(docs)

    logger.info("모든 .java 파일 Qdrant에 저장 완료")
