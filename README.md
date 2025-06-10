# SSOM-LLM

**SSOM-LLM**는 SSOM 프로젝트의 모니터링 시스템에 적용된 LLM 기반 RAG(Retrieval-Augmented Generation) 서버입니다.

- GitHub 저장소를 클론하여 코드를 임베딩하고,
- 임베딩 결과를 벡터 DB(Qdrant)에 저장합니다.
- 사용자가 쿼리를 입력하면, 관련 코드를 검색해 LLM에 전달하여 답변을 생성합니다.
- 프롬프트 엔지니어링 및 구조화된 출력(structured output) 기능을 적용하여,
    - 사용자 코드 기반 로그 요약
    - 로그 기반 GitHub 이슈 초안 작성  
  기능을 제공합니다.

---

## 기술 스택

- **Qdrant** 1.14.0
- **Langchain** 0.3.24
- **Python** 3.13
- **FastAPI** 0.115.12

---

## 실행 방법

### 1. 사전 준비

- Docker 및 Docker Compose가 설치된 환경이 필요합니다.

### 2. 환경 변수 설정

- `docker-compose.yaml` 파일이 위치한 디렉토리에 `.env` 파일을 아래와 같이 작성합니다.

    ```env
    # .env
    # OpenAI API 키
    OPENAI_API_KEY=YOUR_API_KEY

    # 임베딩 모델 (필요에 따라 변경)
    EMBEDDING_MODEL=text-embedding-3-small

    # LLM 모델 (필요에 따라 변경)
    LLM_MODEL=gpt-4o-mini
    LLM_TEMPERATURE=0.2

    # 벡터 DB에서 검색할 상위 문서 개수 (필요에 따라 변경)
    RETRIEVER_TOP_K=3

    # 벡터 DB 컬렉션 이름 (필요에 따라 변경)
    COLLECTION_NAME=java-files
    ```

### 3. 서비스 실행

```bash
docker-compose up -d
```

### 4. 서비스 중지

```bash
docker-compose down
```
