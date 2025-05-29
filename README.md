## SSOK-SANDBOX
### rag-sample
SSOK 프로젝트의 모니터링 시스템에 적용 예정인 LLM 프로토타입입니다.
</br></br>
깃허브 클론 기반으로 코드를 불러와 임베딩하고,</br>
벡터 DB에 저장하는 형태로 구현하였으며,</br>
쿼리 질문 시, 질문에 해당하는 코드를 LLM이 함깨 입력받아 답변을 하는 RAG 기반 시스템입니다.

</br></br>

**기술스택**
- Qdrant 1.14.0
- Langchain 0.3.24
- Python 3.13
- FastAPI 0.115.12

</br></br>

**구동 방법**
1. Docker 및 Docker Compose가 설치된 환경이 필요합니다.
2. docker-compose.yaml 파일이 있는 디렉토리에서, .env 파일을 아래와 같이 작성해주세요.
```
# .env
# OPENAI API 키
OPENAI_API_KEY=YOUR_API_KEY

# 임베딩 모델 (변경 가능)
EMBEDDING_MODEL=text-embedding-3-small

#LLM 모델 (변경 가능)
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.2

# 벡터 DB에서 가져올 데이터 갯수 (상위 N개) (변경 가능)
RETRIEVER_TOP_K=3

# 벡터 DB에 저장할 컬렉션 이름 (변경 가능)
COLLECTION_NAME=java-files
```
3. docker-compose.yaml 파일이 있는 디렉토리에서, 다음 명령어로 구동합니다.
```
docker-compose up -d
```
4. 다음 명령어로 구동을 중지합니다.
```
docker-compose down
```
