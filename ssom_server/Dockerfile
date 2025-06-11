FROM python:3.13-slim

# git 설치
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# 필요한 파일 복사
COPY server.py .
COPY embedding_service.py .
COPY rag_service.py .
COPY log_summary_prompt.py .
COPY github_issue_prompt.py .
COPY logging_utils.py .
COPY exceptions.py .
COPY settings.py .
COPY requirements.txt .
COPY .env .

# github_repo 폴더 만들기
RUN mkdir -p github_repo

# 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 환경 변수 설정
ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

# 서버 실행
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
