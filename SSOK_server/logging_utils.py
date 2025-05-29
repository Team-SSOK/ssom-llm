import logging

# 로거 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_relevant_docs(relevant_docs):
    logger.info("\n검색된 관련 코드 파일\n")
    if not relevant_docs:
        logger.info("관련 문서를 찾을 수 없음\n")
    else:
        for i, doc in enumerate(relevant_docs, 1):
            logger.info(f"문서 {i}: {doc.metadata.get('source')}")
            # logger.info(f"내용:\n{doc.page_content}\n\n")
