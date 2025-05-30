from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List

# 프롬프트 정의
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert DevOps engineer. Your task is to analyze the provided production error log and related code, then summarize the incident as a GitHub issue for your engineering team.

**Instructions:**
1. Respond only in fluent, clear Korean.
2. Output must be a single valid JSON object using exactly the field names from the schema below. Do not use Korean or any other key names.
3. Do not include any explanations, markdown, or text outside the JSON object.
4. Each field should be concise (1–2 sentences max). Use bullet points only for lists.
5. If information is unclear or inferred, begin the sentence with “불명확” or “추정”.
6. Include only essential error messages or stack traces.

**Field-by-field guidance:**
- "title": Always start with "hotfix: " and write a concise, one-line summary of the issue.  
  Example: "hotfix: 계좌 잔액 확인 시 존재하지 않는 계좌 오류 발생"
- "description": Clearly describe the production issue in one or two sentences.
- "location": Provide the file name and function/method name in English.  
  Example: "file": "AccountController.java", "function": "checkAccountBalance()"
- "cause": Summarize the root cause. If not certain, start with "불명확" or "추정".
- "reproduction_steps": Write a step-by-step, **numbered list** so that another developer can reliably reproduce the error.  
  Each step must begin with an action verb (e.g., "로그인", "입력", "호출", "전송").
  Clearly specify required environment, settings, and input values.
  Number the steps as strings (e.g., "1. ...", "2. ...").
  Example: ["1. 운영 환경에서 정상 계좌번호 사용", "2. /api/bank/account/balance API 호출", "3. 서버 콘솔에 에러 로그 출력 확인"]
- "log": Include only the most essential error message(s) or stack trace(s), up to 2 lines.
- "solution": Suggest a practical, actionable fix from an engineering perspective.
- "references": Provide as a single string. If multiple, separate with commas. If none, write "없음".  
  Example: "AccountController.java, AccountServiceImpl.java" or "없음"

**Schema (use these keys exactly):**
{{
  "title": "",
  "description": "",
  "location": {{
    "file": "",
    "function": ""
  }},
  "cause": "",
  "reproduction_steps": [],
  "log": "",
  "solution": "",
  "references": ""
}}

**Inputs:**
- Error log: {question}
- Reference code: {context}

**Output:**  
Respond ONLY with the JSON object matching the schema above.  
Do not include any text, explanation, or markdown outside the JSON.
"""
)

# Pydantic 스키마 정의
class Location(BaseModel):
    file: str = Field(description="문제가 발생한 파일 (예: `AccountController.java`)")
    function: str = Field(description="문제가 발생한 함수 또는 메소드 (예: `checkAccountBalance()`)")

class IssueResponse(BaseModel):
    title: str = Field(description="문제를 한 줄로 요약한 간결한 제목 작성. 반드시 'hotfix: '로 시작")
    description: str = Field(description="운영 환경에서 발생한 문제를 한두 문장으로 설명")
    location: Location = Field(description="문제가 발생한 코드 내 위치")
    cause: str = Field(description="문제가 발생한 근본 원인을 명확하게 요약")
    reproduction_steps: List[str] = Field(description="1. 어떤 조건/설정/환경에서\n2. 어떤 동작을 했을 때 발생하는지")
    log: str = Field(description="핵심 에러 메시지 또는 스택트레이스 1~2줄")
    solution: str = Field(description="어떻게 해결할 수 있을지 실무 관점에서 간단히 제안")
    references: str = Field(description="참고할 코드/문서/링크 등. 여러 개면 쉼표로 구분. 없으면 '없음'")

def get_prompt_template():
    return prompt_template

def get_output_schema():
    return IssueResponse
