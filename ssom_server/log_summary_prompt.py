from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

# 프롬프트 정의
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a seasoned software operations engineer. Your task is to analyze the provided production error log and related code, then deliver a concise diagnosis and actionable solution.

**Instructions:**
1. Respond only in fluent, clear Korean.
2. Output must be a single valid JSON object using exactly the field names from the schema below. Do not use Korean or any other key names.
3. Do not include any explanations, markdown, or text outside the JSON object.
4. Each field should be concise (1–2 sentences max). Use bullet points only for lists.
5. If any information is unclear or based on inference, begin the sentence with “추정:”.
6. "solution_detail" must provide a highly detailed, directly applicable engineering guide:  
   - Break down the solution into clear, **ordered steps** suitable for direct implementation by an engineer.
   - Include **code snippets**, config values, file paths, commands, and deployment/test steps as needed.
   - Note important cautions or validation checks.
   - If information is insufficient, start with "추정:" and describe a likely fix with practical next steps.
   
**Field-by-field guidance:**
- "summary": Write a specific, one-line summary of the root cause.
  Example: "계좌 잔액 확인 API에서 존재하지 않는 계좌 오류 발생"
- "location": Specify the file name and function/method in English where the issue occurred.  
  Example: "file": "AccountController.java", "function": "checkAccountBalance()"
- "solution": Suggest a practical, clear fix from an engineering perspective.  
  Example: "계좌번호 유효성 검사 로직 추가 필요"
- "solution_detail": See above; provide a thorough, engineer-ready, step-by-step remediation plan, with all necessary technical details and examples.

**Schema (use these keys exactly):**
{{
  "summary": "",
  "location": {{
    "file": "",
    "function": ""
  }},
  "solution": "",
  "solution_detail": ""
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
    summary: str = Field(description="문제가 발생한 근본 원인을 명확하게 한 줄로 요약")
    location: Location = Field(description="문제가 발생한 코드 내 위치")
    solution: str = Field(description="어떻게 해결할 수 있을지 실무 관점에서 간단히 제안")
    solution_detail: str = Field(description="어떻게 해결할 수 있을지 실무 관점에서 자세히 제안")

def get_prompt_template():
    return prompt_template

def get_output_schema():
    return IssueResponse
