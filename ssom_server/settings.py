from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    qdrant_host: str = Field(..., env="QDRANT_HOST")
    qdrant_port: int = Field(6333, env="QDRANT_PORT")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    llm_model: str = Field("gpt-4.1-mini", env="LLM_MODEL")
    llm_temperature: float = Field(0.2, env="LLM_TEMPERATURE")
    retriever_top_k: int = Field(3, env="RETRIEVER_TOP_K")
    collection_name: str = Field("java-files", env="COLLECTION_NAME")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @field_validator("*", mode="before")
    @classmethod
    def empty_string_as_default(cls, v, info):
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return info.field.default if info.field.default is not None else None
        return v

    @field_validator("openai_api_key", "qdrant_host", mode="before")
    @classmethod
    def required_env(cls, v, info):
        if v is None or (isinstance(v, str) and v.strip() == ""):
            raise ValueError(f"필수 환경 변수 '{info.field.name}'가 누락되었습니다.")
        return v

try:
    settings = Settings()
except ValidationError as e:
    print("환경설정 오류:", e)
    raise
