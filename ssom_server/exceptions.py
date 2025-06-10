# exceptions.py
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

class CustomException(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code

# FastAPI exception handler
async def custom_exception_handler(request: Request, exc: CustomException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "isSuccess": False,
            "code": exc.code,
            "message": exc.message,
            "result": []
        }
    )
