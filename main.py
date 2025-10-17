from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

# FastAPI 앱 생성
app = FastAPI(
    title="Dinory AI API",
    description="AI 동화 생성 서비스",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8090"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 루트 엔드포인트
@app.get("/")
async def root():
    return {
        "message": "Dinory AI API is running",
        "status": "healthy",
        "version": "1.0.0"
    }


# 헬스 체크 엔드포인트
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "dinory-ai",
        "port": os.getenv("API_PORT", "8000")
    }


# 테스트 엔드포인트
@app.get("/api/test")
async def test():
    return {
        "message": "Test endpoint is working",
        "openai_key_exists": bool(os.getenv("OPENAI_API_KEY")),
        "pinecone_key_exists": bool(os.getenv("PINECONE_API_KEY"))
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True
    )