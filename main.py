from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from dotenv import load_dotenv
from starlette.middleware.base import BaseHTTPMiddleware
import os, time, uuid, logging, inspect

load_dotenv()

app_logger = logging.getLogger("dinory.http")
if not app_logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[HTTP] %(asctime)s | %(levelname)s | %(message)s"))
    app_logger.addHandler(h)
app_logger.setLevel(logging.INFO)

app = FastAPI(title="Dinory AI API", description="AI 동화 생성 서비스", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8090"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = str(uuid.uuid4())[:8]
        t0 = time.perf_counter()
        app_logger.info(f"[{rid}] -> {request.method} {request.url.path}")
        try:
            response = await call_next(request)
        except Exception as e:
            app_logger.exception(f"[{rid}] !! Exception during request: {e}")
            raise
        dt = (time.perf_counter() - t0) * 1000
        app_logger.info(f"[{rid}] <- {response.status_code} {request.method} {request.url.path} ({dt:.1f} ms)")
        return response

app.add_middleware(AccessLogMiddleware)

# --- 라우터 등록 ---
from app.api.endpoints.story_generation import router as story_router
from app.api.endpoints.chat import router as chat_router
from app.api.endpoints.memory_sync import router as memory_router
from app.api.endpoints.memory_query import router as memory_query_router  # [2025-11-04 김민중 추가]
import app.api.endpoints.story_generation as story_generation_mod
import app.api.endpoints.chat as chat_mod

app.include_router(story_router, prefix="/ai", tags=["ai"])
app.include_router(chat_router,  prefix="/api", tags=["chat"])
app.include_router(memory_router, prefix="/api/memory", tags=["memory"])  # RAG 메모리 동기화
app.include_router(memory_query_router, prefix="/api/memory", tags=["memory"])  # [2025-11-04 김민중 추가] Pinecone 조회

# ★ alias 라우트: 지연 import로 순환/경로 문제 방지
from pydantic import BaseModel
class _NextSceneBody(BaseModel):
    storyId: str | None = None  # 최소 스키마(유효성 422 확인용)
    sceneNumber: int | None = None
    childId: int | None = None

@app.post("/ai/generate-next-scene")
async def _alias_generate_next_scene(req: _NextSceneBody):
    from app.api.endpoints.story_generation import generate_next_scene, NextSceneRequest
    real_req = NextSceneRequest(
        storyId=req.storyId or "dummy",
        childId=req.childId or 0,
        sceneNumber=req.sceneNumber or 1,
        previousChoices=[]
    )
    return await generate_next_scene(real_req)

@app.post("/ai/generate-first-scene")
async def _alias_generate_first_scene(req: _NextSceneBody):
    from app.api.endpoints.story_generation import generate_next_scene, NextSceneRequest
    real_req = NextSceneRequest(
        storyId=req.storyId or "dummy",
        childId=req.childId or 0,
        sceneNumber=1,
        previousChoices=[]
    )
    return await generate_next_scene(real_req)

# --- 예외 핸들러 ---
@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    app_logger.error(f"[VALIDATION] {request.method} {request.url.path} -> {exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    app_logger.warning(f"[404] {request.method} {request.url.path}")
    return JSONResponse(status_code=404, content={"detail": "Not Found"})

@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    app_logger.exception(f"[EXC] {request.method} {request.url.path} -> {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

# --- 디버그: 라우트 덤프/확인 ---
def _dump_routes():
    for r in app.routes:
        if isinstance(r, APIRoute):
            app_logger.info(f"[route] {','.join(sorted(r.methods)):<10} {r.path} -> {r.endpoint.__name__}")

@app.get("/__routes")
async def __routes():
    return {
        "routes": [
            {"methods": sorted(list(r.methods)), "path": r.path, "endpoint": r.endpoint.__name__}
            for r in app.routes if isinstance(r, APIRoute)
        ]
    }

@app.on_event("startup")
async def on_startup():
    app_logger.info("[startup] Dinory AI API Starting…")
    app_logger.info(f"OPENAI_KEY={'set' if os.getenv('OPENAI_API_KEY') else 'unset'}")
    app_logger.info(f"PINECONE_KEY={'set' if os.getenv('PINECONE_API_KEY') else 'unset'}")
    app_logger.info(f"[file] story_generation.py -> {inspect.getfile(story_generation_mod)}")
    app_logger.info(f"[file] chat.py            -> {inspect.getfile(chat_mod)}")
    _dump_routes()

@app.get("/")
async def root():
    return {"message": "Dinory AI API is running", "status": "healthy", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "dinory-ai", "port": os.getenv("API_PORT", "8000")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("API_PORT", 8000)), reload=True)
