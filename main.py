# app/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from dotenv import load_dotenv
from starlette.middleware.base import BaseHTTPMiddleware
import os, time, uuid, logging

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

# ────────────────────────────────
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
# ────────────────────────────────

from app.api.endpoints import story_generation, chat
app.include_router(story_generation.router, prefix="/ai")
app.include_router(chat.router, prefix="/api")

# ────────────────────────────────
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
# ────────────────────────────────

def _dump_routes():
    for r in app.routes:
        if isinstance(r, APIRoute):
            app_logger.info(f"[route] {','.join(sorted(r.methods)):<10} {r.path}")

@app.on_event("startup")
async def on_startup():
    app_logger.info("[startup] Dinory AI API Starting…")
    app_logger.info(f"OPENAI_KEY={'set' if os.getenv('OPENAI_API_KEY') else 'unset'}")
    app_logger.info(f"PINECONE_KEY={'set' if os.getenv('PINECONE_API_KEY') else 'unset'}")
    _dump_routes()

@app.get("/")
async def root():
    return {"message": "Dinory AI API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("API_PORT", 8000)), reload=True)
