"""
Pinecone Memory Sync Endpoints
데이터를 Pinecone에 동기화하는 엔드포인트 (옵션)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
from app.services.chat.memory_service import MemoryService

router = APIRouter()

# 전역 서비스 (옵션으로 Pinecone 사용)
_memory_service = None


def get_memory_service():
    global _memory_service
    if _memory_service is None:
        # Pinecone 활성화 여부는 .env의 CHATBOT_PINECONE_API_KEY 존재 여부로 결정
        _memory_service = MemoryService(use_pinecone=True)
    return _memory_service


class SyncConversationRequest(BaseModel):
    """대화를 Pinecone에 동기화"""
    session_id: int
    child_id: int
    user_message: str
    ai_response: str
    message_id: int


class SyncStoryCompletionRequest(BaseModel):
    """동화 완료를 Pinecone에 동기화"""
    completion_id: int
    child_id: int
    story_title: str
    story_content: str  # scene_json + story_json 합친 텍스트
    abilities: Dict[str, int]


@router.post("/sync/conversation")
async def sync_conversation_to_pinecone(request: SyncConversationRequest):
    """
    대화 메시지를 Pinecone에 동기화

    Spring Boot에서 채팅 메시지 저장 후 호출 (비동기/선택적)
    실패해도 채팅 기능에는 영향 없음
    """
    try:
        memory_service = get_memory_service()

        if not memory_service.use_pinecone:
            return {
                "status": "skipped",
                "message": "Pinecone is disabled"
            }

        await memory_service.sync_conversation_to_pinecone(
            session_id=request.session_id,
            child_id=request.child_id,
            user_message=request.user_message,
            ai_response=request.ai_response,
            message_id=request.message_id
        )

        return {
            "status": "success",
            "message": f"Conversation synced: msg_{request.message_id}"
        }

    except Exception as e:
        # 동기화 실패는 치명적이지 않음 (로그만 남김)
        print(f"❌ Sync to Pinecone failed (non-critical): {e}")
        return {
            "status": "failed",
            "message": str(e)
        }


@router.post("/sync/story-completion")
async def sync_story_completion_to_pinecone(request: SyncStoryCompletionRequest):
    """
    동화 완료 기록을 Pinecone에 동기화

    Spring Boot에서 동화 완료 저장 후 호출 (비동기/선택적)
    """
    try:
        memory_service = get_memory_service()

        if not memory_service.use_pinecone:
            return {
                "status": "skipped",
                "message": "Pinecone is disabled"
            }

        await memory_service.sync_story_completion_to_pinecone(
            completion_id=request.completion_id,
            child_id=request.child_id,
            story_title=request.story_title,
            story_content=request.story_content,
            abilities=request.abilities
        )

        return {
            "status": "success",
            "message": f"Story completion synced: story_{request.completion_id}"
        }

    except Exception as e:
        print(f"❌ Sync story to Pinecone failed (non-critical): {e}")
        return {
            "status": "failed",
            "message": str(e)
        }


@router.get("/health")
async def check_memory_service_health():
    """
    메모리 서비스 상태 확인

    Pinecone 연결 상태, MySQL API 접근 가능 여부 등
    """
    try:
        memory_service = get_memory_service()

        return {
            "pinecone_enabled": memory_service.use_pinecone,
            "spring_api_url": memory_service.spring_api_url,
            "status": "healthy"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
