"""
[2025-11-04 김민중 추가] Pinecone 기반 대화 조회 엔드포인트
MySQL 대신 Pinecone에서 대화 기록을 조회합니다.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.services.chat.memory_service import MemoryService

router = APIRouter()

# 전역 서비스
_memory_service = None


def get_memory_service():
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService(use_pinecone=True)
    return _memory_service


class ConversationHistory(BaseModel):
    """대화 기록 응답"""
    message_id: str
    session_id: int
    child_id: int
    message: str
    response: str
    created_at: str
    score: Optional[float] = None  # 시맨틱 검색 점수


@router.get("/conversations/child/{child_id}")
async def get_conversations_by_child(
    child_id: int,
    limit: int = Query(default=10, ge=1, le=100)
):
    """
    Pinecone에서 특정 아이의 최근 대화 조회

    필터 기반 조회입니다.
    """
    try:
        memory_service = get_memory_service()

        if not memory_service.use_pinecone:
            raise HTTPException(
                status_code=503,
                detail="Pinecone is not enabled. Please set CHATBOT_PINECONE_API_KEY"
            )

        # [2025-11-04 김민중 수정] Pinecone list() 사용하여 필터 기반 조회
        # query()는 벡터 검색용이므로, 메타데이터 필터만으로는 부적합
        # 대신 list()로 ID prefix 조회 후 fetch()로 메타데이터 가져오기

        # msg_ prefix로 시작하는 모든 메시지 조회
        list_response = memory_service.index.list(prefix="msg_", limit=limit * 10)

        if not list_response or not hasattr(list_response, 'vectors'):
            return {
                "total": 0,
                "conversations": []
            }

        # ID 목록 가져오기
        vector_ids = [v.id for v in list_response.vectors] if hasattr(list_response, 'vectors') else []

        if not vector_ids:
            return {
                "total": 0,
                "conversations": []
            }

        # 메타데이터 fetch
        fetch_response = memory_service.index.fetch(ids=vector_ids)

        conversations = []
        for vec_id, vec_data in fetch_response.get('vectors', {}).items():
            metadata = vec_data.get('metadata', {})

            # child_id 필터링
            if metadata.get('child_id') == child_id:
                conversations.append({
                    "message_id": vec_id,
                    "session_id": metadata.get('session_id'),
                    "child_id": metadata.get('child_id'),
                    "message": metadata.get('message', ''),
                    "response": metadata.get('response', ''),
                    "created_at": metadata.get('created_at', '')
                })

        # 최신순으로 정렬
        conversations.sort(key=lambda x: x['created_at'], reverse=True)
        conversations = conversations[:limit]

        return {
            "total": len(conversations),
            "conversations": conversations
        }

    except Exception as e:
        print(f"❌ Error fetching conversations from Pinecone: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/search")
async def search_conversations(
    child_id: int,
    query: str,
    limit: int = Query(default=5, ge=1, le=20)
):
    """
    Pinecone 시맨틱 검색으로 관련 대화 찾기

    예: "동화 추천해줘" → 과거에 동화 관련 대화 검색
    """
    try:
        memory_service = get_memory_service()

        if not memory_service.use_pinecone:
            raise HTTPException(
                status_code=503,
                detail="Pinecone is not enabled"
            )

        # 시맨틱 검색
        similar_convs = await memory_service.search_similar_conversations(
            query=query,
            child_id=child_id,
            top_k=limit
        )

        return {
            "query": query,
            "total": len(similar_convs),
            "similar_conversations": similar_convs
        }

    except Exception as e:
        print(f"❌ Error searching conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/session/{session_id}")
async def get_conversations_by_session(
    session_id: int,
    limit: int = Query(default=50, ge=1, le=200)
):
    """
    Pinecone에서 특정 세션의 대화 조회
    """
    try:
        memory_service = get_memory_service()

        if not memory_service.use_pinecone:
            raise HTTPException(
                status_code=503,
                detail="Pinecone is not enabled"
            )

        # [2025-11-04 김민중 수정] list + fetch 방식 사용
        list_response = memory_service.index.list(prefix="msg_", limit=limit * 10)

        if not list_response or not hasattr(list_response, 'vectors'):
            return {
                "session_id": session_id,
                "total": 0,
                "conversations": []
            }

        vector_ids = [v.id for v in list_response.vectors] if hasattr(list_response, 'vectors') else []

        if not vector_ids:
            return {
                "session_id": session_id,
                "total": 0,
                "conversations": []
            }

        fetch_response = memory_service.index.fetch(ids=vector_ids)

        conversations = []
        for vec_id, vec_data in fetch_response.get('vectors', {}).items():
            metadata = vec_data.get('metadata', {})

            # session_id 필터링
            if metadata.get('session_id') == session_id:
                conversations.append({
                    "message_id": vec_id,
                    "session_id": metadata.get('session_id'),
                    "child_id": metadata.get('child_id'),
                    "message": metadata.get('message', ''),
                    "response": metadata.get('response', ''),
                    "created_at": metadata.get('created_at', '')
                })

        # 시간순으로 정렬
        conversations.sort(key=lambda x: x['created_at'])
        conversations = conversations[:limit]

        return {
            "session_id": session_id,
            "total": len(conversations),
            "conversations": conversations
        }

    except Exception as e:
        print(f"❌ Error fetching session conversations: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
