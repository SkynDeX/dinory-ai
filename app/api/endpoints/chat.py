from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from app.services.chat.chatbot_service import ChatbotService
from app.services.chat.chatbot_service_with_rag import ChatbotServiceWithRAG
from app.services.chat.response_generator import ResponseGenerator

router = APIRouter()

# RAG 메모리 사용 여부 (환경변수로 제어)
USE_RAG = os.getenv("USE_RAG_MEMORY", "false").lower() == "true"
USE_PINECONE = os.getenv("USE_PINECONE_MEMORY", "false").lower() == "true"

# 서비스를 전역 변수로 선언하지만 초기화는 하지 않음
_chatbot_service = None
_response_generator = None

def get_chatbot_service():
    global _chatbot_service
    if _chatbot_service is None:
        if USE_RAG:
            print(f"✅ RAG Memory ENABLED (Pinecone: {USE_PINECONE})")
            _chatbot_service = ChatbotServiceWithRAG(use_pinecone=USE_PINECONE)
        else:
            print("⚠️ RAG Memory DISABLED (using basic chatbot service)")
            _chatbot_service = ChatbotService()
    return _chatbot_service

def get_response_generator():
    global _response_generator
    if _response_generator is None:
        _response_generator = ResponseGenerator()
    return _response_generator


class ChatRequest(BaseModel):
    session_id: int
    message: str
    child_id: Optional[int] = None


class ChatResponse(BaseModel):
    session_id: int
    ai_response: str
    emotion: Optional[str] = None


class StoryCompletionChatRequest(BaseModel):
    session_id: int
    child_id: int
    child_name: str
    story_id: str
    story_title: str
    total_time: Optional[int] = None
    abilities: Dict[str, int]
    choices: List[Dict[str, Any]]
    scenes: Optional[List[Dict[str, Any]]] = None  # [2025-11-04 김민중 추가] Scene 정보


class GenerateChoicesRequest(BaseModel):
    session_id: Optional[int] = None
    child_id: Optional[int] = None
    last_message: Optional[str] = None


class GenerateChoicesResponse(BaseModel):
    choices: List[str]
    emotion: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    아이와의 채팅 처리
    """
    try:
        chatbot_service = get_chatbot_service()
        response_generator = get_response_generator()

        # AI 응답 생성
        ai_response = await chatbot_service.generate_response(
            message=request.message,
            session_id=request.session_id,
            child_id=request.child_id
        )

        # 감정 분석 (선택적)
        emotion = response_generator.analyze_emotion(request.message)

        return ChatResponse(
            session_id=request.session_id,
            ai_response=ai_response,
            emotion=emotion
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/init")
async def init_chat(child_id: int):
    """
    새로운 채팅 세션 초기화
    """
    try:
        response_generator = get_response_generator()
        greeting = response_generator.generate_greeting(child_id)
        return {
            "message": "Chat session initialized",
            "greeting": greeting
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/init-from-story", response_model=ChatResponse)
async def init_chat_from_story(request: StoryCompletionChatRequest):
    """
    동화 완료 후 챗봇 세션 시작
    """
    try:
        chatbot_service = get_chatbot_service()

        # [2025-11-04 김민중 수정] Scene 정보도 함께 전달
        first_message = await chatbot_service.generate_first_message_from_story(
            session_id=request.session_id,
            child_name=request.child_name,
            story_title=request.story_title,
            story_id=request.story_id,
            abilities=request.abilities,
            choices=request.choices,
            total_time=request.total_time,
            scenes=request.scenes  # Scene 정보 추가
        )

        return ChatResponse(
            session_id=request.session_id,
            ai_response=first_message,
            emotion=None
        )

    except Exception as e:
        print(f"Error in init_chat_from_story: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/generate-choices", response_model=GenerateChoicesResponse)
async def generate_choices(request: GenerateChoicesRequest):
    """
    [2025-11-04 김민중 추가] AI 기반 동적 선택지 생성
    대화 맥락에 맞는 선택지를 생성하고, Dino의 감정을 판단합니다.
    """
    try:
        chatbot_service = get_chatbot_service()

        # AI 선택지 생성
        result = await chatbot_service.generate_choices(
            session_id=request.session_id or 0,
            child_id=request.child_id,
            last_message=request.last_message
        )

        return GenerateChoicesResponse(
            choices=result["choices"],
            emotion=result["emotion"]
        )

    except Exception as e:
        print(f"Error in generate_choices: {e}")
        raise HTTPException(status_code=500, detail=str(e))
