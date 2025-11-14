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


class NavigationIntentRequest(BaseModel):
    message: str


class NavigationIntentResponse(BaseModel):
    has_navigation_intent: bool
    target_path: Optional[str] = None
    confidence: float
    reason: Optional[str] = None


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


@router.post("/chat/analyze-navigation", response_model=NavigationIntentResponse)
async def analyze_navigation_intent(request: NavigationIntentRequest):
    """
    사용자 메시지에서 페이지 이동 의도를 분석합니다.
    """
    try:
        from openai import OpenAI
        import json

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # 페이지 매핑 정보
        page_mappings = {
            "홈": "/home",
            "홈페이지": "/home",
            "메인": "/home",
            "메인페이지": "/home",
            "동화": "/story/list",
            "동화목록": "/story/list",
            "동화리스트": "/story/list",
            "동화페이지": "/story/list",
            "이야기": "/story/list",
            "스토리": "/story/list",
            "대시보드": "/parent/dashboard",
            "부모대시보드": "/parent/dashboard",
            "통계": "/parent/dashboard",
            "리포트": "/parent/dashboard",
            "보고서": "/parent/dashboard",
            "자녀선택": "/child/select",
            "아이선택": "/child/select",
            "자녀등록": "/child/registration",
            "아이등록": "/child/registration",
            "자녀추가": "/child/registration",
            "감정선택": "/child/emotion",
            "감정체크": "/child/emotion",
            "기분선택": "/child/emotion",
            "관심사": "/child/interest",
            "관심사선택": "/child/interest",
            "공룡": "/my-dinos",
            "내공룡": "/my-dinos",
            "공룡보기": "/my-dinos",
            "디노": "/my-dinos",
            "프로필": "/profile",
            "내정보": "/profile",
            "랜딩": "/landing",
            "소개": "/landing",
        }

        system_prompt = f"""당신은 사용자의 메시지에서 페이지 이동 의도를 분석하는 전문가입니다.

사용 가능한 페이지 목록:
{json.dumps(page_mappings, ensure_ascii=False, indent=2)}

**중요 규칙:**
1. "이동", "가자", "보여줘", "가줘", "열어줘", "보고싶어" 등의 표현이 있으면 페이지 이동 의도로 판단
2. 위 페이지 목록에 있는 키워드가 포함되면 해당 페이지로 매핑
3. 명확한 이동 요청은 confidence 0.9 이상
4. 애매한 표현도 페이지 키워드가 있으면 confidence 0.7 이상
5. 일반 대화는 confidence 0.0

응답은 반드시 다음 JSON 형식으로만 반환하세요 (마크다운 코드블록 없이):
{{
    "has_navigation_intent": true/false,
    "target_path": "/path/to/page" or null,
    "confidence": 0.0~1.0,
    "reason": "판단 근거"
}}

예시:
- "동화 페이지로 이동해줘" → {{"has_navigation_intent": true, "target_path": "/story/list", "confidence": 0.95, "reason": "명확한 동화 페이지 이동 요청"}}
- "대시보드 보여줘" → {{"has_navigation_intent": true, "target_path": "/parent/dashboard", "confidence": 0.9, "reason": "대시보드 표시 요청"}}
- "대시보드로 이동해줘" → {{"has_navigation_intent": true, "target_path": "/parent/dashboard", "confidence": 0.95, "reason": "명확한 대시보드 이동 요청"}}
- "공룡 보고 싶어" → {{"has_navigation_intent": true, "target_path": "/my-dinos", "confidence": 0.8, "reason": "공룡 페이지 조회 의도"}}
- "홈으로 가자" → {{"has_navigation_intent": true, "target_path": "/home", "confidence": 0.9, "reason": "홈 페이지 이동 요청"}}
- "오늘 기분이 어때?" → {{"has_navigation_intent": false, "target_path": null, "confidence": 0.0, "reason": "일반 대화"}}

**주의:** JSON만 반환하세요. 마크다운 코드블록(```json)은 사용하지 마세요."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message}
            ],
            temperature=0.2,  # 더 일관성 있는 응답을 위해 낮춤
            max_tokens=300,
            response_format={"type": "json_object"}  # JSON 형식 강제
        )

        result_text = response.choices[0].message.content.strip()
        print(f"🤖 AI 원본 응답: {result_text}")

        # JSON 파싱
        try:
            # 코드 블록 제거 (혹시 모를 경우 대비)
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result = json.loads(result_text)
            print(f"✅ JSON 파싱 성공: {result}")

            return NavigationIntentResponse(
                has_navigation_intent=result.get("has_navigation_intent", False),
                target_path=result.get("target_path"),
                confidence=result.get("confidence", 0.0),
                reason=result.get("reason", "")
            )

        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 실패: {result_text}")
            print(f"❌ 에러: {e}")
            # 기본값 반환
            return NavigationIntentResponse(
                has_navigation_intent=False,
                target_path=None,
                confidence=0.0,
                reason=f"JSON 파싱 실패: {str(e)}"
            )

    except Exception as e:
        print(f"Error in analyze_navigation_intent: {e}")
        raise HTTPException(status_code=500, detail=str(e))
