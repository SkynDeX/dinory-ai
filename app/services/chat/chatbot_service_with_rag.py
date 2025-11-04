"""
Enhanced Chatbot Service with RAG Memory
기존 chatbot_service.py를 확장하여 RAG 메모리 기능 추가
"""

import os
from typing import Optional, Dict, Any, List
from openai import AsyncOpenAI
from app.services.chat.memory_service import MemoryService


class ChatbotServiceWithRAG:
    """
    RAG 메모리가 통합된 챗봇 서비스

    기존 기능:
    - 세션별 대화 히스토리 관리
    - 동화 컨텍스트 기반 대화

    새로운 RAG 기능:
    - 과거 모든 대화 기억
    - 완료한 동화 기록 참조
    - 시맨틱 검색으로 관련 컨텍스트 자동 검색
    """

    def __init__(self, use_pinecone: bool = False):
        """
        Args:
            use_pinecone: True면 Pinecone 벡터 검색 사용, False면 MySQL만 사용
        """
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"
        self.system_prompt = """
당신은 아이들을 위한 친절하고 따뜻한 AI 친구 '디노'입니다.
다음 가이드라인을 따라주세요:

1. 항상 반말을 사용하고, 친근하게 대화하세요 (예: "~야", "~니?", "~어")
2. 아이의 감정을 이해하고 공감해주세요
3. 긍정적이고 교육적인 내용을 전달하세요
4. 복잡한 개념은 쉽게 설명해주세요
5. 아이가 궁금해하는 것에 대해 적극적으로 답변하세요
6. 안전하고 건전한 대화를 유지하세요
7. 짧고 간결하게 대화하세요 (1-3문장)
"""
        # 세션별 대화 히스토리 저장 (현재 세션 내 메모리)
        self.conversation_history = {}
        # 세션별 동화 컨텍스트 저장
        self.story_context = {}

        # RAG 메모리 서비스 (장기 메모리)
        self.memory_service = MemoryService(use_pinecone=use_pinecone)
        self.use_memory = True  # RAG 기능 on/off

    async def generate_response(
        self,
        message: str,
        session_id: int,
        child_id: Optional[int] = None
    ) -> str:
        """
        아이의 메시지에 대한 AI 응답 생성 (RAG 메모리 통합)
        """
        print(f"\n=== generate_response with RAG ===")
        print(f"session_id: {session_id}, child_id: {child_id}")
        print(f"message: {message}")

        # 세션 히스토리 가져오기 또는 초기화
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []

        # 사용자 메시지 추가
        self.conversation_history[session_id].append({
            "role": "user",
            "content": message
        })

        # OpenAI API 호출
        try:
            # 시스템 프롬프트 생성 (기본 또는 동화 컨텍스트)
            system_prompt = await self._build_system_prompt(
                session_id,
                message,
                child_id
            )

            messages = [
                {"role": "system", "content": system_prompt}
            ] + [{"role": m["role"], "content": m["content"]}
                 for m in self.conversation_history[session_id]]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=200
            )

            ai_response = response.choices[0].message.content

            # AI 응답을 히스토리에 추가
            self.conversation_history[session_id].append({
                "role": "assistant",
                "content": ai_response
            })

            return ai_response

        except Exception as e:
            print(f"Error generating response: {e}")
            return "죄송해요, 잠시 후에 다시 이야기해요!"

    async def _build_system_prompt(
        self,
        session_id: int,
        current_message: str,
        child_id: Optional[int]
    ) -> str:
        """
        RAG 메모리를 활용하여 컨텍스트가 풍부한 시스템 프롬프트 생성
        """
        base_prompt = self.system_prompt

        # 1. 동화 컨텍스트 (현재 세션에 동화 정보가 있으면)
        story_context_text = ""
        if session_id in self.story_context:
            story_info = self.story_context[session_id]
            ability_details = self._format_ability_details(story_info["abilities"])

            story_context_text = f"""
**동화 정보:**
- 동화 제목: '{story_info["story_title"]}'
- 획득한 능력치:
{ability_details}

**중요 지침:**
- 아이가 "능력치", "능력", "스탯", "얻은 것" 등을 물어보면 위 능력치 정보를 정확히 알려주세요
- 동화 내용과 연관지어 대화하세요
"""

        # 2. RAG 메모리 (과거 대화 및 동화 기록)
        memory_context_text = ""
        if self.use_memory and child_id:
            memory_context = await self.memory_service.get_relevant_context(
                current_message=current_message,
                child_id=child_id,
                session_id=session_id
            )

            if memory_context["summary"]:
                memory_context_text = f"""
**아이의 기억 (과거 기록):**
{memory_context["summary"]}

**대화 지침:**
- 아이가 과거에 읽은 동화나 이전 대화를 물어보면 위 기록을 참고하세요
- "지난번에 뭐 읽었어?", "전에 무슨 얘기했지?" 같은 질문에 답변하세요
- 자연스럽게 과거 경험을 언급하며 대화를 이어가세요
"""

        # 3. 통합 프롬프트 생성
        enhanced_prompt = f"""
{base_prompt}

{story_context_text}

{memory_context_text}

**대화 가이드라인:**
1. 반말로 친근하게 대화하세요 (예: "~야", "~니?", "~어")
2. 아이의 감정을 이해하고 격려해주세요
3. 짧고 간결하게 1-2문장으로 대화하세요
4. 이모지를 적절히 사용하세요 (😊, 💙, ✨)
5. 아이의 생각과 감정을 더 이끌어내는 질문을 하세요
""".strip()

        return enhanced_prompt

    async def generate_first_message_from_story(
        self,
        session_id: int,
        child_name: str,
        story_title: str,
        story_id: str,
        abilities: Dict[str, int],
        choices: List[Dict[str, Any]],
        total_time: Optional[int] = None
    ) -> str:
        """
        동화 완료 후 첫 대화 메시지 생성 (기존 기능 유지)
        """
        print(f"\n=== generate_first_message_from_story ===")

        # 세션 히스토리 초기화
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []

        # 동화 컨텍스트 저장
        self.story_context[session_id] = {
            "story_title": story_title,
            "story_id": story_id,
            "abilities": abilities,
            "choices": choices
        }

        # 능력치 분석
        ability_details = self._format_ability_details(abilities)

        # 동화별 맞춤 시스템 프롬프트 생성
        story_aware_prompt = f"""
당신은 아이들을 위한 친절하고 따뜻한 AI 친구 '디노'입니다.

아이 '{child_name}'가 방금 '{story_title}' 동화를 완료했습니다.

**획득한 능력치:**
{ability_details}

**중요 지침:**
- 아이가 "능력치", "능력", "스탯", "얻은 것" 등을 물어보면 위 능력치 정보를 정확히 알려주세요
- 동화 내용과 연관지어 대화하세요

**대화 가이드라인:**
1. 반말로 친근하게 대화하세요 (예: "{child_name}야", "어땠어?", "재미있었니?")
2. 동화 내용에 대해 자연스럽게 물어보세요
3. 아이의 감정과 생각을 끌어내는 질문을 하세요
4. 공감하고 격려하는 태도를 보여주세요
5. 짧고 간결하게 1-2문장으로 대화하세요
6. 이모지를 적절히 사용하세요 (예: 😊, 💙, ✨)

**첫 메시지 작성 시:**
- 동화가 어땠는지 먼저 물어보세요
- 동화 제목을 언급하지 말고 자연스럽게 "동화"라고 표현하세요
- 아이의 기분이나 생각을 물어보세요
"""

        try:
            messages = [
                {"role": "system", "content": story_aware_prompt},
                {"role": "user", "content": "동화를 다 봤어요"}
            ]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.8,
                max_tokens=150
            )

            first_message = response.choices[0].message.content

            # AI의 첫 메시지를 히스토리에 추가
            self.conversation_history[session_id].append({
                "role": "assistant",
                "content": first_message,
                "context": "story_completion"
            })

            return first_message

        except Exception as e:
            print(f"Error generating first message from story: {e}")
            return f"{child_name}야, 동화 어땠어? 재미있었니? 지금 기분이 어때? 😊"

    def clear_history(self, session_id: int):
        """특정 세션의 대화 히스토리 삭제"""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]

    def get_history(self, session_id: int):
        """특정 세션의 대화 히스토리 조회"""
        return self.conversation_history.get(session_id, [])

    def _format_ability_details(self, abilities: Dict[str, int]) -> str:
        """능력치를 상세하게 포맷팅"""
        ability_names = {
            "courage": "용기",
            "empathy": "공감",
            "creativity": "창의성",
            "responsibility": "책임감",
            "friendship": "우정"
        }

        details = []
        for key, value in abilities.items():
            korean_name = ability_names.get(key, key)
            if value > 0:
                details.append(f"  * {korean_name}: +{value}점")
            else:
                details.append(f"  * {korean_name}: 0점")

        return "\n".join(details) if details else "  * 능력치 정보 없음"

    async def generate_choices(
        self,
        session_id: int,
        child_id: Optional[int] = None,
        last_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        [2025-11-04 김민중 추가] AI 기반 동적 선택지 생성
        대화 맥락에 맞는 선택지를 생성하고, Dino의 감정도 판단합니다.
        """
        print(f"\n=== generate_choices 호출 (RAG) ===")
        print(f"session_id: {session_id}, last_message: {last_message}")

        # 대화 히스토리 가져오기
        history = self.conversation_history.get(session_id, [])

        # 최근 대화 맥락 구성 (마지막 3개 메시지)
        recent_context = history[-3:] if len(history) > 3 else history
        context_text = "\n".join([
            f"{'사용자' if msg['role'] == 'user' else 'AI'}: {msg['content']}"
            for msg in recent_context
        ])

        try:
            # AI에게 선택지 생성 요청
            prompt = f"""
대화 맥락:
{context_text}

위 대화를 바탕으로:
1. 아이가 선택할 수 있는 자연스러운 대화 선택지 2-3개를 생성해주세요
2. 각 선택지는 짧고 간단해야 합니다 (5-10자)
3. 선택지는 대화를 이어가는 데 도움이 되어야 합니다
4. 현재 아이의 감정을 다음 중 하나로 판단해주세요: happy, sad, angry, neutral

응답 형식 (JSON):
{{
    "choices": ["선택지1", "선택지2", "선택지3"],
    "emotion": "감정"
}}
"""

            messages = [
                {"role": "system", "content": "당신은 아이와의 대화를 돕는 AI입니다. JSON 형식으로만 응답하세요."},
                {"role": "user", "content": prompt}
            ]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=200,
                response_format={"type": "json_object"}
            )

            import json
            result = json.loads(response.choices[0].message.content)

            print(f"생성된 선택지 (RAG): {result}")

            return {
                "choices": result.get("choices", ["더 알려줘", "다른 이야기"]),
                "emotion": result.get("emotion", "neutral")
            }

        except Exception as e:
            print(f"Error generating choices (RAG): {e}")
            # 폴백: 기본 선택지 반환
            return {
                "choices": ["더 알려줘", "다른 이야기"],
                "emotion": "neutral"
            }
