import os
from typing import Optional, Dict, Any, List
from openai import AsyncOpenAI


class ChatbotService:
    def __init__(self):
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
        # 세션별 대화 히스토리 저장
        self.conversation_history = {}
        # 세션별 동화 컨텍스트 저장
        self.story_context = {}

    async def generate_response(
        self,
        message: str,
        session_id: int,
        child_id: Optional[int] = None
    ) -> str:
        """
        아이의 메시지에 대한 AI 응답 생성
        """
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
            # 동화 컨텍스트가 있으면 시스템 프롬프트에 추가
            system_prompt = self.system_prompt
            if session_id in self.story_context:
                story_info = self.story_context[session_id]
                ability_analysis = self._analyze_abilities(story_info["abilities"])

                system_prompt = f"""
당신은 아이들을 위한 친절하고 따뜻한 AI 친구 '디노'입니다.

**동화 컨텍스트:**
아이가 방금 '{story_info["story_title"]}' 동화를 완료했습니다.
- 선택한 능력: {ability_analysis}

**대화 가이드라인:**
1. 반말로 친근하게 대화하세요 (예: "~야", "~니?", "~어")
2. 동화 내용과 연관지어 공감하고 이야기하세요
3. 아이의 감정을 이해하고 격려해주세요
4. 짧고 간결하게 1-2문장으로 대화하세요
5. 이모지를 적절히 사용하세요 (😊, 💙, ✨)
6. 아이의 생각과 감정을 더 이끌어내는 질문을 하세요
"""

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

    def clear_history(self, session_id: int):
        """
        특정 세션의 대화 히스토리 삭제
        """
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]

    def get_history(self, session_id: int):
        """
        특정 세션의 대화 히스토리 조회
        """
        return self.conversation_history.get(session_id, [])

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
        동화 완료 후 첫 대화 메시지 생성
        """
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
        ability_analysis = self._analyze_abilities(abilities)

        # 동화별 맞춤 시스템 프롬프트 생성
        story_aware_prompt = f"""
당신은 아이들을 위한 친절하고 따뜻한 AI 친구 '디노'입니다.

아이 '{child_name}'가 방금 '{story_title}' 동화를 완료했습니다.

**동화에서의 선택 정보:**
- {ability_analysis}

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
            # 폴백 메시지
            return f"{child_name}야, 동화 어땠어? 재미있었니? 지금 기분이 어때? 😊"

    def _analyze_abilities(self, abilities: Dict[str, int]) -> str:
        """
        능력치를 분석하여 텍스트로 변환
        """
        ability_names = {
            "courage": "용기",
            "empathy": "공감",
            "creativity": "창의성",
            "responsibility": "책임감",
            "friendship": "우정"
        }

        analysis_parts = []
        for key, value in abilities.items():
            if value > 0:
                korean_name = ability_names.get(key, key)
                analysis_parts.append(f"{korean_name} +{value}")

        if analysis_parts:
            return ", ".join(analysis_parts)
        else:
            return "특별한 선택을 했어요"