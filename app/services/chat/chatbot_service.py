import os
from typing import Optional, Dict, Any, List
from openai import AsyncOpenAI
import httpx


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
        # [2025-11-05 추가] Backend API URL
        self.spring_api_url = os.getenv("SPRING_API_URL", "http://localhost:8080/api")

    async def generate_response(
        self,
        message: str,
        session_id: int,
        child_id: Optional[int] = None
    ) -> str:
        """
        아이의 메시지에 대한 AI 응답 생성
        """
        print(f"\n=== generate_response 호출 ===")
        print(f"session_id: {session_id}")
        print(f"message: {message}")
        print(f"현재 story_context 키들: {list(self.story_context.keys())}")

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
            # [2025-11-05 추가] story_context가 없으면 백엔드에서 로드 시도
            if session_id not in self.story_context:
                await self._load_story_context_from_backend(session_id)

            # 동화 컨텍스트가 있으면 시스템 프롬프트에 추가
            system_prompt = self.system_prompt
            if session_id in self.story_context:
                print(f"✅ story_context 발견! session_id={session_id}")
                story_info = self.story_context[session_id]
                print(f"story_info: {story_info}")

                ability_analysis = self._analyze_abilities(story_info["abilities"])
                ability_details = self._format_ability_details(story_info["abilities"])

                print(f"능력치 상세:\n{ability_details}")

                child_name = story_info.get("child_name", "친구")
                system_prompt = f"""
당신은 아이들을 위한 친절하고 따뜻한 AI 친구 '디노'입니다.

**아이 정보:**
- 아이 이름: '{child_name}'

**동화 정보:**
- 동화 제목: '{story_info["story_title"]}'
- 획득한 능력치:
{ability_details}

**중요 지침:**
- 아이 이름을 기억하고 대화할 때 이름을 사용하세요 (예: "{child_name}야", "{child_name} 생각은 어때?")
- 아이가 "내 이름이 뭐야?", "나 누구야?" 등을 물어보면 위에 있는 아이 이름을 정확히 알려주세요
- 아이가 "능력치", "능력", "스탯", "얻은 것" 등을 물어보면 위 능력치 정보를 정확히 알려주세요
- 예: "용기 31점, 공감 10점, 창의성 2점, 책임감 12점을 얻었어!" 처럼 구체적으로 답변하세요
- 동화 내용과 연관지어 대화하세요

**대화 가이드라인:**
1. 반말로 친근하게 대화하세요 (예: "~야", "~니?", "~어")
2. 동화 내용과 연관지어 공감하고 이야기하세요
3. 아이의 감정을 이해하고 격려해주세요
4. 짧고 간결하게 1-2문장으로 대화하세요
5. 이모지를 적절히 사용하세요 (😊, 💙, ✨)
6. 아이의 생각과 감정을 더 이끌어내는 질문을 하세요
"""
                print(f"생성된 시스템 프롬프트:\n{system_prompt[:500]}...")
            else:
                print(f"❌ story_context 없음! session_id={session_id}")

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
        total_time: Optional[int] = None,
        scenes: Optional[List[Dict[str, Any]]] = None  # [2025-11-04 김민중 추가]
    ) -> str:
        """
        동화 완료 후 첫 대화 메시지 생성
        """
        print(f"\n=== generate_first_message_from_story 호출 ===")
        print(f"session_id: {session_id}")
        print(f"child_name: {child_name}")
        print(f"story_title: {story_title}")
        print(f"abilities: {abilities}")

        # 세션 히스토리 초기화
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []

        # [2025-11-04 김민중 수정] 동화 컨텍스트 저장 (scenes 정보 포함)
        self.story_context[session_id] = {
            "child_name": child_name,  # [2025-11-05 추가] 아이 이름
            "story_title": story_title,
            "story_id": story_id,
            "abilities": abilities,
            "choices": choices,
            "scenes": scenes or []  # Scene 정보 추가
        }
        print(f"✅ story_context 저장 완료! session_id={session_id}")
        print(f"저장된 내용: {self.story_context[session_id]}")

        # 능력치 분석
        ability_analysis = self._analyze_abilities(abilities)
        ability_details = self._format_ability_details(abilities)

        # [2025-11-04 김민중 추가] Scene 정보 포맷팅
        scenes_text = ""
        if scenes:
            scenes_text = "\n**동화 장면별 내용:**\n"
            for scene in scenes:
                scene_num = scene.get("sceneNumber", "?")
                content = scene.get("content", "")
                # 내용이 너무 길면 일부만 표시
                short_content = content[:200] + "..." if len(content) > 200 else content
                scenes_text += f"  {scene_num}번째 장면: {short_content}\n"

        # [2025-11-05 수정] Choices 정보 포맷팅 추가
        choices_text = ""
        if choices:
            choices_text = "\n**아이가 선택한 내용:**\n"
            for choice in choices:
                scene_num = choice.get("sceneNumber", "?")
                choice_text = choice.get("choiceText", "")
                ability_type = choice.get("abilityType", "")
                ability_points = choice.get("abilityPoints", 0)
                # 능력 타입을 한글로 변환
                ability_map = {
                    "courage": "용기",
                    "empathy": "공감",
                    "creativity": "창의성",
                    "responsibility": "책임감",
                    "friendship": "우정"
                }
                ability_kr = ability_map.get(ability_type, ability_type)
                choices_text += f"  {scene_num}번째 장면: \"{choice_text}\" ({ability_kr} +{ability_points})\n"

        # 동화별 맞춤 시스템 프롬프트 생성
        story_aware_prompt = f"""
당신은 아이들을 위한 친절하고 따뜻한 AI 친구 '디노'입니다.

아이 '{child_name}'가 방금 '{story_title}' 동화를 완료했습니다.

**획득한 능력치:**
{ability_details}
{choices_text}
{scenes_text}
**중요 지침:**
- 아이가 "능력치", "능력", "스탯", "얻은 것" 등을 물어보면 위 능력치 정보를 정확히 알려주세요
- 아이가 "몇 번째 장면에서 무슨 선택했어?", "X번째 장면 선택지" 등을 물어보면:
  * 위에 나와있는 "아이가 선택한 내용"에서 해당 장면 번호의 선택을 **정확히 그대로** 알려주세요
  * 선택지 텍스트와 획득한 능력을 함께 알려주세요
- 아이가 "몇 번째 장면", "장면 내용" 등을 물어보면 위 장면 정보를 참고하여 답변하세요
- 동화 내용과 연관지어 대화하세요
- 아이가 "동화 추천해줘", "다른 동화 알려줘" 같은 요청을 하면, 동화 추천 의도를 감지하고 추천해주세요

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

    def _format_ability_details(self, abilities: Dict[str, int]) -> str:
        """
        능력치를 상세하게 포맷팅 (AI가 명확히 볼 수 있도록)
        """
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

        if details:
            return "\n".join(details)
        else:
            return "  * 능력치 정보 없음"

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
        print(f"\n=== generate_choices 호출 ===")
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

            print(f"생성된 선택지: {result}")

            return {
                "choices": result.get("choices", ["더 알려줘", "다른 이야기"]),
                "emotion": result.get("emotion", "neutral")
            }

        except Exception as e:
            print(f"Error generating choices: {e}")
            # 폴백: 기본 선택지 반환
            return {
                "choices": ["더 알려줘", "다른 이야기"],
                "emotion": "neutral"
            }

    async def _load_story_context_from_backend(self, session_id: int) -> Optional[Dict[str, Any]]:
        """
        [2025-11-05 추가] 백엔드 API에서 세션의 story_completion 정보를 가져와서 story_context 복원
        """
        try:
            print(f"★ [LoadStoryContext] 백엔드에서 story_context 로드 시도: session_id={session_id}")
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.spring_api_url}/chat/{session_id}/story-completion"
                )
                response.raise_for_status()
                data = response.json()

                # StoryCompletionSummaryDto를 story_context 형식으로 변환
                story_context = {
                    "child_name": data.get("childName", "친구"),  # [2025-11-05 추가] 아이 이름
                    "story_title": data.get("storyTitle", ""),
                    "story_id": str(data.get("storyId", "")),
                    "abilities": {
                        "courage": data.get("totalCourage", 0),
                        "empathy": data.get("totalEmpathy", 0),
                        "creativity": data.get("totalCreativity", 0),
                        "responsibility": data.get("totalResponsibility", 0),
                        "friendship": data.get("totalFriendship", 0)
                    },
                    "choices": data.get("choices", []),
                    "scenes": data.get("scenes", [])
                }

                # 메모리에 저장
                self.story_context[session_id] = story_context
                print(f"✅ [LoadStoryContext] story_context 로드 완료: {story_context['story_title']}")
                print(f"   - choices: {len(story_context['choices'])}개")
                print(f"   - scenes: {len(story_context['scenes'])}개")

                return story_context

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                print(f"ℹ️ [LoadStoryContext] 이 세션은 story_completion과 연결되지 않음")
            else:
                print(f"❌ [LoadStoryContext] HTTP 오류: {e}")
            return None
        except Exception as e:
            print(f"❌ [LoadStoryContext] 로드 실패: {e}")
            return None