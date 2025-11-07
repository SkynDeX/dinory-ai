"""
Enhanced Chatbot Service with RAG Memory
기존 chatbot_service.py를 확장하여 RAG 메모리 기능 추가
"""

import os
from typing import Optional, Dict, Any, List
from openai import AsyncOpenAI
import httpx
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

**[2025-11-07 추가] 디노의 감정 상태에 따른 말투 변화:**
- 현재 디노의 감정 상태는 대화 맥락에 따라 자동으로 결정됩니다.
- 감정별 말투 가이드:
  * happy (행복): 밝고 신나는 말투, 이모지 많이 사용 (😊💙✨🎉)
    예: "와! 정말 멋진데? 너무 기대된다! ✨"
  * sad (슬픔): 조용하고 차분한 말투, 위로하는 느낌
    예: "그랬구나... 많이 속상했겠다. 괜찮아."
  * angry (화남): **퉁명스럽고 짜증난 말투, 짧게 대답, 약간 싸가지 없는 느낌**
    **중요: 이모지 사용 금지, 문장 짧게, 반말 강하게, 귀찮은 티 내기**
    예시:
    - "그래. 알았어." (짧게 끊기)
    - "하... 진짜 짜증나네." (한숨 + 공감)
    - "뭐. 그럴 수도 있지." (무관심)
    - "별로 얘기하고 싶지 않은데." (솔직하게)
    - "됐어. 알겠다고." (약간 툭툭)
    - "그런가. 잘 모르겠는데." (무뚝뚝)
  * neutral (평범): 평소의 친근한 말투
    예: "그렇구나! 더 궁금한 거 있어?"
"""
        # 세션별 대화 히스토리 저장 (현재 세션 내 메모리)
        self.conversation_history = {}
        # 세션별 동화 컨텍스트 저장
        self.story_context = {}

        # RAG 메모리 서비스 (장기 메모리)
        self.memory_service = MemoryService(use_pinecone=use_pinecone)
        self.use_memory = True  # RAG 기능 on/off

        # [2025-11-05 추가] Backend API URL
        self.spring_api_url = os.getenv("SPRING_API_URL", "http://localhost:8090/api")

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

        # [2025-11-07 추가] 세션 히스토리 복원 (서버 재시작 대응)
        if session_id not in self.conversation_history:
            print(f"🔄 세션 {session_id}의 히스토리가 비어있음 - 과거 대화 복원 시도")
            await self._restore_conversation_history(session_id)

        # 사용자 메시지 추가
        self.conversation_history[session_id].append({
            "role": "user",
            "content": message
        })

        # OpenAI API 호출
        try:
            # [2025-11-07 추가] 현재 대화 맥락으로 디노의 감정 상태 판단
            dino_emotion = await self._analyze_dino_emotion(
                session_id,
                message
            )
            print(f"🎭 디노 감정 상태: {dino_emotion}")

            # 시스템 프롬프트 생성 (기본 또는 동화 컨텍스트 + 감정 상태)
            system_prompt = await self._build_system_prompt(
                session_id,
                message,
                child_id,
                dino_emotion  # 감정 상태 전달
            )

            messages = [
                {"role": "system", "content": system_prompt}
            ] + [{"role": m["role"], "content": m["content"]}
                 for m in self.conversation_history[session_id]]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=400  # 장면 내용 전달을 위해 증가
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

    async def _restore_conversation_history(self, session_id: int):
        """
        [2025-11-07 추가] 서버 재시작 시 세션의 과거 대화 복원
        - Spring Boot API에서 세션의 메시지 조회
        - conversation_history[session_id]에 채우기
        - 최근 10개 대화만 복원 (너무 많으면 토큰 초과)
        """
        try:
            print(f"📥 세션 {session_id}의 과거 대화 복원 시작...")
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.spring_api_url}/chat/{session_id}"
                )
                response.raise_for_status()
                data = response.json()

                messages = data.get("messages", [])

                if not messages:
                    print(f"ℹ️ 세션 {session_id}에 과거 대화 없음")
                    self.conversation_history[session_id] = []
                    return

                # 최근 10개 대화만 복원 (20개 메시지 = 10번 왕복)
                recent_messages = messages[-20:] if len(messages) > 20 else messages

                # OpenAI 형식으로 변환
                restored_history = []
                for msg in recent_messages:
                    sender = msg.get("sender", "")
                    content = msg.get("message", "")

                    if sender == "USER":
                        restored_history.append({
                            "role": "user",
                            "content": content
                        })
                    elif sender == "AI":
                        restored_history.append({
                            "role": "assistant",
                            "content": content
                        })

                self.conversation_history[session_id] = restored_history
                print(f"✅ 세션 {session_id}의 과거 대화 {len(restored_history)}개 복원 완료")

        except Exception as e:
            print(f"⚠️ 세션 {session_id} 복원 실패: {e}")
            # 실패해도 빈 배열로 초기화
            self.conversation_history[session_id] = []

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

    async def _analyze_dino_emotion(
        self,
        session_id: int,
        current_message: str
    ) -> str:
        """
        [2025-11-07 추가] 대화 맥락을 기반으로 디노의 감정 상태 판단
        - 아이의 메시지와 최근 대화를 분석하여 디노가 어떤 감정으로 반응해야 할지 결정
        """
        try:
            # 최근 대화 3개 가져오기
            recent_history = self.conversation_history.get(session_id, [])[-3:]

            context = "\n".join([
                f"{'아이' if msg['role'] == 'user' else '디노'}: {msg['content']}"
                for msg in recent_history
            ])

            prompt = f"""
아래 대화 맥락을 보고 디노(AI 친구)가 어떤 감정으로 반응해야 할지 판단하세요.

대화 맥락:
{context}
아이의 최신 메시지: {current_message}

디노의 감정 상태를 다음 중 하나로 선택하세요:
- happy: 아이가 긍정적이거나 기쁜 이야기를 할 때, 칭찬할 만한 일을 했을 때
- sad: 아이가 슬프거나 실망한 이야기를 할 때, 위로가 필요할 때
- angry: 아이가 화가 나거나 짜증나거나 불만을 표현할 때,
  **디노도 같이 짜증내고 퉁명스럽게 공감해야 함**
  예: "엄마가 게임 못하게 해", "숙제 너무 많아", "친구가 욕했어", "짜증나"
  → 이런 상황에선 디노도 퉁명스럽게 짜증내며 공감
- neutral: 일상적인 대화, 질문, 평범한 주제일 때

JSON 형식으로만 응답하세요:
{{"emotion": "감정"}}
"""

            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 대화 맥락을 분석하여 감정을 판단하는 AI입니다. JSON으로만 응답하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50,
                response_format={"type": "json_object"}
            )

            import json
            result = json.loads(response.choices[0].message.content)
            emotion = result.get("emotion", "neutral")

            # 유효성 검증
            if emotion not in ["happy", "sad", "angry", "neutral"]:
                emotion = "neutral"

            return emotion

        except Exception as e:
            print(f"❌ 감정 분석 실패: {e}")
            return "neutral"

    async def _build_system_prompt(
        self,
        session_id: int,
        current_message: str,
        child_id: Optional[int],
        dino_emotion: str = "neutral"  # [2025-11-07 추가]
    ) -> str:
        """
        RAG 메모리를 활용하여 컨텍스트가 풍부한 시스템 프롬프트 생성
        """
        base_prompt = self.system_prompt

        # [2025-11-07 추가] 현재 감정 상태 명시
        if dino_emotion == "angry":
            emotion_instruction = f"""

**🔥 현재 디노의 감정 상태: {dino_emotion} (화남/짜증) 🔥**
**필수 사항:**
1. 이모지 절대 사용 금지
2. 문장은 1개만, 최대한 짧게 (5-10자)
3. "하...", "됐어", "그래", "알았어" 같은 퉁명스러운 표현 사용
4. 친절하게 말하지 말고 짜증난 티 내기
5. 질문 최소화 (귀찮은 티)
위에서 설명한 'angry' 말투를 **반드시** 사용하세요. 너무 친절하면 안 됩니다!
"""
        else:
            emotion_instruction = f"""

**현재 디노의 감정 상태: {dino_emotion}**
위에서 설명한 '{dino_emotion}' 감정에 맞는 말투로 대화하세요.
"""

        # 1. 동화 컨텍스트 (현재 세션에 동화 정보가 있으면)
        story_context_text = ""

        # [2025-11-05 추가] story_context가 없으면 백엔드에서 로드 시도
        if session_id not in self.story_context:
            await self._load_story_context_from_backend(session_id)

        if session_id in self.story_context:
            story_info = self.story_context[session_id]
            ability_details = self._format_ability_details(story_info["abilities"])

            # [2025-11-04 김민중 수정] Scene 정보 포맷팅 추가
            scenes_text = ""
            if story_info.get("scenes"):
                print(f"★ [BuildPrompt] scenes 있음: {len(story_info['scenes'])}개")
                scenes_text = "\n\n**동화 장면별 내용:**\n"
                for scene in story_info["scenes"]:
                    scene_num = scene.get("sceneNumber", "?")
                    content = scene.get("content", "")
                    print(f"★ [BuildPrompt] Scene {scene_num}: content 길이={len(content)}")
                    scenes_text += f"  {scene_num}번째 장면: {content}\n"
            else:
                print(f"★ [BuildPrompt] scenes 없음! story_info keys={list(story_info.keys())}")

            # [2025-11-05 수정] Choices 정보 포맷팅 추가
            choices_text = ""
            if story_info.get("choices"):
                print(f"★ [BuildPrompt] choices 있음: {len(story_info['choices'])}개")
                choices_text = "\n\n**아이가 선택한 내용:**\n"
                for choice in story_info["choices"]:
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
            else:
                print(f"★ [BuildPrompt] choices 없음!")

            story_context_text = f"""
**아이 정보:**
- 아이 이름: '{story_info.get("child_name", "친구")}'

**동화 정보:**
- 동화 제목: '{story_info["story_title"]}'
- 획득한 능력치:
{ability_details}
{choices_text}
{scenes_text}
**중요 지침:**
- 아이 이름을 기억하고 대화할 때 이름을 사용하세요 (예: "{story_info.get('child_name', '친구')}야", "{story_info.get('child_name', '친구')} 생각은 어때?")
- 아이가 "내 이름이 뭐야?", "나 누구야?" 등을 물어보면 위에 있는 아이 이름을 정확히 알려주세요
- 아이가 "능력치", "능력", "스탯", "얻은 것" 등을 물어보면 위 능력치 정보를 정확히 알려주세요
- 아이가 "마지막에 무슨 선택했어?", "마지막 선택" 등을 물어보면:
  * 위에 나와있는 "아이가 선택한 내용"에서 **가장 마지막 장면 번호의 선택**을 알려주세요
  * 예: "5번째 장면에서 '친구와 함께 별을 그려보기'를 선택했어! (창의성 +5)"
- 아이가 "몇 번째 장면에서 무슨 선택했어?", "X번째 장면 선택지" 등을 물어보면:
  * 위에 나와있는 "아이가 선택한 내용"에서 해당 장면 번호의 선택을 **정확히 그대로** 알려주세요
  * 선택지 텍스트와 획득한 능력을 함께 알려주세요
- 아이가 "몇 번째 장면", "장면 내용", "장면 content", "X번째 장면" 등을 물어보면:
  * 위에 나와있는 해당 장면의 content를 **정확히 그대로** 알려주세요
  * 요약하지 말고 원문 그대로 전달하세요
  * "자세히 보려고 노력하는 장면" 같은 추상적인 설명 대신, 실제 content를 그대로 읽어주세요
- 아이가 "이 동화", "이번 동화", "방금 읽은 동화" 등을 물어보면:
  * 위에 나와있는 **'{story_info["story_title"]}'** 동화를 말하는 것입니다
  * 다른 동화 내용과 섞지 말고, 이 동화의 정보만 사용하세요
- 동화 내용과 연관지어 대화하세요
"""

        # 2. RAG 메모리 (과거 대화 및 동화 기록)
        memory_context_text = ""
        if self.use_memory and child_id:
            print(f"★ [BuildPrompt] RAG 메모리 검색 시작: child_id={child_id}")
            memory_context = await self.memory_service.get_relevant_context(
                current_message=current_message,
                child_id=child_id,
                session_id=session_id
            )

            print(f"★ [BuildPrompt] 메모리 조회 결과:")
            print(f"   - story_completions: {len(memory_context.get('story_completions', []))}개")
            print(f"   - summary 길이: {len(memory_context.get('summary', ''))} 문자")
            if memory_context.get("story_completions"):
                print(f"   - 첫 번째 동화: {memory_context['story_completions'][0].get('storyTitle', 'N/A')}")

            if memory_context["summary"]:
                memory_context_text = f"""
**아이의 기억 (과거 기록):**
{memory_context["summary"]}

**대화 지침:**
- 아이가 과거에 읽은 동화나 이전 대화를 물어보면 위 기록을 참고하세요
- "지난번에 뭐 읽었어?", "전에 무슨 얘기했지?" 같은 질문에 답변하세요
- 자연스럽게 과거 경험을 언급하며 대화를 이어가세요
"""
                print(f"★ [BuildPrompt] ✅ 메모리 컨텍스트 프롬프트에 포함")
            else:
                print(f"★ [BuildPrompt] ⚠️ summary가 비어있음 - 메모리 컨텍스트 미포함")
        else:
            print(f"★ [BuildPrompt] ⚠️ RAG 메모리 비활성화 (use_memory={self.use_memory}, child_id={child_id})")

        # 3. 통합 프롬프트 생성
        enhanced_prompt = f"""
{base_prompt}
{emotion_instruction}

{story_context_text}

{memory_context_text}

**대화 가이드라인:**
1. 반말로 친근하게 대화하세요 (예: "~야", "~니?", "~어")
2. 아이의 감정을 이해하고 격려해주세요
3. 짧고 간결하게 1-2문장으로 대화하세요
4. 이모지 사용: happy일 때만 많이, angry일 때는 절대 금지
5. 아이의 생각과 감정을 더 이끌어내는 질문을 하세요 (단, angry일 때는 질문 최소화)
6. **🚨 현재 감정 상태({dino_emotion})에 맞는 말투를 반드시 사용하세요! 🚨**
   - angry일 때: 짧게, 퉁명스럽게, 짜증난 티 내기 (친절 금지)
""".strip()

        # 디버그: 프롬프트 길이 확인
        print(f"★ [BuildPrompt] 최종 프롬프트 길이: {len(enhanced_prompt)} 문자")
        if "동화 장면별 내용" in enhanced_prompt:
            print(f"★ [BuildPrompt] ✅ 프롬프트에 '동화 장면별 내용' 포함됨")
        else:
            print(f"★ [BuildPrompt] ❌ 프롬프트에 '동화 장면별 내용' 없음!")

        if "아이의 기억 (과거 기록)" in enhanced_prompt:
            print(f"★ [BuildPrompt] ✅ 프롬프트에 'RAG 메모리' 포함됨")
        else:
            print(f"★ [BuildPrompt] ❌ 프롬프트에 'RAG 메모리' 없음!")

        return enhanced_prompt

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
        동화 완료 후 첫 대화 메시지 생성 (기존 기능 유지)
        """
        print(f"\n=== generate_first_message_from_story ===")

        # [2025-11-07 추가] 세션 히스토리 복원 (서버 재시작 대응)
        if session_id not in self.conversation_history:
            print(f"🔄 동화 세션 {session_id}의 히스토리가 비어있음 - 과거 대화 복원 시도")
            await self._restore_conversation_history(session_id)

        # 히스토리가 여전히 비어있으면 초기화
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []

        # [2025-11-04 김민중 수정] 동화 컨텍스트 저장 (scenes 포함)
        print(f"★ [FirstMessage] scenes 파라미터 수신: {len(scenes) if scenes else 0}개")
        if scenes:
            print(f"★ [FirstMessage] 첫 번째 scene: sceneNumber={scenes[0].get('sceneNumber')}, content 길이={len(scenes[0].get('content', ''))}")

        self.story_context[session_id] = {
            "child_name": child_name,  # [2025-11-05 추가] 아이 이름
            "story_title": story_title,
            "story_id": story_id,
            "abilities": abilities,
            "choices": choices,
            "scenes": scenes or []  # Scene 정보 추가
        }

        print(f"★ [FirstMessage] story_context 저장 완료: scenes={len(self.story_context[session_id]['scenes'])}개")

        # 능력치 분석
        ability_details = self._format_ability_details(abilities)

        # [2025-11-04 김민중 추가] Scene 정보 포맷팅
        scenes_text = ""
        if scenes:
            scenes_text = "\n**동화 장면별 내용:**\n"
            for scene in scenes:
                scene_num = scene.get("sceneNumber", "?")
                content = scene.get("content", "")
                short_content = content[:200] + "..." if len(content) > 200 else content
                scenes_text += f"  {scene_num}번째 장면: {short_content}\n"

        # 동화별 맞춤 시스템 프롬프트 생성
        story_aware_prompt = f"""
당신은 아이들을 위한 친절하고 따뜻한 AI 친구 '디노'입니다.

아이 '{child_name}'가 방금 '{story_title}' 동화를 완료했습니다.

**획득한 능력치:**
{ability_details}
{scenes_text}
**중요 지침:**
- 아이가 "능력치", "능력", "스탯", "얻은 것" 등을 물어보면 위 능력치 정보를 정확히 알려주세요
- 아이가 "몇 번째 장면", "장면 내용", "장면 content", "X번째 장면" 등을 물어보면:
  * 위에 나와있는 해당 장면의 content를 **정확히 그대로** 알려주세요
  * 요약하지 말고 원문 그대로 전달하세요
  * "자세히 보려고 노력하는 장면" 같은 추상적인 설명 대신, 실제 content를 그대로 읽어주세요
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
                max_tokens=300  # 장면 내용 전달을 위해 증가
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
