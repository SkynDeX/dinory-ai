from openai import OpenAI
import os
import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger("dinory.openai")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[OPENAI] %(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

class OpenAIService:
    """OpenAI GPT를 사용한 동화 생성 서비스"""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY를 찾을 수 없습니다.")
            self.client = None
            return
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        logger.info("OpenAI 서비스가 초기화되었습니다.")

    def generate_personalized_stroy(
            self,
            story_id: str,
            child_name: str,
            emotion: str,
            interests: List[str],
            original_story_data: Optional[Dict] = None       
    ) -> List[Dict]:
        """
        아이 맞춤형 동화 생성(8씬)

        Args:
            story_id: 동화 ID
            child_name: 아이 이름
            emotion: 현재 감정
            interests: 관심사 리스트
            original_story_data: 원본 동화 데이터(Pinecone에서 가져온 것)

        Retruns:
            8개 씬 리스트
        """ 
        if not self.client:
            logger.error("OpenAI 클라이언트가 초기화되지 않았습니다.")
            return self._get_dummy_scenes(child_name)
        
        try:
            # 프롬포트 생성
            prompt = self._create_story_prompt(
                child_name, emotion, interests, original_story_data
            )

            logger.info(f'{child_name}에 대한 스토리를 감정 {emotion}으로 생성합니다.')

            # OpenAI 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 어린이를 위한 창의적이고 따뜻한 동화 작가입니다. 아이의 감정을 이해하고 긍정적인 가치를 전달하는 이야기를 만듭니다."
                    },
                    {
                        "role":  "user",
                        "content": prompt
                    }
                ],
                temperature=0.8,
                max_tokens=3000,
                response_format={"type": "json_object"}
            )

            # 응답 파싱
            content = response.choices[0].message.content
            result = json.loads(content)
            scenes = result.get('scenes', [])

            logger.info(f'{len(scenes)}개의 장면이 성공적으로 생성되었습니다.')
            return scenes

        except Exception as e:
            logger.error(f'스토리 생성 중 오류 발생: {e}')
            return self._get_dummy_scenes(child_name)
        
    def _create_story_prompt(
            self,
            child_name: str,
            emotion: str,
            interests: List[str],
            original_story_data: Optional[Dict]
    ) -> str:
        """동화 생성 프롬포트 작성"""

        interests_text = ", ".join(interests) if interests else "친구와 우정"

        prompt = f"""
            {child_name}라는 아이를 위한 인터랙티브 동화를 만들어주세요.
            
            **아이 정보:**
            - 이름: {child_name}
            - 현재 감정: {emotion}
            - 관심사 : {interests_text}

            **요구사항:**
            1. 총 8개의 씬(scene)으로 구성
            2. 주인공 이름은 {child_name}로 설정
            3. {emotion} 감정을 다루는 내용 포함 (감정 인정 → 긍정적 변화)
            4. {interests_text} 관련 요소 포함
            5. 각 씬마다 3개의 선택지 제공
            6. 선택지는 다양한 능력치(용기, 공감, 창의성, 책임감, 우정) 향상

            **능력치 유형:**
            - 용기: 두려움을 극복하고 도전하는 행동
            - 공감: 다른 사람의 감정을 이해하는 행동
            - 창의성: 새로운 아이디어를 내고 문제를 해결하는 행동
            - 책임감: 자신의 행동에 책임을 지고 약속을 지키는 행동
            - 우정: 친구와 좋은 관계를 만드는 행동

            **출력 형식 (JSON):**
            {{
            "scenes": [
                {{
                "sceneNumber": 1,
                "content": "씬 내용 (3-5문장, 유아용 쉬운 문장)",
                "imagePrompt": "이미지 생성용 영어 프롬프트",
                "choices": [
                    {{
                    "choiceId": 1,
                    "choiceText": "선택지 1 (아이가 이해하기 쉬운 문장)",
                    "abilityType": "용기",
                    "abilityScore": 10
                    }},
                    {{
                    "choiceId": 2,
                    "choiceText": "선택지 2",
                    "abilityType": "공감",
                    "abilityScore": 15
                    }},
                    {{
                    "choiceId": 3,
                    "choiceText": "선택지 3",
                    "abilityType": "창의성",
                    "abilityScore": 10
                    }}
                ]
                }},
                ... (총 8개 씬)
            ]
            }}

            **스토리 구조:**
            - 씬 1-2: 도입 (주인공 소개, 감정 상황 제시)
            - 씬 3-5: 전개 (문제 해결 과정, 다양한 시도)
            - 씬 6-7: 절정 (중요한 선택, 감정 변화)
            - 씬 8: 결말 (긍정적 해결, 교훈)

            **이미지 프롬프트 예시:**
            "A cute [동물/캐릭터] named {child_name} in [배경], children's book illustration style, warm colors, friendly atmosphere"

            동화를 만들어주세요!
        """

        return prompt
    
    def anlyze_custom_choice(
            self,
            custom_text: str,
            scene_context: Optional[str] = None
    ) -> Dict:
        """
        아이가 직접 입력한 선택지 분석

        Args:
            custom_text: 아이가 입력한 텍스트
            scene_context: 현재 씬 내용
        
        Returns:
            분석 결과 (능력치, 점수, 피드백)
        """

        if not self.client:
            logger.error("OpenAI 클라이언트가 초기화되지 않았습니다.")
            return {
                "abilityType": "용기",
                "abilityScore": 10,
                "feedback": "좋은 선택이에요!",
                "nextSceneBranch": None
            }
        try:
            prompt = f"""
            다음은 동화를 읽던 아이가 직접 입력한 선택입니다.
            이 선택을 분석하고 적절한 능력치와 피드백을 제공해주세요.

            **아이의 선택:**
            "{custom_text}"

            **현재 씬:**
            {scene_context or "정보 없음"}

            **분석 기준:**
            - 용기: 두려움을 극복하거나 도전하는 내용
            - 공감: 다른 사람의 감정을 이해하는 내용
            - 창의성: 새로운 아이디어를 내거나 문제를 해결하는 내용
            - 책임감: 자신의 행동에 책임을 지거나 약속을 지키는 내용
            - 우정: 친구와의 관계를 중요하게 생각하는 내용

            **출력 형식 (JSON):**
            {{
            "abilityType": "용기/공감/창의성/책임감/우정 중 하나",
            "abilityScore": 10-15,
            "feedback": "아이에게 전할 긍정적인 피드백 (1-2문장)",
            "nextSceneBranch": null
            }}

            분석 결과를 JSON으로 출력해주세요.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 아이의 선택을 분석하는 교육 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"Analyzed custom choice: {result['abilityType']} +{result['abilityScore']}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing custom choice: {e}")
            return {
                "abilityType": "용기",
                "abilityScore": 10,
                "feedback": "멋진 선택이에요!",
                "nextSceneBranch": None
            }
    
    def generate_next_scene(
            self,
            story_id: str,
            story_title: str,
            story_description: str,
            emotion: str,
            interests: List[str],
            concerns: List[str],  # [2025-11-11 김광현 추가] 자녀 우려사항
            scene_number: int,
            previous_choices: List[Dict],
            story_context: Optional[str] = None,
            character_description: Optional[str] = None  # [2025-11-05 추가] 캐릭터 일관성
    ) -> Dict:
        """
        이전 선택을 기반으로 다음 씬 생성 (분기형 스토리)

        Args:
            story_id: 동화 ID
            story_title: 동화 제목 (예: "새로운 동생을 맞이하는 아이의 이야기")
            story_description: 동화 설명
            emotion: 현재 감정
            interests: 관심사 리스트
            concerns: 자녀 우려사항 리스트
            scene_number: 생성할 씬 번호 (1~8)
            previous_choices: 이전 선택들 [{"sceneNumber": 1, "choiceText": "...", "abilityType": "용기"}]
            story_context: 이전까지의 스토리 흐름 (optional)

        Returns:
            단일 씬 Dict
        """
        if not self.client:
            logger.error("OpenAI 클라이언트가 초기화되지 않았습니다.")
            return self._get_dummy_single_scene(story_title, scene_number)

        try:
            # 프롬프트 생성
            prompt = self._create_next_scene_prompt(
                story_title, story_description, emotion, interests, concerns, scene_number, previous_choices, story_context, character_description
            )

            logger.info(f'씬 {scene_number} 생성 중... (스토리: {story_title}, 이전 선택: {len(previous_choices)}개)')

            # OpenAI 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 어린이를 위한 창의적이고 따뜻한 인터랙티브 동화 작가입니다. 아이의 이전 선택을 반영하여 스토리가 자연스럽게 분기되도록 만듭니다. 반드시 순수 한글로만 작성하고, 주인공을 '네가', '너는' 같은 2인칭이 아닌 '작은 토끼가', '꼬마 로봇은' 같은 3인칭 캐릭터 호칭으로 지칭하세요. 각 문장은 줄바꿈으로 구분하여 읽기 쉽게 작성하세요. **중요: 동화 제목 생성 시 절대 원본 제목과 비슷하게 만들지 말고, 완전히 새로운 모험적인 제목을 창작하세요.**"
                    },
                    {
                        "role":  "user",
                        "content": prompt
                    }
                ],
                temperature=0.9,  # 분기형이라 좀 더 창의적으로
                max_tokens=800,
                response_format={"type": "json_object"}
            )

            # 응답 파싱
            content = response.choices[0].message.content
            logger.info(f'OpenAI 원본 응답: {content[:200]}...')  # 처음 200자만 로그
            logger.info(f'OpenAI 원본 응답 전체: {content}')  

            result = json.loads(content)
            logger.info(f'파싱된 JSON 키들: {list(result.keys())}')

            scene = result.get('scene', result)  # 'scene' 키가 없으면 result 자체를 씬으로 사용

            # scene이 비어있으면 result 전체가 scene일 가능성
            if not scene or not scene.get('sceneNumber'):
                scene = result

            # 씬 1인 경우 storyTitle과 characterDescription 추출하여 응답에 포함
            response = {"scene": scene, "isEnding": scene.get("isEnding", scene_number >= 8)}
            logger.info(f'scene_number={scene_number}, result에 storyTitle 있는지: {result.get("storyTitle")}')

            logger.info(f'scene_number={scene_number}, result keys={list(result.keys())}')
            logger.info(f'result에 storyTitle 있는지: {result.get("storyTitle")}')

            if scene_number == 1:
                if result.get('storyTitle'):
                    response['storyTitle'] = result.get('storyTitle')
                    logger.info(f'동화 제목 생성됨: {response["storyTitle"]}')
                else:
                    logger.warning(f'scene=1인데 storyTitle이 없음! result keys={list(result.keys())}')

                # [2025-11-05 추가] 캐릭터 설명 추출
                if result.get('characterDescription'):
                    response['characterDescription'] = result.get('characterDescription')
                    logger.info(f'캐릭터 설명 생성됨: {response["characterDescription"]}')
                elif scene.get('characterDescription'):
                    response['characterDescription'] = scene.get('characterDescription')
                    logger.info(f'캐릭터 설명 생성됨 (scene에서): {response["characterDescription"]}')

            logger.info(f'씬 {scene_number} 생성 완료: content={len(scene.get("content", ""))}자, choices={len(scene.get("choices", []))}개')
            return response

        except Exception as e:
            logger.error(f'씬 {scene_number} 생성 중 오류 발생: {e}')
            return self._get_dummy_single_scene(story_title, scene_number)

    def _create_next_scene_prompt(
            self,
            story_title: str,
            story_description: str,
            emotion: str,
            interests: List[str],
            concerns: List[str],  # [2025-11-11 추가] 자녀 우려사항
            scene_number: int,
            previous_choices: List[Dict],
            story_context: Optional[str],
            character_description: Optional[str] = None  # [2025-11-05 추가]
    ) -> str:
        """[2025-10-28 수정] 다음 씬 생성 프롬프트 작성

        story_title과 story_description 기반으로 스토리 생성
        childName은 주인공 이름으로 사용하지 않음
        [2025-11-11 추가] concerns를 통한 맞춤형 동화 생성
        """

        interests_text = ", ".join(interests) if interests else "친구와 우정"
        concerns_text = ", ".join(concerns) if concerns else None

        # 이전 선택 요약 및 능력치 분석
        choices_summary = ""
        used_abilities = set()
        if previous_choices:
            choices_summary = "\n**아이의 이전 선택들과 그 영향:**\n"
            for i, choice in enumerate(previous_choices, 1):
                ability = choice.get('abilityType')
                choice_text = choice.get('choiceText', '')

                if ability:
                    used_abilities.add(ability)
                
                # 마지막 선택지 저장
                if i == len(previous_choices):
                    last_choice_text = choice_text
                    last_ability_type = ability

                choices_summary += f"- 씬 {choice.get('sceneNumber')}: \"{choice_text}\" ({ability})\n"

            # 마지막 선택의 영향을 명시적으로 표시
            if last_choice_text and last_ability_type:
                choices_summary += f"\n**[중요] 방금 아이가 선택한 \"{last_choice_text}\"의 결과가 이번 씬에 반드시 반영되어야 합니다!**\n"
                choices_summary += f"- {last_ability_type} 능력치를 발휘한 선택이므로, 그에 맞는 긍정적인 결과를 보여주세요.\n"
                choices_summary += f"- 예: 용기 → 두려움을 극복한 결과, 공감 → 친구가 기뻐하는 모습, 창의성 → 문제가 해결됨 등\n"

        # 아직 안 나온 능력치 찾기
        all_abilities = {"용기", "공감", "창의성", "책임감", "우정"}
        unused_abilities = all_abilities - used_abilities

        if unused_abilities:
            choices_summary += f"\n**[중요] 아직 안 나온 능력치: {', '.join(unused_abilities)} - 이 중에서 우선적으로 선택지를 만들어주세요!**\n"

        # 씬 단계별 가이드
        stage_guide = ""
        if scene_number == 1:
            stage_guide = "**씬 1 (시작):** 주인공 소개, 현재 감정 상황 제시"
        elif scene_number <= 3:
            stage_guide = f"**씬 {scene_number} (도입/전개):** 문제 상황 제시, 갈등 시작"
        elif scene_number <= 5:
            stage_guide = f"**씬 {scene_number} (전개):** 문제 해결 시도, 선택의 영향 나타남"
        elif scene_number <= 7:
            stage_guide = f"**씬 {scene_number} (절정):** 중요한 선택의 순간, 감정 변화"
        else:
            stage_guide = f"**씬 {scene_number} (결말):** 긍정적 해결, 교훈, 마무리"

        is_ending = scene_number >= 8
        ending_note = ""
        if is_ending:
            ending_note = """
                        **중요: 이것이 마지막 씬입니다.**
                        - 선택지 중 하나는 "이야기를 마치고 돌아가기" 같은 종료 선택지여야 합니다.
                        - 스토리를 긍정적으로 마무리하세요.
                        - 아이가 배운 교훈을 자연스럽게 담으세요.
                        """

        # [2025-11-05 추가] 캐릭터 일관성 지시사항
        character_note = ""
        if scene_number == 1:
            character_note = """
            **[매우 중요] 캐릭터 일관성:**
            - 씬 1에서는 주인공 캐릭터를 정의해야 합니다!
            - characterDescription을 반드시 생성하세요. 예: "a cute white rabbit with pink ears", "a brave little bear with brown fur"
            - 영어로 작성하고, 구체적인 외모 특징을 포함하세요 (종류, 색상, 특징)
            - 이 설명은 모든 씬의 이미지 생성에 사용됩니다
            """
        elif character_description:
            character_note = f"""
            **[매우 중요] 캐릭터 일관성:**
            - 주인공 캐릭터: {character_description}
            - 모든 씬에서 이 캐릭터를 정확히 유지하세요
            - 캐릭터의 종류나 외모를 절대 바꾸지 마세요
            """

        # 씬별 스토리 가이드 (기승전결)
        story_phase = ""
        if scene_number == 1:
            story_phase = """
            **[씬 1 - 기(起): 시작]**
            - 주인공과 배경 소개
            - 평화롭거나 일상적인 상황에서 시작
            - 앞으로 펼쳐질 모험의 단서 제시
            """
        elif scene_number <= 3:
            story_phase = """
            **[씬 2-3 - 승(承): 전개]**
            - 사건이 시작되거나 문제가 등장
            - 주인공이 새로운 상황에 직면
            - 호기심을 자극하는 요소 추가
            """
        elif scene_number <= 6:
            story_phase = """
            **[씬 4-6 - 전(轉): 절정]**
            - 갈등이나 도전이 최고조에 달함
            - 주인공의 선택이 중요해지는 순간
            - 긴장감 있는 상황 연출
            """
        else:  # scene_number == 7 or 8
            story_phase = """
            **[씬 7-8 - 결(結): 결말]**
            - 이야기의 마무리 단계
            - 지금까지의 선택과 행동의 결과 보여주기
            - 따뜻하고 긍정적인 결말로 마무리
            """

        # 마지막 씬 처리
        ending_note = ""
        if scene_number == 8:
            ending_note = """

            **[최종 씬 - 선택지 없음!]**
            - 이 씬은 동화의 마지막이므로 **choices를 빈 배열 []로 반환**하세요
            - 주인공이 지금까지의 모험을 통해 배운 교훈이나 성장 포함
            - "그리하여 [주인공]은 행복하게 살았답니다" 같은 동화 결말 문구 사용
            - 아이에게 따뜻한 메시지 전달 (예: "용기", "친구", "배려")
            """

        # 우려사항 안내 추가
        concerns_note = ""
        if concerns_text:
            concerns_note = f"""

            **[매우 중요] 자녀 우려사항 반영:**
            부모가 다음과 같은 우려사항을 가지고 있습니다: {concerns_text}
            - 이 우려사항과 관련된 상황을 동화에 자연스럽게 포함시키세요
            - 주인공이 이러한 문제를 긍정적으로 해결하는 모습을 보여주세요
            - 아이가 배울 수 있는 교훈이나 올바른 행동을 제시하세요
            - 예시: "낯가림"이 우려사항이면 → 새로운 친구를 만나 용기내어 인사하는 이야기
            - 예시: "떼쓰기"가 우려사항이면 → 참을성 있게 기다리고 좋은 결과를 얻는 이야기
            """
        # [2025-11-12 김광현] 제목 생성 규칙 대폭 강화 - 원본과 완전히 다르게!
        scene_1_instruction = f"""**[최우선 명령!!!] 씬 1에서는 storyTitle과 characterDescription 생성 필수!**

        🚨 **storyTitle 생성 시 절대 금지 사항 (위반 시 실패!):**
        ❌ 원본 제목 "{story_title}"을 그대로 또는 비슷하게 사용 금지!
        ❌ 감정 단어("{emotion}")를 제목에 직접 사용 금지! (예: "걱정 많은", "슬픈", "화난" 등)
        ❌ 관심사 단어({interests_text})를 단순 조합 금지! (예: "공룡의 모험", "친구와 함께" 등)
        ❌ 설명문 형태 금지! (예: "~하는 아이의 이야기", "~을 배우는 동화")
        ❌ 교훈적 표현 금지! (예: "용기를 배우는", "극복하기", "해결하는 법")

        ✅ **storyTitle 올바른 생성 방법:**
        1. 원본 "{story_title}"의 **핵심 교훈/주제**만 머릿속에 기억
        2. 완전히 다른 **동화 캐릭터 중심 제목** 창작
        3. 형식: "형용사 + 캐릭터 + 명사" (예: "용감한 꼬마 토끼의 모험")
        4. 길이: 3-7어절
        5. 톤: 모험적, 판타지적, 긍정적

        **제목 생성 단계별 가이드:**
        Step 1: 원본 제목의 주제 파악 (예: 형제관계, 용기, 우정 등)
        Step 2: 아이 관심사({interests_text})에서 **주인공 캐릭터** 선택 (공룡→"꼬마 트리케라톱스", 동물→"작은 토끼")
        Step 3: 주제를 **모험/사건**으로 변환 (형제관계→"동생을 구한", 용기→"어둠을 이긴")
        Step 4: 조합하여 창작 (예: "어둠을 이긴 꼬마 트리케라톱스")

        **실전 예시 (반드시 참고!):**
        원본: "걱정을 극복하는 이야기"
        → ❌ "걱정 많은 공룡의 모험" (감정 단어 직접 사용!)
        → ✅ "어둠 속을 헤쳐 나간 꼬마 공룡"
        → ✅ "무서움을 이긴 작은 용사"
        → ✅ "용감한 트리케라톱스의 첫 여행"

        원본: "새로운 동생을 맞이하는 아이의 이야기"
        → ❌ "새로운 동생과의 하루" (원본과 유사!)
        → ✅ "꼬마 형이 된 작은 토끼"
        → ✅ "동생을 지킨 용감한 곰"
        → ✅ "둘이서 함께한 마법의 모험"

        원본: "친구와의 갈등 해결"
        → ❌ "친구 관계 개선" (교훈적!)
        → ✅ "친구를 구한 작은 별"
        → ✅ "마법의 숲에서 만난 친구"
        → ✅ "우정의 씨앗을 심은 날"

        **characterDescription 생성 규칙:**
        - 영어로 주인공의 외모를 구체적으로 작성
        - 동물이나 판타지 캐릭터로 설정
        - 예: "a brave little triceratops with green scales and a yellow horn", "a cute white rabbit with big blue eyes wearing a tiny backpack"
        """

        scene_continue_instruction = """**[중요]** 이전 씬의 선택 결과가 이번 씬 내용에 명확하게 드러나야 합니다! 아이가 선택한 행동의 결과를 구체적으로 보여주세요."""
        
        prompt = f"""
             '{story_title}' 동화의 씬 {scene_number}을 생성해주세요.

            **동화 정보:**
            {f'- 동화 주제: {story_description}' if scene_number == 1 else f'- 제목: {story_title}'}
            {'' if scene_number == 1 else f'- 줄거리: {story_description}'}
            - 주제/감정: {emotion}
            - 관심 요소: {interests_text}
            {concerns_note}

            {character_note}

            {story_phase}

            {stage_guide}

            {choices_summary}

            **이전 스토리 흐름:**
            {story_context or "첫 번째 씬입니다."}

           **요구사항:**
            1. {scene_1_instruction if scene_number == 1 else scene_continue_instruction}
            2. {emotion} 감정을 다루는 따뜻한 이야기
            3. {interests_text} 요소를 포함
            4. **스토리 연결성**: 아이의 선택이 스토리를 바꿨다는 느낌을 주도록 작성
            5. **선택지 작성 원칙 (매우 중요!):**
            - 나쁜 예: "용기를 낸다", "친구에게 도움을 청한다" (너무 추상적)
            - 좋은 예: "무서워도 큰 나무 위로 올라가본다", "숲 속 다람쥐에게 길을 물어본다" (구체적 행동)
            - **반드시 현재 씬의 상황에 맞는 구체적인 행동**을 선택지로 제시하세요
            - 선택지는 "~한다", "~해본다" 형태로 작성
            - 각 선택지는 **씬 내용에 등장한 요소나 상황을 직접 언급**해야 함
            - 3개의 선택지는 **서로 다른 능력치**를 대표해야 함 (전체 5가지 골고루 배치)
            6. **선택지 점수**: 10~15점 범위 (일반적 행동 10점, 적극적 행동 12점, 매우 훌륭한 행동 15점)
            7. **씬 내용 작성 규칙:**
            - 3-5문장으로 작성하되, 2-3개의 의미 단락으로 묶기
            - 각 의미 단락은 빈 줄(\\n\\n)로 구분
            - 한 단락 안에서는 띄어쓰기로만 구분 (줄바꿈 금지)
            - 마지막 문장(질문/결말)은 독립된 단락으로 구분
            - 유아가 이해하기 쉬운 한글 문장 사용
            - 각 문장은 짧고 명확하게
            8. 주인공은 동화 속 캐릭터로 설정 (특정 아이 이름 사용 금지)
            9. **[매우 중요] 언어 및 문체 규칙:**
            - **모든 동화 내용은 반드시 100% 순수 한글로만 작성** (영어 단어, 외래어 최소화)
            - **주인공 호칭 규칙**: "네가", "너는", "당신" 같은 2인칭 절대 금지!
              ✅ 좋은 예: "작은 토끼가", "꼬마 로봇은", "아기 곰이"
              ❌ 나쁜 예: "네가", "너는", "당신이"
            - **문체**: 동화책 스타일의 3인칭 서술형으로 작성
              ✅ 좋은 예: "작은 토끼가 무서운 숲 속을 용기 내어 걸어갔어요."
              ❌ 나쁜 예: "네가 무서운 숲을 용기 내어 걸어가고 있어."
            - **선택지도 3인칭 주어 사용**: 
              ✅ 좋은 예: "토끼가 큰 나무 위로 올라간다"
              ❌ 나쁜 예: "큰 나무 위로 올라간다" (주어 생략 금지)
            - **캐릭터 지칭 일관성**: 한 번 정한 호칭(예: "작은 토끼")을 계속 사용
            {ending_note}

            **출력 형식 (JSON):**
            """

        if scene_number == 1:
            prompt += f"""
            {{
                "storyTitle": "동화 제목 (한글! 원본 '{story_title}'을 참고만 하고 완전히 새롭게! 예: 용감한 꼬마 토끼의 모험, 친구를 구한 작은 별, 무지개를 찾아 떠난 여행)",
                "characterDescription": "주인공 캐릭터 설명 (영어로 필수! 예: a cute white rabbit with pink ears, a brave little bear with brown fur)",
                "scene": {{
                    "sceneNumber": 1,
                    "content": "씬 내용 (3-5문장, '{story_title}'에 맞는 내용)",
                    "imagePrompt": "DALL-E용 영어 프롬프트",
                    "choices": [
                        {{
                            "choiceId": 101,
                            "choiceText": "선택지 1 텍스트",
                            "abilityType": "용기/공감/창의성/책임감/우정 (한글)",
                            "abilityScore": 10-15
                        }},
                        {{
                            "choiceId": 102,
                            "choiceText": "선택지 2 텍스트",
                            "abilityType": "용기/공감/창의성/책임감/우정 (한글)",
                            "abilityScore": 10-15
                        }},
                        {{
                            "choiceId": 103,
                            "choiceText": "선택지 3 텍스트",
                            "abilityType": "용기/공감/창의성/책임감/우정 (한글)",
                            "abilityScore": 10-15
                        }}
                    ],
                    "isEnding": false
                }}
            }}
            """
        else:
            prompt += f"""
            {{
                "scene": {{
                    "sceneNumber": {scene_number},
                    "content": "씬 내용 (3-5문장, '{story_title}'에 맞는 내용)",
                    "imagePrompt": "DALL-E용 영어 프롬프트",
                    "choices": [
                        {{
                            "choiceId": {scene_number * 100 + 1},
                            "choiceText": "선택지 1 텍스트 (구체적인 행동)",
                            "abilityType": "용기/공감/창의성/책임감/우정 (한글)",
                            "abilityScore": 10-15
                        }},
                        {{
                            "choiceId": {scene_number * 100 + 2},
                            "choiceText": "선택지 2 텍스트 (구체적인 행동)",
                            "abilityType": "용기/공감/창의성/책임감/우정 (한글)",
                            "abilityScore": 10-15
                        }},
                        {{
                            "choiceId": {scene_number * 100 + 3},
                            "choiceText": "선택지 3 텍스트 (구체적인 행동)",
                            "abilityType": "용기/공감/창의성/책임감/우정 (한글)",
                            "abilityScore": 10-15
                        }}
                    ],
                    "isEnding": {str(is_ending).lower()}
                }}
            }}
            **선택지 작성 예시:**
            만약 씬 내용이 "작은 토끼가 높은 산을 마주쳤어요. 정상까지 가려면 험한 바위를 올라가야 해요."라면,

            좋은 선택지:
            - "무서워도 바위를 하나씩 잡고 조심조심 올라간다" (용기, 12점)
            - "산 아래서 쉬고 있는 친구 거북이에게 함께 가자고 한다" (우정, 12점)
            - "나뭇가지로 지팡이를 만들어 균형을 잡으며 올라간다" (창의성, 15점)

            나쁜 선택지:
            - "용기를 낸다" (추상적)
            - "도움을 청한다" (누구에게? 무엇을?)
            - "창의적으로 해결한다" (어떻게?)
            """

        # [2025-11-04 김광현] 스토리 작성 팁 추가 (씬 2 이상에서만)
        story_tips = ""
        if scene_number > 1 and previous_choices:
            last_choice = previous_choices[-1] if previous_choices else None
            if last_choice:
                last_choice_text = last_choice.get('choiceText', '')
                story_tips = f"""

                **스토리 작성 팁:**
                - 아이의 선택 "{last_choice_text}"의 직접적인 결과를 씬 내용에 포함하세요
                - "네가 [선택한 행동] 덕분에..." 같은 문구로 인과관계를 명확히 하세요
                - 선택지도 이전 선택을 반영한 새로운 상황에서 나와야 합니다
                """

        prompt += f"""

        **[최종 체크리스트 - 반드시 확인!]**
        ✅ 씬 1에서는 storyTitle을 **scene 밖에** 별도로 포함
        ✅ 씬 8(마지막)은 **choices를 빈 배열 []로 반환** (선택지 없음)
        ✅ 씬 번호에 맞는 스토리 단계(기승전결) 준수
        ✅ 모든 동화 내용(content, choiceText)은 **100% 순수 한글** (영어 단어 금지!)
        ✅ 주인공 호칭: "네가", "너는" 금지 → "작은 토끼가", "꼬마 로봇은" 사용
        ✅ 3인칭 서술: "작은 토끼가 ~했어요" (O) / "네가 ~했어" (X)
        ✅ content는 2-3개의 의미 단락으로 구성, 단락 간 빈 줄(\\n\\n)로 구분
            예시: "작은 토끼가 숲 속을 걷다가 갑자기 큰 나무를 발견했어요. 나무 위에서 다람쥐가 손을 흔들고 있었어요.\\n\\n토끼는 용기를 내어 나무를 올라가보기로 했어요.\\n\\n이제 어떻게 할지 생각해보아야 했어요."
        ✅ 동화 제목: 아이가 이해하기 쉬운 한글 (예: "용감한 작은 토끼", "친구를 도운 꼬마 별")
        ✅ **imagePrompt만 영어로 작성** (씬 내용 구체적 묘사)
          예시: "A cute little rabbit standing bravely in a magical forest, children's book illustration style, warm pastel colors, friendly atmosphere, digital art"
        ✅ imagePrompt에는 씬의 주요 장면, 캐릭터, 분위기, 배경 포함
        ✅ 위 JSON 형식 정확히 준수
        {story_tips}
        
        **지금 바로 씬 {scene_number}을 위 규칙에 따라 JSON으로 생성해주세요!**
        """

        return prompt

    def _get_dummy_single_scene(self, story_title: str, scene_number: int) -> Dict:
        """더미 단일 씬 데이터 (OpenAI 연결 실패시)"""
        scene = {
            "sceneNumber": scene_number,
            "content": f"씬 {scene_number}: '{story_title}' 이야기가 계속됩니다. 주인공은 친구들과 함께 즐거운 하루를 보냈어요.",
            "imagePrompt": f"Children's book illustration for '{story_title}', scene {scene_number}, warm and friendly atmosphere",
            "choices": [
                {
                    "choiceId": scene_number * 100 + 1,
                    "choiceText": "친구에게 다가가서 말을 걸어요",
                    "abilityType": "용기",
                    "abilityScore": 10
                },
                {
                    "choiceId": scene_number * 100 + 2,
                    "choiceText": "친구를 도와줘요",
                    "abilityType": "책임감",
                    "abilityScore": 10
                },
                {
                    "choiceId": scene_number * 100 + 3,
                    "choiceText": "친구의 이야기를 들어줘요",
                    "abilityType": "공감",
                    "abilityScore": 10
                }
            ],
            "isEnding": scene_number >= 8
        }
        return {"scene": scene, "isEnding": scene_number >= 8}

    def _get_dummy_scenes(self, child_name: str) -> List[Dict]:
        """더미 씬 데이터 (OpenAI 연결 실패시)"""
        scenes = []
        for i in range(1, 9):
            scene = {
                "sceneNumber": i,
                "content": f"씬 {i}: 옛날 옛날 {child_name}는 친구들과 함께 즐거운 하루를 보냈어요.",
                "imagePrompt": f"A cute character named {child_name}, scene {i}, children's book style",
                "choices": [
                    {
                        "choiceId": i * 10 + 1,
                        "choiceText": "친구에게 다가가서 말을 걸어요",
                        "abilityType": "용기",
                        "abilityScore": 10
                    },
                    {
                        "choiceId": i * 10 + 2,
                        "choiceText": "친구를 도와줘요",
                        "abilityType": "책임감",
                        "abilityScore": 10
                    },
                    {
                        "choiceId": i * 10 + 3,
                        "choiceText": "친구의 이야기를 들어줘요",
                        "abilityType": "공감",
                        "abilityScore": 10
                    }
                ]
            }
            scenes.append(scene)
        return scenes

    async def generate_text_async(self, prompt: str) -> str:
        """간단한 텍스트 생성 (async 래퍼)"""
        if not self.client:
            return "텍스트 생성 불가"
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 어린이를 위한 따뜻한 동화 작가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"텍스트 생성 실패: {e}")
            return "텍스트 생성에 실패했습니다."

    async def generate_next_scene_async(
            self,
            story_id: str,
            story_title: str,
            story_description: str,
            emotion: str,
            interests: List[str],
            concerns: List[str],  # [2025-11-11 김광현 추가] 자녀 우려사항
            scene_number: int,
            previous_choices: List[Dict],
            story_context: Optional[str] = None,
            character_description: Optional[str] = None  # [2025-11-05 추가]
    ) -> Dict:
        """
        이전 선택을 기반으로 다음 씬 생성 (분기형 스토리) - async 버전

        [2025-10-28 수정] story_title, story_description 추가
        [2025-11-05 수정] character_description 추가
        [2025-11-11 수정] concerns 추가
        childName 제거 - 동화 주인공으로 사용하지 않음
        """
        return self.generate_next_scene(
            story_id, story_title, story_description, emotion, interests, concerns,
            scene_number, previous_choices, story_context, character_description
        )

    async def generate_image_async(self, prompt: str, size: str = "1024x1024") -> str:
        """
        DALL-E를 사용한 이미지 생성

        Args:
            prompt: 이미지 생성 프롬프트 (영어)
            size: 이미지 크기 ("1024x1024", "1792x1024", "1024x1792")

        Returns:
            이미지 URL
        """
        if not self.client:
            logger.error("OpenAI 클라이언트가 초기화되지 않았습니다.")
            raise Exception("OpenAI 클라이언트 없음")

        try:
            logger.info(f"DALL-E 이미지 생성 중: {prompt[:50]}... (size: {size})")

            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality="standard",
                n=1,
            )

            image_url = response.data[0].url
            logger.info(f"이미지 생성 완료: {image_url}")
            return image_url

        except Exception as e:
            logger.error(f"DALL-E 이미지 생성 실패: {e}")
            raise

    # [2025-11-12 김광현] 동화 제목 기반 줄거리 생성
    async def generate_story_summary(self, story_title: str) -> str:
        """
        Pinecone 동화 제목을 기반으로 1-2문장의 간단한 줄거리를 생성합니다.
        동화 추천 카드에 표시할 설명문입니다.
        
        Args:
            story_title: Pinecone에 저장된 동화 제목
            
        Returns:
            1-2문장의 간단한 줄거리 (40-60자 이내)
            
        Example:
            >>> await generate_story_summary("공포를 극복하는 공룡 친구들")
            "무서움을 이겨내고 용기를 배우는 공룡들의 우정 이야기예요."
        """
        if not self.client:
            logger.warning("OpenAI 클라이언트가 없어 기본 줄거리 반환")
            return f"{story_title}의 따뜻한 이야기예요."
        
        try:
            prompt = f"""다음 동화 제목을 보고, 어린이에게 보여줄 1-2문장의 간단한 줄거리를 작성해주세요.

            동화 제목: "{story_title}"

            요구사항:
            1. 1-2문장으로 작성 (40-60자 이내)
            2. 한글로만 작성
            3. 어린이가 이해하기 쉬운 표현
            4. 동화의 핵심 주제/교훈을 담기
            5. 흥미롭고 따뜻한 톤
            6. "~이야기예요", "~배워요", "~느껴요" 등으로 끝맺기

            좋은 예시:
            - 제목: "공포를 극복하는 공룡 친구들" 
            줄거리: "무서움을 이겨내고 용기를 배우는 공룡들의 우정 이야기예요."

            - 제목: "새로운 동생을 맞이하는 아이" 
            줄거리: "새로운 가족을 맞이하며 형/언니가 되는 기쁨을 느껴요."

            - 제목: "친구와의 갈등 해결" 
            줄거리: "친구와 다투고 화해하며 우정의 소중함을 깨달아요."

            나쁜 예시:
            - "이 동화는 공포를 극복하는 내용입니다" (딱딱하고 설명적)
            - "공룡 친구들" (너무 짧고 줄거리 없음)
            - "Once upon a time..." (영어 사용)
            - "공포 극복에 대한 교육적인 이야기입니다" (딱딱함)

            줄거리 (40-60자):"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # 빠르고 저렴한 모델
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 어린이 동화 줄거리를 작성하는 전문가입니다. 간결하고 따뜻한 문체로 1-2문장의 설명을 작성합니다."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=100,
                timeout=5.0
            )
            
            summary = response.choices[0].message.content.strip()
            
            # 불필요한 따옴표, 줄바꿈 제거
            summary = summary.strip('"').strip("'").strip().replace('\n', ' ').replace('\r', '')
            
            # "줄거리:" 같은 접두어 제거
            if summary.startswith("줄거리:"):
                summary = summary[4:].strip()
            
            # 너무 길면 자르기 (80자 이내)
            if len(summary) > 80:
                summary = summary[:77] + "..."
            
            logger.info(f"[AI 줄거리 생성] {story_title} → {summary}")
            
            return summary
            
        except Exception as e:
            logger.error(f"[AI 줄거리 생성 실패] {story_title}, 에러: {str(e)}")
            # 실패 시 기본 줄거리 반환
            return f"{story_title}의 따뜻하고 감동적인 이야기예요."