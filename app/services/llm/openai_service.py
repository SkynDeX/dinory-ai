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
            "abilityScore": 8-15,
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
                story_title, story_description, emotion, interests, scene_number, previous_choices, story_context, character_description
            )

            logger.info(f'씬 {scene_number} 생성 중... (스토리: {story_title}, 이전 선택: {len(previous_choices)}개)')

            # OpenAI 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 어린이를 위한 창의적이고 따뜻한 인터랙티브 동화 작가입니다. 아이의 이전 선택을 반영하여 스토리가 자연스럽게 분기되도록 만듭니다."
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
            scene_number: int,
            previous_choices: List[Dict],
            story_context: Optional[str],
            character_description: Optional[str] = None  # [2025-11-05 추가]
    ) -> str:
        """[2025-10-28 수정] 다음 씬 생성 프롬프트 작성

        story_title과 story_description 기반으로 스토리 생성
        childName은 주인공 이름으로 사용하지 않음
        """

        interests_text = ", ".join(interests) if interests else "친구와 우정"

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

        prompt = f"""
            '{story_title}' 동화의 씬 {scene_number}을 생성해주세요.

            **동화 정보:**
            - 제목: {story_title}
            - 줄거리: {story_description}
            - 주제/감정: {emotion}
            - 관심 요소: {interests_text}

            {character_note}

            {stage_guide}

            {choices_summary}

            **이전 스토리 흐름:**
            {story_context or "첫 번째 씬입니다."}

           **요구사항:**
            1. {f'**[중요]** 씬 1에서는 반드시 storyTitle과 characterDescription을 생성해야 합니다! storyTitle은 한글로 작성하세요. 예: "용감한 꼬마 호랑이", "마법의 숲 탐험", "친구를 도운 작은 별" 등. 원본 제목 "{story_title}"을 참고하되 더 매력적이고 아이가 이해하기 쉬운 제목으로 만드세요. characterDescription은 영어로 주인공의 외모를 구체적으로 작성하세요.' if scene_number == 1 else '**[중요]** 이전 씬의 선택 결과가 이번 씬 내용에 명확하게 드러나야 합니다! 아이가 선택한 행동의 결과를 구체적으로 보여주세요.'}
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
            7. 씬 내용은 3-5문장, 유아가 이해하기 쉬운 한글 문장
            8. 주인공은 동화 속 캐릭터로 설정 (특정 아이 이름 사용 금지)
            {ending_note}

            **출력 형식 (JSON):**
            """

        if scene_number == 1:
            prompt += f"""
            {{
                "storyTitle": "동화 제목 (한글로 필수! 예: 용감한 꼬마 호랑이, 친구를 도운 작은 별 등)",
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

        **중요:**
        - 씬 1에서는 storyTitle을 **scene 밖에** 별도로 포함해야 합니다!
        - 모든 텍스트는 한글로 작성
        - 동화 제목은 아이가 이해하기 쉬운 한글로 (예: "용감한 작은 토끼", "친구를 도운 꼬마 별")
        - **imagePrompt는 반드시 영어로 작성하고, 씬 내용을 구체적으로 묘사해야 합니다!**
        예시: "A cute little rabbit standing bravely in a magical forest, children's book illustration style, warm pastel colors, friendly atmosphere, digital art"
        - imagePrompt에는 씬의 주요 장면, 캐릭터, 분위기, 배경을 영어로 상세히 포함하세요
        - 반드시 위 JSON 형식을 정확히 따라주세요!
        {story_tips}
        씬 {scene_number}을 JSON 형식으로 생성해주세요!
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
            scene_number: int,
            previous_choices: List[Dict],
            story_context: Optional[str] = None,
            character_description: Optional[str] = None  # [2025-11-05 추가]
    ) -> Dict:
        """
        이전 선택을 기반으로 다음 씬 생성 (분기형 스토리) - async 버전

        [2025-10-28 수정] story_title, story_description 추가
        [2025-11-05 수정] character_description 추가
        childName 제거 - 동화 주인공으로 사용하지 않음
        """
        return self.generate_next_scene(
            story_id, story_title, story_description, emotion, interests,
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