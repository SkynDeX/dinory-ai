from openai import OpenAI
import os
import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

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
            6. 선택지는 다양한 능력치(친절, 용기, 공감, 우정, 자존감) 향상

            **능력치 유형:**
            - 친절: 다른 사람을 배려하고 도와주는 행동
            - 용기: 두려움을 극복하고 도전하는 행동
            - 공감: 다른 사람의 감정을 이해하는 행동
            - 우정: 친구와 좋은 관계를 만드는 행동
            - 자존감: 자신을 사랑하고 믿는 행동

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
                    "abilityType": "친절",
                    "abilityScore": 10
                    }},
                    {{
                    "choiceId": 2,
                    "choiceText": "선택지 2",
                    "abilityType": "용기",
                    "abilityScore": 15
                    }},
                    {{
                    "choiceId": 3,
                    "choiceText": "선택지 3",
                    "abilityType": "공감",
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
                "abilityType": "친절",
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
            - 친절: 다른 사람을 배려하거나 도와주는 내용
            - 용기: 두려움을 극복하거나 도전하는 내용
            - 공감: 다른 사람의 감정을 이해하는 내용
            - 우정: 친구와의 관계를 중요하게 생각하는 내용
            - 자존감: 자신을 믿거나 긍정하는 내용

            **출력 형식 (JSON):**
            {{
            "abilityType": "친절/용기/공감/우정/자존감 중 하나",
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
                "abilityType": "친절",
                "abilityScore": 10,
                "feedback": "멋진 선택이에요!",
                "nextSceneBranch": None
            }
    
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
                        "abilityType": "친절",
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