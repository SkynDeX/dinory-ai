import random
from typing import Optional


class ResponseGenerator:
    """
    채팅 응답 생성 및 감정 분석을 위한 유틸리티 클래스
    """

    def __init__(self):
        self.greetings = [
            "안녕하세요! 오늘 하루는 어땠나요?",
            "만나서 반가워요! 무슨 이야기를 하고 싶나요?",
            "안녕! 오늘은 어떤 재미있는 일이 있었나요?",
            "반가워요! 궁금한 게 있으면 무엇이든 물어보세요!",
            "안녕하세요! 함께 즐거운 이야기를 나눠봐요!"
        ]

        self.emotion_keywords = {
            "happy": ["기쁘", "행복", "좋아", "신나", "재밌", "웃겨", "즐거"],
            "sad": ["슬프", "우울", "속상", "아쉽", "눈물", "힘들"],
            "angry": ["화나", "짜증", "싫어", "미워", "분해"],
            "scared": ["무서", "두려", "겁나", "불안"],
            "excited": ["설레", "기대", "궁금", "신기"],
            "tired": ["피곤", "지쳐", "힘들", "졸려"]
        }

    def generate_greeting(self, child_id: Optional[int] = None) -> str:
        """
        환영 인사 생성
        """
        return random.choice(self.greetings)

    def analyze_emotion(self, message: str) -> Optional[str]:
        """
        메시지에서 감정 분석
        """
        message_lower = message.lower()

        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    return emotion

        return "neutral"

    def generate_empathy_response(self, emotion: str) -> str:
        """
        감정에 맞는 공감 응답 생성
        """
        empathy_responses = {
            "happy": [
                "정말 좋은 일이네요! 저도 기뻐요!",
                "와! 정말 기분 좋은 일이에요!",
                "함께 기뻐하니 더 좋네요!"
            ],
            "sad": [
                "속상했겠어요. 괜찮으신가요?",
                "힘든 일이 있었나 봐요. 제가 들어줄게요.",
                "슬플 때는 이야기하면 조금 나아져요."
            ],
            "angry": [
                "많이 화가 났나 봐요. 무슨 일이 있었어요?",
                "화가 나는 건 당연해요. 천천히 이야기해봐요.",
                "짜증나는 일이 있었군요. 제가 들어줄게요."
            ],
            "scared": [
                "무서웠겠어요. 이제 괜찮아요.",
                "걱정하지 마세요. 함께 있을게요.",
                "두려운 마음이 들 때는 깊게 숨을 쉬어봐요."
            ],
            "excited": [
                "정말 설레는 일이네요!",
                "기대되는 일이 있나 봐요! 더 이야기해줘요!",
                "와! 정말 신나는 일이에요!"
            ],
            "tired": [
                "많이 피곤하신가 봐요. 조금 쉬어요.",
                "힘든 하루였나 봐요. 수고했어요!",
                "무리하지 말고 푹 쉬세요."
            ]
        }

        responses = empathy_responses.get(emotion, ["네, 그렇군요!"])
        return random.choice(responses)

    def generate_followup_question(self, emotion: str) -> str:
        """
        대화를 이어갈 후속 질문 생성
        """
        followup_questions = {
            "happy": [
                "어떤 일이 있었는지 더 자세히 들려주실래요?",
                "그래서 어떻게 됐어요?",
                "정말 멋진 일이네요! 다른 재미있는 일도 있었나요?"
            ],
            "sad": [
                "무슨 일이 있었는지 이야기하고 싶으세요?",
                "제가 도와줄 수 있는 일이 있을까요?",
                "이야기를 나누면 조금 나아질 거예요."
            ],
            "angry": [
                "어떤 일 때문에 화가 났어요?",
                "속상한 일을 이야기해보면 어떨까요?",
                "무슨 일이 있었는지 들려주세요."
            ],
            "neutral": [
                "오늘 하루는 어땠나요?",
                "더 이야기하고 싶은 게 있나요?",
                "궁금한 게 있으면 물어보세요!"
            ]
        }

        questions = followup_questions.get(emotion, followup_questions["neutral"])
        return random.choice(questions)