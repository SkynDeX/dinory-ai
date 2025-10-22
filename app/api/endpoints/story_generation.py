from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai", tags=["ai"])

# 서비스 import 및 초기화
from app.services.story.story_generator import StorySearchService
from app.services.llm.openai_service import OpenAIService

story_generator = StorySearchService()
openai_service = OpenAIService()

# ==================== Request Models ====================

class RecommendStoriesRequest(BaseModel):
    """추천 동화 검색 요청 (Pinecone)"""
    emotion: Optional[str] = None
    interests: Optional[List[str]] = None
    childId: Optional[int] = None
    limit: int = 5


class GenerateStoryRequest(BaseModel):
    """AI 동화 생성 요청"""
    storyId: str
    childId: int
    emotion: str
    interests: List[str]
    childName: Optional[str] = None  # 아이 이름 (개인화용)
    originalStoryData: Optional[Dict[str, Any]] = None  # 원본 동화 데이터


class AnalyzeCustomChoiceRequest(BaseModel):
    """커스텀 선택지 분석 요청"""
    childId: int
    storyId: str
    sceneId: int
    customText: str
    sceneContext: Optional[str] = None  # 씬 내용


# ==================== Response Models ====================

class StoryRecommendation(BaseModel):
    """추천 동화 아이템"""
    storyId: str
    title: str
    matchingScore: int  # 매칭 점수 0-100
    metadata: Dict[str, Any]


class SceneChoice(BaseModel):
    """씬 선택지"""
    choiceId: int
    choiceText: str
    abilityType: str  # 친절, 용기, 공감, 우정
    abilityScore: int


class GeneratedScene(BaseModel):
    """생성된 씬"""
    sceneNumber: int
    content: str
    imagePrompt: Optional[str] = None  # DALL-E 이미지 생성용 프롬프트
    choices: List[SceneChoice]


class GenerateStoryResponse(BaseModel):
    """동화 생성 응답"""
    storyId: str
    totalScenes: int
    scenes: List[GeneratedScene]


class CustomChoiceAnalysis(BaseModel):
    """커스텀 선택지 분석 결과"""
    abilityType: str
    abilityScore: int
    feedback: str  # AI 피드백
    nextSceneBranch: Optional[int] = None  # 분기 씬 번호


# ==================== API Endpoints ====================

@router.post("/recommend-stories", response_model=List[StoryRecommendation])
async def recommend_stories(request: RecommendStoriesRequest):
    """
    [AI] Pinecone에서 감정/관심사 기반 동화 추천
    
    Spring Boot의 GET /api/stories/recommended 에서 호출
    """
    try:
        logger.info(f"AI recommend - emotion: {request.emotion}, interests: {request.interests}")
        
        # Pinecone 검색 구현
        search_results = story_generator.search_stories(
            emotion=request.emotion,
            interests=request.interests,
            top_k=request.limit
        )

        # 응답 변환
        recommendations = []
        for story in search_results:
            metadata = story.get('metadata',{})

            rec = StoryRecommendation(
                storyId=str(story.get('story_id', '')),
                title=story.get('title', metadata.get('title', '제목 없음')),
                matchingScore=story.get('matching_score', 0),
                metadata=metadata
            )
            recommendations.append(rec)
        
        # # 임시 더미 데이터
        # dummy_results = [
        #     StoryRecommendation(
        #         storyId="9791193449196",
        #         title="정글에서 친구 찾기",
        #         matchingScore=95,
        #         metadata={
        #             "author": "이지영",
        #             "classification": "의사소통",
        #             "readAge": "유아",
        #             "plotSummary": "원숭이가 친구를 찾아 떠나는 모험"
        #         }
        #     ),
        #     StoryRecommendation(
        #         storyId="9791193449197",
        #         title="화난 토끼의 하루",
        #         matchingScore=92,
        #         metadata={
        #             "author": "테스트",
        #             "classification": "감정조절",
        #             "readAge": "유아",
        #             "plotSummary": "토끼가 화를 조절하는 법을 배워요"
        #         }
        #     )
        # ]
        
        # logger.info(f"{len(recommendations)}개의 추천을 반환합니다.")
        # return dummy_results[:request.limit]
        logger.info(f"{len(recommendations)}개의 추천을 반환합니다.")
        return recommendations
        
    except Exception as e:
        logger.error(f"Error in recommend_stories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-story", response_model=GenerateStoryResponse)
async def generate_story(request: GenerateStoryRequest):
    """
    [AI] OpenAI로 8씬 인터랙티브 동화 생성
    
    Spring Boot의 POST /api/story/{storyId}/generate 에서 호출
    """
    try:
        logger.info(f"AI generate story - storyId: {request.storyId}, child: {request.childId}")
        
        # OpenAI로 동화 생성 구현
        scenes_data = openai_service.generate_personalized_stroy(
            story_id=request.storyId,
            child_name=request.childName or "친구",
            emotion=request.emotion,
            interests=request.interests,
            original_story_data=request.originalStoryData
        )

        # 응답 변환
        scenes = []
        for scene_data in scenes_data:
            choices = [
                SceneChoice(
                    choiceId=choice['choiceId'],
                    choiceText=choice['choiceText'],
                    abilityType=choice['abilityType'],
                    abilityScore=choice['abilityScore']
                )
                for choice in scene_data.get('choices', [])
            ]

            scene = GeneratedScene(
                sceneNumber=scene_data['sceneNumber'],
                content=scene_data['content'],
                imagePrompt=scene_data.get('imagePrompt'),
                choices=choices
            )
            scenes.append(scene)

        response = GenerateStoryResponse(
            storyId=request.storyId,
            totalScenes=len(scenes),
            scenes=scenes
        )

        logger.info(f'OpenAI로 생성된 {len(scenes)} 장면')
        return response
        
        # # 임시 더미 데이터
        # child_name = request.childName or "친구"
        # scenes = []
        
        # for i in range(1, 9):  # 8씬 생성
        #     scene = GeneratedScene(
        #         sceneNumber=i,
        #         content=f"씬 {i}: 옛날 옛날 정글에 {child_name}라는 원숭이가 살았어요. 오늘은 친구와 바나나를 두고 다퉜어요.",
        #         imagePrompt=f"A cute monkey named {child_name} in the jungle, children's book illustration style",
        #         choices=[
        #             SceneChoice(
        #                 choiceId=i * 10 + 1,
        #                 choiceText="친구에게 미안하다고 말하기",
        #                 abilityType="친절",
        #                 abilityScore=10
        #             ),
        #             SceneChoice(
        #                 choiceId=i * 10 + 2,
        #                 choiceText="혼자 바나나 먹기",
        #                 abilityScore=5,
        #                 abilityType="자존감"
        #             ),
        #             SceneChoice(
        #                 choiceId=i * 10 + 3,
        #                 choiceText="다른 친구 찾으러 가기",
        #                 abilityType="용기",
        #                 abilityScore=10
        #             )
        #         ]
        #     )
        #     scenes.append(scene)
        
        # response = GenerateStoryResponse(
        #     storyId=request.storyId,
        #     totalScenes=len(scenes),
        #     scenes=scenes
        # )
        
        # logger.info(f"Generated {len(scenes)} scenes")
        # return response
        
    except Exception as e:
        logger.error(f"generate_story 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-custom-choice", response_model=CustomChoiceAnalysis)
async def analyze_custom_choice(request: AnalyzeCustomChoiceRequest):
    """
    [AI] 커스텀 선택지 텍스트 분석
    
    Spring Boot의 POST /api/story/custom-choice 에서 호출
    아이가 직접 입력한 텍스트를 분석하여 능력치와 피드백 제공
    """
    try:
        logger.info(f"AI가 맞춤형 선택을 분석 - text: {request.customText}")
        
        # OpenAI로 텍스트 분석
        result = openai_service.anlyze_custom_choice(
            custom_text=request.customText,
            scene_context=request.sceneContext
        )

        analysis = CustomChoiceAnalysis(
            abilityType=result['abilityType'],
            abilityScore=result['abilityScore'],
            feedback=result['feedback'],
            nextSceneBranch=result.get('nextSceneBranch')
        )
        
        logger.info(f"Analysis result: {analysis.abilityType} +{analysis.abilityScore}")
        return analysis
        
        # # 임시 더미 분석
        # analysis = CustomChoiceAnalysis(
        #     abilityType="친절",
        #     abilityScore=12,
        #     feedback="정말 좋은 선택이에요! 친구를 배려하는 마음이 느껴져요.",
        #     nextSceneBranch=None
        # )
        
        # logger.info(f"Analysis result: {analysis.abilityType} +{analysis.abilityScore}")
        # return analysis
        
    except Exception as e:
        logger.error(f"Error in analyze_custom_choice: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-image")
async def generate_image(scene_content: str, style: str = "children's book illustration"):
    """
    [AI] DALL-E로 씬 이미지 생성
    
    Spring Boot에서 필요시 호출
    """
    try:
        logger.info(f"AI generate image for scene")
        
        # TODO: DALL-E 이미지 생성
        # image_url = dalle_service.generate(scene_content, style)
        
        # 임시 더미 URL
        image_url = "https://via.placeholder.com/1024x1024.png?text=Story+Scene"
        
        return {
            "success": True,
            "imageUrl": image_url
        }
        
    except Exception as e:
        logger.error(f"Error in generate_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """AI 서비스 헬스체크"""
    return {
        "status": "healthy",
        "service": "dinory-ai",
        "pinecone": "not_connected",  # TODO: 실제 체크
        "openai": "not_configured"  # TODO: 실제 체크
    }