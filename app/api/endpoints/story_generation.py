from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, AliasChoices, ConfigDict
from typing import List, Optional, Dict, Any
import time, random, logging, traceback, json

logger = logging.getLogger("dinory.storygen")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[STORYGEN] %(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

router = APIRouter(tags=["ai"])

# [2025-11-05 추가] 스토리별 캐릭터 설명 저장소 (메모리)
# Key: storyId, Value: characterDescription
CHARACTER_DESCRIPTIONS = {}

# ────────────────────────────────────────────────
try:
    from app.services.story.story_generator import StorySearchService
except Exception as e:
    logger.warning(f"StorySearchService import 실패: {e}")
    StorySearchService = None

try:
    from app.services.llm.openai_service import OpenAIService
except Exception as e:
    logger.warning(f"OpenAIService import 실패: {e}")
    OpenAIService = None

# ==================== 모델 ====================

class RecommendStoriesRequest(BaseModel):
    model_config = ConfigDict(extra='ignore', populate_by_name=True)

    emotion: Optional[str] = None
    interests: Optional[List[str]] = None
    childId: Optional[int] = Field(default=None, validation_alias=AliasChoices('childId', 'child_id'))
    limit: int = 5


class Choice(BaseModel):
    model_config = ConfigDict(extra='ignore', populate_by_name=True)

    id: Optional[str] = None
    choiceId: Optional[str] = Field(default=None, validation_alias=AliasChoices('choiceId', 'choice_id'))
    label: str
    text: Optional[str] = None
    abilityType: Optional[str] = Field(default=None, validation_alias=AliasChoices('abilityType', 'ability_type'))
    abilityPoints: Optional[int] = Field(default=0, validation_alias=AliasChoices('abilityPoints', 'ability_points'))
    abilityScore: Optional[int] = Field(default=0, validation_alias=AliasChoices('abilityScore', 'ability_score'))


class Scene(BaseModel):
    model_config = ConfigDict(extra='ignore', populate_by_name=True)

    sceneNumber: int = Field(validation_alias=AliasChoices('sceneNumber', 'scene_number'))
    text: Optional[str] = None
    content: Optional[str] = None
    choices: List[Choice] = Field(default_factory=list)


class GenerateStoryBody(BaseModel):
    model_config = ConfigDict(extra='ignore', populate_by_name=True)

    childId: int = Field(validation_alias=AliasChoices('childId', 'child_id'))
    childName: Optional[str] = Field(default=None, validation_alias=AliasChoices('childName', 'child_name'))
    emotion: Optional[str] = None
    interests: Optional[List[str]] = None


class GenerateStoryRequest(BaseModel):
    model_config = ConfigDict(extra='ignore', populate_by_name=True)

    storyId: str = Field(validation_alias=AliasChoices('storyId', 'story_id'))
    body: Optional[GenerateStoryBody] = None


class GenerateStoryResponse(BaseModel):
    completionId: int | str
    story: Dict[str, Any]
    firstScene: Scene


class NextSceneRequest(BaseModel):
    """다음 씬 생성 요청 (분기형 스토리) - Spring Boot와 호환"""
    model_config = ConfigDict(extra='ignore', populate_by_name=True)

    storyId: str = Field(validation_alias=AliasChoices('storyId', 'story_id'))
    storyTitle: Optional[str] = Field(default=None, validation_alias=AliasChoices('storyTitle', 'story_title'))
    storyDescription: Optional[str] = Field(default=None, validation_alias=AliasChoices('storyDescription', 'story_description'))
    childId: int = Field(validation_alias=AliasChoices('childId', 'child_id'))
    emotion: Optional[str] = None
    interests: Optional[List[str]] = None
    sceneNumber: int = Field(validation_alias=AliasChoices('sceneNumber', 'scene_number'))
    previousChoices: Optional[List[Dict[str, Any]]] = Field(default_factory=list, validation_alias=AliasChoices('previousChoices', 'previous_choices'))


class AnalyzeCustomChoiceRequest(BaseModel):
    model_config = ConfigDict(extra='ignore', populate_by_name=True)

    completionId: int | str = Field(validation_alias=AliasChoices('completionId', 'completion_id'))
    sceneNumber: int = Field(validation_alias=AliasChoices('sceneNumber', 'scene_number'))
    text: str


class GenerateImageRequest(BaseModel):
    model_config = ConfigDict(extra='ignore', populate_by_name=True)

    prompt: str
    size: Optional[str] = "1024x1024"


class CreateImagePromptRequest(BaseModel):
    model_config = ConfigDict(extra='ignore', populate_by_name=True)

    koreanText: str = Field(validation_alias=AliasChoices('koreanText', 'korean_text'))
    maxLength: Optional[int] = Field(default=150, validation_alias=AliasChoices('maxLength', 'max_length'))
    characterDescription: Optional[str] = Field(default=None, validation_alias=AliasChoices('characterDescription', 'character_description'))
    storyId: Optional[str] = Field(default=None, validation_alias=AliasChoices('storyId', 'story_id'))  # [2025-11-05 추가] 캐릭터 설명 자동 조회용


# ==================== 폴백 유틸 ====================

def _fallback_first_scene(story_id: str, child_name: Optional[str]) -> Scene:
    logger.debug(f"폴백 첫장면 생성 story_id={story_id}, child_name={child_name}")
    text = f"'{story_id}' 이야기가 시작돼요. {child_name or '주인공'}가 첫 걸음을 내딛습니다."
    return Scene(
        sceneNumber=1,
        text=text,
        choices=[
            Choice(id="c1", label="용기를 내본다", abilityType="용기", abilityPoints=2),
            Choice(id="c2", label="친구에게 묻는다", abilityType="우정", abilityPoints=2),
            Choice(id="c3", label="천천히 관찰한다", abilityType="창의성", abilityPoints=2),
        ],
    )


def _fallback_next_scene(scene_number: int, story_title: str, previous_choices: list) -> Dict[str, Any]:
    logger.debug(f"폴백 다음장면 생성 scene={scene_number}, story_title={story_title}")
    is_ending = scene_number >= 8  # 8장면 이상이면 종료

    # 이전 선택 기반 간단한 텍스트 생성
    if previous_choices:
        last_choice = previous_choices[-1]
        ability = last_choice.get("abilityType", "능력")
        txt = f"{scene_number}번째 장면입니다. '{story_title}' 이야기가 계속됩니다. {ability} 능력이 성장하고 있어요."
    else:
        txt = f"'{story_title}' 이야기가 시작됩니다. 첫 번째 선택을 해보세요!"

    if is_ending:
        txt += " 이제 이야기가 끝나가고 있어요."

    scene = Scene(
        sceneNumber=scene_number,
        text=txt,
        choices=[] if is_ending else [
            Choice(id=f"c{scene_number}1", choiceId=f"c{scene_number}1", label="용기있게 행동하기", abilityType="용기", abilityPoints=2),
            Choice(id=f"c{scene_number}2", choiceId=f"c{scene_number}2", label="친구와 함께하기", abilityType="우정", abilityPoints=2),
            Choice(id=f"c{scene_number}3", choiceId=f"c{scene_number}3", label="창의적으로 해결하기", abilityType="창의성", abilityPoints=2),
        ],
    )
    return {"scene": scene, "isEnding": is_ending}


def _scene_from_payload(payload: Dict[str, Any]) -> Scene:
    scene_number = payload.get("sceneNumber") or payload.get("scene_number") or payload.get("number") or 1
    text = payload.get("text") or payload.get("content")
    raw_choices = payload.get("choices") or []
    choices = [
        Choice(
            id=str(ch.get("id") or ch.get("choiceId") or ch.get("choice_id") or ""),
            choiceId=str(ch.get("choiceId") or ch.get("choice_id") or ch.get("id") or ""),
            label=ch.get("label") or ch.get("choiceText") or ch.get("choice_text") or ch.get("text") or "선택",
            text=ch.get("text") or ch.get("choiceText") or ch.get("choice_text"),
            abilityType=ch.get("abilityType") or ch.get("ability_type"),
            abilityPoints=ch.get("abilityPoints") or ch.get("ability_points") or ch.get("abilityScore") or ch.get("ability_score") or 0,
            abilityScore=ch.get("abilityScore") or ch.get("ability_score") or ch.get("abilityPoints") or ch.get("ability_points") or 0,
        )
        for ch in raw_choices
    ]
    return Scene(sceneNumber=int(scene_number), text=text, content=text, choices=choices)


# ==================== 엔드포인트 ====================

@router.post("/recommend-stories")
async def recommend_stories(req: RecommendStoriesRequest):
    logger.info(f"추천 요청: emotion={req.emotion}, interests={req.interests}")
    try:
        if StorySearchService:
            svc = StorySearchService()
            items = await svc.recommend_stories_async(req.emotion, req.interests or [], req.childId, req.limit)
            logger.info(f"추천 결과 {len(items)}건")
            # return {"items": items}
            return items
        
        logger.info("StorySearchService 없음 → 폴백 사용")
        samples = [
            {"storyId": "new_sibling", "title": "새 동생과의 하루"},
            {"storyId": "brave_little_star", "title": "작은 별의 용기"},
            {"storyId": "forest_friends", "title": "숲속 친구들"},
        ]
        # return {"items": samples[: req.limit]}
        return samples[: req.limit]
    except Exception as e:
        logger.exception("recommend-stories 실패")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-story", response_model=GenerateStoryResponse)
async def generate_story(req: GenerateStoryRequest):
    logger.info(f"스토리 생성 요청: storyId={req.storyId}")
    try:
        story_id = req.storyId
        body = req.body or GenerateStoryBody(childId=0)
        child_name = body.childName or "아이"

        first_scene: Optional[Scene] = None
        if OpenAIService:
            try:
                llm = OpenAIService()
                prompt = f"{story_id} 이야기를 300자 이내로. 주인공: {child_name}, 감정: {body.emotion or '중립'}."
                out = await llm.generate_text_async(prompt)
                first_scene = Scene(sceneNumber=1, text=out.strip())
                logger.info("OpenAI LLM 스토리 생성 성공")
            except Exception as e:
                logger.warning(f"OpenAI 실패, 폴백 사용: {e}")
                first_scene = _fallback_first_scene(story_id, child_name)
        else:
            logger.info("OpenAIService 없음 → 폴백 사용")
            first_scene = _fallback_first_scene(story_id, child_name)

        completion_id = int(time.time() * 1000) + random.randint(0, 999)
        resp = GenerateStoryResponse(
            completionId=completion_id,
            story={"title": story_id.replace("_", " ").title(), "scenes": [first_scene.model_dump()]},
            firstScene=first_scene,
        )
        logger.info(f"스토리 생성 완료 id={completion_id}")
        return resp
    except Exception as e:
        logger.error(f"generate-story 실패: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-next-scene")
async def generate_next_scene(req: NextSceneRequest):
    logger.info(f"다음 장면 요청: storyId={req.storyId}, scene={req.sceneNumber}, childId={req.childId}")
    try:
        # OpenAI 서비스 사용하여 분기형 스토리 생성
        if OpenAIService:
            try:
                llm = OpenAIService()

                # 이전 선택들로부터 스토리 맥락 구축
                story_context = ""
                for choice in req.previousChoices or []:
                    scene_no = choice.get("sceneNumber", "")
                    choice_text = choice.get("choiceText", "")
                    if choice_text:
                        story_context += f"Scene {scene_no}: {choice_text}\n"

                # [2025-11-05 추가] 캐릭터 설명 가져오기
                character_description = CHARACTER_DESCRIPTIONS.get(req.storyId)
                logger.info(f"캐릭터 설명 조회: storyId={req.storyId}, found={'Yes' if character_description else 'No'}")

                # [2025-10-28 수정] Story의 title과 description을 OpenAI로 전달
                # [2025-11-05 수정] character_description 추가
                # childName은 제거 - 동화 주인공으로 사용하지 않음
                result = await llm.generate_next_scene_async(
                    story_id=req.storyId,
                    story_title=req.storyTitle or req.storyId,
                    story_description=req.storyDescription or "",
                    emotion=req.emotion or "중립",
                    interests=req.interests or [],
                    scene_number=req.sceneNumber,
                    previous_choices=req.previousChoices or [],
                    story_context=story_context if story_context else None,
                    character_description=character_description
                )

                # Scene 객체로 변환
                scene = _scene_from_payload(result["scene"])
                is_ending = result.get("isEnding", False)

                # # [2025-10-30 김광현] 이미지 생성 추가
                # image_url = None
                # # OpenAI가 생성한 imagePrompt를 우선 사용
                # image_prompt = result["scene"].get("imagePrompt") or scene.text

                # if image_prompt:
                #     try:
                #         # imagePrompt가 한글이면 영어로 번역 필요
                #         if any('\uac00' <= c <= '\ud7a3' for c in image_prompt):
                #             # 씬 내용을 기반으로 영어 프롬프트 생성
                #             scene_content = result["scene"].get("content") or scene.text or ""
                #             image_prompt = f"Children's book illustration style, warm and friendly atmosphere, featuring the story: {scene_content[:100]}"

                #         logger.info(f"씬 {req.sceneNumber} 이미지 생성 시작: {image_prompt[:100]}...")
                #         image_url = await llm.generate_image_async(image_prompt, size="1024x1024")
                #         logger.info(f"씬 {req.sceneNumber} 이미지 생성 완료: {image_url[:80]}...")
                #     except Exception as img_error:
                #         logger.warning(f"이미지 생성 실패 (계속 진행): {img_error}")
                #         # 이미지 생성 실패해도 스토리는 계속 진행

                logger.info(f"OpenAI로 다음 장면 생성 완료 scene={scene.sceneNumber}, isEnding={is_ending}")

                # [2025-11-05 추가] 첫 번째 씬일 때 캐릭터 설명 저장
                if req.sceneNumber == 1 and result.get("characterDescription"):
                    CHARACTER_DESCRIPTIONS[req.storyId] = result["characterDescription"]
                    logger.info(f"캐릭터 설명 저장됨: storyId={req.storyId}, characterDescription={result['characterDescription']}")

                # [2025-10-30 김광현] storyTitle이 있으면 응답에 포함
                response = {"scene": scene.model_dump(), "isEnding": is_ending}

                # if image_url:
                #     response["imageUrl"] = image_url

                if result.get("storyTitle"):
                    response["storyTitle"] = result["storyTitle"]
                    logger.info(f"동화 제목 포함: {result['storyTitle']}")

                return response

            except Exception as e:
                logger.warning(f"OpenAI 실패, 폴백 사용: {e}")
                # 폴백으로 처리
                result = _fallback_next_scene(req.sceneNumber, req.storyTitle or "동화", req.previousChoices or [])
                scene_payload = result["scene"].model_dump() if isinstance(result["scene"], Scene) else result["scene"]
                scene = _scene_from_payload(scene_payload)
                return {"scene": scene.model_dump(), "isEnding": bool(result.get("isEnding", False))}
        else:
            logger.info("OpenAIService 없음 → 폴백 사용")
            result = _fallback_next_scene(req.sceneNumber, req.storyTitle or "동화", req.previousChoices or [])
            scene_payload = result["scene"].model_dump() if isinstance(result["scene"], Scene) else result["scene"]
            scene = _scene_from_payload(scene_payload)
            return {"scene": scene.model_dump(), "isEnding": bool(result.get("isEnding", False))}

    except Exception as e:
        logger.error(f"generate-next-scene 실패: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# @router.post("/analyze-custom-choice")
# async def analyze_custom_choice(req: AnalyzeCustomChoiceRequest):
#     logger.info(f"선택 분석 요청: text={req.text}")
#     try:
#         # OpenAI를 사용한 정교한 분석
#         llm = OpenAIService()
#         if llm and llm.client:
#             try:
#                 prompt = f"""
#                 사용자가 동화에서 입력한 선택지를 분석하여 어떤 능력치가 향상되는지 판단해주세요.

#                 선택지: "{req.text}"
                                
#                 **1단계: 부정적 표현 검사**
#                 다음과 같은 부정적 표현이 포함되어 있는지 확인하세요:
#                 - 폭력적 표현 (때리기, 죽이기, 부수기, 싸우기)
#                 - 욕설이나 비속어
#                 - 타인을 해치는 행동 (괴롭히기, 무시하기, 따돌리기)
#                 - 극단적으로 부정적인 감정 (미워하기, 싫어하기, 증오하기)
#                 - 위험한 행동 제안 (불장난, 높은 곳에서 뛰어내리기 등)

#                 **2단계: 능력치 분석 (부정적이지 않은 경우)**
#                 다음 5가지 능력치 중 가장 적합한 것을 선택하세요:
#                 - 용기: 두려움을 극복하고 도전하는 행동
#                 - 공감: 다른 사람의 감정을 이해하고 배려하는 행동
#                 - 창의성: 새로운 아이디어나 독특한 해결책을 제시하는 행동
#                 - 책임감: 의무를 다하고 약속을 지키는 행동
#                 - 우정: 친구와의 관계를 중요시하고 함께하는 행동

#                 **점수 부여 기준:**
#                 - 매우 창의적이거나 긍정적인 선택: 15점
#                 - 적절하고 좋은 선택: 12점  
#                 - 평범하지만 긍정적인 선택: 10점

#                 JSON 형식으로 응답해주세요:
#                 {{
#                     "isNegative": true/false,
#                     "negativeReason": "부정적인 경우 구체적인 이유 (한글로)",
#                     "feedback": "사용자에게 보여줄 메시지",
#                     "abilityType": "능력치 이름 (부정적이지 않은 경우)",
#                     "abilityPoints": 점수 10~15 (부정적이지 않은 경우),
#                     "reason": "능력치 선정 이유"
#                 }}
#                 """

#                 response = llm.client.chat.completions.create(
#                     model="gpt-4o-mini",
#                     messages=[{"role": "user", "content": prompt}],
#                     response_format={"type": "json_object"},
#                     temperature=0.7
#                 )

#                 result = json.loads(response.choices[0].message.content)

#                 # 부정문 체크
#                 is_negative = result.get("isNegative", False)

#                 if is_negative:
#                     negative_reason = result.get("negativeReason", "")
#                     feedback = result.get("feedback", "부정적인 표현이 감지되었습니다.")
#                     logger.info(f"부정문 감지: {req.text} - 이유: {negative_reason}")

#                     return {
#                         "isNegative": True,
#                         "feedback": feedback,
#                         "negativeReason": negative_reason,
#                         "abilityType": None,
#                         "abilityPoints": 0
#                     }

#                 # 긍정적인 경우 능력치 반환
#                 ability_type = result.get("abilityType", "책임감")
#                 ability_points = result.get("abilityPoints", 12)  # 기본값 12로 상향
#                 reason = result.get("reason", "")

#                 # 커스텀 선택지 보너스 +2점(최대는 17점으로 제한함)
#                 ability_points = min(ability_points + 2, 17)

#                 logger.info(f"AI 분석 결과 (커스텀 보너스 포함): {ability_type} +{ability_points}")

#                 return {
#                     "isNegative": False,
#                     "abilityType": ability_type,
#                     "abilityPoints": ability_points,
#                     "feedback": f"와! 정말 멋진 선택이에요! {ability_type} 능력이 크게 성장했어요 (+{ability_points}점)"
#                 }

#             except Exception as e:
#                 logger.warning(f"OpenAI 분석 실패, 폴백 사용: {e}")
#                 # 폴백으로 기존 키워드 매칭 사용

#                 # 폴백: 키워드 매칭
#                 txt = req.text.lower()

#                 # 부정적 키워드 체크
#                 negative_keywords = [
#                     # 폭력/공격
#                     "때리", "패", "죽이", "죽여", "부수", "찌르", "찔러", "폭행", "폭력",
#                     "싸우", "싸움", "밀치", "발로", "주먹", "때림", "혼내", "혼낼",
#                     "덤벼", "때릴", "때렸", "찌를", "차버", "발로차", "하대",

#                     # 괴롭힘/왕따
#                     "괴롭히", "놀려", "놀림", "비웃", "무시", "따돌리", "왕따", "업신",
#                     "멸시", "따돌림", "깎아내", "욕보",

#                     # 욕설/비하
#                     "욕", "쌍욕", "욕설", "나쁜말", "바보", "멍청", "멍청이", "미친",
#                     "또라이", "변태", "바보야", "멍청아", "쓰레기", "개같", "개색", 
#                     "하찮", "저질", "재수없", "못생", "뚱뚱", "게으르",

#                     # 미움/증오
#                     "미워", "싫어", "증오", "저주", "혐오", "싫다", "미워하",

#                     # 명령형 / 공격 의도 표현
#                     "죽어", "꺼져", "사라져", "입닥쳐", "닥쳐", "조용히해", "가버려",

#                     # 약한 욕/아이들이 자주 쓰는 표현
#                     "바보같", "멍텅", "멍청", "멍청하", "멍충", "멍텅구리",
#                     "못해", "못하", "너때문", "이상해", "무식",

#                     # 줄임/변형 표현 (필터링용)
#                     "ㅂㅅ", "ㅅㅂ", "ㅈㄹ", "ㄷㅊ", "ㅁㅊ", "미쳣", "미첬", "븅",
#                 ]

#                 if any(k in txt for k in negative_keywords):
#                     return {
#                         "isNegative": True,
#                         "feedback": "친구를 존중하고 배려하는 선택을 해보면 어떨까요?",
#                         "negativeReason": "부정적인 표현이 포함되어 있습니다",
#                         "abilityType": None,
#                         "abilityPoints": 0
#                     }

#                 # 긍정적인 경우 기존 로직 (점수 상향)
#                 if any(k in txt for k in ["용기", "brave", "courage", "도전", "혼자", "앞으로"]):
#                     ability, pts = "용기", 12
#                 elif any(k in txt for k in ["친구", "friend", "우정", "같이", "함께"]):
#                     ability, pts = "우정", 12
#                 elif any(k in txt for k in ["아이디어", "idea", "창의", "발명", "만들"]):
#                     ability, pts = "창의성", 12
#                 elif any(k in txt for k in ["공감", "empathy", "이해", "위로", "도와"]):
#                     ability, pts = "공감", 12
#                 else:
#                     ability, pts = "책임감", 10

#                 feedback = f"스스로 생각한 선택이 정말 훌륭해요! {ability} 능력이 크게 성장했어요 (+{pts}점)"
#                 logger.info(f"키워드 매칭 결과: ability={ability}, pts={pts}")
#                 return {"abilityType": ability, "abilityPoints": pts, "feedback": feedback}
                        
#     except Exception as e:
#         logger.error(f"analyze-custom-choice 실패: {e}\n{traceback.format_exc()}")
#         raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-custom-choice")
async def analyze_custom_choice(req: AnalyzeCustomChoiceRequest):
    logger.info(f"선택 분석 요청: text={req.text}")

    txt = req.text.lower()

    PRODUCTION_PROMPT = f"""
        당신은 어린이 동화 속 선택지를 분석하는 전문 평가자입니다.
        사용자가 입력한 선택지가 어떤 능력치와 관련되는지 정확하게 판단하세요.

        # 1단계: 부정적 표현 여부 판단
        다음 범주에 해당하면 무조건 "isNegative": true 로 처리하세요.

        ## [부정 표현 기준]
        - 폭력: 때리기, 죽이기, 패기, 부수기, 찌르기, 위협하기 등
        - 괴롭힘: 놀리기, 비웃기, 무시하기, 따돌리기
        - 욕설/비하: 욕설, 비속어, 모욕, 심한 비난
        - 감정적 공격: 미워하기, 싫어하기, 증오, 혐오, 저주
        - 위험한 행동: 불장난, 뛰어내리기, 위험 유도

        부정적이면 즉시:
        {{
            "isNegative": true,
            "negativeReason": "왜 부정인지 한 문장으로 설명",
            "feedback": "아이에게 부드럽게 안내하는 문장"
        }}
        만 출력하세요.

        # 2단계: 긍정 능력 판단 (부정이 아닌 경우만)
        다음 5가지 중 가장 적합한 능력을 선택하세요.

        ### 용기 (Courage)
        - 두려움 극복 / 도전 / 첫걸음 / 혼자 시도
        ### 공감 (Empathy)
        - 이해 / 위로 / 배려 / 감정 공감
        ### 창의성 (Creativity)
        - 아이디어 / 발명 / 새로운 시도 / 독창적 해결
        ### 책임감 (Responsibility)
        - 스스로 해결 / 약속 / 정리 / 맡은 일 수행
        ### 우정 (Friendship)
        - 친구와 협력 / 함께하기 / 관계 유지

        # 점수 규칙:
        - 매우 훌륭한 선택 → 15점
        - 명확하게 긍정적인 선택 → 12점
        - 기본적으로 괜찮은 선택 → 10점

        # 출력(JSON만)
        {{
        "isNegative": false,
        "abilityType": "능력 1개",
        "abilityPoints": 10~15 정수,
        "reason": "이 능력치를 선택한 이유",
        "feedback": "아이에게 보여줄 칭찬 문장"
        }}

        선택지: "{req.text}"
        """

    NEGATIVE_KEYWORDS = {
        "strong": [
            "죽이", "죽여", "패버", "패주", "패죽", "찌르", "폭행", "부숴",
            "욕해", "욕함", "ㅅㅂ", "ㅈㄹ", "개새", "sex", "죽어"
        ],
        "medium": [
            "때리", "패", "싸우", "밀치", "주먹", "혼내", "놀림", "따돌",
            "비웃", "변태", "쓰레기", "혐오", "증오", "미워"
        ],
        "weak": [
            "바보", "멍청", "못생", "뚱뚱", "이상해", "무식", "재수없",
            "싫어", "싫다", "미워하", "하찮", "저질", "므흣"
        ]
    }

    POSITIVE_KEYWORDS = {
        "용기": {
            "strong": ["도전", "용감", "두렵지만", "첫걸음", "해볼게", "포기하지"],
            "medium": ["용기", "brave", "혼자서", "앞으로 나아"],
            "weak": ["시도", "가볼게", "해볼까", "무서워도"]
        },
        "공감": {
            "strong": ["위로해", "마음 알아", "힘들었겠다", "슬펐겠다", "배려해"],
            "medium": ["공감", "이해해", "empathy", "도와줄게"],
            "weak": ["괜찮아", "힘내", "도와"]
        },
        "창의성": {
            "strong": ["새로운 아이디어", "발명", "창조해", "기발한"],
            "medium": ["아이디어", "창의", "독창", "상상"],
            "weak": ["생각해냈", "만들어"]
        },
        "책임감": {
            "strong": ["약속 지킬게", "맡은 일", "스스로 해결"],
            "medium": ["정리했", "정리할게", "책임감", "챙겨"],
            "weak": ["스스로", "도와줄게"]
        },
        "우정": {
            "strong": ["같이 도와", "함께 해결", "친구 지켜", "협력했어"],
            "medium": ["친구", "friend", "우정", "함께", "같이"],
            "weak": ["같아", "둘이서"]
        }
    }

    # 1) 부정 키워드 폴백 체크 (LLM 실패 대비)
    def check_negative(text: str):
        for level, keywords in NEGATIVE_KEYWORDS.items():
            if any(k in text for k in keywords):
                return True, level
        return False, None

    # 2) 긍정 키워드 매칭
    def match_positive(text: str):
        for ability, levels in POSITIVE_KEYWORDS.items():
            if any(k in text for k in levels["strong"]):
                return ability, 12, "strong"
            if any(k in text for k in levels["medium"]):
                return ability, 10, "medium"
            if any(k in text for k in levels["weak"]):
                return ability, 8, "weak"
        return "책임감", 10, "fallback"

    try:
        llm = OpenAIService()
        if llm and llm.client:
            try:
                prompt = PRODUCTION_PROMPT

                response = llm.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )

                result = json.loads(response.choices[0].message.content)

                # 부정일 경우 바로 반환
                if result.get("isNegative", False):
                    return {
                        "isNegative": True,
                        "negativeReason": result.get("negativeReason", ""),
                        "feedback": result.get("feedback", "부정적인 표현이 있어요!"),
                        "abilityType": None,
                        "abilityPoints": 0
                    }

                # 긍정 결과 + 커스텀 보너스
                ability_type = result.get("abilityType", "책임감")
                ability_points = min(result.get("abilityPoints", 12) + 2, 17)

                return {
                    "isNegative": False,
                    "abilityType": ability_type,
                    "abilityPoints": ability_points,
                    "feedback": result.get(
                        "feedback",
                        f"정말 멋진 선택이에요! {ability_type} 능력이 자랐어요."
                    )
                }

            except Exception as e:
                logger.warning(f"OpenAI 분석 실패 → 폴백 적용: {e}")

    except Exception as e:
        logger.error(f"LLM 초기화 실패 → 폴백 사용: {e}")

    # LLM 실패 시 — 강력한 폴백 로직
    is_neg, level = check_negative(txt)
    if is_neg:
        return {
            "isNegative": True,
            "negativeReason": f"{level} 수준의 부정적 표현 포함",
            "feedback": "친구를 존중하고 배려하는 선택을 해보면 좋겠어요!",
            "abilityType": None,
            "abilityPoints": 0
        }

    ability, base_pts, intensity = match_positive(txt)
    pts = base_pts + 2  # 커스텀 선택 보너스
    pts = min(pts, 17)

    return {
        "isNegative": False,
        "abilityType": ability,
        "abilityPoints": pts,
        "feedback": f"멋진 선택이에요! {ability} 능력이 자랐어요 (+{pts}점)"
    }


@router.post("/generate-image")
async def generate_image(req: GenerateImageRequest):
    logger.info(f"이미지 생성 요청: prompt={req.prompt}, size={req.size}")
    try:
        if OpenAIService():
            try:
                # [2025-10-30 김광현] 이미지 사용하기 위해 코드 변경
                llm = OpenAIService()
                image_url = await llm.generate_image_async(req.prompt, req.size or "1024x1024")
                logger.info(f"DALE-E 이미지 생성 완료 : {image_url}")
                return {"url": image_url, "prompt": req.prompt, "size": req.size}
            except Exception as e:
                logger.warning(f"DALL-E 실패, 더미 이미지 사용: {e}")
                # 폴백: 더미 이미지
                dummy_url = f"https://picsum.photos/seed/{hash(req.prompt) % 100000}/{req.size}"
                return {"url": dummy_url, "prompt": req.prompt, "size": req.size}
        else:
            #  더미 이미지
            dummy_url = f"https://picsum.photos/seed/{hash(req.prompt) % 100000}/{req.size}"
            logger.info(f"이미지 생성 완료: {dummy_url}")
            return {"url": dummy_url, "prompt": req.prompt, "size": req.size}
        
    except Exception as e:
        logger.error(f"generate-image 실패: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    


@router.post("/create-image-prompt")
async def create_image_prompt(req: CreateImagePromptRequest):
    """
    한글 동화 내용을 이미지 생성에 적합한 짧은 영어 프롬프트로 변환

    Args:
        koreanText: 한글 동화 내용
        maxLength: 최대 프롬프트 길이 (기본값: 150자)
        characterDescription: 주인공 캐릭터 설명 (선택)
        storyId: 스토리 ID (characterDescription 자동 조회용, 선택)

    Returns:
        영어 이미지 프롬프트
    """
    logger.info(f"이미지 프롬프트 생성 요청: {req.koreanText[:50]}...")

    try:
        if OpenAIService:
            try:
                llm = OpenAIService()

                # [2025-11-05 추가] storyId가 있으면 캐릭터 설명 자동 조회
                character_description = req.characterDescription
                if not character_description and req.storyId:
                    character_description = CHARACTER_DESCRIPTIONS.get(req.storyId)
                    if character_description:
                        logger.info(f"storyId={req.storyId}로 캐릭터 설명 자동 조회 성공")

                # [2025-11-05 김민중 수정] 일관된 anime style 적용 및 캐릭터 일관성 강화
                # [2025-11-05 추가] characterDescription이 제공되면 이를 프롬프트에 포함
                character_info = ""
                if character_description:
                    character_info = f"\n**주인공 캐릭터 (필수 포함):** {character_description}"

                prompt = f"""
                다음 한글 동화 내용을 이미지 생성 AI(PollinationAI)가 이해할 수 있는 짧고 효과적인 영어 프롬프트로 변환해주세요.

                **한글 동화 내용:**
                {req.koreanText}
{character_info}

**요구사항:**
1. 핵심 시각적 요소만 추출 (캐릭터, 배경, 분위기, 행동)
2. 최대 {req.maxLength}자 이내의 영어로 작성
3. **필수 스타일**: 반드시 "consistent anime art style, Studio Ghibli inspired, kawaii" 포함
4. **캐릭터 일관성 (매우 중요)**:
   - 주인공 캐릭터는 매번 동일하게 묘사: "same character design"
   - {"제공된 캐릭터 설명을 반드시 그대로 사용: " + req.characterDescription if req.characterDescription else "외모를 구체적으로 고정: a cute child with [구체적 특징]"}
   - 캐릭터의 종류(토끼, 사람, 곰 등)와 외모 특징을 매 장면마다 정확히 동일하게 유지
5. **금지 사항**: realistic, photorealistic, real photo, 3D render 같은 실사/3D 스타일 절대 사용 금지
6. **색상 일관성**: "soft pastel color palette, consistent color scheme" 반드시 포함
7. PollinationAI가 이해하기 쉬운 간결한 문장

**좋은 예시:**
- "A cute white rabbit with pink ears walking through a magical forest, consistent anime art style, Studio Ghibli inspired, kawaii, soft pastel colors, same character design"
- "The same white rabbit with pink ears helping a small bird, same character design, anime style, gentle atmosphere, consistent color scheme"

**나쁜 예시 (사용 금지):**
- "realistic portrait" ❌
- "photorealistic rendering" ❌
- "3D cartoon style" ❌
- "different character design" ❌
- 캐릭터 종류가 바뀌는 경우 (토끼 → 사람) ❌

**출력 형식 (JSON):**
{{
    "imagePrompt": "영어 프롬프트 (최대 {req.maxLength}자, 반드시 consistent anime art style과 동일한 캐릭터 설명 포함)",
    "keyElements": ["주요 요소1", "주요 요소2", "주요 요소3"]
}}

                JSON 형식으로 응답해주세요.
                """

                # [2025-11-05 김민중 수정] 시스템 프롬프트에 캐릭터 일관성 강조
                system_prompt = "당신은 Studio Ghibli 스타일 애니메이션 이미지 프롬프트 작성 전문가입니다. 한글 텍스트에서 핵심 시각적 요소를 추출하여 짧고 효과적인 영어 프롬프트를 만듭니다. 반드시 'consistent anime art style, Studio Ghibli inspired, kawaii, same character design, soft pastel color palette' 키워드를 포함하고, 주인공 캐릭터의 외모는 항상 동일하게 유지합니다. realistic, photorealistic, 3D render 같은 실사/3D 스타일은 절대 사용하지 않습니다."
                if req.characterDescription:
                    system_prompt += f" 주인공 캐릭터는 반드시 '{req.characterDescription}' 로 고정하여 모든 장면에서 동일하게 유지해야 합니다. 캐릭터의 종류와 외모 특징을 절대 바꾸지 마세요."

                response = llm.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.5
                )

                result = json.loads(response.choices[0].message.content)
                image_prompt = result.get("imagePrompt", "")
                key_elements = result.get("keyElements", [])

                # [2025-11-05 김민중 수정] 필수 키워드 강제 추가
                required_keywords = ["anime", "consistent", "same character"]
                missing_keywords = []

                for keyword in required_keywords:
                    if keyword not in image_prompt.lower():
                        missing_keywords.append(keyword)

                if missing_keywords:
                    # 누락된 필수 키워드 추가
                    additional = ", consistent anime art style, same character design, Studio Ghibli inspired, soft pastel color palette"
                    image_prompt = image_prompt + additional
                    logger.info(f"필수 키워드 자동 추가됨: {', '.join(missing_keywords)}")

                # [2025-11-05 김민중 수정] 금지된 스타일 키워드 제거 및 대체
                forbidden_keywords = ["realistic", "photorealistic", "real photo", "photograph", "3d render", "3d cartoon"]
                for keyword in forbidden_keywords:
                    if keyword in image_prompt.lower():
                        image_prompt = image_prompt.replace(keyword, "anime style")
                        logger.warning(f"금지된 키워드 '{keyword}' 제거하고 anime style로 대체")

                # 프롬프트 길이 제한
                if len(image_prompt) > req.maxLength:
                    image_prompt = image_prompt[:req.maxLength].rsplit(' ', 1)[0]  # 마지막 단어가 잘리지 않도록

                logger.info(f"프롬프트 생성 완료: {image_prompt}")

                return {
                    "imagePrompt": image_prompt,
                    "keyElements": key_elements,
                    "originalLength": len(req.koreanText),
                    "promptLength": len(image_prompt)
                }

            except Exception as e:
                logger.warning(f"OpenAI 프롬프트 생성 실패, 폴백 사용: {e}")
                # 폴백: 간단한 변환

        # [2025-11-05 김민중 수정] 폴백 프롬프트에 캐릭터 일관성 키워드 추가
        # [2025-11-05 추가] storyId로 캐릭터 설명 조회 또는 제공된 characterDescription 사용
        character_description = req.characterDescription
        if not character_description and req.storyId:
            character_description = CHARACTER_DESCRIPTIONS.get(req.storyId)

        character_part = character_description if character_description else "A cute child character"
        fallback_prompt = f"{character_part}, consistent anime art style, Studio Ghibli inspired, same character design, kawaii, soft pastel color palette, warm and friendly atmosphere, {req.koreanText[:20]}"
        if len(fallback_prompt) > req.maxLength:
            fallback_prompt = fallback_prompt[:req.maxLength].rsplit(' ', 1)[0]

        logger.info(f"폴백 프롬프트 사용: {fallback_prompt}")
        return {
            "imagePrompt": fallback_prompt,
            "keyElements": ["children's book", "illustration", "anime style", "consistent character"],
            "originalLength": len(req.koreanText),
            "promptLength": len(fallback_prompt)
        }

    except Exception as e:
        logger.error(f"create-image-prompt 실패: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    logger.info("health check 요청")
    return {"status": "ok"}
