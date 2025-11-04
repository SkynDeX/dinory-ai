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

                # [2025-10-28 수정] Story의 title과 description을 OpenAI로 전달
                # childName은 제거 - 동화 주인공으로 사용하지 않음
                result = await llm.generate_next_scene_async(
                    story_id=req.storyId,
                    story_title=req.storyTitle or req.storyId,
                    story_description=req.storyDescription or "",
                    emotion=req.emotion or "중립",
                    interests=req.interests or [],
                    scene_number=req.sceneNumber,
                    previous_choices=req.previousChoices or [],
                    story_context=story_context if story_context else None
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


@router.post("/analyze-custom-choice")
async def analyze_custom_choice(req: AnalyzeCustomChoiceRequest):
    logger.info(f"선택 분석 요청: text={req.text}")
    try:
        # OpenAI를 사용한 정교한 분석
        llm = OpenAIService()
        if llm and llm.client:
            try:
                prompt = f"""
                사용자가 동화에서 입력한 선택지를 분석하여 어떤 능력치가 향상되는지 판단해주세요.

                선택지: "{req.text}"

                다음 5가지 능력치 중 가장 적합한 것을 선택하세요:
                - 용기: 두려움을 극복하고 도전하는 행동 (예: 혼자 해결, 앞으로 나아가기, 시도하기)
                - 공감: 다른 사람의 감정을 이해하고 배려하는 행동 (예: 위로하기, 도와주기, 함께 슬퍼하기)
                - 창의성: 새로운 아이디어나 독특한 해결책을 제시하는 행동 (예: 발명, 다른 방법 시도, 상상력)
                - 책임감: 의무를 다하고 약속을 지키는 행동 (예: 청소하기, 약속 지키기, 맡은 일 완수)
                - 우정: 친구와의 관계를 중요시하고 함께하는 행동 (예: 친구 찾기, 같이 놀기, 도움 요청)

                JSON 형식으로 응답해주세요:
                {{
                "abilityType": "능력치 이름 (용기/공감/창의성/책임감/우정)",
                "abilityPoints": 점수 (1~3),
                "reason": "이유 설명"
                }}
                """
                
                response = llm.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )
                
                result = json.loads(response.choices[0].message.content)
                ability_type = result.get("abilityType", "책임감")
                ability_points = result.get("abilityPoints", 2)
                reason = result.get("reason", "")
                
                feedback = f"'{req.text}' → {ability_type} +{ability_points} ({reason})"
                logger.info(f"AI 분석 결과: {ability_type} +{ability_points}")
                
                return {
                    "abilityType": ability_type,
                    "abilityPoints": ability_points,
                    "feedback": feedback
                }
                
            except Exception as e:
                logger.warning(f"OpenAI 분석 실패, 폴백 사용: {e}")
                # 폴백으로 기존 키워드 매칭 사용
        
        # 폴백: 키워드 매칭
        txt = req.text.lower()
        if any(k in txt for k in ["용기", "brave", "courage", "도전", "혼자", "앞으로"]):
            ability, pts = "용기", 2
        elif any(k in txt for k in ["친구", "friend", "우정", "같이", "함께"]):
            ability, pts = "우정", 2
        elif any(k in txt for k in ["아이디어", "idea", "창의", "발명", "만들"]):
            ability, pts = "창의성", 2
        elif any(k in txt for k in ["공감", "empathy", "이해", "위로", "도와"]):
            ability, pts = "공감", 2
        else:
            ability, pts = "책임감", 1

        feedback = f"선택이 {ability}에 긍정적 영향을 줍니다. +{pts}"
        logger.info(f"키워드 매칭 결과: ability={ability}, pts={pts}")
        return {"abilityType": ability, "abilityPoints": pts, "feedback": feedback}
        
    except Exception as e:
        logger.error(f"analyze-custom-choice 실패: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


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

    Returns:
        영어 이미지 프롬프트
    """
    logger.info(f"이미지 프롬프트 생성 요청: {req.koreanText[:50]}...")

    try:
        if OpenAIService:
            try:
                llm = OpenAIService()

                prompt = f"""
다음 한글 동화 내용을 이미지 생성 AI(PollinationAI)가 이해할 수 있는 짧고 효과적인 영어 프롬프트로 변환해주세요.

**한글 동화 내용:**
{req.koreanText}

**요구사항:**
1. 핵심 시각적 요소만 추출 (캐릭터, 배경, 분위기, 행동)
2. 최대 {req.maxLength}자 이내의 영어로 작성
3. 어린이 동화책 일러스트레이션 스타일 명시
4. 구체적이고 명확한 묘사
5. PollinationAI가 이해하기 쉬운 간결한 문장

**좋은 예시:**
- "A cute little rabbit bravely walking through a magical forest, warm pastel colors, children's book illustration style"
- "A young boy helping a small bird with a broken wing, gentle and caring atmosphere, watercolor style"

**출력 형식 (JSON):**
{{
    "imagePrompt": "영어 프롬프트 (최대 {req.maxLength}자)",
    "keyElements": ["주요 요소1", "주요 요소2", "주요 요소3"]
}}

JSON 형식으로 응답해주세요.
"""

                response = llm.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "당신은 이미지 프롬프트 작성 전문가입니다. 한글 텍스트에서 핵심 시각적 요소를 추출하여 짧고 효과적인 영어 프롬프트를 만듭니다."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )

                result = json.loads(response.choices[0].message.content)
                image_prompt = result.get("imagePrompt", "")
                key_elements = result.get("keyElements", [])

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

        # 폴백: 간단한 키워드 추출 및 기본 프롬프트
        fallback_prompt = f"Children's book illustration, warm and friendly atmosphere, {req.koreanText[:50]}"
        if len(fallback_prompt) > req.maxLength:
            fallback_prompt = fallback_prompt[:req.maxLength].rsplit(' ', 1)[0]

        logger.info(f"폴백 프롬프트 사용: {fallback_prompt}")
        return {
            "imagePrompt": fallback_prompt,
            "keyElements": ["children's book", "illustration"],
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
