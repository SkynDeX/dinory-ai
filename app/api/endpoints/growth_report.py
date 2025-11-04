from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, AliasChoices, ConfigDict
from typing import List, Dict, Any, Optional
import logging
import json

logger = logging.getLogger("dinory.growth_report")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[GROWTH_REPORT] %(asctime)s || %(levelname)s || %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

router = APIRouter(tags=["ai"])

try:
    from app.services.llm.openai_service import OpenAIService
except Exception as e:
    logger.warning(f"OpenAIService import 실패: {e}")
    OpenAIService = None

# ================== 모델 ================== 

class GrowthReportRequest(BaseModel):
    model_config = ConfigDict(extra='ignore', populate_by_name=True)

    beforeAbilities: Dict[str, float] = Field(validation_alias=AliasChoices('beforeAbilities', 'before_abilities'))
    afterAbilities: Dict[str, float] = Field(validation_alias=AliasChoices('afterAbilities', 'after_abilities'))
    strengths: Optional[List[Dict[str, Any]]] = []
    growthAreas: Optional[List[Dict[str, Any]]] = Field(default_factory=list, validation_alias=AliasChoices('growthAreas', 'growth_areas'))
    totalStories: int = Field(default=0, validation_alias=AliasChoices('totalStories', 'total_stories'))
    period: str = "month"

# ================== 엔드포인트 ==================     

@router.post("/generate-growth-evaluation")
async def generate_growth_evaluation(req: GrowthReportRequest):
    """AI 종합 평가 생성"""
    logger.info(f"성장 평가 생성 요청: period={req.period}, totalStories={req.totalStories}")
    try:
        if OpenAIService:
            llm = OpenAIService()

            # Before/After 능력치 비교
            before_text = "\n".join([f"- {k}: {v:.0f}점" for k, v in req.beforeAbilities.items()])
            after_text = "\n".join([f"- {k}: {v:.0f}점" for k, v in req.afterAbilities.items()])

            # 능력치 변화 계산
            changes = []
            for ability, after_score in req.afterAbilities.items():
                before_score = req.beforeAbilities.get(ability, 0)
                change = after_score - before_score
                if abs(change) > 5: # 5점 이상 변화만
                    changes.append(f"{ability}: {change:+.0f}점")

            changes_text = ", ".join(changes) if changes else "전반적으로 안정적"

            # 강점 영역
            strengths_text = ", ".join([s.get("area", "") for s in req.strengths[:2]]) if req.strengths else "없음"

            # 성장 가능 영역
            growth_areas_text = ", ".join([g.get("area", "") for g in req.growthAreas[:2]]) if req.growthAreas else "없음"

            period_map = {"month": "한 달", "quarter": "3개월", "halfyear": "6개월"}
            period_text = period_map.get(req.period, "한 달")

            prompt = f"""

당신은 아동심리전문상담가입니다. 아이의 {period_text}간 성장 리포트를 위한 따뜻하고 격려하는 종합 평가를 작성해주세요.

**기본 정보**:
- 완료한 동화: {req.totalStories}개
- 기간: {period_text}

**이전 능력치**:
{before_text}

**현재 능력치**:
{after_text}

**주요 변화**: {changes_text}
**강점 영역**: {strengths_text}
**성장 가능 영역**: {growth_areas_text}

조건:
1. 5-7문장으로 작성 (최소 200자 이상)
2. 각 능력치의 의미를 쉽게 풀어서 설명 (예: 용기 → 새로운 도전을 두려워하지 않는 마음)
3. 가장 크게 성장한 영역을 구체적인 예시와 함께 언급
4. 완료한 동화 개수를 바탕으로 아이의 노력 인정
5. 긍정적이고 성장 가능성에 초점을 맞춘 격려
6. 부모가 이해하기 쉬운 자연스러운 한국어
7. 평가문만 작성 (제목, 인사말, "~드립니다" 같은 결어 제외)
8. 데이터가 부족하더라도 아이의 잠재력과 가능성을 중심으로 풍부하게 작성
"""
            response = llm.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=600
            )

            evaluation = response.choices[0].message.content.strip()
            logger.info(f"AI 평가 생성 완료: {len(evaluation)}자")

            return {"evaluation": evaluation}
        
        else:
            # OpenAI 서비스 없을 때 풀백
            logger.info("OpenAIService 없음 -> 템플릿 사용")
            period_map = {"month": "한 달", "quarter": "3개월", "halfyear": "6개월"}
            fallback = f"이번 {period_map.get(req.period, '한 달')}간 아이는 {req.totalStories}개의 동화를 완료하여 긍정적인 성장을 보였습니다."
            if req.strengths:
                fallback += f" 특히 {req.strengths[0].get('area', '')} 영역에서 뛰어난 모습을 보여주었습니다."
            return {"evaluation": fallback}
            
    except Exception as e:
        logger.exception("generate-growth-evaluation 실패")
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/generate-growth-recommendations")
async def generate_growth_recommendations(req: GrowthReportRequest):
    """AI 기반 추천 활동 생성"""
    logger.info(f"추천 활동 생성 요청: growthAreas={len(req.growthAreas)}개")
    try:
        if OpenAIService:
            llm = OpenAIService()
            
            # 성장 가능 영역 정보
            if not req.growthAreas:
                logger.warning("성장 가능 영역 데이터 없음")
                return {"recommendations": []}
            
            growth_areas_info = "\n".join([
                f"- {g.get('area', '')}: {g.get('score', 0)}점 ({g.get('description', '')})"
                for g in req.growthAreas[:3]  # 최대 3개
            ])
            
            prompt = f"""
아이의 성장을 위한 맞춤 활동을 추천해주세요.

**성장 가능 영역**:
{growth_areas_info}

위 영역들을 고려하여 우선순위가 높은 순서로 3가지 활동을 추천해주세요.

다음 JSON 형식으로만 응답해주세요 (다른 설명 없이):
{{
  "recommendations": [
    {{
      "priority": 1,
      "activity": "활동 이름 (10자 이내)",
      "description": "구체적인 활동 설명 (40자 이내)",
      "targetArea": "타겟 능력치"
    }},
    {{
      "priority": 2,
      "activity": "활동 이름",
      "description": "활동 설명",
      "targetArea": "타겟 능력치"
    }},
    {{
      "priority": 3,
      "activity": "활동 이름",
      "description": "활동 설명",
      "targetArea": "타겟 능력치"
    }}
  ]
}}

조건:
1. 아이가 실제로 할 수 있는 구체적이고 재미있는 활동
2. 부모가 함께 할 수 있는 활동
3. 일상에서 쉽게 실천 가능
4. 정확한 JSON 형식 (쉼표, 괄호 주의)
"""
            
            response = llm.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            result = json.loads(response.choices[0].message.content)
            recommendations = result.get("recommendations", [])
            
            logger.info(f"추천 활동 생성 완료: {len(recommendations)}개")
            return {"recommendations": recommendations}
            
        else:
            # 폴백
            logger.info("OpenAIService 없음 → 기본 추천 사용")
            fallback_recs = []
            for i, area in enumerate(req.growthAreas[:3]):
                fallback_recs.append({
                    "priority": i + 1,
                    "activity": f"{area.get('area', '')} 향상 활동",
                    "description": f"아이와 함께 {area.get('area', '')} 능력을 키우는 활동을 해보세요.",
                    "targetArea": area.get('area', '')
                })
            return {"recommendations": fallback_recs}
            
    except Exception as e:
        logger.exception("generate-growth-recommendations 실패")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-growth-area-descriptions")
async def generate_growth_area_descriptions(req: GrowthReportRequest):
    """성장 가능 영역에 대한 구체적인 설명과 추천 생성"""
    logger.info(f"성장 영역 설명 생성 요청: {len(req.growthAreas)}개")
    try:
        if not OpenAIService or not req.growthAreas:
            return {"descriptions": []}

        llm = OpenAIService()
        results = []

        for area_info in req.growthAreas[:3]:
            area_name = area_info.get("area", "")
            score = area_info.get("score", 0)

            prompt = f"""
아이의 {area_name} 능력(현재 {score}점)을 발전시키기 위한 구체적인 설명과 추천을 작성해주세요.

다음 JSON 형식으로만 응답하세요:
{{
  "description": "{area_name}의 의미를 쉽게 설명하고, 왜 중요한지 30자 이내로",
  "recommendation": "부모가 아이와 함께 할 수 있는 구체적인 활동 1가지를 40자 이내로"
}}

조건:
1. 아동 발달 심리학 관점에서 작성
2. 실생활에서 바로 실천 가능한 내용
3. "~해보세요", "~하면 좋습니다" 등 부드러운 어조
"""

            try:
                response = llm.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )

                result = json.loads(response.choices[0].message.content)
                results.append({
                    "area": area_name,
                    "score": score,
                    "description": result.get("description", f"{area_name} 영역을 더 발전시킬 수 있습니다."),
                    "recommendation": result.get("recommendation", f"{area_name} 관련 동화를 함께 읽어보세요.")
                })
            except Exception as e:
                logger.warning(f"{area_name} 설명 생성 실패: {e}")
                results.append({
                    "area": area_name,
                    "score": score,
                    "description": f"{area_name} 영역을 더 발전시킬 수 있습니다.",
                    "recommendation": f"{area_name} 관련 동화를 함께 읽어보세요."
                })

        logger.info(f"성장 영역 설명 생성 완료: {len(results)}개")
        return {"descriptions": results}

    except Exception as e:
        logger.exception("generate-growth-area-descriptions 실패")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-milestones")
async def generate_milestones(req: GrowthReportRequest):
    """AI 기반 마일스톤 생성"""
    logger.info(f"마일스톤 생성 요청: totalStories={req.totalStories}")
    try:
        if not OpenAIService:
            return {"milestones": []}

        llm = OpenAIService()
        milestones = []

        # 1. 동화 완료 마일스톤
        if req.totalStories >= 5:
            prompt = f"""
아이가 {req.totalStories}개의 동화를 완료했습니다. 이 성취를 축하하는 마일스톤 문구를 작성해주세요.

다음 JSON 형식으로만 응답하세요:
{{
  "achievement": "축하 문구 (20자 이내, 구체적이고 감동적으로)"
}}

조건:
1. 아이의 노력과 꾸준함을 강조
2. 긍정적이고 격려하는 어조
3. 숫자를 자연스럽게 포함
"""
            try:
                response = llm.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )
                result = json.loads(response.choices[0].message.content)
                milestones.append({
                    "achievement": result.get("achievement", f"{req.totalStories}개의 동화를 완료했습니다"),
                    "date": None  # Spring Boot에서 설정
                })
            except Exception as e:
                logger.warning(f"동화 완료 마일스톤 생성 실패: {e}")

        # 2. 높은 능력치 마일스톤
        for ability, score in req.afterAbilities.items():
            if score >= 75:
                prompt = f"""
아이가 {ability} 능력에서 {score:.0f}점을 달성했습니다. 이 성취를 축하하는 마일스톤 문구를 작성해주세요.

다음 JSON 형식으로만 응답하세요:
{{
  "achievement": "축하 문구 (25자 이내, {ability}의 의미를 쉽게 풀어서)"
}}

조건:
1. 능력치 이름을 아이가 이해할 수 있는 말로 풀어서 설명
2. 점수를 자연스럽게 포함
3. 따뜻하고 격려하는 어조
"""
                try:
                    response = llm.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        temperature=0.7
                    )
                    result = json.loads(response.choices[0].message.content)
                    milestones.append({
                        "achievement": result.get("achievement", f"{ability} 능력 {score:.0f}점 달성"),
                        "date": None
                    })
                except Exception as e:
                    logger.warning(f"{ability} 마일스톤 생성 실패: {e}")

        logger.info(f"마일스톤 생성 완료: {len(milestones)}개")
        return {"milestones": milestones}

    except Exception as e:
        logger.exception("generate-milestones 실패")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-strength-descriptions")
async def generate_strength_descriptions(req: GrowthReportRequest):
    """강점 영역에 대한 구체적인 설명 생성"""
    logger.info(f"강점 설명 생성 요청: {len(req.strengths)}개")
    try:
        if not OpenAIService or not req.strengths:
            return {"descriptions": []}

        llm = OpenAIService()
        results = []

        for strength_info in req.strengths[:3]:
            area_name = strength_info.get("area", "")
            score = strength_info.get("score", 0)
            example = strength_info.get("example", "")

            prompt = f"""
아이가 {area_name} 능력에서 {score}점을 기록하며 뛰어난 모습을 보였습니다.
{f"예시: {example}" if example else ""}

이 강점을 칭찬하고 격려하는 설명을 작성해주세요.

다음 JSON 형식으로만 응답하세요:
{{
  "description": "{area_name}의 의미를 쉽게 설명하고, 왜 대단한지 40자 이내로"
}}

조건:
1. 아이의 성취를 구체적으로 칭찬
2. {area_name} 능력의 의미를 쉽게 풀어서 설명
3. 따뜻하고 격려하는 어조
"""

            try:
                response = llm.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )

                result = json.loads(response.choices[0].message.content)
                results.append({
                    "area": area_name,
                    "score": score,
                    "description": result.get("description", f"{area_name} 영역에서 뛰어난 능력을 보여줍니다."),
                    "example": example
                })
            except Exception as e:
                logger.warning(f"{area_name} 강점 설명 생성 실패: {e}")
                results.append({
                    "area": area_name,
                    "score": score,
                    "description": f"{area_name} 영역에서 뛰어난 능력을 보여줍니다.",
                    "example": example
                })

        logger.info(f"강점 설명 생성 완료: {len(results)}개")
        return {"descriptions": results}

    except Exception as e:
        logger.exception("generate-strength-descriptions 실패")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-example-description")
async def generate_example_description(request: Dict[str, Any]):
    """강점 예시를 자연스러운 문장으로 변환"""
    logger.info("예시 설명 생성 요청")
    try:
        story_title = request.get("storyTitle", "")
        choice_text = request.get("choiceText", "")
        ability = request.get("ability", "")

        if not OpenAIService or not story_title or not choice_text:
            return {"example": f"'{story_title}'에서 '{choice_text}'를 선택했습니다."}

        llm = OpenAIService()

        prompt = f"""
아이가 '{story_title}'라는 동화에서 '{choice_text}'라는 선택을 했습니다.
이 선택이 {ability} 능력을 보여준다는 것을 부모에게 설명하는 문장을 작성해주세요.

다음 JSON 형식으로만 응답하세요:
{{
  "example": "자연스럽고 따뜻한 설명 (30자 이내)"
}}

조건:
1. 동화 제목과 선택 내용을 자연스럽게 포함
2. {ability} 능력과 연결하여 설명
3. "~했어요", "~보였어요" 등 과거형으로 작성
4. 아이의 선택을 긍정적으로 평가
"""

        try:
            response = llm.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )

            result = json.loads(response.choices[0].message.content)
            example = result.get("example", f"'{story_title}'에서 '{choice_text}'를 선택했습니다.")
            logger.info(f"예시 설명 생성 완료: {len(example)}자")
            return {"example": example}

        except Exception as e:
            logger.warning(f"예시 설명 생성 실패: {e}")
            return {"example": f"'{story_title}'에서 '{choice_text}'를 선택했습니다."}

    except Exception as e:
        logger.exception("generate-example-description 실패")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health-growth")
async def health_growth():
    """성장 리포트 API 헬스체크"""
    logger.info("health check 요청 (성장 리포트)")
    return {"status": "ok", "service": "growth_report"}