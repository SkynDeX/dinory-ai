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

당신은 아동심리전문상담가입니다. 아이의 {period_text}간 성장 리포트를 위한 따뜻하고 격려하는 객관적인 종합 평가를 작성해주세요.

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
1. 10-15문장으로 작성 (최소 400자 이상)
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
            examples = strength_info.get("examples", [])  # 배열로 받기

            prompt = f"""
아이가 {area_name} 능력에서 {score}점을 기록하며 뛰어난 모습을 보였습니다.
{f"예시: {', '.join(examples)}" if examples else ""}

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
                    "examples": examples  # 배열로 반환
                })
            except Exception as e:
                logger.warning(f"{area_name} 강점 설명 생성 실패: {e}")
                results.append({
                    "area": area_name,
                    "score": score,
                    "description": f"{area_name} 영역에서 뛰어난 능력을 보여줍니다.",
                    "examples": examples  # 배열로 반환
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


@router.post("/generate-all-growth-content")
async def generate_all_growth_content(req: GrowthReportRequest):
    """모든 성장 리포트 AI 콘텐츠를 한 번에 생성 (성능 최적화)"""
    logger.info(f"통합 AI 콘텐츠 생성 요청: totalStories={req.totalStories}, period={req.period}")

    result = {
        "evaluation": "",
        "recommendations": [],
        "milestones": [],
        "strengthDescriptions": [],
        "growthAreaDescriptions": [],
        "examples": {}
    }

    try:
        if not OpenAIService:
            logger.warning("OpenAIService 없음")
            return result

        llm = OpenAIService()

        # 1. AI 종합 평가
        try:
            before_text = "\n".join([f"- {k}: {v:.0f}점" for k, v in req.beforeAbilities.items()])
            after_text = "\n".join([f"- {k}: {v:.0f}점" for k, v in req.afterAbilities.items()])
            changes = []
            for ability, after_score in req.afterAbilities.items():
                before_score = req.beforeAbilities.get(ability, 0)
                change = after_score - before_score
                if abs(change) > 5:
                    changes.append(f"{ability}: {change:+.0f}점")
            changes_text = ", ".join(changes) if changes else "전반적으로 안정적"
            strengths_text = ", ".join([s.get("area", "") for s in req.strengths[:2]]) if req.strengths else "없음"
            growth_areas_text = ", ".join([g.get("area", "") for g in req.growthAreas[:2]]) if req.growthAreas else "없음"
            period_map = {"month": "한 달", "quarter": "3개월", "halfyear": "6개월"}
            period_text = period_map.get(req.period, "한 달")

            eval_prompt = f"""
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
2. 각 능력치의 의미를 쉽게 풀어서 설명
3. 가장 크게 성장한 영역을 구체적인 예시와 함께 언급
4. 완료한 동화 개수를 바탕으로 아이의 노력 인정
5. 긍정적이고 성장 가능성에 초점을 맞춘 격려
6. 부모가 이해하기 쉬운 자연스러운 한국어
7. 평가문만 작성 (제목, 인사말 제외)
8. 데이터가 부족하더라도 아이의 잠재력과 가능성을 중심으로 풍부하게 작성
"""

            eval_response = llm.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.7,
                max_tokens=600
            )
            result["evaluation"] = eval_response.choices[0].message.content.strip()
            logger.info(f"종합 평가 생성 완료: {len(result['evaluation'])}자")
        except Exception as e:
            logger.error(f"종합 평가 생성 실패: {e}")

        # 2. 추천 활동
        if req.growthAreas:
            try:
                growth_areas_info = "\n".join([
                    f"- {g.get('area', '')}: {g.get('score', 0)}점 ({g.get('description', '')})"
                    for g in req.growthAreas[:3]
                ])

                rec_prompt = f"""
아이의 성장을 위한 맞춤 활동을 추천해주세요.

**성장 가능 영역**:
{growth_areas_info}

위 영역들을 고려하여 우선순위가 높은 순서로 3가지 활동을 추천해주세요.

다음 JSON 형식으로만 응답해주세요:
{{
  "recommendations": [
    {{"priority": 1, "activity": "활동 이름", "description": "구체적인 활동 설명 (40자 이내)", "targetArea": "타겟 능력치"}},
    {{"priority": 2, "activity": "활동 이름", "description": "활동 설명", "targetArea": "타겟 능력치"}},
    {{"priority": 3, "activity": "활동 이름", "description": "활동 설명", "targetArea": "타겟 능력치"}}
  ]
}}

조건:
1. 아이가 실제로 할 수 있는 구체적이고 재미있는 활동
2. 부모가 함께 할 수 있는 활동
3. 일상에서 쉽게 실천 가능
"""

                rec_response = llm.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": rec_prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )
                rec_data = json.loads(rec_response.choices[0].message.content)
                result["recommendations"] = rec_data.get("recommendations", [])
                logger.info(f"추천 활동 생성 완료: {len(result['recommendations'])}개")
            except Exception as e:
                logger.error(f"추천 활동 생성 실패: {e}")

        # 3. 마일스톤 (동시 생성 - 효율성)
        milestones = []
        if req.totalStories >= 5:
            try:
                ms_prompt = f"""
아이가 {req.totalStories}개의 동화를 완료했습니다. 축하 문구를 작성해주세요.
JSON: {{"achievement": "축하 문구 (20자 이내)"}}
조건: 노력과 꾸준함 강조, 긍정적 어조
"""
                ms_resp = llm.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": ms_prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )
                ms_data = json.loads(ms_resp.choices[0].message.content)
                milestones.append({"achievement": ms_data.get("achievement", ""), "date": None})
            except Exception as e:
                logger.error(f"동화 완료 마일스톤 실패: {e}")

        for ability, score in req.afterAbilities.items():
            if score >= 75:
                try:
                    ab_prompt = f"""
아이가 {ability} 능력에서 {score:.0f}점 달성. 축하 문구를 작성해주세요.
JSON: {{"achievement": "축하 문구 (25자 이내, {ability}의 의미 쉽게 풀어서)"}}
조건: 능력치 쉽게 설명, 따뜻한 어조
"""
                    ab_resp = llm.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": ab_prompt}],
                        response_format={"type": "json_object"},
                        temperature=0.7
                    )
                    ab_data = json.loads(ab_resp.choices[0].message.content)
                    milestones.append({"achievement": ab_data.get("achievement", ""), "date": None})
                except Exception as e:
                    logger.error(f"{ability} 마일스톤 실패: {e}")

        result["milestones"] = milestones
        logger.info(f"마일스톤 생성 완료: {len(milestones)}개")

        # 4. 강점 영역 설명
        strength_descs = []
        for strength_info in req.strengths[:3]:
            area_name = strength_info.get("area", "")
            score = strength_info.get("score", 0)
            examples = strength_info.get("examples", [])  # 배열로 받기

            try:
                st_prompt = f"""
아이가 {area_name} 능력에서 {score}점을 기록. {f"예시: {', '.join(examples)}" if examples else ""}
칭찬하는 설명을 작성해주세요.
JSON: {{"description": "{area_name}의 의미를 쉽게 설명하고, 왜 대단한지 40자 이내로"}}
조건: 구체적 칭찬, 쉬운 설명, 따뜻한 어조
"""
                st_resp = llm.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": st_prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )
                st_data = json.loads(st_resp.choices[0].message.content)
                strength_descs.append({
                    "area": area_name,
                    "score": score,
                    "description": st_data.get("description", ""),
                    "examples": examples  # 배열로 반환
                })
            except Exception as e:
                logger.error(f"{area_name} 강점 설명 실패: {e}")
                strength_descs.append({
                    "area": area_name,
                    "score": score,
                    "description": f"{area_name} 영역에서 뛰어난 능력을 보여줍니다.",
                    "examples": examples  # 배열로 반환
                })

        result["strengthDescriptions"] = strength_descs
        logger.info(f"강점 설명 생성 완료: {len(strength_descs)}개")

        # 5. 성장가능영역 설명
        growth_descs = []
        for area_info in req.growthAreas[:3]:
            area_name = area_info.get("area", "")
            score = area_info.get("score", 0)

            try:
                ga_prompt = f"""
아이의 {area_name} 능력(현재 {score}점)을 발전시키기 위한 설명과 추천.
JSON: {{"description": "{area_name}의 의미와 왜 중요한지 30자 이내", "recommendation": "구체적 활동 1가지 40자 이내"}}
조건: 실천 가능, 부드러운 어조
"""
                ga_resp = llm.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": ga_prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )
                ga_data = json.loads(ga_resp.choices[0].message.content)
                growth_descs.append({
                    "area": area_name,
                    "score": score,
                    "description": ga_data.get("description", ""),
                    "recommendation": ga_data.get("recommendation", "")
                })
            except Exception as e:
                logger.error(f"{area_name} 성장영역 설명 실패: {e}")
                growth_descs.append({
                    "area": area_name,
                    "score": score,
                    "description": f"{area_name} 영역을 더 발전시킬 수 있습니다.",
                    "recommendation": f"{area_name} 관련 동화를 함께 읽어보세요."
                })

        result["growthAreaDescriptions"] = growth_descs
        logger.info(f"성장영역 설명 생성 완료: {len(growth_descs)}개")

        logger.info("통합 AI 콘텐츠 생성 완료")
        return result

    except Exception as e:
        logger.exception("generate-all-growth-content 실패")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-choice-pattern")
async def analyze_choice_pattern(request: Dict[str, Any]):
    """아이의 선택 패턴을 AI로 분석하여 스타일 분류"""
    logger.info("선택 패턴 분석 요청")
    try:
        ability_type = request.get("abilityType", "")
        ability_ratios = request.get("abilityRatios", {})

        if not OpenAIService or not ability_type:
            # 폴백: 기본 스타일
            default_styles = {
                "용기": "용감한 선택",
                "친절": "배려하는 선택",
                "공감": "배려하는 선택",
                "우정": "협력하는 선택",
                "자존감": "자신있는 선택"
            }
            return {"style": default_styles.get(ability_type, "용감한 선택")}

        llm = OpenAIService()

        # 비율 정보를 텍스트로 변환
        ratios_text = ", ".join([f"{k}: {v:.1f}%" for k, v in ability_ratios.items()])

        prompt = f"""
아이의 선택 패턴을 분석해주세요.

**능력치 분포**:
{ratios_text}

**현재 선택한 능력치**: {ability_type}

아이의 전체적인 선택 패턴을 보고, 이 선택이 어떤 스타일인지 판단해주세요.

다음 JSON 형식으로만 응답하세요:
{{
  "style": "선택 스타일 (아래 6가지 중 1개)"
}}

**가능한 스타일**:
1. "용감한 선택" - 용기를 크게 보이며 과감하게 도전
2. "배려하는 선택" - 타인을 생각하는 따뜻한 마음
3. "협력하는 선택" - 함께하는 것을 중요시
4. "자신있는 선택" - 자존감과 확신을 가지고 행동
5. "도전적인 선택" - 새로운 것에 도전하는 모습
6. "신중한 선택" - 깊이 생각하고 판단

조건:
1. 능력치 분포와 선택한 능력치를 종합적으로 고려
2. 위 6가지 스타일 중 정확히 하나만 선택
3. 능력치 이름을 그대로 사용하지 말고 의미를 해석
"""

        try:
            response = llm.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )

            result = json.loads(response.choices[0].message.content)
            style = result.get("style", "용감한 선택")

            # 유효한 스타일인지 검증
            valid_styles = ["용감한 선택", "배려하는 선택", "협력하는 선택",
                           "자신있는 선택", "도전적인 선택", "신중한 선택"]
            if style not in valid_styles:
                style = "용감한 선택"

            logger.info(f"선택 패턴 분석 완료: {ability_type} → {style}")
            return {"style": style}

        except Exception as e:
            logger.warning(f"AI 선택 패턴 분석 실패: {e}")
            # 폴백
            default_styles = {
                "용기": "용감한 선택",
                "친절": "배려하는 선택",
                "공감": "배려하는 선택",
                "우정": "협력하는 선택",
                "자존감": "자신있는 선택"
            }
            return {"style": default_styles.get(ability_type, "용감한 선택")}

    except Exception as e:
        logger.exception("analyze-choice-pattern 실패")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-chat-pattern")
async def analyze_chat_pattern(request: Dict[str, Any]):
    """대화 패턴 AI 분석"""
    logger.info("대화 패턴 분석 요청")
    try:
        messages = request.get("messages", [])
        child_name = request.get("childName", "아이")

        if not OpenAIService or not messages:
            logger.warning("OpenAIService 없거나 메시지 없음")
            return {
                "conversationStyle": "활발한 대화",
                "vocabularyLevel": "나이에 적절한 어휘 사용",
                "mainInterests": [],
                "emotionPattern": "긍정적인 감정 표현",
                "participationLevel": "적극적인 참여",
                "insights": "아이가 대화에 잘 참여하고 있습니다."
            }

        llm = OpenAIService()

        # 아이의 메시지만 추출
        child_messages = [msg.get("message", "") for msg in messages if msg.get("sender") == "CHILD"]

        if not child_messages:
            return {
                "conversationStyle": "대화 시작 단계",
                "vocabularyLevel": "분석 데이터 부족",
                "mainInterests": [],
                "emotionPattern": "분석 데이터 부족",
                "participationLevel": "대화 시작",
                "insights": "아직 대화가 충분하지 않아 패턴을 분석하기 어렵습니다."
            }

        # 대화 내용 텍스트로 결합
        conversation_text = "\n".join([f"- {msg}" for msg in child_messages[:20]])  # 최근 20개
        message_count = len(child_messages)
        avg_length = sum(len(msg) for msg in child_messages) / len(child_messages) if child_messages else 0

        prompt = f"""
{child_name}의 챗봇 대화 패턴을 분석해주세요.

**대화 메시지 ({message_count}개)**:
{conversation_text}

**통계**:
- 총 메시지 수: {message_count}개
- 평균 메시지 길이: {avg_length:.1f}자

아동심리 전문가 관점에서 대화 패턴을 분석하고, 다음 JSON 형식으로 응답해주세요:

{{
  "conversationStyle": "대화 스타일 (20자 이내)",
  "vocabularyLevel": "어휘 수준 평가 (30자 이내)",
  "mainInterests": ["관심사1", "관심사2", "관심사3"],
  "emotionPattern": "감정 표현 패턴 (30자 이내)",
  "participationLevel": "참여도 평가 (20자 이내)",
  "insights": "부모를 위한 인사이트 (50자 이내)"
}}

**대화 스타일 예시**: "호기심 많고 질문이 많은 탐구형", "감정 표현이 풍부한 감성형", "짧고 명확한 실용형" 등

**어휘 수준**: 아이의 연령대를 고려하여 평가

**관심사**: 대화에서 자주 나오는 주제나 키워드 (최대 3개)

**감정 패턴**: "긍정적 감정 위주", "다양한 감정 표현", "조심스러운 감정 표현" 등

**참여도**: "매우 적극적", "적극적", "보통", "소극적" 등

**인사이트**: 부모가 아이와 대화할 때 도움이 될 만한 조언
"""

        try:
            response = llm.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )

            result = json.loads(response.choices[0].message.content)
            logger.info(f"대화 패턴 분석 완료: style={result.get('conversationStyle')}")
            return result

        except Exception as e:
            logger.error(f"AI 대화 패턴 분석 실패: {e}")
            # 폴백
            return {
                "conversationStyle": "활발한 대화",
                "vocabularyLevel": "나이에 적절한 어휘 사용",
                "mainInterests": ["동화", "친구", "놀이"],
                "emotionPattern": "긍정적인 감정 표현",
                "participationLevel": "적극적인 참여",
                "insights": f"{child_name}가 대화에 잘 참여하고 있습니다."
            }

    except Exception as e:
        logger.exception("analyze-chat-pattern 실패")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-chat-topics")
async def extract_chat_topics(request: Dict[str, Any]):
    """채팅 메시지에서 주요 관심 주제 추출"""
    logger.info("대화 주제 추출 요청")
    try:
        messages = request.get("messages", [])

        if not OpenAIService or not messages:
            logger.warning("OpenAIService 없거나 메시지 없음")
            return {"topics": []}

        llm = OpenAIService()

        # 아이의 메시지만 추출
        child_messages = [msg.get("message", "") for msg in messages if msg.get("sender") in ["USER", "CHILD"]]

        if not child_messages:
            return {"topics": []}

        # 대화 샘플 (최근 30개)
        sample_messages = "\n".join([f"- {msg}" for msg in child_messages[-30:]])

        prompt = f"""
다음은 아이와의 대화 내용입니다. 대화에서 아이가 관심있어 하는 주제나 키워드를 추출해주세요.

**대화 내용**:
{sample_messages}

아이가 자주 언급하거나 관심있어 하는 주제를 최대 10개까지 추출해주세요.

다음 JSON 형식으로만 응답하세요:
{{
  "topics": [
    {{"text": "주제1", "count": 빈도}},
    {{"text": "주제2", "count": 빈도}},
    ...
  ]
}}

**조건**:
1. 구체적인 명사형 키워드로 추출 (예: "동물", "공룡", "우주", "친구", "학교" 등)
2. 너무 추상적인 단어는 제외 (예: "것", "이야기" 등)
3. 대화에서 실제로 언급된 주제만 포함
4. count는 대략적인 빈도 (1~10)
5. 빈도가 높은 순서로 정렬
6. 최대 10개
"""

        try:
            response = llm.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )

            result = json.loads(response.choices[0].message.content)
            topics = result.get("topics", [])

            logger.info(f"대화 주제 추출 완료: {len(topics)}개")
            return {"topics": topics}

        except Exception as e:
            logger.error(f"AI 주제 추출 실패: {e}")
            return {"topics": []}

    except Exception as e:
        logger.exception("extract-chat-topics 실패")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-dashboard-insights")
async def generate_dashboard_insights(request: Dict[str, Any]):
    """
    대시보드 AI 인사이트 생성 (2개)
    1. Quick 인사이트 (종합 현황 탭)
    2. 추천 활동 (능력 발달 탭)
    """
    try:
        abilities = request.get("abilities", {})
        choices = request.get("choices", [])
        total_stories = request.get("totalStories", 0)
        period = request.get("period", "week")

        if not OpenAIService:
            return {
                "quickInsight": "아이와 함께 동화를 읽으며 성장해보세요!",
                "recommendation": {
                    "ability": "용기",
                    "message": "용기 관련 동화를 함께 읽어보세요."
                }
            }

        llm = OpenAIService()

        # 1. Quick 인사이트 생성
        top_ability = max(abilities.items(), key=lambda x: x[1]) if abilities else None
        low_ability = min(abilities.items(), key=lambda x: x[1]) if abilities else None
        top_choice = choices[0] if choices else None

        period_text = {"day": "오늘", "week": "이번 주", "month": "이번 달"}.get(period, "이번 주")

        quick_prompt = f"""
아이의 {period_text} 활동 데이터를 분석한 결과입니다:
- 완료한 동화: {total_stories}개
- 가장 높은 능력: {top_ability[0]} ({top_ability[1]:.0f}점) (최고인 능력)
- 가장 낮은 능력: {low_ability[0]} ({low_ability[1]:.0f}점) (개선이 필요한 능력)
- 주요 선택 스타일: {top_choice['name']} ({top_choice['value']}%)

부모에게 전달할 따뜻하고 격려하는 한 줄 인사이트를 작성해주세요.

조건:
1. 40자 이내
2. 아이의 강점을 칭찬하고, 개선점을 부드럽게 제안
3. 구체적인 능력명과 수치 언급
4. "~해요", "~보세요" 등 친근한 어조

JSON 형식으로만 응답:
{{
  "insight": "한 줄 인사이트"
}}
"""

        try:
            quick_response = llm.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": quick_prompt}],
                response_format={"type": "json_object"},
                temperature=0.8
            )
            quick_data = json.loads(quick_response.choices[0].message.content)
            quick_insight = quick_data.get("insight", "아이와 함께 동화를 읽으며 성장해보세요!")
        except Exception as e:
            logger.error(f"Quick 인사이트 생성 실패: {e}")
            quick_insight = f"{top_ability[0] if top_ability else '능력'}이 높고, {top_choice['name'] if top_choice else '좋은 선택'}을 주로 하고 있어요."

        # 2. 능력 추천 활동 생성
        rec_prompt = f"""
아이의 능력치 분석 결과:
- 가장 낮은 능력: {low_ability[0]} ({low_ability[1]:.0f}점)

이 능력을 키울 수 있는 추천 메시지를 작성해주세요.

조건:
1. 30자 이내
2. {low_ability[0]} 능력을 키우는 구체적인 활동 제안
3. "~해보세요", "~는 어떨까요?" 등 제안하는 어조

JSON 형식으로만 응답:
{{
  "message": "추천 메시지"
}}
"""

        try:
            rec_response = llm.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": rec_prompt}],
                response_format={"type": "json_object"},
                temperature=0.8
            )
            rec_data = json.loads(rec_response.choices[0].message.content)
            rec_message = rec_data.get("message", f"{low_ability[0] if low_ability else '능력'} 관련 동화를 함께 읽어보세요.")
        except Exception as e:
            logger.error(f"추천 활동 생성 실패: {e}")
            rec_message = f"{low_ability[0] if low_ability else '능력'} 관련 동화를 함께 읽으면서 키워보는 건 어떨까요?"

        logger.info("대시보드 인사이트 생성 완료")
        return {
            "quickInsight": quick_insight,
            "recommendation": {
                "ability": low_ability[0] if low_ability else "용기",
                "message": rec_message
            }
        }

    except Exception as e:
        logger.exception("generate-dashboard-insights 실패")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health-growth")
async def health_growth():
    """성장 리포트 API 헬스체크"""
    logger.info("health check 요청 (성장 리포트)")
    return {"status": "ok", "service": "growth_report"}