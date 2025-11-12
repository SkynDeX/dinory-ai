# app/services/story/story_generator.py
import os
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger("dinory.story")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[STORY] %(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


class StorySearchService:
    """
    Pinecone + 임베딩 기반 동화 검색 서비스.
    - 모든 외부 의존성(Pinecone, OpenAI, SBERT)이 없어도 안전 폴백으로 동작.
    - 엔드포인트에서 await을 쓸 수 있도록 async 래퍼 제공.
    """

    def __init__(self):
        self.index = None
        self.pc = None
        self.openai_client = None
        self.embedder = None  # lazy
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "story-embeddings")

        # Pinecone 초기화
        try:
            if os.getenv("PINECONE_DISABLE", "0") == "1":
                logger.warning("Pinecone 비활성화(PINECONE_DISABLE=1). 더미로 동작")
            else:
                api_key = os.getenv("PINECONE_API_KEY")
                if not api_key:
                    logger.warning("PINECONE_API_KEY 없음. 더미로 동작")
                else:
                    from pinecone import Pinecone  # optional import
                    self.pc = Pinecone(api_key=api_key)
                    try:
                        self.index = self.pc.Index(self.index_name)
                        logger.info(f"Pinecone 인덱스 연결: {self.index_name}")
                    except Exception as e:
                        logger.error(f"Pinecone 인덱스 연결 실패: {e}")
                        self.index = None
        except Exception as e:
            logger.error(f"Pinecone 초기화 오류: {e}")
            self.pc, self.index = None, None

        # OpenAI 임베딩(옵션)
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                from openai import OpenAI  # optional import
                self.openai_client = OpenAI(api_key=openai_key)
                logger.info("OpenAI 클라이언트(임베딩) 준비")
            else:
                logger.warning("OPENAI_API_KEY 없음. SBERT 폴백 예정")
        except Exception as e:
            logger.error(f"OpenAI 초기화 실패: {e}")
            self.openai_client = None

    # ──────────────────────────────────────────────────────────────────────────
    # 내부 유틸
    def _lazy_load_sbert(self) -> None:
        if self.embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer(
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                logger.info("SBERT 임베딩기 초기화")
            except Exception as e:
                logger.error(f"SBERT 초기화 실패: {e}")
                self.embedder = None

    def create_search_query(self, emotion: Optional[str], interests: Optional[List[str]]) -> str:
        emotion_map = {
            "기뻐요": "기쁨 행복 즐거움 웃음 축하 신남 좋아함",
            "슬퍼요": "슬픔 눈물 위로 공감 아픔 상처 헤어짐 그리움",
            "화나요": "화남 분노 짜증 싸움 갈등 미안함 용서 화해",
            "무서워요": "두려움 공포 무서움 용기 극복 도전 강함",
            "신나요": "신남 모험 탐험 재미 활기 에너지 활동",
            "피곤해요": "피곤 휴식 평온 편안 잠 쉼 여유",
        }
        emotion_text = emotion_map.get(emotion or "", emotion or "")
        interests_text = " ".join(interests or [])
        query = f"{emotion_text} {interests_text} 동화 이야기".strip() or "아이 감정 공감 모험 우정 동화 이야기"
        logger.info(f"검색어 생성: {query}")
        return query

    def _embed(self, text: str) -> Optional[List[float]]:
        # 1) OpenAI
        try:
            if self.openai_client:
                resp = self.openai_client.embeddings.create(
                    model="text-embedding-3-large",
                    input=text,
                )
                return resp.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI 임베딩 실패: {e}")

        # 2) SBERT
        try:
            self._lazy_load_sbert()
            if self.embedder:
                vec = self.embedder.encode([text])[0]
                return vec.tolist() if hasattr(vec, "tolist") else list(vec)
        except Exception as e:
            logger.error(f"SBERT 임베딩 실패: {e}")

        return None

    def _normalize(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        출력 표준화:
        {
            storyId, title, matchingScore, metadata
        }
        """
        normed = []
        for it in items:
            normed.append({
                "storyId": it.get("story_id") or it.get("id") or it.get("storyId"),
                "title": it.get("title") or it.get("metadata", {}).get("title", "제목 없음"),
                "matchingScore": int(it.get("matching_score") or it.get("score") or 0),
                "metadata": it.get("metadata") or {},
            })
        return normed

    # ──────────────────────────────────────────────────────────────────────────
    # 퍼블릭 API
    def search_stories(
        self,
        emotion: Optional[str],
        interests: Optional[List[str]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Pinecone가 없으면 더미. 있으면 임베딩 쿼리.
        반환은 _normalize 형식 전제.
        """
        if not self.index:
            logger.warning("Pinecone 인덱스 미연결 → 더미 반환")
            return self._normalize(self._get_dummy_stories(emotion, interests or [], top_k))

        query_text = self.create_search_query(emotion, interests)
        vec = self._embed(query_text)
        if vec is None:
            logger.error("임베딩 생성 실패 → 더미 반환")
            return self._normalize(self._get_dummy_stories(emotion, interests or [], top_k))

        try:
            # results = self.index.query(vector=vec, top_k=top_k, include_metadata=True) # 기존 코드
            # [2025-10-29 김광현]중복이 있을 수 있으므로 top_k보다 더많은 동화 찾기(수정코드)
            results = self.index.query(vector=vec, top_k=top_k * 2, include_metadata=True)
            matches = getattr(results, "matches", None) or getattr(results, "data", None) or results.get("matches", [])  # type: ignore[attr-defined]
            
            stories: List[Dict[str, Any]] = []
            seen_ids = set()    # [2025-10-29 김광현] ID 중복 체크용
            seen_titles = set()  # [2025-11-12 추가] 제목 중복 체크용

            for m in matches:
                mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None)
                score = getattr(m, "score", None) or (m.get("score") if isinstance(m, dict) else 0.0)
                meta = getattr(m, "metadata", None) or (m.get("metadata") if isinstance(m, dict) else {}) or {}
                title = meta.get("title", "제목 없음")

                # [2025-11-12 수정] ID와 제목 둘 다 중복 체크
                if mid in seen_ids or title in seen_titles:
                    continue

                seen_ids.add(mid)
                seen_titles.add(title)

                stories.append(
                    {
                        "story_id": mid,
                        "title": title,
                        "matching_score": int(float(score) * 100),
                        "metadata": meta,
                    }
                )

                # # [2025-10-29 김광현] 원하는 개수만큼 모이면 중단
                if len(stories) >= top_k:
                    break

            logger.info(f"Pinecone 검색 결과 (ID/제목 중복 제거 전/후): {len(matches)}/{len(stories)}개")
            return self._normalize(stories)
        
        except Exception as e:
            logger.error(f"Pinecone 검색 오류 → 더미 반환: {e}")
            return self._normalize(self._get_dummy_stories(emotion, interests or [], top_k))

    async def recommend_stories_async(
        self,
        emotion: Optional[str],
        interests: Optional[List[str]],
        child_id: Optional[int],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        엔드포인트에서 await 가능하도록 제공하는 async 래퍼.

        [2025-11-12 김광현] AI 줄거리 생성 기능 추가
        - Pinecone 검색 후 각 동화 제목으로 AI가 줄거리 생성
        - metadata["ai_summary"]에 저장하여 백엔드로 전달

        [2025-11-12 수정] 이미 읽은 동화 제외
        - child_id로 완료한 동화 목록 조회
        - 추천 결과에서 중복 제거
        """
        import httpx
        import os

        # [2025-11-12 추가] 이미 읽은 동화 ID 목록 가져오기
        read_story_ids = set()
        if child_id:
            try:
                spring_api_url = os.getenv("SPRING_API_URL", "http://localhost:8090/api")
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        f"{spring_api_url}/story/completions/child/{child_id}",
                        params={"limit": 100}
                    )
                    if response.status_code == 200:
                        completions = response.json()
                        read_story_ids = {str(c.get("storyId")) for c in completions if c.get("storyId")}
                        logger.info(f"✅ 아이 {child_id}의 읽은 동화 {len(read_story_ids)}개 제외")
                    else:
                        logger.warning(f"⚠️ 읽은 동화 목록 조회 실패: {response.status_code}")
            except Exception as e:
                logger.warning(f"⚠️ 읽은 동화 목록 조회 실패: {e}")

        # 기존 동기 검색 (더 많이 가져와서 중복 제거 후 limit 맞추기)
        stories = self.search_stories(emotion, interests, top_k=limit * 3)

        # [2025-11-12 추가] 이미 읽은 동화 제외
        filtered_stories = [
            story for story in stories
            if story.get("storyId") not in read_story_ids
        ][:limit]  # 필터링 후 limit만큼만

        logger.info(f"📚 전체 추천: {len(stories)}개 → 중복 제거 후: {len(filtered_stories)}개")

        # 각 동화에 AI 줄거리 추가 (병렬 처리로 속도 개선)
        from app.services.llm.openai_service import OpenAIService
        import asyncio

        openai_service = OpenAIService()

        async def add_ai_summary(story: Dict[str, Any]) -> Dict[str, Any]:
            """각 동화에 AI 생성 줄거리 추가"""
            title = story.get("title", "제목 없음")
            metadata = story.get("metadata", {})

            try:
                # AI로 줄거리 생성
                ai_summary = await openai_service.generate_story_summary(title)
                metadata["ai_summary"] = ai_summary
                logger.info(f"✅ AI 줄거리 생성: {title[:20]}... → {ai_summary[:30]}...")
            except Exception as e:
                logger.warning(f"⚠️ AI 줄거리 생성 실패: {title}, {str(e)}")
                # 실패 시 plotSummaryText 사용하거나 기본 문구
                fallback = metadata.get("plotSummaryText") or f"{title}의 이야기예요."
                metadata["ai_summary"] = fallback

            story["metadata"] = metadata
            return story

        # 모든 동화에 대해 병렬로 AI 줄거리 생성 (필터링된 동화만)
        try:
            enriched_stories = await asyncio.gather(*[add_ai_summary(s) for s in filtered_stories])
            logger.info(f"✅ {len(enriched_stories)}개 동화에 AI 줄거리 추가 완료")
            return enriched_stories
        except Exception as e:
            logger.error(f"❌ AI 줄거리 일괄 생성 실패: {str(e)}")
            # 실패해도 필터링된 stories 반환
            return filtered_stories

    def get_story_by_id(self, story_id: str) -> Optional[Dict[str, Any]]:
        if not self.index:
            logger.warning("Pinecone 미연결 → None")
            return None
        try:
            result = self.index.fetch(ids=[story_id])
            vectors = getattr(result, "vectors", None) or result.get("vectors", {})
            if story_id in vectors:
                meta = vectors[story_id].get("metadata", {}) or {}
                return {"storyId": story_id, "title": meta.get("title", "제목 없음"), "metadata": meta}
            logger.info(f"Pinecone에 없음: {story_id}")
            return None
        except Exception as e:
            logger.error(f"{story_id} 조회 오류: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # 더미 데이터
    def _get_dummy_stories(self, emotion: Optional[str], interests: List[str], top_k: int) -> List[Dict[str, Any]]:
        dummy = [
            {
                "story_id": "new_sibling",
                "title": "새 동생과의 하루",
                "matching_score": 96,
                "metadata": {"classification": "가족", "readAge": "유아", "plotSummaryText": "새로운 가족을 맞이하며 배우는 공감"},
            },
            {
                "story_id": "brave_little_star",
                "title": "작은 별의 용기",
                "matching_score": 93,
                "metadata": {"classification": "용기", "readAge": "유아", "plotSummaryText": "두려움을 이겨내는 모험"},
            },
            {
                "story_id": "forest_friends",
                "title": "숲속 친구들",
                "matching_score": 89,
                "metadata": {"classification": "우정", "readAge": "유아", "plotSummaryText": "서로 돕는 친구들의 이야기"},
            },
            {
                "story_id": "angry_rabbit",
                "title": "화난 토끼의 하루",
                "matching_score": 85,
                "metadata": {"classification": "감정조절", "readAge": "유아", "plotSummaryText": "화를 다루는 법 배우기"},
            },
            {
                "story_id": "magic_adventure",
                "title": "마법의 모험",
                "matching_score": 82,
                "metadata": {"classification": "모험", "readAge": "유아", "plotSummaryText": "작은 마법사의 성장기"},
            },
        ]
        logger.info(f"더미 {top_k}개 반환")
        return dummy[:top_k]
