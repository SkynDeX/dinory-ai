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
            results = self.index.query(vector=vec, top_k=top_k, include_metadata=True)
            matches = getattr(results, "matches", None) or getattr(results, "data", None) or results.get("matches", [])  # type: ignore[attr-defined]
            stories: List[Dict[str, Any]] = []
            for m in matches:
                mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None)
                score = getattr(m, "score", None) or (m.get("score") if isinstance(m, dict) else 0.0)
                meta = getattr(m, "metadata", None) or (m.get("metadata") if isinstance(m, dict) else {}) or {}
                stories.append(
                    {
                        "story_id": mid,
                        "title": meta.get("title", "제목 없음"),
                        "matching_score": int(float(score) * 100),
                        "metadata": meta,
                    }
                )
            logger.info(f"Pinecone 검색 결과: {len(stories)}개")
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
        """
        return self.search_stories(emotion, interests, top_k=limit)

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
