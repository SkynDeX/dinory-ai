"""
RAG Memory Service for DinoCharacter Chatbot
Supports both MySQL-based and Pinecone-based memory retrieval
"""

import os
import httpx
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from datetime import datetime


class MemoryService:
    """
    RAG 메모리 서비스

    사용자의 과거 대화와 동화 완료 기록을 검색하여 컨텍스트 제공

    지원하는 메모리 소스:
    1. MySQL (via Spring Boot API) - 최근 대화, 구조화된 데이터
    2. Pinecone Vector DB - 시맨틱 검색, 전체 히스토리
    """

    def __init__(self, use_pinecone: bool = False):
        """
        Args:
            use_pinecone: True면 Pinecone 사용, False면 MySQL만 사용
        """
        self.use_pinecone = use_pinecone
        self.spring_api_url = os.getenv("SPRING_API_URL", "http://localhost:8090/api")
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Pinecone 설정 (옵션)
        if self.use_pinecone:
            self._init_pinecone()

    def _init_pinecone(self):
        """Pinecone 초기화 (별도 챗봇용 인덱스)"""
        try:
            from pinecone import Pinecone

            # 챗봇 전용 Pinecone 설정 (스토리용과 별도)
            api_key = os.getenv("CHATBOT_PINECONE_API_KEY")
            index_name = os.getenv("CHATBOT_PINECONE_INDEX_NAME", "chatbot-memory-index")

            if not api_key:
                print("⚠️ CHATBOT_PINECONE_API_KEY not set. Pinecone disabled.")
                self.use_pinecone = False
                return

            self.pc = Pinecone(api_key=api_key)
            self.index = self.pc.Index(index_name)
            print(f"✅ Pinecone initialized: {index_name}")

        except Exception as e:
            print(f"❌ Pinecone initialization failed: {e}")
            self.use_pinecone = False

    # ========== MySQL 기반 메모리 검색 ==========

    async def get_recent_conversations(
        self,
        child_id: int,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        MySQL에서 최근 대화 기록 조회 (Spring Boot API 호출)

        Args:
            child_id: 아이 ID
            limit: 가져올 메시지 수 (최신순)

        Returns:
            [
                {
                    "session_id": 1,
                    "message": "안녕!",
                    "sender": "USER",
                    "created_at": "2025-10-29T..."
                },
                ...
            ]
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Spring Boot API 엔드포인트 호출
                response = await client.get(
                    f"{self.spring_api_url}/chat/history/child/{child_id}",
                    params={"limit": limit}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"❌ Failed to fetch conversations from MySQL: {e}")
            return []

    async def get_story_completions(
        self,
        child_id: int,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        MySQL에서 동화 완료 기록 조회

        Args:
            child_id: 아이 ID
            limit: 가져올 동화 수

        Returns:
            [
                {
                    "completion_id": 1,
                    "story_id": 1,
                    "story_title": "용감한 디노",
                    "abilities": {...},
                    "choices": [...],
                    "completed_at": "2025-10-29T..."
                },
                ...
            ]
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.spring_api_url}/story/completions/child/{child_id}",
                    params={"limit": limit}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"❌ Failed to fetch story completions from MySQL: {e}")
            return []

    # ========== Pinecone 기반 시맨틱 검색 ==========

    async def search_similar_conversations(
        self,
        query: str,
        child_id: int,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Pinecone에서 시맨틱 유사도 검색

        Args:
            query: 검색 쿼리 (현재 사용자 메시지)
            child_id: 아이 ID
            top_k: 가져올 유사 대화 수

        Returns:
            유사한 과거 대화 목록
        """
        if not self.use_pinecone:
            print("⚠️ Pinecone disabled. Skipping semantic search.")
            return []

        try:
            # 1. 쿼리 임베딩 생성
            embedding = await self._get_embedding(query)

            # 2. Pinecone 검색
            results = self.index.query(
                vector=embedding,
                filter={"child_id": child_id},  # 해당 아이의 대화만
                top_k=top_k,
                include_metadata=True
            )

            # 3. 결과 포맷팅
            similar_conversations = []
            for match in results['matches']:
                similar_conversations.append({
                    "message": match['metadata'].get('message', ''),
                    "response": match['metadata'].get('response', ''),
                    "session_id": match['metadata'].get('session_id'),
                    "score": match['score'],
                    "created_at": match['metadata'].get('created_at')
                })

            return similar_conversations

        except Exception as e:
            print(f"❌ Pinecone search failed: {e}")
            return []

    async def _get_embedding(self, text: str) -> List[float]:
        """OpenAI Embedding API로 텍스트 벡터화"""
        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            return []

    # ========== 통합 메모리 검색 (Hybrid) ==========

    async def get_relevant_context(
        self,
        current_message: str,
        child_id: int,
        session_id: int,
        use_semantic_search: bool = None
    ) -> Dict[str, Any]:
        """
        현재 메시지와 관련된 모든 컨텍스트 수집 (하이브리드)

        Args:
            current_message: 현재 사용자 메시지
            child_id: 아이 ID
            session_id: 현재 세션 ID
            use_semantic_search: None이면 self.use_pinecone 사용

        Returns:
            {
                "recent_conversations": [...],  # 최근 대화 10개
                "similar_conversations": [...], # 유사한 과거 대화 5개 (Pinecone)
                "story_completions": [...],     # 완료한 동화 5개
                "summary": "요약 텍스트"
            }
        """
        if use_semantic_search is None:
            use_semantic_search = self.use_pinecone

        print(f"\n=== Memory Retrieval Start ===")
        print(f"child_id: {child_id}, session_id: {session_id}")
        print(f"use_semantic_search: {use_semantic_search}")

        # 병렬로 모든 데이터 가져오기
        recent_convs = await self.get_recent_conversations(child_id, limit=10)
        story_completions = await self.get_story_completions(child_id, limit=5)

        similar_convs = []
        if use_semantic_search:
            similar_convs = await self.search_similar_conversations(
                current_message,
                child_id,
                top_k=5
            )

        # 컨텍스트 요약 생성 (아이 이름 포함)
        summary_result = self._create_context_summary(
            recent_convs,
            story_completions,
            similar_convs
        )

        print(f"✅ Memory retrieved: {len(recent_convs)} recent, {len(similar_convs)} similar, {len(story_completions)} stories")
        if summary_result.get("child_name"):
            print(f"👤 아이 이름 추출: {summary_result['child_name']}")

        return {
            "recent_conversations": recent_convs,
            "similar_conversations": similar_convs,
            "story_completions": story_completions,
            "summary": summary_result["summary"],
            "child_name": summary_result.get("child_name")  # [2025-11-11 추가]
        }

    def _create_context_summary(
        self,
        recent_convs: List[Dict],
        story_completions: List[Dict],
        similar_convs: List[Dict]
    ) -> Dict[str, Any]:
        """
        [2025-11-11 수정] 컨텍스트를 AI가 이해하기 쉬운 텍스트로 요약
        Returns:
            {
                "summary": "요약 텍스트",
                "child_name": "아이 이름" (있으면)
            }
        """

        summary_parts = []
        child_name = None

        # 1. 동화 완료 기록
        if story_completions:
            summary_parts.append("**완료한 동화:**")
            for story in story_completions[:3]:  # 최근 3개만
                title = story.get('storyTitle', '알 수 없음')  # camelCase로 변경!

                # [2025-11-11 추가] 첫 번째 동화에서 아이 이름 추출
                if child_name is None:
                    child_name = story.get('childName')

                # 능력치 정보 추출 (Java DTO의 total* 필드들)
                abilities = {
                    'courage': story.get('totalCourage', 0),
                    'empathy': story.get('totalEmpathy', 0),
                    'creativity': story.get('totalCreativity', 0),
                    'responsibility': story.get('totalResponsibility', 0),
                    'friendship': story.get('totalFriendship', 0)
                }
                ability_text = self._format_abilities(abilities)
                summary_parts.append(f"  - '{title}' ({ability_text})")

        # 2. 최근 대화 요약
        if recent_convs:
            summary_parts.append("\n**최근 대화 주제:**")
            topics = self._extract_topics_from_conversations(recent_convs[:5])
            summary_parts.append(f"  - {topics}")

        # 3. 유사한 과거 대화 (있으면) - [2025-11-11 수정] 전체 내용 포함
        if similar_convs:
            summary_parts.append("\n**관련된 과거 대화 (시맨틱 검색 결과):**")
            for idx, conv in enumerate(similar_convs[:5], 1):  # 상위 5개
                user_msg = conv.get('message', '')
                ai_response = conv.get('response', '')
                score = conv.get('score', 0)

                # 전체 내용 포함 (잘라내지 않음)
                summary_parts.append(f"  {idx}. [유사도: {score:.2f}]")
                summary_parts.append(f"     사용자: {user_msg}")
                summary_parts.append(f"     AI: {ai_response}")

        summary_text = "\n".join(summary_parts) if summary_parts else "이전 대화 기록 없음"

        return {
            "summary": summary_text,
            "child_name": child_name
        }

    def _format_abilities(self, abilities: Dict[str, int]) -> str:
        """능력치를 간단한 텍스트로"""
        ability_names = {
            "courage": "용기", "empathy": "공감", "creativity": "창의성",
            "responsibility": "책임감", "friendship": "우정"
        }

        parts = []
        for key, value in abilities.items():
            if value > 0:
                name = ability_names.get(key, key)
                parts.append(f"{name}+{value}")

        return ", ".join(parts) if parts else "능력치 없음"

    def _extract_topics_from_conversations(self, convs: List[Dict]) -> str:
        """최근 대화에서 주요 주제 추출 (간단한 키워드 추출)"""
        # 간단한 구현: 메시지들을 합쳐서 요약
        messages = []
        for conv in convs:
            msg = conv.get('message', '')
            if msg and len(msg) > 3:
                messages.append(msg)

        if messages:
            combined = " / ".join(messages[:3])  # 최근 3개 메시지
            return combined[:100] + "..." if len(combined) > 100 else combined

        return "대화 기록 없음"

    # ========== Pinecone 데이터 동기화 (관리용) ==========

    async def sync_conversation_to_pinecone(
        self,
        session_id: int,
        child_id: int,
        user_message: str,
        ai_response: str,
        message_id: int
    ):
        """
        새로운 대화를 Pinecone에 저장

        MySQL에 저장된 후 호출되어야 함
        """
        if not self.use_pinecone:
            return

        try:
            # 1. 임베딩 생성 (사용자 메시지 기준)
            embedding = await self._get_embedding(user_message)

            if not embedding:
                return

            # 2. 메타데이터 구성
            metadata = {
                "child_id": child_id,
                "session_id": session_id,
                "message": user_message,
                "response": ai_response,
                "created_at": datetime.utcnow().isoformat(),
                "message_id": message_id
            }

            # 3. Pinecone에 저장
            self.index.upsert(
                vectors=[
                    {
                        "id": f"msg_{message_id}",
                        "values": embedding,
                        "metadata": metadata
                    }
                ]
            )

            print(f"✅ Synced conversation to Pinecone: msg_{message_id}")

        except Exception as e:
            print(f"❌ Failed to sync to Pinecone: {e}")

    async def sync_story_completion_to_pinecone(
        self,
        completion_id: int,
        child_id: int,
        story_title: str,
        story_content: str,  # scene_json, story_json 합친 텍스트
        abilities: Dict[str, int]
    ):
        """
        동화 완료 기록을 Pinecone에 저장

        나중에 "어떤 동화 읽었지?" 같은 질문에 검색 가능
        """
        if not self.use_pinecone:
            return

        try:
            # 스토리 내용 요약 생성
            story_summary = f"동화 제목: {story_title}\n내용: {story_content[:500]}"

            # 임베딩 생성
            embedding = await self._get_embedding(story_summary)

            if not embedding:
                return

            # 메타데이터
            metadata = {
                "child_id": child_id,
                "completion_id": completion_id,
                "story_title": story_title,
                "abilities": abilities,
                "type": "story_completion",
                "completed_at": datetime.utcnow().isoformat()
            }

            # Pinecone에 저장
            self.index.upsert(
                vectors=[
                    {
                        "id": f"story_{completion_id}",
                        "values": embedding,
                        "metadata": metadata
                    }
                ]
            )

            print(f"✅ Synced story completion to Pinecone: story_{completion_id}")

        except Exception as e:
            print(f"❌ Failed to sync story to Pinecone: {e}")
