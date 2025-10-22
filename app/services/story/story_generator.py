from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class StorySearchService:
    """Pinecone을 사용한 동화 검색 서비스"""

    def __init__(self):
        try:
            # Pinecone 초기화
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                logger.warning("PINECONE_API_KEY 찾을 수 없음")
                self.index = None
                self.openai_client = None
                return
            
            self.pc = Pinecone(api_key=api_key)
            self.index_name = os.getenv("PINECONE_INDEX_NAME", "story-embeddings")
            
            # 인덱스 연결
            try:
                self.index = self.pc.Index(self.index_name)
                logger.info(f"Pinecone 인덱스에 연결됨: {self.index_name}")
            except Exception as e:
                logger.error(f"Pinecone 인덱스에 연결하지 못했습니다.: {e}")
                self.index = None
            
            # OpenAI 클라이언트 초기화 (임베딩용)
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=openai_key)
                logger.info("임베딩을 위해 OpenAI 클라이언트가 초기화되었습니다.")
            else:
                logger.warning("OPENAI_API_KEY를 찾을 수 없습니다. 더미 데이터 사용")
                self.openai_client = None
            
        except Exception as e:
            logger.error(f"StorySearchService 초기화 오류: {e}")
            self.index = None
            self.openai_client = None

    def create_search_query(self, emotion: str, interests: List[str]) -> str:
        """
        감정과 관심사를 조합해서 검색 쿼리생성

        Args:
            emotion: 선택한 감정 ("기뻐요", "슬퍼요", "화나요" 등)
            interests: 관심사 리스트(["친구", "동물"] 등)

        Returns:
            검색용 쿼리 문자열
        """
        # 감정을 주제로 매핑
        emotion_map = {
            "기뻐요": "기쁨 행복 즐거움 웃음 축하 신남 좋아함",
            "슬퍼요": "슬픔 눈물 위로 공감 아픔 상처 헤어짐 그리움",
            "화나요": "화남 분노 짜증 싸움 갈등 미안함 용서 화해",
            "무서워요": "두려움 공포 무서움 용기 극복 도전 강함",
            "신나요": "신남 모험 탐험 재미 활기 에너지 활동",
            "피곤해요": "피곤 휴식 평온 편안 잠 쉼 여유"
        }

        emotion_text = emotion_map.get(emotion, emotion)
        interests_text = " ".join(interests)

        # 검색 쿼리 조합
        query = f'{emotion_text} {interests_text} 동화 이야기'
        logger.info(f'검색어가 생성 : {query}')
        return query

    def search_stories(self, emotion: str, interests: List[str], top_k: int = 5) -> List[Dict]:
        """
        Pinecone에서 감정과 관심사에 맞는 동화 검색
        
        Args:
            emotion: 선택한 감정
            interests: 관심사 리스트
            top_k: 반환할 결과 개수
        
        Returns:
            검색된 동화 리스트
        """
        if not self.index:
            logger.error("인덱스가 초기화되지 않음")
            return self._get_dummy_stories(emotion, interests, top_k)
        
        if not self.openai_client:
            logger.error("OpenAI 클라이언트가 초기화되지 않음")
            return self._get_dummy_stories(emotion, interests, top_k)
        
        try:
            # 검색 쿼리 생성
            query_text = self.create_search_query(emotion, interests)
            
            # OpenAI로 쿼리 임베딩 생성 (3072 차원)
            logger.info("OpenAI로 임베딩 만드는 중...")
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=query_text
            )
            query_embedding = response.data[0].embedding
            logger.info(f"임베딩 생성됨: {len(query_embedding)}")
            
            # Pinecone 검색
            logger.info("Pinecone 검색중...")
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # 결과 변환
            stories = []
            for match in results['matches']:
                story = {
                    'story_id': match['id'],
                    'title': match['metadata'].get('title', '제목 없음'),
                    'matching_score': int(match['score'] * 100),  # 0-1 점수를 0-100으로 변환
                    'metadata': match['metadata']
                }
                stories.append(story)
            
            logger.info(f"Pinecone에서 {len(stories)}개의 스토리 찾음")
            return stories
            
        except Exception as e:
            logger.error(f"Pinecone에서 스토리를 검색하는 중 오류가 발생: {e}")
            # 에러 발생시 더미 데이터 반환
            return self._get_dummy_stories(emotion, interests, top_k)
    
    def _get_dummy_stories(self, emotion: str, interests: List[str], top_k: int) -> List[Dict]:
        """
        테스트용 더미 동화 데이터 반환
        (Pinecone 연결 안될 때 사용)
        """
        dummy_stories = [
            {
                'story_id': '9791193449196',
                'title': '정글에서 친구 찾기',
                'matching_score': 95,
                'metadata': {
                    'title': '정글에서 친구 찾기',
                    'author': '테스트 작가',
                    'classification': '의사소통',
                    'readAge': '유아',
                    'plotSummaryText': '원숭이가 친구를 찾아 떠나는 모험 이야기'
                }
            },
            {
                'story_id': '9791193449197',
                'title': '화난 토끼의 하루',
                'matching_score': 92,
                'metadata': {
                    'title': '화난 토끼의 하루',
                    'author': '테스트 작가',
                    'classification': '감정조절',
                    'readAge': '유아',
                    'plotSummaryText': '화가 난 토끼가 감정을 조절하는 법을 배우는 이야기'
                }
            },
            {
                'story_id': '9791193449198',
                'title': '용감한 강아지',
                'matching_score': 88,
                'metadata': {
                    'title': '용감한 강아지',
                    'author': '테스트 작가',
                    'classification': '용기',
                    'readAge': '유아',
                    'plotSummaryText': '겁이 많은 강아지가 용기를 내는 이야기'
                }
            },
            {
                'story_id': '9791193449199',
                'title': '숲속 친구들',
                'matching_score': 85,
                'metadata': {
                    'title': '숲속 친구들',
                    'author': '테스트 작가',
                    'classification': '우정',
                    'readAge': '유아',
                    'plotSummaryText': '숲속 동물들이 서로 도우며 우정을 나누는 이야기'
                }
            },
            {
                'story_id': '9791193449200',
                'title': '마법의 모험',
                'matching_score': 80,
                'metadata': {
                    'title': '마법의 모험',
                    'author': '테스트 작가',
                    'classification': '모험',
                    'readAge': '유아',
                    'plotSummaryText': '작은 마법사가 친구들과 함께 모험을 떠나는 이야기'
                }
            }
        ]
        
        logger.info(f"테스트를 위해 {top_k}개의 더미 스토리를 반환합니다.")
        return dummy_stories[:top_k]
    
    def get_story_by_id(self, story_id: str) -> Dict:
        """
        특정 story_id로 동화 조회
        
        Args:
            story_id: 조회할 동화 ID
        
        Returns:
            동화 데이터
        """
        if not self.index:
            logger.error("Pinecone 초기화되지 않았습니다")
            return None
        
        try:
            # Pinecone에서 특정 ID 조회
            result = self.index.fetch(ids=[story_id])
            
            if story_id in result['vectors']:
                vector_data = result['vectors'][story_id]
                return {
                    'story_id': story_id,
                    'title': vector_data['metadata'].get('title', '제목 없음'),
                    'metadata': vector_data['metadata']
                }
            else:
                logger.warning(f"스토리 {story_id}를 Pinecone에서 찾을 수 없습니다.")
                return None
                
        except Exception as e:
            logger.error(f"스토리를 가져오는 중 오류가 발생 {story_id}: {e}")
            return None