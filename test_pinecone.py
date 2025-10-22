"""
Pinecone 연동 테스트 스크립트
"""
from dotenv import load_dotenv
import sys
import os

# 환경변수 로드
load_dotenv()

# 현재 디렉토리를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print(f"Current directory: {current_dir}")
print(f"Python path: {sys.path[0]}")

# import 전에 경로 확인
story_service_path = os.path.join(current_dir, 'app', 'services', 'story', 'story_generator.py')
print(f"Story service file exists: {os.path.exists(story_service_path)}")

try:
    from app.services.story.story_generator import StorySearchService
    print("✅ Import 성공!")
except ImportError as e:
    print(f"❌ Import 실패!: {e}")
    sys.exit(1)

def test_story_search():
    print("=" * 50)
    print("Pinecone 동화 검색 테스트")
    print("=" * 50)
    
    # 서비스 초기화
    print("\n1. StorySearchService 초기화 중...")
    service = StorySearchService()
    
    if not service.index:
        print("⚠️  Pinecone 연결 안됨 - 더미 데이터로 테스트합니다")
    else:
        print("✅ Pinecone 연결 성공!")
    
    # 검색 테스트 케이스
    test_cases = [
        {
            "emotion": "화나요",
            "interests": ["친구", "동물"],
            "description": "화가 났을 때 친구와 동물 관련 동화"
        },
        {
            "emotion": "슬퍼요",
            "interests": ["가족"],
            "description": "슬플 때 가족 관련 동화"
        },
        {
            "emotion": "신나요",
            "interests": ["모험", "우주"],
            "description": "신날 때 모험과 우주 관련 동화"
        }
    ]
    
    for idx, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 50}")
        print(f"테스트 케이스 {idx}: {test_case['description']}")
        print(f"{'=' * 50}")
        print(f"감정: {test_case['emotion']}")
        print(f"관심사: {', '.join(test_case['interests'])}")
        
        # 검색 실행
        results = service.search_stories(
            emotion=test_case['emotion'],
            interests=test_case['interests'],
            top_k=3
        )
        
        print(f"\n검색 결과: {len(results)}개")
        for i, story in enumerate(results, 1):
            print(f"\n{i}. {story['title']}")
            print(f"   - 매칭 점수: {story['matching_score']}점")
            print(f"   - Story ID: {story['story_id']}")
            if 'metadata' in story:
                print(f"   - 분류: {story['metadata'].get('classification', 'N/A')}")
                print(f"   - 요약: {story['metadata'].get('plotSummaryText', 'N/A')[:50]}...")
    
    print("\n" + "=" * 50)
    print("테스트 완료!")
    print("=" * 50)

if __name__ == "__main__":
    test_story_search()