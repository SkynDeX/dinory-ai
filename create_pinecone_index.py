"""
Pinecone 챗봇 메모리 인덱스 생성 스크립트

이 스크립트를 실행하면 'chatbot-memory-index' 인덱스가 자동 생성됩니다.
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# .env 파일 로드
load_dotenv()

def create_chatbot_index():
    """챗봇 메모리용 Pinecone 인덱스 생성"""

    # API Key 확인
    api_key = os.getenv("CHATBOT_PINECONE_API_KEY")
    index_name = os.getenv("CHATBOT_PINECONE_INDEX_NAME", "chatbot-memory-index")

    if not api_key:
        print("❌ Error: CHATBOT_PINECONE_API_KEY not found in .env")
        return False

    print(f"\n🔧 Creating Pinecone index for chatbot memory...")
    print(f"Index Name: {index_name}")
    print(f"API Key: {api_key[:20]}...")

    try:
        # Pinecone 초기화
        pc = Pinecone(api_key=api_key)

        # 기존 인덱스 확인
        existing_indexes = pc.list_indexes()
        index_names = [idx.name for idx in existing_indexes]

        if index_name in index_names:
            print(f"✅ Index '{index_name}' already exists!")

            # 인덱스 정보 출력
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            print(f"\nIndex Stats:")
            print(f"  - Total vectors: {stats.get('total_vector_count', 0)}")
            print(f"  - Dimension: {stats.get('dimension', 'N/A')}")
            print(f"  - Index fullness: {stats.get('index_fullness', 0)}")

            return True

        # 새 인덱스 생성
        print(f"\n📦 Creating new index '{index_name}'...")

        pc.create_index(
            name=index_name,
            dimension=1536,  # text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        print(f"✅ Index '{index_name}' created successfully!")
        print(f"\n📊 Index Configuration:")
        print(f"  - Dimension: 1536 (text-embedding-3-small)")
        print(f"  - Metric: cosine")
        print(f"  - Cloud: AWS")
        print(f"  - Region: us-east-1")
        print(f"\n🎉 Ready to use! You can now start the FastAPI server.")

        return True

    except Exception as e:
        print(f"\n❌ Error creating index: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Check your API key in .env")
        print(f"2. Make sure you have Pinecone credits")
        print(f"3. Verify region availability (us-east-1)")
        return False


def check_index_health():
    """인덱스 상태 확인"""

    api_key = os.getenv("CHATBOT_PINECONE_API_KEY")
    index_name = os.getenv("CHATBOT_PINECONE_INDEX_NAME", "chatbot-memory-index")

    if not api_key:
        print("❌ CHATBOT_PINECONE_API_KEY not found")
        return

    try:
        pc = Pinecone(api_key=api_key)

        # 인덱스 목록 조회
        indexes = pc.list_indexes()
        index_names = [idx.name for idx in indexes]

        print(f"\n📋 Available Pinecone Indexes:")
        for idx in indexes:
            print(f"  - {idx.name} ({idx.dimension} dimensions, {idx.metric} metric)")

        if index_name in index_names:
            print(f"\n✅ Chatbot index '{index_name}' is available!")

            index = pc.Index(index_name)
            stats = index.describe_index_stats()

            print(f"\n📊 Index Stats:")
            print(f"  - Total vectors: {stats.get('total_vector_count', 0)}")
            print(f"  - Dimension: {stats.get('dimension', 'N/A')}")
            print(f"  - Namespaces: {list(stats.get('namespaces', {}).keys())}")

        else:
            print(f"\n⚠️ Chatbot index '{index_name}' not found!")
            print(f"Run this script to create it.")

    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Pinecone Chatbot Memory Index Setup")
    print("=" * 60)

    # 메뉴 선택
    print("\nWhat would you like to do?")
    print("1. Create chatbot-memory-index (if not exists)")
    print("2. Check index health")
    print("3. Both")

    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice == "1":
        create_chatbot_index()
    elif choice == "2":
        check_index_health()
    elif choice == "3":
        create_chatbot_index()
        check_index_health()
    else:
        print("Invalid choice. Running both...")
        create_chatbot_index()
        check_index_health()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
