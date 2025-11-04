# RAG Memory Implementation Guide

DinoCharacter 챗봇에 장기 메모리 기능을 추가하는 완전한 구현 가이드

---

## 📌 Quick Start (5분 안에 시작하기)

### 1단계: Spring Boot API 엔드포인트 추가

`dinory-backend/REQUIRED_API_ENDPOINTS.md` 파일을 열고 따라하세요:

1. `ChatMessageRepository.java`에 메서드 추가
2. `StoryCompletionRepository.java`에 메서드 추가
3. `ChatController.java`에 엔드포인트 추가
4. `StoryController.java`에 엔드포인트 추가

### 2단계: FastAPI 활성화

```bash
# .env 파일 수정
echo "USE_RAG_MEMORY=true" >> .env
```

### 3단계: FastAPI 재시작

```bash
cd d:\react\3stproject\dinory-ai
python main.py
```

### 4단계: 테스트

브라우저에서 DinoCharacter 클릭 후:
```
사용자: "지난번에 뭐 읽었어?"
디노: "지난번에 '용감한 디노' 동화 읽었잖아! 그때 용기를 31점이나 얻었어 😊"
```

끝! 이제 디노가 과거 기록을 기억합니다.

---

## 🎯 구현된 기능

### MySQL 기반 RAG (기본)
- ✅ 과거 대화 기록 조회 (최근 10개)
- ✅ 완료한 동화 기록 조회 (최근 5개)
- ✅ 동화 능력치 정보 참조
- ✅ 컨텍스트 요약 자동 생성
- ✅ AI에게 자동으로 전달

### Pinecone 기반 RAG (옵션)
- ⚙️ 시맨틱 검색 (의미 기반 유사도)
- ⚙️ 대량 데이터 고속 검색
- ⚙️ 자동 데이터 동기화
- ⚙️ "비슷한 대화 했었는데..." 같은 고급 기능

---

## 📂 생성된 파일 목록

### FastAPI (dinory-ai)
```
app/
├── services/
│   └── chat/
│       ├── memory_service.py          # ⭐ RAG 메모리 서비스 (핵심)
│       ├── chatbot_service.py         # 기존 서비스 (유지)
│       └── chatbot_service_with_rag.py # ⭐ RAG 통합 서비스 (새로운)
├── api/
│   └── endpoints/
│       ├── chat.py                    # 수정 필요 (아래 참고)
│       └── memory_sync.py             # ⭐ Pinecone 동기화 엔드포인트
└── main.py                             # ✅ 이미 업데이트됨

.env                                    # ✅ CHATBOT_PINECONE 설정 추가됨
RAG_SETUP_GUIDE.md                      # 📖 아키텍처 가이드
IMPLEMENTATION_GUIDE.md                 # 📖 이 파일
```

### Spring Boot (dinory-backend)
```
REQUIRED_API_ENDPOINTS.md               # 📖 추가할 엔드포인트 가이드

추가 필요:
├── ChatController.java                 # /api/chat/history/child/{childId} 엔드포인트
├── StoryController.java                # /api/story/completions/child/{childId} 엔드포인트
├── ChatMessageRepository.java          # findRecentMessagesByChildId() 메서드
└── StoryCompletionRepository.java      # findRecentCompletionsByChildId() 메서드
```

---

## 🚀 FastAPI 적용 방법

### 방법 1: 기존 chatbot_service.py 대체 (완전 전환)

```bash
# 백업
mv app/services/chat/chatbot_service.py app/services/chat/chatbot_service_old.py

# 새 서비스를 기본으로 사용
mv app/services/chat/chatbot_service_with_rag.py app/services/chat/chatbot_service.py
```

**장점**: 간단함
**단점**: 롤백 시 번거로움

### 방법 2: 환경변수로 제어 (추천)

`app/api/endpoints/chat.py` 파일 수정:

```python
import os
from app.services.chat.chatbot_service import ChatbotService  # 기존
from app.services.chat.chatbot_service_with_rag import ChatbotServiceWithRAG  # 새로운

# 환경변수로 RAG 사용 여부 결정
USE_RAG = os.getenv("USE_RAG_MEMORY", "false").lower() == "true"
USE_PINECONE = os.getenv("USE_PINECONE_MEMORY", "false").lower() == "true"

_chatbot_service = None

def get_chatbot_service():
    global _chatbot_service
    if _chatbot_service is None:
        if USE_RAG:
            print(f"✅ RAG Memory ENABLED (Pinecone: {USE_PINECONE})")
            _chatbot_service = ChatbotServiceWithRAG(use_pinecone=USE_PINECONE)
        else:
            print("⚠️ RAG Memory DISABLED (using basic service)")
            _chatbot_service = ChatbotService()
    return _chatbot_service
```

**.env 설정:**
```bash
# MySQL 기반 RAG만 사용
USE_RAG_MEMORY=true
USE_PINECONE_MEMORY=false

# MySQL + Pinecone 하이브리드 (API Key 필요)
USE_RAG_MEMORY=true
USE_PINECONE_MEMORY=true
CHATBOT_PINECONE_API_KEY=pcsk_YOUR_KEY_HERE
```

**장점**: 언제든지 on/off 가능
**단점**: 설정 파일 관리 필요

---

## 🔧 Pinecone 설정 (선택사항)

### 1. Pinecone 인덱스 생성

https://app.pinecone.io 에서:

```
Index Name: chatbot-memory-index
Dimensions: 1536
Metric: cosine
Pod Type: p1.x1 (Starter)
Replicas: 1
```

### 2. API Key 발급

Settings → API Keys → Create API Key

### 3. .env 설정

```bash
CHATBOT_PINECONE_API_KEY=pcsk_YOUR_API_KEY_HERE
CHATBOT_PINECONE_INDEX_NAME=chatbot-memory-index
```

### 4. FastAPI 활성화

```bash
USE_PINECONE_MEMORY=true
```

### 5. 데이터 동기화 설정

Spring Boot `ChatService.java`의 `sendMessage()` 메서드 끝에 추가:

```java
// Pinecone 동기화 (비동기, 실패해도 무시)
try {
    String syncUrl = aiServerUrl + "/api/memory/sync/conversation";
    Map<String, Object> syncBody = new HashMap<>();
    syncBody.put("session_id", sessionId.intValue());
    syncBody.put("child_id", session.getChildId().intValue());
    syncBody.put("user_message", userMessage.getMessage());
    syncBody.put("ai_response", aiMessage.getMessage());
    syncBody.put("message_id", aiMessage.getId().intValue());

    HttpEntity<Map<String, Object>> syncEntity = new HttpEntity<>(syncBody, headers);
    restTemplate.postForObject(syncUrl, syncEntity, Map.class);
} catch (Exception e) {
    log.warn("Pinecone sync failed (non-critical): ", e);
}
```

---

## 🧪 테스트 시나리오

### 시나리오 1: 과거 동화 기억
```
[사용자] "지난번에 뭐 읽었어?"

[AI 응답 - RAG 없으면]
"뭘 읽었는지 잘 기억이 안 나..."

[AI 응답 - RAG 있으면]
"지난번에 '용감한 디노' 동화 읽었잖아! 그때 용기를 31점이나 얻었어! 기억나니? 😊"
```

### 시나리오 2: 과거 대화 참조
```
[이전 대화]
사용자: "나 오늘 학교에서 친구랑 싸웠어"
AI: "속상했겠다... 무슨 일이 있었어?"

[나중에...]
사용자: "친구랑 화해했어!"

[AI 응답 - RAG 없으면]
"잘했어! 무슨 일이었는데?"

[AI 응답 - RAG 있으면]
"진짜? 학교에서 싸웠던 친구랑? 정말 잘했어! 어떻게 화해했어? 😊"
```

### 시나리오 3: 능력치 질문
```
[동화 완료 후]
사용자: "내가 무슨 능력치 얻었어?"

[AI 응답]
"용기 31점, 공감 10점, 창의성 2점, 책임감 12점을 얻었어! 용기가 제일 많이 올랐네! 👍"
```

---

## 📊 로그 확인

### FastAPI 로그
```bash
=== Memory Retrieval Start ===
child_id: 1, session_id: 50
use_semantic_search: False
✅ Memory retrieved: 10 recent, 0 similar, 3 stories

**아이의 기억 (과거 기록):**
**완료한 동화:**
  - '용감한 디노' (용기+31, 공감+10, 창의성+2, 책임감+12)
  - '친구를 돕는 디노' (우정+25, 공감+15)
  - '숲속의 보물' (창의성+20, 책임감+10)

**최근 대화 주제:**
  - 안녕! / 안녕! 반가워~ / 오늘 기분이 어때?
```

### Spring Boot 로그
```bash
INFO  ChatController - GET /api/chat/history/child/1?limit=10
INFO  ChatController - Returned 10 messages for child 1
```

---

## 🐛 문제 해결

### "Memory retrieval failed: 404"
→ Spring Boot API 엔드포인트가 없음. `REQUIRED_API_ENDPOINTS.md` 참고.

### "Pinecone is disabled"
→ `.env`에 `CHATBOT_PINECONE_API_KEY` 없음. 설정하거나 MySQL만 사용.

### AI가 과거 기억을 참조하지 않음
→ `USE_RAG_MEMORY=true` 설정 확인. FastAPI 재시작.

### Spring Boot API 호출 실패
→ CORS 설정 확인. `@CrossOrigin(origins = "*")` 또는 전역 CORS 설정.

---

## 📈 성능 최적화

### MySQL 인덱스 추가
```sql
-- chat_message 테이블
CREATE INDEX idx_chat_session_child_id ON chat_session(child_id);
CREATE INDEX idx_chat_message_created_at ON chat_message(created_at DESC);

-- story_completion 테이블
CREATE INDEX idx_story_completion_child_created ON story_completion(child_id, created_at DESC);
```

### 캐싱 추가 (향후)
```python
# memory_service.py에 추가
from functools import lru_cache

@lru_cache(maxsize=100)
async def get_recent_conversations_cached(child_id, limit):
    # 최근 대화는 1분간 캐싱
    pass
```

---

## 🎁 추가 기능 아이디어

1. **대화 요약**: 긴 대화를 요약하여 컨텍스트 크기 축소
2. **감정 추적**: 아이의 감정 변화 기록 및 참조
3. **관심사 학습**: 자주 언급하는 주제 파악
4. **추천 시스템**: 과거 선호도 기반 동화 추천
5. **부모 리포트**: 대화 내용 요약하여 부모에게 전달

---

## ✅ 체크리스트

### Spring Boot
- [ ] ChatMessageRepository에 `findRecentMessagesByChildId` 추가
- [ ] StoryCompletionRepository에 `findRecentCompletionsByChildId` 추가
- [ ] ChatController에 `/api/chat/history/child/{childId}` 추가
- [ ] StoryController에 `/api/story/completions/child/{childId}` 추가
- [ ] CORS 설정 확인 (`@CrossOrigin` 또는 전역 설정)
- [ ] Spring Boot 재시작

### FastAPI
- [ ] `.env`에 `USE_RAG_MEMORY=true` 추가
- [ ] `chat.py`에서 `ChatbotServiceWithRAG` 사용하도록 수정
- [ ] FastAPI 재시작
- [ ] `/api/memory/health` 엔드포인트 테스트

### 테스트
- [ ] DinoCharacter 클릭하여 챗봇 오픈
- [ ] "지난번에 뭐 읽었어?" 질문
- [ ] AI가 과거 동화 기록 참조하는지 확인
- [ ] FastAPI 로그에서 "Memory retrieved" 메시지 확인

### Pinecone (옵션)
- [ ] Pinecone 인덱스 생성
- [ ] `.env`에 `CHATBOT_PINECONE_API_KEY` 추가
- [ ] `USE_PINECONE_MEMORY=true` 설정
- [ ] Spring Boot에서 동기화 로직 추가
- [ ] 시맨틱 검색 테스트

---

## 📞 도움말

- **RAG 아키텍처**: `RAG_SETUP_GUIDE.md` 참고
- **Spring Boot 엔드포인트**: `dinory-backend/REQUIRED_API_ENDPOINTS.md` 참고
- **Pinecone 설정**: https://docs.pinecone.io
- **OpenAI Embeddings**: https://platform.openai.com/docs/guides/embeddings

---

## 🎉 완료!

이제 DinoCharacter가 사용자의 과거 대화와 동화 기록을 기억하며 더욱 개인화된 대화를 제공합니다!
