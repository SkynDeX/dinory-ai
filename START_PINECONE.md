# 🚀 DinoCharacter RAG + Pinecone 시작하기

**5분 안에 설정 완료!**

---

## ✅ 현재 상태

모든 코드와 설정이 완료되었습니다!

- ✅ `.env`에 Pinecone API Key 설정됨
- ✅ RAG 메모리 활성화됨 (`USE_RAG_MEMORY=true`)
- ✅ Pinecone 활성화됨 (`USE_PINECONE_MEMORY=true`)
- ✅ `chat.py`가 RAG 서비스 사용하도록 수정됨
- ✅ 모든 필요한 파일 생성됨

---

## 🎯 해야 할 일 (3단계)

### 1️⃣ Pinecone 인덱스 생성 (5분)

Windows PowerShell 또는 CMD에서:

```cmd
cd D:\react\3stproject\dinory-ai
python create_pinecone_index.py
```

**실행 화면:**
```
What would you like to do?
1. Create chatbot-memory-index (if not exists)
2. Check index health
3. Both

Enter choice (1/2/3): 1
```

**입력**: `1` 입력 후 엔터

**성공 메시지:**
```
✅ Index 'chatbot-memory-index' created successfully!
🎉 Ready to use! You can now start the FastAPI server.
```

---

### 2️⃣ Spring Boot API 엔드포인트 추가 (30분)

파일 열기: `dinory-backend/REQUIRED_API_ENDPOINTS.md`

다음 2개 파일 수정:

#### A. `ChatMessageRepository.java`에 메서드 추가
```java
@Query("SELECT cm FROM ChatMessage cm JOIN cm.chatSession cs WHERE cs.childId = :childId ORDER BY cm.createdAt DESC")
List<ChatMessage> findRecentMessagesByChildId(@Param("childId") Long childId, Pageable pageable);
```

#### B. `ChatController.java`에 엔드포인트 추가
```java
@GetMapping("/chat/history/child/{childId}")
public List<ChatMessageDto> getChatHistoryByChild(
    @PathVariable Long childId,
    @RequestParam(defaultValue = "10") int limit
) {
    // ... (REQUIRED_API_ENDPOINTS.md 참고)
}
```

#### C. `StoryCompletionRepository.java`에 메서드 추가
```java
@Query("SELECT sc FROM StoryCompletion sc WHERE sc.childId = :childId ORDER BY sc.createdAt DESC")
List<StoryCompletion> findRecentCompletionsByChildId(Long childId, Pageable pageable);
```

#### D. `StoryController.java`에 엔드포인트 추가
```java
@GetMapping("/story/completions/child/{childId}")
public List<StoryCompletionSummaryDto> getStoryCompletionsByChild(
    @PathVariable Long childId,
    @RequestParam(defaultValue = "5") int limit
) {
    // ... (REQUIRED_API_ENDPOINTS.md 참고)
}
```

**자세한 코드**: `dinory-backend/REQUIRED_API_ENDPOINTS.md` 파일 전체 참고

---

### 3️⃣ 서버 재시작 (2분)

#### Spring Boot 재시작
```bash
cd D:\react\3stproject\dinory-backend
# Spring Boot 재시작 (IntelliJ 또는 mvn spring-boot:run)
```

#### FastAPI 재시작
```bash
cd D:\react\3stproject\dinory-ai
python main.py
```

**로그 확인 (FastAPI):**
```
✅ RAG Memory ENABLED (Pinecone: True)
✅ Pinecone initialized: chatbot-memory-index
[startup] Dinory AI API Starting…
```

---

## 🧪 테스트

### 1. 메모리 서비스 상태 확인

브라우저에서:
```
http://localhost:8000/api/memory/health
```

**예상 응답:**
```json
{
  "pinecone_enabled": true,
  "spring_api_url": "http://localhost:8090/api",
  "status": "healthy"
}
```

### 2. Spring Boot API 테스트

브라우저에서:
```
http://localhost:8090/api/chat/history/child/1?limit=10
```

**예상 응답:**
```json
[
  {
    "sessionId": 50,
    "message": "안녕!",
    "sender": "USER",
    "createdAt": "2025-10-29T15:30:00"
  }
]
```

### 3. DinoCharacter 챗봇 테스트

1. `http://localhost:3000` 접속
2. DinoCharacter(공룡) 클릭
3. 채팅창 열림

**테스트 대화:**
```
[사용자] "지난번에 뭐 읽었어?"

[AI - RAG 작동 시]
"지난번에 '용감한 디노' 동화 읽었잖아!
그때 용기를 31점이나 얻었어! 기억나니? 😊"
```

---

## 📊 로그 확인 방법

### FastAPI 콘솔에서 다음 로그 확인:

```
=== generate_response with RAG ===
session_id: 50, child_id: 1
message: 지난번에 뭐 읽었어?

=== Memory Retrieval Start ===
child_id: 1, session_id: 50
use_semantic_search: True

✅ Memory retrieved: 10 recent, 5 similar, 3 stories

**아이의 기억 (과거 기록):**
**완료한 동화:**
  - '용감한 디노' (용기+31, 공감+10, 창의성+2, 책임감+12)
  - '친구를 돕는 디노' (우정+25, 공감+15)

**최근 대화 주제:**
  - 안녕! / 안녕! 반가워~ / 오늘 기분이 어때?
```

이 로그가 보이면 **RAG + Pinecone이 정상 작동 중**입니다!

---

## 🐛 문제 해결

### "Index not found: chatbot-memory-index"
→ 1단계(Pinecone 인덱스 생성) 다시 실행

### "Failed to fetch conversations from MySQL: 404"
→ 2단계(Spring Boot API) 확인. 엔드포인트 추가했는지 확인.

### "Pinecone initialization failed"
→ `.env` 파일 확인:
```
CHATBOT_PINECONE_API_KEY=pcsk_Q8CsM_...
```

### AI가 과거를 기억 못 함
→ FastAPI 콘솔에서 "Memory Retrieval" 로그 확인
→ 없으면 `USE_RAG_MEMORY=true` 설정 확인 후 재시작

---

## 🎁 완료 후 기능

DinoCharacter가 이제:

✅ **과거 대화 기억**
- "전에 무슨 얘기했지?" → 정확히 기억

✅ **동화 기록 참조**
- "지난번에 뭐 읽었어?" → "용감한 디노 읽었잖아!"

✅ **시맨틱 검색**
- "용기 관련 동화 읽었을 때..." → 유사한 과거 대화 검색

✅ **개인화된 대화**
- 각 아이의 성향, 관심사, 감정 기억

---

## 📚 추가 문서

- **아키텍처**: `RAG_SETUP_GUIDE.md`
- **구현 가이드**: `IMPLEMENTATION_GUIDE.md`
- **Pinecone 설정**: `PINECONE_SETUP.md`
- **의사결정 가이드**: `DECISION_GUIDE.md`
- **Spring Boot 코드**: `dinory-backend/REQUIRED_API_ENDPOINTS.md`

---

## ✅ 체크리스트

- [ ] 1단계: `python create_pinecone_index.py` 실행
- [ ] 2단계: Spring Boot API 엔드포인트 4개 추가
- [ ] 3단계: Spring Boot + FastAPI 재시작
- [ ] 4단계: `/api/memory/health` 테스트
- [ ] 5단계: DinoCharacter 챗봇 테스트
- [ ] 6단계: FastAPI 로그 확인

---

**🚀 시작하세요! 1단계부터 차근차근 진행하면 됩니다!**

문제 발생 시 각 문서의 "문제 해결" 섹션 참고하세요.
