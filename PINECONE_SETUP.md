# Pinecone 챗봇 메모리 설정 가이드

DinoCharacter가 RAG + Pinecone으로 과거 대화와 동화 기록을 기억하도록 설정

---

## ✅ 현재 상태

- ✅ .env에 `CHATBOT_PINECONE_API_KEY` 설정됨
- ✅ .env에 `USE_RAG_MEMORY=true` 설정됨
- ✅ .env에 `USE_PINECONE_MEMORY=true` 설정됨
- ✅ `chat.py`가 `ChatbotServiceWithRAG` 사용하도록 수정됨
- ⚠️ Pinecone 인덱스 생성 필요 (아래 단계 진행)

---

## 🚀 1단계: Pinecone 인덱스 생성

### 방법 1: 자동 생성 스크립트 (추천)

```bash
cd d:\react\3stproject\dinory-ai
python create_pinecone_index.py
```

**실행 예시:**
```
============================================================
Pinecone Chatbot Memory Index Setup
============================================================

What would you like to do?
1. Create chatbot-memory-index (if not exists)
2. Check index health
3. Both

Enter choice (1/2/3): 1

🔧 Creating Pinecone index for chatbot memory...
Index Name: chatbot-memory-index
API Key: pcsk_Q8CsM_7S4Xe8xDHC...

📦 Creating new index 'chatbot-memory-index'...
✅ Index 'chatbot-memory-index' created successfully!

📊 Index Configuration:
  - Dimension: 1536 (text-embedding-3-small)
  - Metric: cosine
  - Cloud: AWS
  - Region: us-east-1

🎉 Ready to use! You can now start the FastAPI server.
```

### 방법 2: Pinecone 콘솔에서 수동 생성

https://app.pinecone.io 접속 후:

1. **Create Index** 클릭
2. 다음 정보 입력:
   ```
   Index Name: chatbot-memory-index
   Dimensions: 1536
   Metric: cosine
   Cloud Provider: AWS
   Region: us-east-1
   ```
3. **Create Index** 클릭

---

## 🔧 2단계: Spring Boot API 엔드포인트 추가

FastAPI가 MySQL에서 데이터를 가져오려면 Spring Boot에 API가 필요합니다.

### 필수 엔드포인트 2개

#### 1. 최근 대화 기록 조회
`ChatController.java`에 추가:

```java
@GetMapping("/chat/history/child/{childId}")
public List<ChatMessageDto> getChatHistoryByChild(
    @PathVariable Long childId,
    @RequestParam(defaultValue = "10") int limit
) {
    Pageable pageable = PageRequest.of(0, limit);
    List<ChatMessage> messages = chatMessageRepository
        .findRecentMessagesByChildId(childId, pageable);

    return messages.stream()
        .map(msg -> ChatMessageDto.builder()
            .sessionId(msg.getChatSession().getId())
            .message(msg.getMessage())
            .sender(msg.getSender())
            .createdAt(msg.getCreatedAt())
            .build())
        .collect(Collectors.toList());
}
```

#### 2. 동화 완료 기록 조회
`StoryController.java`에 추가:

```java
@GetMapping("/story/completions/child/{childId}")
public List<StoryCompletionSummaryDto> getStoryCompletionsByChild(
    @PathVariable Long childId,
    @RequestParam(defaultValue = "5") int limit
) {
    Pageable pageable = PageRequest.of(0, limit);
    List<StoryCompletion> completions = storyCompletionRepository
        .findRecentCompletionsByChildId(childId, pageable);

    return completions.stream()
        .map(StoryCompletionSummaryDto::from)
        .collect(Collectors.toList());
}
```

### Repository 메서드 추가

`ChatMessageRepository.java`:
```java
@Query("SELECT cm FROM ChatMessage cm JOIN cm.chatSession cs WHERE cs.childId = :childId ORDER BY cm.createdAt DESC")
List<ChatMessage> findRecentMessagesByChildId(@Param("childId") Long childId, Pageable pageable);
```

`StoryCompletionRepository.java`:
```java
@Query("SELECT sc FROM StoryCompletion sc WHERE sc.childId = :childId ORDER BY sc.createdAt DESC")
List<StoryCompletion> findRecentCompletionsByChildId(Long childId, Pageable pageable);
```

**자세한 코드**: `dinory-backend/REQUIRED_API_ENDPOINTS.md` 참고

---

## 🎯 3단계: FastAPI 재시작

```bash
cd d:\react\3stproject\dinory-ai
python main.py
```

**로그 확인:**
```
✅ RAG Memory ENABLED (Pinecone: True)
✅ Pinecone initialized: chatbot-memory-index
```

---

## 🧪 4단계: 테스트

### 테스트 1: 메모리 서비스 상태 확인

```bash
curl http://localhost:8000/api/memory/health
```

**예상 응답:**
```json
{
  "pinecone_enabled": true,
  "spring_api_url": "http://localhost:8090/api",
  "status": "healthy"
}
```

### 테스트 2: DinoCharacter 챗봇 테스트

1. 브라우저에서 `http://localhost:3000` 접속
2. DinoCharacter 클릭
3. 채팅 시작

**시나리오 1: 동화 기록 참조**
```
[사용자] "지난번에 뭐 읽었어?"

[AI - RAG 작동 시]
"지난번에 '용감한 디노' 동화 읽었잖아!
그때 용기를 31점이나 얻었어! 기억나니? 😊"
```

**시나리오 2: 과거 대화 참조**
```
[이전 대화]
사용자: "나 오늘 학교에서 친구랑 싸웠어"
AI: "속상했겠다... 무슨 일이 있었어?"

[나중에...]
사용자: "친구랑 화해했어!"

[AI - RAG 작동 시]
"진짜? 학교에서 싸웠던 친구랑?
정말 잘했어! 어떻게 화해했어? 😊"
```

### 테스트 3: FastAPI 로그 확인

채팅 시 콘솔에 다음과 같은 로그가 표시되어야 합니다:

```
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

**관련된 과거 대화:**
  - '나 오늘 학교에서 친구랑 싸웠어...'
  - '속상했겠다... 무슨 일이 있었어?...'
```

---

## 📊 5단계: Pinecone 데이터 동기화 (선택사항)

대화가 MySQL에 저장될 때 자동으로 Pinecone에도 저장하려면:

### ChatService.java의 sendMessage() 메서드에 추가

```java
@Transactional
public ChatResponseDto sendMessage(ChatMessageRequest request) {
    // ... 기존 코드 (사용자 메시지 저장, AI 응답 생성) ...

    chatMessageRepository.save(aiMessage);

    // ✨ Pinecone 동기화 (비동기, 실패해도 무시)
    syncToPinecone(userMessage, aiMessage, session.getChildId());

    return response;
}

private void syncToPinecone(ChatMessage userMsg, ChatMessage aiMsg, Long childId) {
    try {
        String syncUrl = aiServerUrl + "/api/memory/sync/conversation";

        Map<String, Object> syncBody = new HashMap<>();
        syncBody.put("session_id", userMsg.getChatSession().getId().intValue());
        syncBody.put("child_id", childId.intValue());
        syncBody.put("user_message", userMsg.getMessage());
        syncBody.put("ai_response", aiMsg.getMessage());
        syncBody.put("message_id", aiMsg.getId().intValue());

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(syncBody, headers);

        // 비동기로 호출 (실패해도 채팅은 계속 진행)
        restTemplate.postForObject(syncUrl, entity, Map.class);

        log.info("✅ Synced to Pinecone: msg_{}", aiMsg.getId());
    } catch (Exception e) {
        log.warn("⚠️ Pinecone sync failed (non-critical): ", e);
        // 실패해도 무시 (채팅 기능에 영향 없음)
    }
}
```

---

## 🐛 문제 해결

### "Pinecone is disabled"
→ `.env` 파일 확인:
```bash
CHATBOT_PINECONE_API_KEY=pcsk_...
USE_RAG_MEMORY=true
USE_PINECONE_MEMORY=true
```

### "Index not found: chatbot-memory-index"
→ Pinecone 인덱스 생성:
```bash
python create_pinecone_index.py
```

### "Memory retrieval failed: 404"
→ Spring Boot API 엔드포인트 확인:
- GET `/api/chat/history/child/{childId}`
- GET `/api/story/completions/child/{childId}`

### AI가 과거 기록을 참조하지 않음
→ FastAPI 로그 확인:
```
=== Memory Retrieval Start ===
✅ Memory retrieved: X recent, Y similar, Z stories
```
로그가 없으면 `USE_RAG_MEMORY=true` 설정 확인

### "Failed to get embedding"
→ OpenAI API Key 확인:
```bash
OPENAI_API_KEY=sk-proj-...
```

---

## 📈 성능 최적화

### MySQL 인덱스 추가
```sql
-- 채팅 메시지 조회 속도 향상
CREATE INDEX idx_chat_session_child_id ON chat_session(child_id);
CREATE INDEX idx_chat_message_created_at ON chat_message(created_at DESC);

-- 동화 완료 조회 속도 향상
CREATE INDEX idx_story_completion_child_created ON story_completion(child_id, created_at DESC);
```

### Pinecone 쿼리 최적화
```python
# memory_service.py에서 top_k 조정 (기본값: 5)
results = self.index.query(
    vector=embedding,
    filter={"child_id": child_id},
    top_k=3,  # 5 → 3으로 줄여서 속도 향상
    include_metadata=True
)
```

---

## 🎉 완료!

이제 DinoCharacter가:
- ✅ 과거 대화 기억
- ✅ 완료한 동화 기록 참조
- ✅ 시맨틱 검색으로 유사한 대화 찾기
- ✅ 개인화된 대화 제공

---

## 📞 다음 단계

1. ✅ Pinecone 인덱스 생성
2. ✅ Spring Boot API 엔드포인트 추가
3. ✅ FastAPI 재시작 및 테스트
4. ⚙️ (옵션) Pinecone 자동 동기화 설정
5. 🚀 프로덕션 배포

---

## 📚 참고 문서

- **아키텍처**: `RAG_SETUP_GUIDE.md`
- **구현 가이드**: `IMPLEMENTATION_GUIDE.md`
- **Spring Boot 코드**: `dinory-backend/REQUIRED_API_ENDPOINTS.md`
- **Pinecone Docs**: https://docs.pinecone.io
