# RAG Memory System Setup Guide

DinoCharacter 챗봇에 장기 메모리 기능을 추가하기 위한 설정 가이드

---

## 📋 목차

1. [아키텍처 개요](#아키텍처-개요)
2. [Option A: MySQL만 사용 (추천)](#option-a-mysql만-사용-추천)
3. [Option B: Pinecone 벡터 DB 추가](#option-b-pinecone-벡터-db-추가)
4. [Option C: 하이브리드 (MySQL + Pinecone)](#option-c-하이브리드-mysql--pinecone)
5. [Spring Boot API 엔드포인트 추가](#spring-boot-api-엔드포인트-추가)
6. [FastAPI 적용 방법](#fastapi-적용-방법)

---

## 아키텍처 개요

### 현재 시스템
```
[Frontend] → [Spring Boot] → [MySQL]
                ↓
           [FastAPI AI]
```

### RAG 시스템 추가
```
[Frontend] → [Spring Boot] → [MySQL]
                ↓              ↑
           [FastAPI AI] ←──────┘
                ↓
           [Pinecone] (옵션)
```

### 데이터 흐름

**대화 시:**
1. 사용자 메시지 → Spring Boot → MySQL 저장
2. FastAPI AI ← Spring Boot API 호출하여 과거 대화/동화 기록 조회
3. FastAPI AI → MySQL 데이터 + (옵션) Pinecone 시맨틱 검색
4. 컨텍스트 통합 → OpenAI GPT 호출
5. AI 응답 → 사용자

---

## Option A: MySQL만 사용 (추천)

### 장점
- ✅ 추가 인프라 불필요
- ✅ 빠른 구현
- ✅ 비용 절감
- ✅ 데이터 동기화 불필요

### 단점
- ❌ 키워드 검색만 가능 (시맨틱 검색 X)
- ❌ 대화량이 많아지면 느려질 수 있음

### 적합한 경우
- 사용자당 대화량이 적음 (수백 개 이하)
- 최근 대화만 참조하면 충분
- 빠른 MVP 구현 필요

### 설정 방법

**1. Spring Boot API 엔드포인트 추가** (아래 섹션 참고)

**2. FastAPI 설정**
```python
# chat.py에서 서비스 초기화
from app.services.chat.chatbot_service_with_rag import ChatbotServiceWithRAG

# MySQL만 사용 (Pinecone 비활성화)
chatbot_service = ChatbotServiceWithRAG(use_pinecone=False)
```

**3. .env 설정**
```bash
SPRING_API_URL=http://localhost:8090/api  # 이미 설정됨
```

끝! 추가 설정 불필요.

---

## Option B: Pinecone 벡터 DB 추가

### 장점
- ✅ 시맨틱 검색 (의미 기반 유사도)
- ✅ 대량 데이터에서도 빠른 검색
- ✅ "지난번에 용기 관련 동화 읽었을 때..." 같은 복잡한 질문 가능

### 단점
- ❌ 추가 서비스 비용
- ❌ 데이터 동기화 필요 (MySQL → Pinecone)
- ❌ 복잡한 설정

### 적합한 경우
- 사용자당 대화량이 많음 (수천 개 이상)
- 시맨틱 검색이 중요함
- "예전에 비슷한 얘기 했었는데..." 같은 기능 필요

### 설정 방법

**1. Pinecone 인덱스 생성**

Pinecone Console (https://app.pinecone.io)에서:
```
Index Name: chatbot-memory-index
Dimensions: 1536
Metric: cosine
Cloud: AWS (또는 GCP)
Region: us-east-1 (가장 가까운 지역)
```

**2. .env 설정**
```bash
CHATBOT_PINECONE_API_KEY=pcsk_YOUR_API_KEY_HERE
CHATBOT_PINECONE_INDEX_NAME=chatbot-memory-index
```

**3. FastAPI 설정**
```python
# Pinecone 활성화
chatbot_service = ChatbotServiceWithRAG(use_pinecone=True)
```

**4. 데이터 동기화 설정**

Spring Boot에서 채팅 메시지 저장 후 FastAPI 호출:
```java
// ChatService.java의 sendMessage() 메서드에 추가
@Transactional
public ChatResponseDto sendMessage(ChatMessageRequest request) {
    // ... 기존 코드 ...

    chatMessageRepository.save(aiMessage);

    // Pinecone 동기화 (비동기)
    syncToPinecone(userMessage, aiMessage);

    return response;
}

private void syncToPinecone(ChatMessage userMsg, ChatMessage aiMsg) {
    try {
        // FastAPI /api/chat/sync-to-pinecone 호출
        // (아래 엔드포인트 추가 필요)
    } catch (Exception e) {
        log.warn("Pinecone sync failed (non-critical): ", e);
    }
}
```

---

## Option C: 하이브리드 (MySQL + Pinecone)

### 추천 전략
- **MySQL**: 최근 10개 대화 (빠른 조회)
- **Pinecone**: 전체 히스토리에서 유사한 대화 검색 (스마트)

### 설정
Option A + Option B 모두 적용

```python
# 하이브리드 활성화
chatbot_service = ChatbotServiceWithRAG(use_pinecone=True)

# memory_service.py에서 자동으로:
# 1. MySQL에서 최근 대화 가져옴
# 2. Pinecone에서 유사한 대화 검색
# 3. 두 결과를 통합하여 AI에게 제공
```

---

## Spring Boot API 엔드포인트 추가

FastAPI의 `MemoryService`가 MySQL 데이터를 조회하려면 Spring Boot에 다음 엔드포인트가 필요합니다.

### 필요한 엔드포인트

#### 1. 최근 대화 기록 조회
```java
// ChatController.java

@GetMapping("/chat/history/child/{childId}")
public List<ChatMessageDto> getChatHistoryByChild(
    @PathVariable Long childId,
    @RequestParam(defaultValue = "10") int limit
) {
    List<ChatMessage> messages = chatMessageRepository
        .findTop10ByChildIdOrderByCreatedAtDesc(childId, PageRequest.of(0, limit));

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
```java
// StoryController.java (또는 ChatController.java)

@GetMapping("/story/completions/child/{childId}")
public List<StoryCompletionSummaryDto> getStoryCompletionsByChild(
    @PathVariable Long childId,
    @RequestParam(defaultValue = "5") int limit
) {
    List<StoryCompletion> completions = storyCompletionRepository
        .findTopNByChildIdOrderByCreatedAtDesc(childId, PageRequest.of(0, limit));

    return completions.stream()
        .map(StoryCompletionSummaryDto::from)
        .collect(Collectors.toList());
}
```

### Repository 메서드 추가

```java
// ChatMessageRepository.java
public interface ChatMessageRepository extends JpaRepository<ChatMessage, Long> {
    // 기존 메서드들...

    @Query("SELECT cm FROM ChatMessage cm JOIN cm.chatSession cs WHERE cs.childId = :childId ORDER BY cm.createdAt DESC")
    List<ChatMessage> findTop10ByChildIdOrderByCreatedAtDesc(
        @Param("childId") Long childId,
        Pageable pageable
    );
}

// StoryCompletionRepository.java
public interface StoryCompletionRepository extends JpaRepository<StoryCompletion, Long> {
    // 기존 메서드들...

    List<StoryCompletion> findTopNByChildIdOrderByCreatedAtDesc(
        Long childId,
        Pageable pageable
    );
}
```

### DTO 클래스

```java
// ChatMessageDto.java
@Data
@Builder
public class ChatMessageDto {
    private Long sessionId;
    private String message;
    private String sender;  // "USER" or "AI"
    private LocalDateTime createdAt;
}
```

---

## FastAPI 적용 방법

### 1. 기존 chatbot_service.py 대체

**옵션 1: 완전 대체**
```bash
# 기존 파일 백업
mv app/services/chat/chatbot_service.py app/services/chat/chatbot_service_old.py

# 새 파일을 기본으로 사용
mv app/services/chat/chatbot_service_with_rag.py app/services/chat/chatbot_service.py
```

**옵션 2: 점진적 전환** (추천)
```python
# chat.py에서 조건부 사용
import os
from app.services.chat.chatbot_service import ChatbotService  # 기존
from app.services.chat.chatbot_service_with_rag import ChatbotServiceWithRAG  # 새로운

# 환경변수로 제어
USE_RAG = os.getenv("USE_RAG_MEMORY", "false").lower() == "true"

def get_chatbot_service():
    global _chatbot_service
    if _chatbot_service is None:
        if USE_RAG:
            _chatbot_service = ChatbotServiceWithRAG(use_pinecone=False)
        else:
            _chatbot_service = ChatbotService()
    return _chatbot_service
```

**.env에 추가**
```bash
USE_RAG_MEMORY=true  # RAG 활성화
```

### 2. 의존성 확인

이미 `requirements.txt`에 있음:
```txt
httpx==0.28.1       # MySQL API 호출용
pinecone==5.4.2     # Pinecone용 (옵션)
openai==1.55.3      # Embedding 생성용
```

### 3. 테스트

```bash
# FastAPI 재시작
cd d:\react\3stproject\dinory-ai
python main.py

# 테스트 요청
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": 1,
    "message": "지난번에 뭐 읽었어?",
    "child_id": 1
  }'
```

로그에서 확인:
```
=== Memory Retrieval Start ===
child_id: 1, session_id: 1
✅ Memory retrieved: 10 recent, 0 similar, 3 stories
```

---

## 성능 비교

### MySQL만 사용
- **조회 속도**: ~50-100ms (인덱스 있을 때)
- **정확도**: 최근 대화만 가능
- **비용**: 무료
- **추천**: 초기 MVP, 소규모 서비스

### Pinecone 추가
- **조회 속도**: ~100-200ms (임베딩 생성 포함)
- **정확도**: 의미 기반 유사도 검색
- **비용**: 월 $70~ (100K vectors, 1 pod)
- **추천**: 대규모 서비스, 고급 기능 필요

---

## 다음 단계

### 즉시 시작하려면 (Option A)
1. ✅ Spring Boot에 2개 엔드포인트 추가
2. ✅ FastAPI `USE_RAG_MEMORY=true` 설정
3. ✅ 테스트

### Pinecone 추가하려면 (Option B)
1. ✅ Pinecone 인덱스 생성
2. ✅ .env에 API KEY 입력
3. ✅ `use_pinecone=True` 설정
4. ✅ 데이터 동기화 로직 추가

---

## 문제 해결

### "Memory retrieval failed"
→ Spring Boot API 엔드포인트가 없음. 위 섹션 참고하여 추가.

### "Pinecone disabled"
→ `.env`에 `CHATBOT_PINECONE_API_KEY` 설정 확인.

### "No similar conversations found"
→ Pinecone에 데이터가 없음. 동기화 로직 확인.

---

## 요약

| 기능 | Option A (MySQL) | Option B (Pinecone) | Option C (Hybrid) |
|------|-----------------|---------------------|-------------------|
| 최근 대화 기억 | ✅ | ✅ | ✅ |
| 동화 기록 참조 | ✅ | ✅ | ✅ |
| 시맨틱 검색 | ❌ | ✅ | ✅ |
| 추가 비용 | ❌ | ✅ | ✅ |
| 구현 난이도 | 쉬움 | 중간 | 중간 |
| 추천 사용 | MVP, 소규모 | 대규모 | 대규모 |

**첫 시작은 Option A (MySQL만)로, 필요시 나중에 Pinecone 추가 권장!**
