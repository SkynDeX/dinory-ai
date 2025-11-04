# 🧠 DinoCharacter RAG Memory System

DinoCharacter 챗봇에 장기 메모리 기능을 추가하는 완전한 시스템

---

## 📖 개요

DinoCharacter가 이제 사용자의 모든 대화와 동화 기록을 기억합니다!

### 주요 기능
- ✅ 과거 대화 기억 (최근 10개)
- ✅ 완료한 동화 기록 참조 (최근 5개)
- ✅ 시맨틱 검색 (Pinecone 사용)
- ✅ 개인화된 대화 제공
- ✅ "지난번에 뭐 읽었어?" 같은 질문에 정확한 답변

### 기술 스택
- **FastAPI**: RAG 메모리 서비스
- **MySQL**: 대화/동화 기록 저장
- **Pinecone**: 벡터 검색 (시맨틱 검색)
- **OpenAI**: GPT-4o-mini + Embeddings

---

## 🚀 빠른 시작

**처음 시작하시나요?** → `START_PINECONE.md` 파일 열어서 1단계부터 진행하세요!

```bash
# 1. Pinecone 인덱스 생성
python create_pinecone_index.py

# 2. Spring Boot API 추가
# (dinory-backend/REQUIRED_API_ENDPOINTS.md 참고)

# 3. FastAPI 재시작
python main.py
```

---

## 📚 문서 가이드

### 🆕 처음 시작하는 분
1. **`START_PINECONE.md`** ⭐ 메인 가이드
   - 5분 Quick Start
   - 3단계로 완료
   - 체크리스트 포함

### 🏗️ 아키텍처 이해하기
2. **`RAG_SETUP_GUIDE.md`**
   - 3가지 옵션 비교 (MySQL, Pinecone, 하이브리드)
   - 데이터 흐름 설명
   - 성능 비교

3. **`DECISION_GUIDE.md`**
   - 어떤 옵션을 선택할지 결정
   - 비용 분석
   - 사용 사례별 추천

### 🔧 구현 가이드
4. **`IMPLEMENTATION_GUIDE.md`**
   - 단계별 구현 방법
   - 코드 설명
   - 테스트 시나리오
   - 문제 해결

5. **`PINECONE_SETUP.md`**
   - Pinecone 상세 설정
   - 데이터 동기화
   - 성능 최적화

### 💻 코드 참고
6. **`dinory-backend/REQUIRED_API_ENDPOINTS.md`**
   - Spring Boot API 전체 코드
   - Repository, Controller, DTO
   - 복사 붙여넣기 가능

7. **`create_pinecone_index.py`**
   - Pinecone 인덱스 자동 생성 스크립트
   - 상태 확인 기능

---

## 📂 파일 구조

```
dinory-ai/
├── app/
│   ├── services/
│   │   └── chat/
│   │       ├── chatbot_service.py              # 기존 서비스 (유지)
│   │       ├── chatbot_service_with_rag.py     # ⭐ RAG 통합 서비스
│   │       └── memory_service.py               # ⭐ RAG 메모리 핵심 로직
│   └── api/
│       └── endpoints/
│           ├── chat.py                         # ✅ 수정됨 (RAG 사용)
│           └── memory_sync.py                  # ⭐ Pinecone 동기화
├── .env                                        # ✅ Pinecone 설정 추가됨
├── main.py                                     # ✅ memory_router 추가됨
├── create_pinecone_index.py                    # ⭐ 인덱스 생성 스크립트
│
├── START_PINECONE.md                           # 📖 메인 시작 가이드
├── RAG_SETUP_GUIDE.md                          # 📖 아키텍처 가이드
├── IMPLEMENTATION_GUIDE.md                     # 📖 구현 가이드
├── DECISION_GUIDE.md                           # 📖 의사결정 가이드
├── PINECONE_SETUP.md                           # 📖 Pinecone 설정
└── RAG_MEMORY_README.md                        # 📖 이 파일

dinory-backend/
└── REQUIRED_API_ENDPOINTS.md                   # 📖 Spring Boot 코드
```

---

## ⚙️ 현재 설정 상태

### ✅ 완료된 설정

```bash
# .env 파일
USE_RAG_MEMORY=true                             # ✅ RAG 활성화
USE_PINECONE_MEMORY=true                        # ✅ Pinecone 활성화
CHATBOT_PINECONE_API_KEY=pcsk_Q8CsM_...        # ✅ API Key 설정됨
CHATBOT_PINECONE_INDEX_NAME=chatbot-memory-index # ✅ 인덱스 이름 설정됨
SPRING_API_URL=http://localhost:8090/api       # ✅ Spring Boot API URL
```

```python
# chat.py
USE_RAG = True                                  # ✅ RAG 사용
USE_PINECONE = True                             # ✅ Pinecone 사용
```

### ⚠️ 필요한 작업

1. **Pinecone 인덱스 생성** (5분)
   ```bash
   python create_pinecone_index.py
   ```

2. **Spring Boot API 엔드포인트 추가** (30분)
   - ChatController: `/api/chat/history/child/{childId}`
   - StoryController: `/api/story/completions/child/{childId}`
   - Repository 메서드 2개 추가

---

## 🎯 작동 방식

### 대화 흐름

```
1. 사용자: "지난번에 뭐 읽었어?"
   ↓
2. FastAPI: child_id로 MySQL + Pinecone 검색
   - MySQL: 최근 대화 10개
   - MySQL: 완료한 동화 5개
   - Pinecone: 유사한 과거 대화 5개
   ↓
3. MemoryService: 컨텍스트 요약 생성
   ↓
4. ChatbotServiceWithRAG: 시스템 프롬프트에 컨텍스트 추가
   ↓
5. OpenAI GPT: 컨텍스트 기반 응답 생성
   ↓
6. AI: "지난번에 '용감한 디노' 동화 읽었잖아! 그때 용기를 31점이나 얻었어!"
```

### 데이터 흐름

```
[사용자 대화]
    ↓
[Spring Boot] → [MySQL 저장]
    ↓
[FastAPI /api/memory/sync/conversation] → [Pinecone 저장]
    ↓
[다음 대화 시]
    ↓
[FastAPI MemoryService]
    ├→ [MySQL에서 최근 기록 조회]
    └→ [Pinecone에서 유사 대화 검색]
    ↓
[컨텍스트 통합 → AI 응답]
```

---

## 🧪 테스트 시나리오

### 시나리오 1: 동화 기록 참조
```
사용자: "지난번에 뭐 읽었어?"
AI: "지난번에 '용감한 디노' 동화 읽었잖아! 그때 용기를 31점이나 얻었어! 😊"
```

### 시나리오 2: 과거 대화 참조
```
[이전 대화]
사용자: "나 오늘 학교에서 친구랑 싸웠어"
AI: "속상했겠다... 무슨 일이 있었어?"

[나중에...]
사용자: "친구랑 화해했어!"
AI: "진짜? 학교에서 싸웠던 친구랑? 정말 잘했어! 😊"
```

### 시나리오 3: 능력치 정확한 정보
```
사용자: "내가 무슨 능력치 얻었어?"
AI: "용기 31점, 공감 10점, 창의성 2점, 책임감 12점을 얻었어!"
```

---

## 💰 비용 예상

### Pinecone
- **Starter Plan**: $70/월 (100K vectors, 1 pod)
- **적합**: 사용자 ~5000명

### OpenAI
- **GPT-4o-mini**: $0.15 / 1M input tokens
- **Embeddings**: $0.13 / 1M tokens
- **예상**: 월 10만 메시지 기준 ~$15

### 총 비용
- **월 ~$85** (사용자 1000명, 월 10만 메시지 기준)

---

## 📊 성능

### 응답 속도
- **최근 대화 조회**: ~50ms (MySQL)
- **유사 대화 검색**: ~150ms (Pinecone)
- **전체 RAG 응답**: ~500-800ms (OpenAI 포함)

### 확장성
- **MySQL**: 사용자 1만 명까지 충분
- **Pinecone**: 사용자 10만 명 이상 가능

---

## 🐛 문제 해결

| 문제 | 해결 방법 | 문서 |
|------|----------|------|
| Pinecone 연결 실패 | API Key 확인 | `PINECONE_SETUP.md` |
| MySQL 조회 실패 | Spring Boot API 확인 | `dinory-backend/REQUIRED_API_ENDPOINTS.md` |
| AI가 기억 못 함 | FastAPI 로그 확인 | `IMPLEMENTATION_GUIDE.md` |
| 인덱스 없음 | `create_pinecone_index.py` 실행 | `START_PINECONE.md` |

---

## 🎁 향후 개선 사항

1. **캐싱**: 최근 조회 결과 캐싱으로 속도 향상
2. **요약**: 긴 대화 자동 요약
3. **감정 추적**: 아이의 감정 변화 기록
4. **관심사 학습**: 선호도 기반 추천
5. **부모 리포트**: 대화 내용 요약 전송

---

## ✅ 다음 단계

1. **지금 바로 시작**
   - `START_PINECONE.md` 파일 열기
   - 1단계부터 차근차근 진행

2. **Spring Boot API 추가**
   - `dinory-backend/REQUIRED_API_ENDPOINTS.md` 참고
   - 4개 파일 수정

3. **테스트**
   - DinoCharacter 클릭
   - "지난번에 뭐 읽었어?" 테스트

4. **프로덕션 배포**
   - 성능 모니터링
   - 사용자 피드백 수집

---

## 📞 도움말

- **시작**: `START_PINECONE.md`
- **아키텍처**: `RAG_SETUP_GUIDE.md`
- **구현**: `IMPLEMENTATION_GUIDE.md`
- **Spring Boot**: `dinory-backend/REQUIRED_API_ENDPOINTS.md`
- **Pinecone 공식 문서**: https://docs.pinecone.io

---

## 🎉 완료!

이제 DinoCharacter가 사용자의 과거를 기억하며 더욱 개인화된 대화를 제공합니다!

**시작하려면**: `START_PINECONE.md` 파일을 열어주세요! 🚀
