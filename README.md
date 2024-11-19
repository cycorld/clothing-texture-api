# 의류 이미지 처리 API

의류 텍스처를 제품 이미지에 자동으로 적용하는 AI 기반 API 서비스입니다.

## 주요 기능

- 의류 이미지에서 텍스처 패턴 자동 추출
- 제품 이미지에 텍스처 자연스럽게 매핑
- RESTful API를 통한 이미지 처리 서비스 제공
- 투명 배경 지원 (PNG 알파 채널)
- 디버그 모드를 통한 처리 과정 시각화

## 시작하기

### 사전 요구사항

- Python 3.8 이상
- OpenCV 호환 운영체제
- 가상환경 사용 권장

### 설치 방법

1. 저장소 복제
```bash
git clone https://github.com/username/clothing-texture-api.git
cd clothing-texture-api
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv

# Linux/Mac의 경우
source venv/bin/activate

# Windows의 경우
.\venv\Scripts\activate
```

3. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

## API 사용법

### 서버 실행
```bash
# 기본 실행 (8000번 포트)
uvicorn api.main:app --reload

# 다른 포트로 실행 (예: 9000번 포트)
uvicorn api.main:app --reload --port 9000

# 호스트 지정과 함께 실행 (외부 접속 허용)
uvicorn api.main:app --reload --host 0.0.0.0 --port 9000
```

### 데몬 서버 임시로 켜기
```bash
# 백그라운드로 실행
nohup uvicorn api.main:app --host 0.0.0.0 --port 9000 > api.log 2>&1 &

# 프로세스 확인
ps aux | grep uvicorn

# 종료하려면 (PID는 ps 명령어로 확인한 프로세스 ID)
kill <PID>
```

### 주요 uvicorn 옵션
- `--port`: 포트 번호 지정 (기본값: 8000)
- `--host`: 호스트 주소 지정 (기본값: 127.0.0.1)
- `--reload`: 코드 변경 시 자동 재시작
- `--workers`: 워커 프로세스 수 지정

### API 엔드포인트

#### 1. 상태 확인
- 엔드포인트: `GET /`
- 설명: API 서버 상태 확인

#### 2. 이미지 처리
- 엔드포인트: `POST /process-image/`
- Content-Type: `multipart/form-data`
- 요청 파라미터:
  - `clothing_image`: 의류 텍스처 이미지 파일 (JPG/PNG)
  - `product_name`: 제품 이미지 이름 (확장자 제외)
- 응답: 처리된 이미지 (PNG)

### 이미지 요구사항

- 제품 이미지:
  - 형식: PNG
  - 위치: 프로젝트 루트 디렉토리
  - 파일명: `{product_name}.png`

- 마스크 이미지:
  - 형식: PNG
  - 위치: 제품 이미지와 동일한 디렉토리
  - 파일명: `{product_name}-mask.png`

- 의류 텍스처:
  - 형식: JPG/PNG
  - 요구사항: 선명한 텍스처 패턴이 있는 영역 포함

## API 테스트

### cURL을 사용한 테스트

1. 서버 상태 확인
```bash
curl http://localhost:8000/
```

2. 이미지 처리 요청
```bash
curl -X POST http://localhost:8000/process-image/ \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "clothing_image=@/path/to/texture.jpg" \
  -F "product_name=sample_product" \
  -o result.png
```

### Windows에서 테스트
```bash
curl -X POST http://localhost:8000/process-image/ ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "clothing_image=@C:\path\to\texture.jpg" ^
  -F "product_name=sample_product" ^
  -o result.png
```

### Python requests를 사용한 테스트
```python
import requests

url = "http://localhost:8000/process-image/"
files = {
    'clothing_image': ('texture.jpg', open('path/to/texture.jpg', 'rb'), 'image/jpeg')
}
data = {
    'product_name': 'sample_product'
}

response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    with open('result.png', 'wb') as f:
        f.write(response.content)
```

## 개발자 참고사항

### 디버그 모드
- 활성화 시 각 처리 단계별 중간 이미지 저장
- 텍스처 추출 과정 시각화
- 매핑 결과 확인 가능

### 성능 최적화
- 텍스처 추출은 이미지 중앙 50% 영역에서 수행
- PNG 압축 레벨 9 적용
- 메모리 효율적 처리를 위한 임시 파일 관리

### 오류 처리
- 잘못된 이미지 형식에 대한 검증
- 파일 저장/로드 실패 시 명확한 에러 메시지
- 처리 과정 중 예외 발생 시 자동 정리

## 라이선스

MIT License
