# RAG v2

PDF→OCR→LayoutLMv3→Milvus 파이프라인과 Go 백엔드, Angular 프론트엔드 골격.

## 개발 환경
- CPU: Intel(R) Core(TM) i5-14600KF 3.50 GHz
- RAM: 64GB
- GPU: NVIDIA GeForce RTX 5060 Ti

## 파이썬 파이프라인

```bash
pip install transformers==4.39.0 paddleocr pdf2image pymilvus
python pipeline/rag_pipeline.py sample.pdf
```

## Go 백엔드

```bash
cd backend
# go mod init example.com/rag && go get github.com/gin-gonic/gin
go run main.go
```

## Angular 프론트엔드

`frontend` 디렉터리는 최소한의 Angular 구성 요소를 포함한다. Angular 프로젝트를 생성한 뒤 파일을 복사한다:

```bash
npm install -g @angular/cli
ng new rag-ui
# frontend/src를 rag-ui/src로 복사
```
