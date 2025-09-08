# RAG v2

Skeleton for a PDF→OCR→LayoutLMv3→Milvus pipeline with a Go backend and Angular frontend.

## Python pipeline

```bash
pip install transformers==4.39.0 paddleocr pdf2image pymilvus
python pipeline/rag_pipeline.py sample.pdf
```

## Go backend

```bash
cd backend
# go mod init example.com/rag && go get github.com/gin-gonic/gin
go run main.go
```

## Angular frontend

The `frontend` directory contains minimal Angular components. Create an Angular project and copy the files:

```bash
npm install -g @angular/cli
ng new rag-ui
# copy frontend/src into rag-ui/src
```
