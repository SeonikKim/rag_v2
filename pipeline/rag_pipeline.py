import numpy as np
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from transformers import AutoTokenizer, AutoModel
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection


def pdf_to_images(pdf_path: str):
    """PDF 파일을 PIL 이미지 목록으로 변환합니다."""
    return convert_from_path(pdf_path)


def run_ocr(images):
    """각 이미지 페이지에서 PaddleOCR을 실행합니다."""
    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    results = []
    for page_no, img in enumerate(images, start=1):
        page_result = ocr.ocr(np.array(img), cls=True)
        results.append({"page": page_no, "result": page_result})
    return results


def embed_layoutlm(ocr_results):
    """OCR 결과의 각 줄에 대해 LayoutLMv3-base 임베딩을 생성합니다."""
    tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")
    model = AutoModel.from_pretrained("microsoft/layoutlmv3-base")
    texts, embeddings = [], []
    for page in ocr_results:
        for line in page["result"]:
            text = line[1][0]
            enc = tokenizer(text, return_tensors="pt")
            outputs = model(**enc)
            emb = outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]
            texts.append(text)
            embeddings.append(emb.tolist())
    return texts, embeddings


def init_milvus(collection_name: str = "documents", dim: int = 768):
    """Milvus에 연결하고 컬렉션이 없으면 생성합니다."""
    connections.connect(alias="default", host="localhost", port="19530")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields)
    if collection_name not in connections.list_collections():
        Collection(name=collection_name, schema=schema)
    return Collection(collection_name)


def insert_milvus(collection: Collection, texts, embeddings):
    """임베딩을 Milvus에 삽입합니다."""
    collection.insert([texts, embeddings])
    collection.flush()


def process_pdf(pdf_path: str):
    """PDF를 처리하여 텍스트 임베딩을 저장합니다."""
    images = pdf_to_images(pdf_path)
    ocr_results = run_ocr(images)
    texts, embeddings = embed_layoutlm(ocr_results)
    collection = init_milvus()
    insert_milvus(collection, texts, embeddings)
    return len(texts)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python rag_pipeline.py <pdf-path>")
        raise SystemExit(1)
    count = process_pdf(sys.argv[1])
    print(f"Milvus에 {count}개의 세그먼트를 삽입했습니다")
