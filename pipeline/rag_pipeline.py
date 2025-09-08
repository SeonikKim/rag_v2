import numpy as np
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from transformers import AutoTokenizer, AutoModel
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection


def pdf_to_images(pdf_path: str):
    """Convert a PDF into a list of PIL images."""
    return convert_from_path(pdf_path)


def run_ocr(images):
    """Run PaddleOCR on each image page."""
    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    results = []
    for page_no, img in enumerate(images, start=1):
        page_result = ocr.ocr(np.array(img), cls=True)
        results.append({"page": page_no, "result": page_result})
    return results


def embed_layoutlm(ocr_results):
    """Generate embeddings with LayoutLMv3-base for each OCR line."""
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
    """Connect to Milvus and create a collection if it doesn't exist."""
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
    """Insert embeddings into Milvus."""
    collection.insert([texts, embeddings])
    collection.flush()


def process_pdf(pdf_path: str):
    images = pdf_to_images(pdf_path)
    ocr_results = run_ocr(images)
    texts, embeddings = embed_layoutlm(ocr_results)
    collection = init_milvus()
    insert_milvus(collection, texts, embeddings)
    return len(texts)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rag_pipeline.py <pdf-path>")
        raise SystemExit(1)
    count = process_pdf(sys.argv[1])
    print(f"Inserted {count} segments into Milvus")
