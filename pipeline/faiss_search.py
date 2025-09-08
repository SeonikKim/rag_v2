import numpy as np
import faiss
from pathlib import Path
import json
import torch
from transformers import AutoProcessor, LayoutLMv3Model
from PIL import Image


def encode_query(text, processor, model, device="cpu"):
    dummy_image = Image.new("RGB", (1000, 1000), (255, 255, 255))
    enc = processor(
        images=dummy_image,
        text=[text],
        boxes=[[0, 0, 1000, 1000]],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc)
        cls_embed = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
    return cls_embed.astype("float32")


def search_faiss(query_embed: np.ndarray, index_path: Path, meta_path: Path, top_k=3):
    index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    query_embed = query_embed.reshape(1, -1)
    D, I = index.search(query_embed, top_k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        results.append({
            "page_index": meta[idx]["page_index"],
            "distance": float(dist),
            # "sentence": meta[idx]["sentence"]
            "text": meta[idx].get("sample_text", "")  # ✅ 수정
        })
    return results


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "pipeline" / "out"

    emb_path = next(out_dir.glob("*_layoutlmv3_base_pages.npy"))
    meta_path = next(out_dir.glob("*_layoutlmv3_base_pages.meta.json"))
    faiss_path = out_dir / f"{emb_path.stem}.faiss"

    # 인덱스 빌드 (문장 단위)
    embeds = np.load(emb_path).astype("float32")
    index = faiss.IndexFlatL2(embeds.shape[1])
    index.add(embeds)
    faiss.write_index(index, str(faiss_path))

    # 모델 준비
    device = "cpu"
    model_name = "microsoft/layoutlmv3-base"
    processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
    model = LayoutLMv3Model.from_pretrained(model_name).to(device)
    model.eval()

    # 🔍 검색어 입력
    query = input("검색어를 입력하세요: ")
    query_embed = encode_query(query, processor, model, device)

    results = search_faiss(query_embed, faiss_path, meta_path, top_k=3)
    print("\n=== 검색 결과 ===")
    for r in results:
        print(f"- Page {r['page_index']} (distance={r['distance']:.4f})")
        # print(f"  Sentence: {r['sentence']}")
        print(f"  Sentence: {r['text']}")
