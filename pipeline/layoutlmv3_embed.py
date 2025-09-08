import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, LayoutLMv3Model
import ast


def select_device():
    if torch.cuda.is_available():
        try:
            capability = torch.cuda.get_device_capability()
            print(f"[INFO] CUDA GPU: {torch.cuda.get_device_name(0)}, capability={capability}")
            if capability[0] >= 12:
                print("[WARN] sm_120 이상 GPU는 PyTorch 빌드에서 지원 안 됨 → CPU fallback")
                return "cpu"
            return "cuda"
        except Exception as e:
            print(f"[WARN] GPU 체크 실패 → CPU fallback ({e})")
            return "cpu"
    else:
        return "cpu"


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "pipeline" / "out"

    # OCR JSON 로드 (가장 최근 결과 1개 사용)
    ocr_json = sorted(out_dir.glob("*_ocr*.json"))[-1]
    with open(ocr_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    device = select_device()
    model_name = "microsoft/layoutlmv3-base"
    processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
    model = LayoutLMv3Model.from_pretrained(model_name).to(device)
    model.eval()

    all_embeds, all_meta = [], []
    dummy_image = Image.new("RGB", (1000, 1000), (255, 255, 255))

    for page in data["pages"]:
        words = page.get("words", [])
        boxes = page.get("boxes", [])

        # 타입 보정
        if isinstance(words, str):
            try:
                words = ast.literal_eval(words)
            except Exception:
                words = []
        if words is None:
            words = []
        if boxes is None:
            boxes = []

        # 길이 맞추기
        if len(words) != len(boxes):
            min_len = min(len(words), len(boxes))
            words = words[:min_len]
            boxes = boxes[:min_len]

        if not words:
            continue  # OCR 결과 없는 페이지는 건너뜀

        # LayoutLMv3 인코딩
        enc = processor(
            images=dummy_image,
            text=words,       # ✅ 단어 리스트 그대로 전달
            boxes=boxes,      # ✅ 같은 길이의 박스
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            cls_embed = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

        all_embeds.append(cls_embed)
        all_meta.append({
            "page_index": page["page_index"],
            "sample_text": " ".join(words)[:200]  # 앞부분만 저장
        })

      # 저장
    if not all_embeds:
        print("[ERROR] 임베딩 결과가 없습니다. OCR JSON에 words/boxes가 비어 있는지 확인하세요.")
        return

    emb_path = out_dir / f"{Path(data['pdf']).stem}_layoutlmv3_base_pages.npy"
    np.save(emb_path, np.vstack(all_embeds))

    meta_path = out_dir / f"{Path(data['pdf']).stem}_layoutlmv3_base_pages.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] 페이지 임베딩 저장: {emb_path} (shape={np.vstack(all_embeds).shape})")
    print(f"[OK] 메타 저장: {meta_path}")


if __name__ == "__main__":
    main()
