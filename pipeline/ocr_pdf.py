import os
# OpenMP 충돌 회피
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import re
import fitz  # PyMuPDF
import numpy as np
import cv2
from paddleocr import PaddleOCR
from PIL import Image
from pathlib import Path
from datetime import datetime


# --- 후처리 함수들 ---
def fix_roman_numerals(text: str) -> str:
    """숫자 오인식 → 로마 숫자 교정 (간단 룰 기반)"""
    # 2022 → Ⅱ (예시)
    replacements = {
        "2022": "Ⅱ",
        "2023": "Ⅲ",
        "2024": "Ⅳ",
        "2025": "Ⅴ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def fix_korean_spacing(text: str) -> str:
    """자모 분리 → 붙여쓰기 교정 (연속 한글 자모 붙임)"""
    return text.replace(" ", "")


def normalize_box(x0, y0, x1, y1, page_width, page_height):
    """좌표를 0~1000 범위로 정규화"""
    return [
        int(x0 * 1000 / page_width),
        int(y0 * 1000 / page_height),
        int(x1 * 1000 / page_width),
        int(y1 * 1000 / page_height),
    ]


def main():
    root = Path(__file__).resolve().parents[1]
    pdf_dir = root / "pipeline" / "pdf_in"
    out_dir = root / "pipeline" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = pdf_dir / "test.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF가 없습니다: {pdf_path}")

    # PaddleOCR 초기화 (보완용)
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="korean",
        use_gpu=True,
        det_limit_side_len=2048,
        rec_batch_num=10,  # VRAM 16GB → 배치 크게 가능
        drop_score=0.5
    )

    doc = fitz.open(str(pdf_path))

    ocr_pack = {
        "pdf": str(pdf_path.name),
        "pages": []
    }

    for page_index, page in enumerate(doc):
        words, boxes = [], []

        # --- 1. PyMuPDF 텍스트 시도 ---
        pymupdf_words = page.get_text("words")

        if pymupdf_words:
            for x0, y0, x1, y1, word, *_ in pymupdf_words:
                if not word.strip():
                    continue
                # 후처리 적용
                word = fix_roman_numerals(word)
                word = fix_korean_spacing(word)
                words.append(word)
                boxes.append(normalize_box(x0, y0, x1, y1, page.rect.width, page.rect.height))

            print(f"\n=== Page {page_index} (PyMuPDF 우선) ===")
            print(" ".join(words))

        else:
            # --- 2. PyMuPDF 실패 → PaddleOCR 보완 ---
            pix = page.get_pixmap(dpi=600)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            result = ocr.ocr(np.array(img), cls=True)

            if result and result[0]:
                for line in result[0]:
                    box = line[0]
                    text = line[1][0]
                    if not text.strip():
                        continue
                    text = fix_roman_numerals(text)
                    text = fix_korean_spacing(text)
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    words.append(text)
                    boxes.append(normalize_box(x1, y1, x2, y2, pix.width, pix.height))

            print(f"\n=== Page {page_index} (PaddleOCR 보완) ===")
            print(" ".join(words) if words else "(단어 인식 실패)")

        ocr_pack["pages"].append({
            "page_index": page_index,
            "words": words,
            "boxes": boxes
        })

    # 저장 파일명에 날짜·시간 추가
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"{pdf_path.stem}_hybrid_{timestamp}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ocr_pack, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] OCR JSON 저장: {out_json}")


if __name__ == "__main__":
    main()
