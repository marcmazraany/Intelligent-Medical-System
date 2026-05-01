from fastapi import HTTPException
from paddleocr import PaddleOCR
import cv2
import numpy as np
from typing import List, Dict, Any
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
ocr_model = None


def get_ocr_model():
    """Lazy loading of OCR model using local model folders."""
    global ocr_model

    if ocr_model is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

        det_path = os.path.join(base_dir, "models", "en_PP-OCRv3_det_infer")
        rec_path = os.path.join(base_dir, "models", "en_PP-OCRv3_rec_infer")
        cls_path = os.path.join(base_dir, "models", "ch_ppocr_mobile_v2.0_cls_infer")

        if not os.path.isdir(det_path):
            raise HTTPException(
                status_code=500,
                detail=f"Detection model folder not found: {det_path}"
            )

        if not os.path.isdir(rec_path):
            raise HTTPException(
                status_code=500,
                detail=f"Recognition model folder not found: {rec_path}"
            )

        if not os.path.isdir(cls_path):
            raise HTTPException(
                status_code=500,
                detail=f"Classification model folder not found: {cls_path}"
            )

        ocr_model = PaddleOCR(
            det_model_dir=det_path,
            rec_model_dir=rec_path,
            cls_model_dir=cls_path,
            use_angle_cls=True,
            lang="en",
            show_log=True
        )

    return ocr_model


def detect_text_from_image(image_bytes: bytes) -> List[Dict[str, Any]]:
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image")

        ocr = get_ocr_model()

        # Use classification because your images perform better with it
        result = ocr.ocr(img, cls=True)

        detected_texts = []

        if result and len(result) > 0 and result[0]:
            for line in result[0]:
                # line format:
                # [box, (text, confidence)]
                if len(line) >= 2 and isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                    text = line[1][0]
                    score = line[1][1]

                    detected_texts.append({
                        "text": text,
                        "confidence": float(score)
                    })

        return detected_texts

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


def extract_drug_infos_with_gpt(detected_text: List[Dict[str, Any]]) -> List[Dict]:
    text = " ".join([d["text"] for d in detected_text if d.get("text")])

    if not text.strip():
        return []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is missing.")
        return []

    client = OpenAI(api_key=api_key)

    prompt = f"""
    You are an expert pharmacist and data extractor. Extract drug package information from the following OCR text. 
    The text may contain information for one or more drug packages.
    
    CRITICAL RULES:
    1. ONLY extract actual medication names. Do NOT create objects for random words, instructions (e.g., "Keep out of reach", "Caution"), ingredients, or company names unless they are clearly tied to a specific medication brand name.
    2. A valid drug object MUST have a recognizable "Brand Name". If you only see numbers, stray letters, or general text, DO NOT include it as a drug.
    3. If no clear medication package is identified, return an empty array for "drugs".
    
    Return a JSON object with a single key "drugs" containing an array of objects. 
    Each object must have the exact following keys:
    "Brand Name", "Manufacturer", "Quantity", "Form", "Dosage".
    If a value is not found, use an empty string.
    
    OCR Text:
    {text}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output strict JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=300,
        )

        content = response.choices[0].message.content
        data = json.loads(content)
        return data.get("drugs", [])

    except Exception as e:
        print(f"Error calling GPT API: {e}")
        return []