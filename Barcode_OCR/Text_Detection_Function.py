from fastapi import HTTPException
from paddleocr import PaddleOCR
import cv2
import numpy as np
from typing import List, Dict
import pandas as pd
import re
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
ocr_model = None

def get_ocr_model():
    """Lazy loading of OCR model"""
    global ocr_model
    if ocr_model is None:
        ocr_model = PaddleOCR(lang='en')
    return ocr_model

def detect_text_from_image(image_bytes: bytes) -> List[Dict[str, any]]:
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        ocr = get_ocr_model()
        result = ocr.predict(img)
        detected_texts = []
        
        if result and result[0]:
            data = result[0]    
            if 'rec_texts' in data:
                texts = data['rec_texts']
                scores = data.get('rec_scores', [])
                for text, score in zip(texts, scores):
                    detected_texts.append({
                        "text": text,
                        "confidence": float(score)
                    })
        return detected_texts
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
def extract_drug_infos_with_gpt(detected_text) -> List[Dict]:
    # join all detected text into a single string
    text = ' '.join([d['text'] for d in detected_text])
    
    if not text.strip():
        return []
        
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
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
            response_format={ "type": "json_object" },
            max_completion_tokens=300,
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        return data.get("drugs", [])
    except Exception as e:
        print(f"Error calling GPT API: {e}")
        return []
