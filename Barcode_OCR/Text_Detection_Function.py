from fastapi import HTTPException
from paddleocr import PaddleOCR
import cv2
import numpy as np
from typing import List, Dict
import pandas as pd
import re
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
    
def return_name(text) -> str:
    # iterate through each word in text and return the first word found in brand_names.csv
    brand_names_df = pd.read_csv("csvs/brand_names.csv")
    brand_names = set(brand_names_df['brand_name'].str.lower().tolist())
    words = text.lower().split()
    for word in words:
        if word in brand_names:
            return word
        

def return_quantity(text) -> str:
    # search for patterns like '30 tablets', '100 ml', '20 capsules' in text
    pattern = r'(\b\d+\s*(?:tablets?|capsules?|tabs?|caps?|vials?))'
    matches = re.findall(pattern, text.lower())
    if matches:
        return ' '.join(matches[0])
    return ""

def return_dosage(text) -> str:
    # search for patterns like '500 mg', '250 mcg', '5 ml' in text
    pattern = r'(\d+(\.\d+)?)\s*(mg|mcg|g|ml|units|IU)'
    matches = re.findall(pattern, text.lower())
    if matches:
        return ' '.join(matches[0][:3])
    return ""

def return_description(detected_text):
    # join all detected text into a single string
    text = ' '.join([d['text'] for d in detected_text])
    desc = {"name": return_name(text),
            "quantity": return_quantity(text),
            "dosage": return_dosage(text)}
    return desc
