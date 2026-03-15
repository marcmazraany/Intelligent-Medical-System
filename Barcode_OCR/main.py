from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import Dict
from Text_Detection_Function import detect_text_from_image, return_description
from barcode import barcode_infos, lookup_drug_by_gtin, load_and_clean_drug_excel
import cv2
import pandas as pd
import numpy as np

excel_path = 'Gtin_db/GTIN.xls'
df = pd.read_excel(excel_path, dtype=str)
df = load_and_clean_drug_excel(excel_path)
app = FastAPI(
    title="Text Detection API",
    description="API for detecting and recognizing text in images using PaddleOCR",
    version="1.0.0"
)

ocr_model = None

@app.get("/")
def root():
    return {"Hello": "World"}

# @app.post("/detect-text", response_model=Dict)
# async def detect_text(file: UploadFile = File(...)):
#     try:
#         image_bytes = await file.read()
#         detected_texts = detect_text_from_image(image_bytes)
#         desc = return_description(detected_texts)
#         return {
#             "success": True,
#             "filename": file.filename,
#             "detected_texts": detected_texts,
#             "description": desc
#         }
#     except HTTPException as he:
#         raise he
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
@app.post("/barcode-info", response_model=Dict)
async def get_barcode_info(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        infos = barcode_infos(img)
        if not infos:
            raise ValueError("No barcode data found")

        detections = []
        for info in infos:
            gtin = info.get('gtin14')
            if not gtin:
                continue

            final_info = lookup_drug_by_gtin(gtin, df)
            if not final_info:
                continue

            detections.append(
                {
                    "GTIN": gtin,
                    "Brand Name": final_info.get("Brand name", "Unknown"),
                    "Manufacturer": final_info.get("Manufacturer", "Unknown"),
                    "Quantity": final_info.get("Presentation", "Unknown"),
                    "Form": final_info.get("Form", "Unknown"),
                    "Dosage": final_info.get("Strength", "Unknown"),
                    "Expiry Date": info.get("expiry_date", "Unknown"),
                }
            )

        if not detections:
            raise ValueError("No recognizable GTIN/drug match found in detected barcode(s)")

        return {
            "success": True,
            "detected_count": len(detections),
            "Drug Details": detections,
            }

    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
