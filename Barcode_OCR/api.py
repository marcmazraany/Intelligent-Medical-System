from typing import Dict, List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from barcode import barcode_infos, load_and_clean_drug_excel, lookup_drug_by_gtin


app = FastAPI(
    title="Drug Barcode-Only API",
    description="Upload an image and get barcode-based drug recognition only.",
    version="1.0.0",
)

drug_db = load_and_clean_drug_excel("Gtin_db/GTIN.xls")


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "API running"}


@app.post("/barcode-info", response_model=Dict)
async def get_barcode_info(file: UploadFile = File(...)) -> Dict:
    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode uploaded image")

        infos = barcode_infos(img)
        if not infos:
            infos = []

        detections: List[Dict] = []
        for info in infos:
            gtin = info.get("gtin14")
            if not gtin:
                continue

            drug = lookup_drug_by_gtin(gtin, drug_db)
            if not drug:
                continue

            detections.append(
                {
                    "GTIN": gtin,
                    "Brand Name": drug.get("Brand name", "Unknown"),
                    "Manufacturer": drug.get("Manufacturer", "Unknown"),
                    "Quantity": drug.get("Presentation", "Unknown"),
                    "Form": drug.get("Form", "Unknown"),
                    "Dosage": drug.get("Strength", "Unknown"),
                    "Expiry Date": info.get("expiry_date", "Unknown"),
                }
            )

        # Barcode first: if we got a valid brand name, return immediately.
        if detections:
            return {
                "success": True,
                "source": "barcode",
                "detected_count": len(detections),
                "Drug Details": detections,
            }

        raise ValueError("Could not retrieve drug information from barcode")

    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error: {str(exc)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)