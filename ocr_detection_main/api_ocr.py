from typing import Dict

from fastapi import FastAPI, File, HTTPException, UploadFile

from Text_Detection_Function import detect_text_from_image, extract_drug_infos_with_gpt


app = FastAPI(
    title="Drug OCR-Only API",
    description="Upload an image and get OCR-based drug recognition only.",
    version="1.0.0",
)


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "OCR API running"}


@app.post("/ocr-info", response_model=Dict)
async def get_ocr_info(file: UploadFile = File(...)) -> Dict:
    try:
        image_bytes = await file.read()
        detected_texts = detect_text_from_image(image_bytes)
        extracted_drugs = extract_drug_infos_with_gpt(detected_texts)

        return {
            "success": True,
            "source": "text_detection",
            "detected_count": len(extracted_drugs) if extracted_drugs else 0,
            "Drug Details": extracted_drugs if extracted_drugs else [],
            "detected_texts": detected_texts,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error: {str(exc)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8001)
