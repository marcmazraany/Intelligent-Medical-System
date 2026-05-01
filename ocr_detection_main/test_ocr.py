import os
import glob
from Text_Detection_Function import detect_text_from_image, extract_drug_infos_with_gpt

test_images_dir = "OCR_test_images"

def test():
    image_paths = glob.glob(os.path.join(test_images_dir, "*.jpeg")) + glob.glob(os.path.join(test_images_dir, "*.jpg")) + glob.glob(os.path.join(test_images_dir, "*.png"))
    
    for img_path in image_paths:
        print(f"\nProcessing {img_path}...")
        try:
            with open(img_path, "rb") as f:
                image_bytes = f.read()
            
            print("Detecting text using PaddleOCR...")
            detected_texts = detect_text_from_image(image_bytes)
            
            print("Extracting drug info with GPT...")
            extracted_drugs = extract_drug_infos_with_gpt(detected_texts)
            
            print("Results:")
            import json
            print(json.dumps(extracted_drugs, indent=2))
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

if __name__ == "__main__":
    test()
