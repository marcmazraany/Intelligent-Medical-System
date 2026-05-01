import os
import cv2
import zxingcpp
import pandas as pd
from barcode import barcode_infos, lookup_drug_by_gtin, load_and_clean_drug_excel, read_barcodes_robust

drug_db = load_and_clean_drug_excel('Gtin_db/GTIN.xls')
image_folder = 'multiple_barcode_images'
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
brand_names = []
failed_to_read = 0
failed_barcodes = 0
failed_GTIN = 0
failed_BrandName = 0

print("Running Barcode_TB_v2 with read_barcode_robust...")

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    print(f"\nProcessing image: {image_file}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image {image_file}")
        brand_names.append(None)
        failed_to_read += 1
        continue
    
    # Use robust multi-barcode detection
    results = read_barcodes_robust(img)

    if not results:
        print(f"No barcode detected in image {image_file}")
        brand_names.append(None)
        failed_barcodes += 1
        continue

    print(f"Detected {len(results)} barcode(s) in this image.")

    image_infos = barcode_infos(img)
    image_brand_names = []

    for idx, result in enumerate(results):
        print(f"  Barcode {idx + 1}: Text: {result.text[:30]}... Format: {result.format}")
        barcode_info_dict = image_infos[idx] if idx < len(image_infos) else {}

        if 'gtin14' not in barcode_info_dict:
            print("    No GTIN found in barcode.")
            failed_GTIN += 1
            continue

        gtin14 = barcode_info_dict['gtin14']
        print(f"    Found GTIN: {gtin14}")
        drug_info = lookup_drug_by_gtin(gtin14, drug_db)

        if drug_info and 'Brand name' in drug_info:
            brand_name = drug_info['Brand name']
            print(f"    Brand name: {brand_name}")
            image_brand_names.append(brand_name)
        else:
            print(f"    No 'Brand name' found for GTIN: {gtin14}")
            failed_BrandName += 1

    if image_brand_names:
        brand_names.append(image_brand_names)
    else:
        brand_names.append(None)

print("\n------------------------------------------------")
print("Final list of Brand names (per image):")
print(brand_names)
print(brand_names.count(None), "images had no detected Brand name.")
print(len(brand_names) - brand_names.count(None), "images had detected Brand name.")
print("Total images processed:", len(brand_names))
print("Total images failed to read:", failed_to_read)
print("Total images with no barcode detected:", failed_barcodes)
print("Total images with no GTIN found:", failed_GTIN)
print("Total images with no Brand name found:", failed_BrandName)
print("------------------------------------------------")
