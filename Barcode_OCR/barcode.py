import cv2
import zxingcpp 
import re
from datetime import date
import pandas as pd
# img = cv2.imread('images/barcode_img.jpeg')
AI_PATTERNS = {
    "01": r"(?P<gtin>\d{14})",      
    "17": r"(?P<expiry>\d{6})",       
    "10": r"(?P<lot>[^\x1D)]+)",      
}

def parse_gs1(text):
    out = {}
    s = text.replace("\x1D", "").strip()
    m01 = re.search(r"\(01\)" + AI_PATTERNS["01"], s)
    if m01:
        out["gtin14"] = m01.group("gtin")
    m17 = re.search(r"\(17\)" + AI_PATTERNS["17"], s)
    if m17:
        yy, mm, dd = m17.group("expiry")[0:2], m17.group("expiry")[2:4], m17.group("expiry")[4:6]
        yyyy = int("20" + yy)  
        out["expiry_date"] = f"{yyyy:04d}-{int(mm):02d}-{int(dd):02d}"
    m10 = re.search(r"\(10\)" + AI_PATTERNS["10"], s)
    if m10:
        out["lot"] = m10.group("lot")
    return out

import numpy as np


def _decode_all_barcodes(img):
    if hasattr(zxingcpp, "read_barcodes"):
        results = zxingcpp.read_barcodes(img)
        return list(results) if results else []

    # Fallback for environments that only expose single-barcode API
    single_result = zxingcpp.read_barcode(img)
    return [single_result] if single_result else []


def _dedupe_results(results):
    seen = set()
    unique = []
    for result in results:
        key = (str(result.format), result.text.strip())
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)
    return unique


def read_barcodes_robust(img):
    attempts = [img]

    # Prepare grayscale once for image transforms
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharp = cv2.filter2D(gray, -1, kernel_sharp)

    attempts.extend([gray, resized, sharp])

    for attempt in attempts:
        results = _decode_all_barcodes(attempt)
        if results:
            return _dedupe_results(results)

    return []

def read_barcode_robust(img):
    results = read_barcodes_robust(img)
    return results[0] if results else None


def _barcode_result_to_info(result):
    if result is None:
        return {}

    # Handle GS1 DataMatrix (Standard case)
    if result.format == zxingcpp.BarcodeFormat.DataMatrix:
        info = parse_gs1(result.text)
        return info
    
    # Handle EAN-13 / UPC (Consumer products, sometimes used as GTIN)
    # EAN-13 is a 13-digit number. GTIN-14 is 14 digits.
    # We pad EAN-13 with a leading zero to make it a valid GTIN-14.
    if result.format in [zxingcpp.BarcodeFormat.EAN13, zxingcpp.BarcodeFormat.UPCA, zxingcpp.BarcodeFormat.EAN8]:
        # Clean text just in case
        text = result.text.strip()
        
        # Simple padding logic
        if len(text) == 13:
            gtin14 = "0" + text
        elif len(text) == 12: # UPC-A
             gtin14 = "00" + text
        else:
            gtin14 = text.zfill(14) # Generic padding

        return {"gtin14": gtin14}

    # Fallback: try parsing as GS1 anyway if it looks like it
    info = parse_gs1(result.text)
    return info


def barcode_infos(img):
    results = read_barcodes_robust(img)
    infos = []
    for result in results:
        infos.append(_barcode_result_to_info(result))
    return infos


def barcode_info(img):
    infos = barcode_infos(img)
    return infos[0] if infos else {}

def load_and_clean_drug_excel(excel_path):
    df = pd.read_excel(excel_path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if 'GTIN' in df.columns:
        df['GTIN'] = (
            df['GTIN']
            .astype(str)
            .str.strip()
            .str.replace(' ', '', regex=False)
            .str.replace(r'\.0$', '', regex=True) 
            .str.lstrip('0')
        )
    return df

def lookup_drug_by_gtin(gtin, df):
    """
    Look up drug details in a preloaded and cleaned DataFrame.
    """
    clean_gtin = str(gtin).strip().lstrip('0')
    result = df.loc[df['GTIN'] == clean_gtin]

    if result.empty:
        return None
    print(result.iloc[0].to_dict())
    return result.iloc[0].to_dict()
# print(barcode_info(img))
# print(lookup_drug_by_gtin(barcode_info(img)['gtin14'], load_and_clean_drug_excel('Gtin_db/GTIN.xls')))