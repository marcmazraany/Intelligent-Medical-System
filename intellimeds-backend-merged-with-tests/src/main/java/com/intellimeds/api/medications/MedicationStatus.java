package com.intellimeds.api.medications;

public enum MedicationStatus {
    available,
    low_stock,
    expiring_soon,
    needs_review  // New: For OCR-detected medications missing critical info (like expiry date)
}