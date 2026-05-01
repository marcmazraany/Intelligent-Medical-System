package com.intellimeds.api.medications;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "scanner.service")
public record MedicationScannerProperties(
        String barcodeUrl,  // Full URL: http://localhost:8000/barcode-info
        String ocrUrl       // Full URL: http://localhost:8001/ocr-info
) {}