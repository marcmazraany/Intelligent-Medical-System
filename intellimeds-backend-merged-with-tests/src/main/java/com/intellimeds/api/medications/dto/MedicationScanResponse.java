package com.intellimeds.api.medications.dto;

import lombok.Builder;

import java.util.Map;

@Builder
public record MedicationScanResponse(
        String source,
        String gtin,
        String name,
        String manufacturer,
        String dosage,
        String quantity,
        String form,
        String expiryDate,
        Map<String, Object> raw
) {}