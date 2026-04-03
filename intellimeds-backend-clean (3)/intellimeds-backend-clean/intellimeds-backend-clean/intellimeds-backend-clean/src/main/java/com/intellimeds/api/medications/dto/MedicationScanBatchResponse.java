package com.intellimeds.api.medications.dto;

import lombok.Builder;

import java.util.List;
import java.util.Map;

@Builder
public record MedicationScanBatchResponse(
        String source,
        Integer detectedCount,
        List<MedicationScanSaveItem> items,
        Map<String, Object> raw
) {}