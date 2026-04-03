package com.intellimeds.api.medications.dto;

import lombok.Builder;

@Builder
public record MedicationScanSaveItem(
        String action,
        MedicationDto medication,
        MedicationScanResponse scan
) {}