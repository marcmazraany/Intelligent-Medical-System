package com.intellimeds.api.medications.dto;

import lombok.Builder;

@Builder
public record MedicationScanSaveResponse(
        String action,
        MedicationDto medication,
        MedicationScanResponse scan
) {}