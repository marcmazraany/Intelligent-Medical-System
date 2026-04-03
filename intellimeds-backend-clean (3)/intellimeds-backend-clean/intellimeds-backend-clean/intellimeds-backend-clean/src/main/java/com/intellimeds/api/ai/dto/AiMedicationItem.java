package com.intellimeds.api.ai.dto;

public record AiMedicationItem(
        String name,
        String dosage,
        Integer quantity,
        String frequency
) {}