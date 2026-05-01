package com.intellimeds.api.medications.dto;

import lombok.Builder;

import java.time.LocalDate;
import java.util.List;

@Builder
public record MedicationDto(
        String id,
        String name,
        String dosage,
        LocalDate expiryDate,
        String frequency,
        Integer quantity,
        List<String> reminderTimes,
        String status,
        String notes
) {}