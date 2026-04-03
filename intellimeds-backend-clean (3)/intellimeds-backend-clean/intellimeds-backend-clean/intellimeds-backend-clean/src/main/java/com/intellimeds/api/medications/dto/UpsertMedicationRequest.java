package com.intellimeds.api.medications.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

import java.time.LocalDate;
import java.util.List;

public record UpsertMedicationRequest(
        @NotBlank(message = "name must not be blank")
        String name,

        @NotBlank(message = "dosage must not be blank")
        String dosage,

        @NotNull(message = "expiryDate must not be null")
        LocalDate expiryDate,

        @NotBlank(message = "frequency must not be blank")
        String frequency,

        Integer quantity,

        List<String> reminderTimes,

        @NotBlank(message = "status must not be blank")
        String status,

        String notes
) {}