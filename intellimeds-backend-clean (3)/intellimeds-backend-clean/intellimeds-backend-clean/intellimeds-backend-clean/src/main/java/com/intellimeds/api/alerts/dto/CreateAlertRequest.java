package com.intellimeds.api.alerts.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

import java.math.BigDecimal;

public record CreateAlertRequest(
        @NotBlank String     medicationName,
        @NotNull  BigDecimal maxPrice,
        boolean    emailEnabled,
        Double     maxDistance    // optional km radius
) {}
