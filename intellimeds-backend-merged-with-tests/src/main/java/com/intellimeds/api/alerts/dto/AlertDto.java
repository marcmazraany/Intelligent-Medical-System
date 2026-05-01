package com.intellimeds.api.alerts.dto;

import lombok.Builder;

import java.math.BigDecimal;
import java.time.Instant;

@Builder
public record AlertDto(
        String id,
        String medicationName,
        BigDecimal maxPrice,
        boolean emailEnabled,
        String createdDate,
        String lastNotified,
        String status,
        boolean active
) {}
