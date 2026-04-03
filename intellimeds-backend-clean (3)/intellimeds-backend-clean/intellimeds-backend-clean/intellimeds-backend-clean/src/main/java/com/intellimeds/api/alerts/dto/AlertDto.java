package com.intellimeds.api.alerts.dto;

import lombok.Builder;

import java.math.BigDecimal;

@Builder
public record AlertDto(
        String     id,
        String     medicationName,
        BigDecimal maxPrice,
        boolean    emailEnabled,
        String     createdDate,
        String     lastNotified,
        String     status,
        boolean    active,
        Double     maxDistance   // km radius, nullable = no distance filter
) {}
