package com.intellimeds.api.alerts.dto;

import java.math.BigDecimal;

public record PatchAlertRequest(
        Boolean    active,
        Boolean    emailEnabled,
        BigDecimal maxPrice,
        String     status,
        Double     maxDistance   // optional km radius
) {}
