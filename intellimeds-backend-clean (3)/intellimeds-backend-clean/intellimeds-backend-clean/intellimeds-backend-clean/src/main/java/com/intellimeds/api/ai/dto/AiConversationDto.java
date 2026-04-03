package com.intellimeds.api.ai.dto;

import lombok.Builder;

import java.time.Instant;

@Builder
public record AiConversationDto(
        String id,
        String title,
        Instant createdAt,
        Instant updatedAt
) {}