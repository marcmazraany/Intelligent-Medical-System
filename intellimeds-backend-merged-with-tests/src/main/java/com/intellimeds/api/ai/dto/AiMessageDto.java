package com.intellimeds.api.ai.dto;

import lombok.Builder;

import java.time.Instant;

@Builder
public record AiMessageDto(
        String id,
        String role,
        String content,
        Instant createdAt
) {}