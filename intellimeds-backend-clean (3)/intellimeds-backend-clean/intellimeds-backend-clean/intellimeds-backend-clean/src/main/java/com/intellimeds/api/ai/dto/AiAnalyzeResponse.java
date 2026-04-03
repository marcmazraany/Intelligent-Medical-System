package com.intellimeds.api.ai.dto;

import lombok.Builder;

@Builder
public record AiAnalyzeResponse(
        String conversationId,
        String reply
) {}