package com.intellimeds.api.ai.dto;

import jakarta.validation.constraints.NotBlank;

public record AiAnalyzeRequest(
        @NotBlank(message = "message is required")
        String message,

        String conversationId
) {}